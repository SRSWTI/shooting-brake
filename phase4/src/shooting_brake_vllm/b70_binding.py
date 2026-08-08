"""Phase-7 Python ctypes binding for the B70 NVFP4 MoE provider.

Wraps the C ABI shared library (``libsb_b70_provider.so``) so the actual Intel
Arc Pro B70 can compute routed-expert during the vLLM forward pass, in-process
alongside CUDA. SYCL and CUDA coexist (validated: separate driver stacks).

Lifecycle::

    provider = B70ProviderClient(lib_path)
    provider.load(bank_path, generation=1, resident_experts=[128,129,...,255])
    output = provider.dispatch(layer=0, hidden_fp16=..., ids=..., weights=..., M=8)
    provider.shutdown()

The ``dispatch`` method issues + takes synchronously (correctness-first).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


class B70ProviderError(RuntimeError):
    """The B70 provider returned an error."""


class B70ProviderClient:
    """ctypes wrapper around the Phase 1 B70 NVFP4 MoE provider."""

    def __init__(self, lib_path: str | Path) -> None:
        if not os.path.isfile(lib_path):
            raise B70ProviderError(f"B70 provider library not found: {lib_path}")
        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_signatures()
        self._handle: ctypes.c_void_p | None = None
        self._loaded = False
        self._resident_count = 0
        self._sequence = 0

    def _setup_signatures(self) -> None:
        lib = self._lib
        lib.sb_b70_create.restype = ctypes.c_void_p
        lib.sb_b70_create.argtypes = []

        lib.sb_b70_load.restype = ctypes.c_int
        lib.sb_b70_load.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_size_t,
        ]

        lib.sb_b70_issue.restype = ctypes.c_int
        lib.sb_b70_issue.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_size_t, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]

        lib.sb_b70_take.restype = ctypes.c_int
        lib.sb_b70_take.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]

        lib.sb_b70_num_resident.restype = ctypes.c_size_t
        lib.sb_b70_num_resident.argtypes = [ctypes.c_void_p]

        lib.sb_b70_shutdown.restype = None
        lib.sb_b70_shutdown.argtypes = [ctypes.c_void_p]

        lib.sb_b70_destroy.restype = None
        lib.sb_b70_destroy.argtypes = [ctypes.c_void_p]

        lib.sb_b70_poll_create.restype = ctypes.c_void_p
        lib.sb_b70_poll_create.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

        lib.sb_b70_poll_register.restype = ctypes.c_int
        lib.sb_b70_poll_register.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p,          # signal, completion
            ctypes.c_void_p,                           # hidden (fp16)
            ctypes.POINTER(ctypes.c_int32),            # ids
            ctypes.POINTER(ctypes.c_float),            # weights
            ctypes.POINTER(ctypes.c_float),            # output
            ctypes.c_size_t,                           # topk
        ]

        lib.sb_b70_poll_start.restype = ctypes.c_int
        lib.sb_b70_poll_start.argtypes = [ctypes.c_void_p]

        lib.sb_b70_poll_stop.restype = None
        lib.sb_b70_poll_stop.argtypes = [ctypes.c_void_p]

        for counter in (
            "sb_b70_poll_dispatch_count",
            "sb_b70_poll_error_count",
            "sb_b70_poll_service_ns",
            "sb_b70_poll_kernel_ns",
        ):
            fn = getattr(lib, counter)
            fn.restype = ctypes.c_uint64
            fn.argtypes = [ctypes.c_void_p]

        lib.sb_b70_poll_destroy.restype = None
        lib.sb_b70_poll_destroy.argtypes = [ctypes.c_void_p]

    # -- lifecycle -------------------------------------------------------

    def load(
        self,
        bank_path: str | Path,
        generation: int = 1,
        resident_experts: np.ndarray | None = None,
        max_batch: int = 128,
    ) -> None:
        """Create the provider, load the expert bank, select residents."""
        self._handle = ctypes.c_void_p(self._lib.sb_b70_create())
        if not self._handle:
            raise B70ProviderError("sb_b70_create returned NULL")

        bank = str(bank_path).encode("utf-8")
        if resident_experts is not None:
            resident = np.ascontiguousarray(resident_experts, dtype=np.int32)
            ids_ptr = resident.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            count = ctypes.c_size_t(len(resident))
        else:
            ids_ptr = None
            count = ctypes.c_size_t(0)

        status = self._lib.sb_b70_load(
            self._handle, bank, ctypes.c_uint64(generation),
            ids_ptr, count, ctypes.c_size_t(max_batch),
        )
        if status != 0:
            raise B70ProviderError(f"sb_b70_load failed with status {status}")

        self._resident_count = self._lib.sb_b70_num_resident(self._handle)
        self._loaded = True
        # Per-layer resident count
        self._resident_per_layer = self._resident_count // 32

    @property
    def resident_per_layer(self) -> int:
        return self._resident_per_layer

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def lib(self) -> ctypes.CDLL:
        """The loaded shared library, for callers that drive the C ABI
        directly (the Tier 3 native poller)."""
        return self._lib

    @property
    def handle(self) -> ctypes.c_void_p:
        """The provider handle, for callers that drive the C ABI directly."""
        if not self._handle:
            raise B70ProviderError("provider not loaded")
        return self._handle

    # -- compute ---------------------------------------------------------

    def issue(
        self,
        layer: int,
        hidden_fp16: np.ndarray,
        ids: np.ndarray,
        weights: np.ndarray,
        generation: int = 1,
    ) -> int:
        """Submit one MoE dispatch to the B70 (non-blocking kernel launch).

        The SYCL queue enqueues H2D copies + the NVFP4 kernel and returns
        immediately.  Call :meth:`take` with the returned sequence number
        to collect the result once the kernel is done.

        Args:
            layer: NVFP4 layer index (0-31).
            hidden_fp16: [M, 2048] float16 (numpy), row-major, contiguous.
            ids: [M, topk] int32 — compact B70 slot indices, -1 to skip.
            weights: [M, topk] float32 — original routing weights.
            generation: Placement generation id.

        Returns:
            The dispatch sequence number (monotonically increasing).
        """
        if not self._loaded or not self._handle:
            raise B70ProviderError("provider not loaded")

        hidden = np.ascontiguousarray(hidden_fp16, dtype=np.float16)
        ids_c = np.ascontiguousarray(ids, dtype=np.int32)
        weights_c = np.ascontiguousarray(weights, dtype=np.float32)
        M = hidden.shape[0]
        topk = ids_c.shape[1]
        assert hidden.shape[1] == 2048
        assert ids_c.shape == (M, topk)
        assert weights_c.shape == (M, topk)

        self._sequence += 1
        seq = self._sequence

        status = self._lib.sb_b70_issue(
            self._handle,
            ctypes.c_uint64(generation), ctypes.c_uint64(seq),
            ctypes.c_size_t(layer),
            hidden.ctypes.data_as(ctypes.c_void_p),
            ids_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            weights_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(M),
        )
        if status != 0:
            raise B70ProviderError(
                f"sb_b70_issue failed (status={status}, layer={layer}, M={M})"
            )
        return seq

    def take(
        self,
        sequence: int,
        M: int,
        generation: int = 1,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        """Collect the result of a completed dispatch (blocks until ready).

        Blocks on the SYCL ``kernel_end`` event.  If the B70 kernel finished
        before this call (the common case when CUDA work ran between issue
        and take), the event is already signaled and the call returns
        immediately.

        Args:
            sequence: Sequence number returned by :meth:`issue`.
            M: Number of token rows in the dispatch.
            generation: Placement generation id.
            output: Optional pre-allocated [M, 2048] float32 buffer to write
                into (e.g. a pinned-memory numpy view).  If ``None`` a fresh
                array is allocated.

        Returns:
            [M, 2048] float32 — the routing-weighted B70 partial per token.
        """
        if not self._loaded or not self._handle:
            raise B70ProviderError("provider not loaded")

        if output is None:
            output = np.empty((M, 2048), dtype=np.float32)
        elif not output.flags["C_CONTIGUOUS"] or output.dtype != np.float32:
            output = np.ascontiguousarray(output, dtype=np.float32)

        status = self._lib.sb_b70_take(
            self._handle,
            ctypes.c_uint64(generation), ctypes.c_uint64(sequence),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(M * 2048),
        )
        if status != 0:
            raise B70ProviderError(
                f"sb_b70_take failed (status={status}, seq={sequence}, M={M})"
            )
        return output

    def dispatch(
        self,
        layer: int,
        hidden_fp16: np.ndarray,
        ids: np.ndarray,
        weights: np.ndarray,
        generation: int = 1,
    ) -> np.ndarray:
        """Issue one MoE dispatch and synchronously collect the result.

        Convenience wrapper: :meth:`issue` + :meth:`take`.  Prefer the
        separate methods when overlapping B70 compute with CUDA work.
        """
        seq = self.issue(layer, hidden_fp16, ids, weights, generation)
        M = np.ascontiguousarray(hidden_fp16, dtype=np.float16).shape[0]
        return self.take(seq, M, generation)

    # -- teardown --------------------------------------------------------

    def shutdown(self) -> None:
        if self._handle:
            self._lib.sb_b70_shutdown(self._handle)
            self._lib.sb_b70_destroy(self._handle)
            self._handle = None
            self._loaded = False

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
