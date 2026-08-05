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

    # -- compute ---------------------------------------------------------

    def dispatch(
        self,
        layer: int,
        hidden_fp16: np.ndarray,
        ids: np.ndarray,
        weights: np.ndarray,
        generation: int = 1,
    ) -> np.ndarray:
        """Issue one MoE dispatch and synchronously collect the result.

        Args:
            layer: NVFP4 layer index (0-31).
            hidden_fp16: [M, 2048] float16 (numpy), row-major, contiguous.
            ids: [M, topk] int32 — compact B70 slot indices, -1 to skip.
            weights: [M, topk] float32 — original routing weights.
            generation: Placement generation id.

        Returns:
            [M, 2048] float32 — the routing-weighted B70 partial per token.
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

        output = np.empty((M, 2048), dtype=np.float32)
        status = self._lib.sb_b70_take(
            self._handle,
            ctypes.c_uint64(generation), ctypes.c_uint64(seq),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(M * 2048),
        )
        if status != 0:
            raise B70ProviderError(
                f"sb_b70_take failed (status={status}, layer={layer}, M={M})"
            )

        return output

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
