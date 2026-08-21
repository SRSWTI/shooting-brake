"""Phase-7 Python ctypes binding for the B70 NVFP4 MoE provider.

Wraps the C ABI shared library (``libsb_b70_provider.so``) so the actual Intel
Arc Pro B70 can compute routed-expert during the vLLM forward pass, in-process
alongside CUDA. SYCL and CUDA coexist (validated: separate driver stacks).

Lifecycle::

    provider = B70ProviderClient(lib_path)
    provider.load(bank_path, top_k=8, generation=1,
                  resident_experts=[128, 129, ..., 255])
    output = provider.dispatch(layer=0, hidden_fp16=..., ids=..., weights=..., M=8)
    provider.shutdown()

The ``dispatch`` method issues + takes synchronously (correctness-first).
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


#: Minimum ``sb_b70_abi_version()`` this binding accepts. Bumped whenever the
#: C ABI gains a contract the Python side depends on -- version 2 introduced
#: ``sb_b70_load_v2`` with a runtime routing width. A shared object older than
#: this does not export the version symbol at all, which is deliberate: it
#: turns a silent width mismatch into a loud import-time failure.
REQUIRED_ABI_VERSION = 2


class _NativeB70Health(ctypes.Structure):
    _fields_ = [
        ("generation", ctypes.c_uint64),
        ("dispatches", ctypes.c_uint64),
        ("allocations", ctypes.c_uint64),
        ("last_error_bytes", ctypes.c_uint64),
        ("loaded", ctypes.c_uint32),
        ("pending", ctypes.c_uint32),
        ("stopped", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class B70ProviderHealth:
    generation: int
    dispatches: int
    allocations: int
    loaded: bool
    pending: bool
    stopped: bool
    last_error: str


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
        self._out_dtype = None
        self._hidden = 0
        self._resident_count = 0
        self._sequence = 0
        # Routing width this provider was LOADED with. Its device buffers are
        # sized from it, so a later caller dispatching a different width is a
        # silent-wrong-output bug rather than an error; callers reusing a
        # cached provider must compare against this.
        self._top_k = 0

    def _setup_signatures(self) -> None:
        lib = self._lib
        lib.sb_b70_create.restype = ctypes.c_void_p
        lib.sb_b70_create.argtypes = []

        # ABI staleness guard. A shared object predating the versioned load
        # entry point does not export this symbol AT ALL, so the lookup
        # below raises instead of letting a width-8 provider serve width-10
        # routing. That failure mode is otherwise silent: the provider would
        # process 8 of 10 routes per token and still return finite output.
        try:
            lib.sb_b70_abi_version.restype = ctypes.c_size_t
            lib.sb_b70_abi_version.argtypes = []
        except AttributeError as exc:
            raise B70ProviderError(
                "libsb_b70_provider.so predates the versioned load ABI "
                "(no sb_b70_abi_version symbol). Rebuild it: "
                "source /opt/intel/oneapi/setvars.sh --force && "
                "make -C src/phase7 b70"
            ) from exc
        abi = int(lib.sb_b70_abi_version())
        if abi < REQUIRED_ABI_VERSION:
            raise B70ProviderError(
                f"libsb_b70_provider.so reports ABI version {abi}, this "
                f"plugin requires >= {REQUIRED_ABI_VERSION}"
            )

        # v2 carries top_k. v1 still exists in the C ABI, frozen at 8 for
        # older callers, but nothing in Python may use it: a caller that
        # forgets the width is exactly the bug this guard exists to stop.
        lib.sb_b70_load_v2.restype = ctypes.c_int
        lib.sb_b70_load_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
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

        # Result wire width. Absent on providers built before the fp16 wire, so
        # this stays optional -- a missing symbol means fp32, not an error.
        if hasattr(lib, "sb_b70_out_fp16"):
            lib.sb_b70_out_fp16.restype = ctypes.c_int
            lib.sb_b70_out_fp16.argtypes = [ctypes.c_void_p]

        lib.sb_b70_device_memory.restype = ctypes.c_int
        lib.sb_b70_device_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]

        lib.sb_b70_health.restype = ctypes.c_int
        lib.sb_b70_health.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NativeB70Health),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]

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
        lib.sb_b70_poll_start.argtypes = [ctypes.c_void_p, ctypes.c_int]

        lib.sb_b70_poll_stop.restype = None
        lib.sb_b70_poll_stop.argtypes = [ctypes.c_void_p]
        lib.sb_b70_poll_reset.restype = None
        lib.sb_b70_poll_reset.argtypes = [ctypes.c_void_p]


        for counter in (
            "sb_b70_poll_dispatch_count",
            "sb_b70_poll_row_count",
            "sb_b70_poll_error_count",
            "sb_b70_poll_service_ns",
            "sb_b70_poll_total_ns",
            "sb_b70_poll_kernel_ns",
        ):
            fn = getattr(lib, counter)
            fn.restype = ctypes.c_uint64
            fn.argtypes = [ctypes.c_void_p]

        lib.sb_b70_poll_m_bucket_count.restype = ctypes.c_uint64
        lib.sb_b70_poll_m_bucket_count.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
        ]

        lib.sb_b70_poll_destroy.restype = None
        lib.sb_b70_poll_destroy.argtypes = [ctypes.c_void_p]

    # -- lifecycle -------------------------------------------------------

    def load(
        self,
        bank_path: str | Path,
        *,
        top_k: int,
        generation: int = 1,
        resident_experts: np.ndarray | None = None,
        max_batch: int = 128,
        device_selector: str = "",
    ) -> None:
        """Create the provider, load the expert bank, select residents.

        ``device_selector`` picks which physical B70 this provider owns:
        empty keeps the legacy first-enumerated device, a decimal string is
        a zero-based index, anything else is matched as a PCI BDF. Multi-
        card configs must pass BDFs — enumeration order once silently put
        the Gen3 card first and cost 40% for weeks.
        """
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

        status = self._lib.sb_b70_load_v2(
            self._handle, bank, ctypes.c_uint64(generation),
            ids_ptr, count, ctypes.c_size_t(max_batch),
            device_selector.encode("utf-8") if device_selector else None,
            ctypes.c_size_t(top_k),
        )
        if status != 0:
            raise B70ProviderError(
                f"sb_b70_load_v2 failed with status {status} "
                f"(top_k={top_k}, max_batch={max_batch})"
            )

        self._resident_count = self._lib.sb_b70_num_resident(self._handle)
        self._loaded = True
        # Per-layer resident count. The layer count comes from the bank's
        # own header, which is what the provider itself adopts — a constant
        # here would silently misreport occupancy for any bank that is not
        # the 32-layer 35B one.
        from .config import read_bank_header

        header = read_bank_header(str(bank_path))
        layers = header.layers
        self._hidden = header.hidden_size
        self._resident_per_layer = (
            self._resident_count // layers if layers else 0
        )
        self._top_k = top_k

    @property
    def resident_per_layer(self) -> int:
        return self._resident_per_layer

    @property
    def top_k(self) -> int:
        """Routing width this provider was loaded with; 0 before ``load``."""
        return self._top_k

    @property
    def device_memory(self) -> tuple[int, int] | None:
        """``(free_bytes, total_bytes)`` on the B70, or ``None``.

        The second card is bought for capacity, so its occupancy is the
        number that decides whether an expert bank fits. Until this existed
        it was the only resource in the system inferred rather than
        measured: 5090 VRAM, host DRAM and the KV cache were all reported
        while the B70's was computed as experts x bytes-per-expert.

        ``None`` when the provider is not loaded or the runtime does not
        expose the free-memory aspect.
        """
        if not self._loaded or self._handle is None:
            return None
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        rc = self._lib.sb_b70_device_memory(
            self._handle, ctypes.byref(free), ctypes.byref(total)
        )
        return (free.value, total.value) if rc == 0 else None

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
            layer: Absolute routed-expert layer index in the loaded bank.
            hidden_fp16: [M, bank.hidden_size] float16, row-major contiguous.
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
        assert hidden.shape[1] == self._hidden, (
            f"hidden dim {hidden.shape[1]} != bank's {self._hidden}"
        )
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

    @property
    def out_dtype(self) -> "np.dtype":
        """Element type take() writes -- asked of the provider, never guessed.

        The provider decides from its own environment; if this side inferred it
        independently the two could disagree and the result would be silent
        garbage rather than an error. Providers without the symbol predate the
        fp16 wire and are fp32.
        """
        if self._out_dtype is None:
            fn = getattr(self._lib, "sb_b70_out_fp16", None)
            if fn is None or not self._handle:
                self._out_dtype = np.dtype(np.float32)
            else:
                self._out_dtype = np.dtype(
                    np.float16 if fn(self._handle) == 1 else np.float32
                )
        return self._out_dtype

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
            output: Optional pre-allocated [M, bank.hidden_size] float32 buffer.
                If ``None``, a fresh array is allocated.

        Returns:
            [M, bank.hidden_size] routing-weighted B70 partial per token.
        """
        if not self._loaded or not self._handle:
            raise B70ProviderError("provider not loaded")

        dtype = self.out_dtype
        if output is None:
            output = np.empty((M, self._hidden), dtype=dtype)
        elif not output.flags["C_CONTIGUOUS"] or output.dtype != dtype:
            output = np.ascontiguousarray(output, dtype=dtype)

        status = self._lib.sb_b70_take(
            self._handle,
            ctypes.c_uint64(generation), ctypes.c_uint64(sequence),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(M * self._hidden),
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

    @property
    def health(self) -> B70ProviderHealth:
        """Snapshot provider-side dispatch state and its last error."""
        if not self._handle:
            raise B70ProviderError("provider not created")
        native = _NativeB70Health()
        status = self._lib.sb_b70_health(
            self._handle, ctypes.byref(native), None, ctypes.c_size_t(0)
        )
        if status != 0:
            raise B70ProviderError(
                f"sb_b70_health size query failed with status {status}"
            )
        error_buffer = ctypes.create_string_buffer(
            max(int(native.last_error_bytes), 1)
        )
        status = self._lib.sb_b70_health(
            self._handle,
            ctypes.byref(native),
            ctypes.cast(error_buffer, ctypes.c_void_p),
            ctypes.c_size_t(len(error_buffer)),
        )
        if status != 0:
            raise B70ProviderError(
                f"sb_b70_health failed with status {status}"
            )
        return B70ProviderHealth(
            generation=int(native.generation),
            dispatches=int(native.dispatches),
            allocations=int(native.allocations),
            loaded=bool(native.loaded),
            pending=bool(native.pending),
            stopped=bool(native.stopped),
            last_error=error_buffer.value.decode("utf-8", errors="replace"),
        )

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
