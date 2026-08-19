"""CUDA stream signaling for graph-compatible B70 dispatch.

Uses ``cuStreamWriteValue32_v2`` / ``cuStreamWaitValue32_v2`` from the CUDA
Driver API — hardware-accelerated stream operations that are natively
capture-compatible with ``torch.cuda.graph()`` (no kernel spinning).

Usage during forward_modular (inside CUDA graph capture):

    stream_signal.write_flag(signal_dev_ptr, 1)    # Signal B70 thread
    y_cuda = super().forward_modular(...)           # CUDA kernel (overlaps B70)
    stream_signal.wait_flag(completion_dev_ptr, 1)  # Wait for B70 completion
    stream_signal.write_flag(completion_dev_ptr, 0) # Reset for next replay
"""

from __future__ import annotations

import ctypes

# ── Load CUDA libraries ─────────────────────────────────────────────────

_cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
_cuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)

_HOSTALLOC_MAPPED = 0x02
_HOSTALLOC_PORTABLE = 0x01
_WAIT_VALUE_EQ = 1

# cuStreamWriteValue32_v2(stream, addr, value, flags)
_write_fn = _cuda.cuStreamWriteValue32_v2
_write_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_uint, ctypes.c_uint]
_write_fn.restype = ctypes.c_int

# cuStreamWaitValue32_v2(stream, addr, value, flags, type)
_wait_fn = _cuda.cuStreamWaitValue32_v2
_wait_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
_wait_fn.restype = ctypes.c_int


# ── Host-mapped memory ──────────────────────────────────────────────────


def alloc_host_mapped_flag(initial: int = 0) -> tuple[int, int]:
    """Allocate a 4-byte host-mapped flag on its own cache lines.

    Returns ``(host_ptr_int, dev_ptr_int)`` — same physical memory,
    accessible from both CPU and GPU.

    The allocation is padded to 256 bytes so no two flags can ever share a
    cache line, regardless of how the runtime suballocates. Two flags on
    one line let one side's write-back eat the other side's store; it
    presents as a random hang after O(100) round trips, not as an obvious
    bug (Bench 4 paid an afternoon for this).
    """
    host_ptr = ctypes.c_void_p()
    _cudart.cudaHostAlloc(
        ctypes.byref(host_ptr),
        ctypes.c_size_t(256),
        ctypes.c_uint(_HOSTALLOC_MAPPED | _HOSTALLOC_PORTABLE),
    )
    dev_ptr = ctypes.c_void_p()
    _cudart.cudaHostGetDevicePointer(
        ctypes.byref(dev_ptr), host_ptr, ctypes.c_uint(0),
    )
    ctypes.c_uint.from_address(host_ptr.value).value = initial
    return host_ptr.value, dev_ptr.value


def read_flag(host_ptr_int: int) -> int:
    """Read flag value from CPU."""
    return ctypes.c_uint.from_address(host_ptr_int).value


def write_flag_host(host_ptr_int: int, value: int) -> None:
    """Write flag value from CPU."""
    ctypes.c_uint.from_address(host_ptr_int).value = value


# ── CUDA stream operations (capture-compatible) ─────────────────────────


def write_flag(dev_ptr_int: int, value: int) -> None:
    """Enqueue a flag write on the current CUDA stream.

    Uses ``cuStreamWriteValue32_v2`` — a hardware stream operation
    captured by ``torch.cuda.graph()`` and replayed without Python.
    """
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _write_fn(stream, dev_ptr_int, value, 0)


def wait_flag(dev_ptr_int: int, target: int) -> None:
    """Enqueue a flag wait on the current CUDA stream.

    Uses ``cuStreamWaitValue32_v2`` — the GPU's DMA engine handles
    the wait without spinning.  In practice ~0μs because the B70
    kernel (44μs) finishes before the CUDA MoE kernel (100μs).
    """
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _wait_fn(stream, dev_ptr_int, target, 0, _WAIT_VALUE_EQ)


# Late import to avoid circular dependency at module load time.
import torch  # noqa: E402

__all__ = [
    "alloc_host_mapped_flag",
    "read_flag",
    "write_flag_host",
    "write_flag",
    "wait_flag",
]
