"""CPU DDR5 expert tier — Python binding for ``libsb_cpu_expert.so``.

The third residency tier. Experts that route rarely live in hugepage-backed
host DRAM and compute on CPU cores, freeing 5090 and B70 VRAM for hotter
experts and KV cache.

Where it sits::

    CUDA (hot)   experts on the 5090       ~1800 GB/s   fastest
    B70  (warm)  experts on the Arc Pro    ~450 GB/s    ~40us/expert
    CPU  (cold)  experts in DDR5           ~48 GB/s     ~195us/expert

A decode-shaped expert pass reads the whole weight set once and reuses none
of it, so those bandwidths *are* the latency ratios. Measured for the
qualified model (hidden=2048, intermediate=768, 9.0 MiB/expert) via
``phase7/cpu_expert_bench.py``:

     1 thread   -> 1340us  @  7.0 GB/s
     8 threads  ->  287us  @ 32.8 GB/s
    12 threads  ->  195us  @ 48.3 GB/s

So ~5x the B70's cost for identical work, and thread count is a requirement
rather than a knob. That ratio is the entire design constraint: this tier is
correct for experts that almost never fire and a latency bug for any expert
that does, which makes frequency-calibrated placement a prerequisite rather
than an optimisation.

Lifecycle::

    host = CpuExpertHost(num_layers=40, num_experts=256,
                         hidden=2048, intermediate=768, max_experts=512)
    host.load_expert(layer, expert, gate, up, down)   # bf16 CPU tensors
    partial = host.moe_forward(layer, hidden, ids, weights)
    host.close()
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

_DEFAULT_LIB = "phase7/libsb_cpu_expert.so"


class CpuExpertError(RuntimeError):
    """The CPU expert tier returned an error."""


class CpuExpertHost:
    """ctypes wrapper around the native CPU expert host.

    Weight layout follows ``nn.Linear`` (``[out_features, in_features]``):
    ``gate``/``up`` are ``[intermediate, hidden]`` and ``down`` is
    ``[hidden, intermediate]``. Passing them untransposed keeps each output
    row's dot product on contiguous memory, which is what the M=1 decode
    path needs.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        hidden: int,
        intermediate: int,
        max_experts: int,
        num_threads: int = 0,
        lib_path: str | Path | None = None,
    ) -> None:
        lib_path = lib_path or os.environ.get("SHOOTING_BRAKE_CPU_LIB", _DEFAULT_LIB)
        if not os.path.isfile(lib_path):
            raise CpuExpertError(f"CPU expert library not found: {lib_path}")
        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_signatures()

        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)

        handle = self._lib.sb_cpu_create(
            self.num_layers, self.num_experts, self.hidden,
            self.intermediate, int(max_experts), int(num_threads),
        )
        if not handle:
            raise CpuExpertError(
                "sb_cpu_create failed (arena mmap or bad dimensions): "
                f"layers={num_layers} experts={num_experts} hidden={hidden} "
                f"intermediate={intermediate} max_experts={max_experts}"
            )
        self._handle = ctypes.c_void_p(handle)

    # -- ctypes plumbing -------------------------------------------------

    def _setup_signatures(self) -> None:
        lib = self._lib
        lib.sb_cpu_create.restype = ctypes.c_void_p
        lib.sb_cpu_create.argtypes = [ctypes.c_size_t] * 6

        lib.sb_cpu_load_expert.restype = ctypes.c_int
        lib.sb_cpu_load_expert.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]

        lib.sb_cpu_has_expert.restype = ctypes.c_int
        lib.sb_cpu_has_expert.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ]

        lib.sb_cpu_expert_forward.restype = ctypes.c_int
        lib.sb_cpu_expert_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]

        lib.sb_cpu_moe_forward.restype = ctypes.c_int
        lib.sb_cpu_moe_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
        ]

        for name in ("sb_cpu_arena_used", "sb_cpu_arena_capacity",
                     "sb_cpu_resident_count"):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_size_t
            fn.argtypes = [ctypes.c_void_p]

        lib.sb_cpu_skipped_routes.restype = ctypes.c_uint64
        lib.sb_cpu_skipped_routes.argtypes = [ctypes.c_void_p]

        lib.sb_cpu_destroy.restype = None
        lib.sb_cpu_destroy.argtypes = [ctypes.c_void_p]

    def _require_open(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise CpuExpertError("CPU expert host is closed")
        return self._handle

    @staticmethod
    def _ptr(t: torch.Tensor) -> ctypes.c_void_p:
        return ctypes.c_void_p(t.data_ptr())

    def _check_weight(self, t: torch.Tensor, rows: int, cols: int,
                      name: str) -> torch.Tensor:
        """Reject anything the native side would read incorrectly.

        A silently transposed or non-contiguous weight produces plausible
        garbage rather than an error, so shape, dtype, device and stride are
        all enforced here where the failure is still legible.
        """
        if t.dtype is not torch.bfloat16:
            raise CpuExpertError(f"{name} must be bfloat16, got {t.dtype}")
        if t.device.type != "cpu":
            raise CpuExpertError(f"{name} must be on CPU, got {t.device}")
        if tuple(t.shape) != (rows, cols):
            raise CpuExpertError(
                f"{name} must be [{rows}, {cols}], got {tuple(t.shape)}"
            )
        return t if t.is_contiguous() else t.contiguous()

    # -- loading ---------------------------------------------------------

    def load_expert(self, layer: int, expert: int, gate: torch.Tensor,
                    up: torch.Tensor, down: torch.Tensor) -> None:
        """Copy one expert's weights into the DRAM arena.

        The native side memcpy's out of these buffers, so they may be freed
        (or their VRAM originals released by surgery) as soon as this returns.
        """
        handle = self._require_open()
        gate = self._check_weight(gate, self.intermediate, self.hidden, "gate")
        up = self._check_weight(up, self.intermediate, self.hidden, "up")
        down = self._check_weight(down, self.hidden, self.intermediate, "down")

        rc = self._lib.sb_cpu_load_expert(
            handle, int(layer), int(expert),
            self._ptr(gate), self._ptr(up), self._ptr(down),
        )
        if rc == -2:
            raise CpuExpertError(
                f"CPU arena exhausted loading (layer={layer}, expert={expert}); "
                f"used {self.arena_used_bytes / 2**30:.2f} GiB of "
                f"{self.arena_capacity_bytes / 2**30:.2f} GiB. "
                "Raise max_experts."
            )
        if rc != 0:
            raise CpuExpertError(
                f"sb_cpu_load_expert failed rc={rc} (layer={layer}, expert={expert})"
            )

    def has_expert(self, layer: int, expert: int) -> bool:
        return bool(self._lib.sb_cpu_has_expert(
            self._require_open(), int(layer), int(expert)))

    # -- compute ---------------------------------------------------------

    def expert_forward(self, layer: int, expert: int,
                       hidden_states: torch.Tensor) -> torch.Tensor:
        """Run one expert's SwiGLU FFN. Returns fp32 ``[M, hidden]``."""
        handle = self._require_open()
        if hidden_states.dtype is not torch.bfloat16:
            raise CpuExpertError(
                f"hidden_states must be bfloat16, got {hidden_states.dtype}")
        if hidden_states.device.type != "cpu":
            raise CpuExpertError("hidden_states must be on CPU")
        hidden_states = hidden_states.contiguous()
        M = hidden_states.shape[0]

        out = torch.empty(M, self.hidden, dtype=torch.float32, device="cpu")
        rc = self._lib.sb_cpu_expert_forward(
            handle, int(layer), int(expert), self._ptr(hidden_states),
            ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)), M,
        )
        if rc != 0:
            raise CpuExpertError(
                f"sb_cpu_expert_forward failed rc={rc} "
                f"(layer={layer}, expert={expert}); expert not resident?"
            )
        return out

    def moe_forward(self, layer: int, hidden_states: torch.Tensor,
                    expert_ids: torch.Tensor,
                    routing_weights: torch.Tensor) -> torch.Tensor:
        """Routing-weighted partial for all CPU-resident routes in a batch.

        Same contract as the B70 provider so both tiers join identically:
        the result is zero where no resident route touched a token, and every
        resident route contributes ``weight * FFN(token)`` exactly once.

        Args:
            hidden_states: bf16 ``[M, hidden]``.
            expert_ids: int32 ``[M, topk]`` global ids; -1 skips a route.
            routing_weights: fp32 ``[M, topk]``.

        Returns:
            fp32 ``[M, hidden]``.
        """
        handle = self._require_open()
        if hidden_states.dtype is not torch.bfloat16:
            raise CpuExpertError(
                f"hidden_states must be bfloat16, got {hidden_states.dtype}")
        if expert_ids.dtype is not torch.int32:
            raise CpuExpertError(
                f"expert_ids must be int32, got {expert_ids.dtype}")
        if routing_weights.dtype is not torch.float32:
            raise CpuExpertError(
                f"routing_weights must be float32, got {routing_weights.dtype}")
        if expert_ids.shape != routing_weights.shape:
            raise CpuExpertError(
                f"expert_ids {tuple(expert_ids.shape)} and routing_weights "
                f"{tuple(routing_weights.shape)} must have the same shape")

        hidden_states = hidden_states.cpu().contiguous()
        expert_ids = expert_ids.cpu().contiguous()
        routing_weights = routing_weights.cpu().contiguous()
        M, topk = expert_ids.shape

        out = torch.zeros(M, self.hidden, dtype=torch.float32, device="cpu")
        rc = self._lib.sb_cpu_moe_forward(
            handle, int(layer), self._ptr(hidden_states),
            ctypes.cast(expert_ids.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(routing_weights.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            M, topk,
            ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise CpuExpertError(f"sb_cpu_moe_forward failed rc={rc}")
        return out

    # -- introspection ---------------------------------------------------

    @property
    def arena_used_bytes(self) -> int:
        return int(self._lib.sb_cpu_arena_used(self._require_open()))

    @property
    def arena_capacity_bytes(self) -> int:
        return int(self._lib.sb_cpu_arena_capacity(self._require_open()))

    @property
    def resident_count(self) -> int:
        return int(self._lib.sb_cpu_resident_count(self._require_open()))

    @property
    def skipped_routes(self) -> int:
        """Routes whose expert was not resident. Nonzero means a partition bug."""
        return int(self._lib.sb_cpu_skipped_routes(self._require_open()))

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._lib.sb_cpu_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> CpuExpertHost:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
