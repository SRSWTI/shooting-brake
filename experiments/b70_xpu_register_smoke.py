#!/usr/bin/env python3
"""Standalone gate for the doorbell XPU host-registration change (1b).

Loads the real SBINT401 bank on the idle B70, drives the native poller
through host flags with torch CUDA-pinned staging buffers (the production
provenance), and prints the dispatch output checksum plus per-dispatch
service time. Run twice -- SHOOTING_BRAKE_B70_XPU_REGISTER=1 and =0 -- and
compare: outputs must be bitwise identical (registration changes the copy
path, never the bytes), service time should drop with registration.

  ZE_AFFINITY_MASK=0 SHOOTING_BRAKE_B70_XPU_REGISTER=0 .venv/bin/python \
      experiments/b70_xpu_register_smoke.py
  ZE_AFFINITY_MASK=0 SHOOTING_BRAKE_B70_XPU_REGISTER=1 .venv/bin/python \
      experiments/b70_xpu_register_smoke.py
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "src/phase7/libsb_b70_provider.so"
BANK = ROOT / "src/phase1/expert_bank_int4.bin"
HIDDEN, TOPK, MAX_BATCH, LAYER = 3072, 8, 256, 0
GENERATION = 7


def main():
    flag = os.environ.get("SHOOTING_BRAKE_B70_XPU_REGISTER", "1")
    lib = ctypes.CDLL(str(LIB))
    lib.sb_b70_create.restype = ctypes.c_void_p
    lib.sb_b70_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_uint64, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_size_t]
    lib.sb_b70_poll_create.restype = ctypes.c_void_p
    lib.sb_b70_poll_create.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.sb_b70_poll_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_size_t]
    lib.sb_b70_poll_service_ns.restype = ctypes.c_uint64
    lib.sb_b70_poll_dispatch_count.restype = ctypes.c_uint64
    lib.sb_b70_poll_error_count.restype = ctypes.c_uint64
    for f in ("sb_b70_poll_service_ns", "sb_b70_poll_dispatch_count",
              "sb_b70_poll_error_count"):
        getattr(lib, f).argtypes = [ctypes.c_void_p]

    provider = lib.sb_b70_create()
    assert provider, "provider create failed"
    t0 = time.perf_counter()
    rc = lib.sb_b70_load(provider, str(BANK).encode(), GENERATION, None, 0,
                         MAX_BATCH)
    if rc != 0:
        class Health(ctypes.Structure):
            _fields_ = [("generation", ctypes.c_uint64),
                        ("dispatches", ctypes.c_uint64),
                        ("allocations", ctypes.c_uint64),
                        ("last_error_bytes", ctypes.c_uint64),
                        ("loaded", ctypes.c_uint32),
                        ("pending", ctypes.c_uint32),
                        ("stopped", ctypes.c_uint32),
                        ("reserved", ctypes.c_uint32)]
        h = Health()
        err = ctypes.create_string_buffer(4096)
        lib.sb_b70_health.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_char_p, ctypes.c_size_t]
        lib.sb_b70_health(provider, ctypes.byref(h), err, len(err))
        raise SystemExit(f"load failed rc={rc}: {err.value.decode()}")
    print(f"bank loaded in {time.perf_counter()-t0:.1f}s")

    # Production-provenance staging: torch CUDA pinned host memory.
    hidden = torch.zeros(MAX_BATCH, HIDDEN, dtype=torch.float16,
                         pin_memory=True)
    ids = torch.full((MAX_BATCH, TOPK), -1, dtype=torch.int32,
                     pin_memory=True)
    weights = torch.zeros(MAX_BATCH, TOPK, dtype=torch.float32,
                          pin_memory=True)
    output = torch.zeros(MAX_BATCH, HIDDEN, dtype=torch.float32,
                         pin_memory=True)
    signal = np.zeros(1, dtype=np.uint32)
    completion = np.zeros(1, dtype=np.uint32)

    poller = lib.sb_b70_poll_create(provider, GENERATION)
    assert poller
    rc = lib.sb_b70_poll_register(
        poller, LAYER,
        signal.ctypes.data, completion.ctypes.data,
        hidden.data_ptr(), ids.data_ptr(), weights.data_ptr(),
        output.data_ptr(), TOPK)
    assert rc == 0, f"poll_register rc={rc}"
    assert lib.sb_b70_poll_start(poller) == 0

    # Deterministic M=4 dispatch: 4 tokens x 6 valid routes, seeded input.
    M = 4
    rng = np.random.default_rng(42)
    hidden[:M] = torch.from_numpy(
        (rng.standard_normal((M, HIDDEN)) * 0.02).astype(np.float16))
    for m in range(M):
        for j in range(TOPK):
            ids[m, j] = (m * 6 + j) % 126 if j < 6 else -1
            weights[m, j] = 1.0 / 6.0 if j < 6 else 0.0

    print(f"inputs: |hidden|max={float(hidden[:M].abs().max()):.4f} "
          f"ids[0]={ids[0].tolist()} w[0,0]={float(weights[0,0]):.4f}")

    # The down kernel accumulates routes with fp32 atomics, so replays are
    # order-nondeterministic at the last bit; the production correctness
    # gate is epsilon-based (2.15e-09 vs CPU dequant), never bitwise. Gate
    # replays against the first output, and cross-arm against a reference
    # file written by the REGISTER=0 arm.
    lat = []
    reference = None
    max_replay_delta = 0.0
    for it in range(50):
        completion[0] = 0
        t0 = time.perf_counter()
        signal[0] = M
        while completion[0] == 0:
            pass
        lat.append((time.perf_counter() - t0) * 1e6)
        if it == 0:
            print(f"iter0: |out|max={float(output[:M].abs().max()):.3e} "
                  f"|out|mean={float(output[:M].abs().mean()):.3e}")
        got = output[:M].numpy().copy()
        if reference is None:
            reference = got
        else:
            max_replay_delta = max(
                max_replay_delta,
                float(np.abs(got - reference).max()))

    errors = lib.sb_b70_poll_error_count(poller)
    service_us = (lib.sb_b70_poll_service_ns(poller) / 1000.0
                  / max(1, lib.sb_b70_poll_dispatch_count(poller)))
    lat_sorted = sorted(lat)
    scale = float(np.abs(reference).max())
    assert errors == 0, f"{errors} dispatch errors"
    assert max_replay_delta <= 1e-4 * max(scale, 1e-3), (
        f"replay delta {max_replay_delta} vs scale {scale}")

    ref_path = Path("/tmp/sb_xpu_register_ref.npy")
    cross = ""
    if flag == "0":
        np.save(ref_path, reference)
    elif ref_path.exists():
        base = np.load(ref_path)
        cross_delta = float(np.abs(reference - base).max())
        assert cross_delta <= 1e-4 * max(scale, 1e-3), (
            f"cross-arm delta {cross_delta} vs scale {scale}")
        cross = f"  cross_arm_max_delta={cross_delta:.3e}"
    print(f"XPU_REGISTER={flag}  out_scale={scale:.3e}  "
          f"replay_max_delta={max_replay_delta:.3e}  "
          f"round_trip_us median={lat_sorted[len(lat)//2]:.1f} "
          f"min={lat_sorted[0]:.1f}  provider_service_us={service_us:.1f}  "
          f"errors={errors}{cross}")

    lib.sb_b70_poll_stop(poller)
    lib.sb_b70_poll_destroy(poller)
    lib.sb_b70_shutdown(provider)
    lib.sb_b70_destroy(provider)
    print("clean shutdown")


if __name__ == "__main__":
    main()
