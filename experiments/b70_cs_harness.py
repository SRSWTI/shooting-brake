#!/usr/bin/env python3
"""Standalone repro harness for the CS-doorbell wedge -- no vLLM on top.

The Doorbell 2.0 serving integration wedged four times, ~10 minutes per blind
boot, stacks inside UR-adapter internals. This harness reproduces the exact
production concurrency shape in SECONDS per iteration and stays gdb-friendly:

  poller thread (provider .so)  <-- the real sb_b70_poll machinery, CS mode
  "CUDA" writer  (this process) --> torch pin_memory pages, host-thread writes

Memory fidelity matters: signal/completion/rings are torch pin_memory
(cudaHostAlloc) exactly like production lanes -- probe #1 showed the CS wait
ACCEPTS CUDA-pinned pages but REJECTS imported ones, so numpy pages would
wedge for the wrong reason.

Protocol per step (mirrors stream_signal): for each layer in order, the writer
stores signal=M, then spin-waits completion==1 with a timeout, clears
completion, and moves on. A layer that never completes = the wedge, and the
process is alive for gdb.

Usage:
  SHOOTING_BRAKE_B70_CS_DOORBELL=1 .venv/bin/python experiments/b70_cs_harness.py \
      --steps 50 --m 1 [--classic]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "src/phase1/expert_bank_jota_118b_r15.bin"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", default="0000:15:00.0")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--layers", type=int, default=47)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--classic", action="store_true",
                    help="force classic poller (unset CS env)")
    a = ap.parse_args()

    if a.classic:
        os.environ.pop("SHOOTING_BRAKE_B70_CS_DOORBELL", None)
    os.environ.setdefault("SHOOTING_BRAKE_B70_GROUPED", "1")
    os.environ.setdefault("SHOOTING_BRAKE_B70_OUT_FP16", "1")
    os.environ.setdefault("SYCL_UR_USE_LEVEL_ZERO_V2", "0")  # inert, kept for parity
    mode = os.environ.get("SHOOTING_BRAKE_B70_CS_DOORBELL", "unset")
    print(f"[harness] CS_DOORBELL={mode} selector={a.selector} M={a.m} "
          f"steps={a.steps}", flush=True)

    import numpy as np
    import torch  # torch pin_memory == cudaHostAlloc == production lane memory

    sys.path.insert(0, str(REPO / "src/phase4/src"))
    from shooting_brake_vllm.b70_binding import B70ProviderClient

    c = B70ProviderClient(str(REPO / "src/phase7/libsb_b70_provider.so"))
    c.load(BANK, top_k=10, max_batch=2048, device_selector=a.selector,
           resident_experts=np.arange(85, dtype=np.int32))
    print("[harness] bank loaded", flush=True)

    lib = c._lib
    handle = c._handle
    H, K, L, M = 3072, 10, a.layers, a.m

    poller = ctypes.c_void_p(lib.sb_b70_poll_create(handle, ctypes.c_uint64(1)))
    assert poller, "poll_create failed"

    # Production-shaped lanes: one pinned set per layer (signal, completion,
    # hidden, ids, weights, output), registered through poll_register exactly
    # as routed_experts does at first forward.
    lanes = []
    for layer in range(L):
        t = dict(
            sig=torch.zeros(16, dtype=torch.int32, pin_memory=True),
            comp=torch.zeros(16, dtype=torch.int32, pin_memory=True),
            hid=torch.randn(2048, H, dtype=torch.float16, pin_memory=True) * 0.3,
            ids=torch.randint(0, 85, (2048, K), dtype=torch.int32,
                              pin_memory=True),
            w=torch.rand(2048, K, dtype=torch.float32, pin_memory=True),
        )
        t["out"] = torch.zeros(2048, H, dtype=torch.float16, pin_memory=True)
        r = lib.sb_b70_poll_register(
            poller, ctypes.c_size_t(layer),
            ctypes.cast(t["sig"].data_ptr(),
                        ctypes.POINTER(ctypes.c_uint32)),
            ctypes.cast(t["comp"].data_ptr(),
                        ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_void_p(t["hid"].data_ptr()),
            ctypes.cast(t["ids"].data_ptr(),
                        ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(t["w"].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(t["out"].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(K),
        )
        assert r == 0, f"poll_register layer {layer} -> {r}"
        lanes.append(t)
    assert lib.sb_b70_poll_start(poller, ctypes.c_int(5)) == 0
    print(f"[harness] poller started, {L} layers registered", flush=True)

    # -- the fake CUDA side ---------------------------------------------------
    sig_views = [t["sig"].numpy() for t in lanes]
    comp_views = [t["comp"].numpy() for t in lanes]
    step_times = []
    wedged = False
    for step in range(a.steps):
        t0 = time.perf_counter()
        for layer in range(L):
            comp_views[layer][0] = 0
            sig_views[layer][0] = M  # ring the doorbell (host store ~= DMA)
            deadline = time.perf_counter() + a.timeout_s
            while comp_views[layer][0] != 1:
                if time.perf_counter() > deadline:
                    print(f"[harness] WEDGE: step {step} layer {layer}: "
                          f"signal={sig_views[layer][0]} completion=0 after "
                          f"{a.timeout_s}s. Process left alive for gdb "
                          f"(pid {os.getpid()}).", flush=True)
                    wedged = True
                    break
            if wedged:
                break
        if wedged:
            break
        step_times.append((time.perf_counter() - t0) * 1e3)
        if step == 0:
            print(f"[harness] first step OK: {step_times[0]:.2f} ms "
                  f"({L} layers)", flush=True)

    if wedged:
        # keep everything alive so gdb sees the true state
        print("[harness] sleeping 600s for inspection...", flush=True)
        time.sleep(600)
        return 2

    st = sorted(step_times[3:]) or step_times
    p50 = st[len(st) // 2]
    per_layer = p50 * 1000.0 / L
    print(f"\n[harness] {len(step_times)} steps clean.")
    print(f"[harness] step p50 = {p50:.2f} ms  -> {per_layer:.1f} us/layer "
          f"(signal->completion round trip incl. kernel)")
    print(f"[harness] min = {st[0]:.2f} ms, max = {st[-1]:.2f} ms")
    print("[harness] classic-poller reference for this shape: ~61 us/layer "
          "handshake + ~68 us service")
    lib.sb_b70_poll_stop(poller)
    lib.sb_b70_poll_destroy(poller)
    c.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
