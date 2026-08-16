#!/usr/bin/env python3
"""cudaHostRegister-on-mmap probe: can we DMA straight from page cache?

Tests the "skip staging entirely" rung of the prefill ladder:

  1. mmap the bank O_RDONLY fd with PROT_READ|PROT_WRITE, MAP_PRIVATE
     (legal on Linux: COW reserves the write option, never fires -- we
     never write; sidesteps cudaHostRegisterReadOnly entirely)
  2. cudaHostRegister the slab range (page-aligned: HEADER_SKIP == 4096)
  3. CUDA-event-timed H2D from the registered mmap vs the known ceilings:
       pinned cudaHostAlloc DMA   52.8 GiB/s  (slab_h2d.json)
       pageable mmap              18.5 GiB/s  (slab_h2d.json)
  4. control: same mapping PROT_READ-only should fail registration
     without cudaHostRegisterReadOnly (documented gotcha -- confirm on
     this driver/GPU rather than trust forum threads)

Also reports cudaHostRegister wall time (the one-time startup tax) and
whether the registered-source copy really takes the async DMA path.

Usage:
  .venv/bin/python benchmarks/hostregister_probe.py \
      --bank src/phase1/expert_bank_int4.bin \
      --out benchmarks/results/slab_h2d/hostregister_probe.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import statistics
import time
from pathlib import Path

import torch

EXPERT_STRIDE = 4_866_048  # SBINT401 v2, validated by the provider
REMOTE_EXPERTS = 126       # split:54 -> experts 54..179 on the B70
HEADER_SKIP = 4096         # bank data_offset alignment (page-aligned)


def buf_addr(mm: mmap.mmap) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mm))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=9)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    cudart = torch.cuda.cudart()
    slab = REMOTE_EXPERTS * EXPERT_STRIDE
    gib = slab / 2**30
    map_len = HEADER_SKIP + slab

    fd = os.open(args.bank, os.O_RDONLY)

    # --- main arm: PROT_READ|PROT_WRITE, MAP_PRIVATE on the O_RDONLY fd ---
    mm = mmap.mmap(fd, map_len, flags=mmap.MAP_PRIVATE,
                   prot=mmap.PROT_READ | mmap.PROT_WRITE)
    base = buf_addr(mm)
    src_t = torch.frombuffer(memoryview(mm)[HEADER_SKIP:HEADER_SKIP + slab],
                             dtype=torch.uint8)
    assert src_t.data_ptr() == base + HEADER_SKIP

    # fault-in (page cache is warm; this materializes PTEs in *this* mapping)
    t0 = time.perf_counter()
    _ = int(src_t[::4096].to(torch.int64).sum())
    fault_s = time.perf_counter() - t0

    # register only the slab range, page-aligned
    t0 = time.perf_counter()
    reg_status = int(cudart.cudaHostRegister(base + HEADER_SKIP, slab, 0))
    register_s = time.perf_counter() - t0
    if reg_status != 0:
        raise SystemExit(f"cudaHostRegister failed: {reg_status}")

    dev = torch.empty(slab, dtype=torch.uint8, device="cuda")
    ev_a, ev_b = torch.cuda.Event(True), torch.cuda.Event(True)

    # warmup
    dev.copy_(src_t, non_blocking=True)
    torch.cuda.synchronize()

    h2d_ts, wall_ts = [], []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        ev_a.record()
        dev.copy_(src_t, non_blocking=True)
        ev_b.record()
        submit_s = time.perf_counter() - w0  # ~0 iff truly async DMA path
        torch.cuda.synchronize()
        h2d_ts.append(ev_a.elapsed_time(ev_b) / 1000.0)
        wall_ts.append(submit_s)

    # correctness: registered-path bytes == file bytes
    with open(args.bank, "rb") as f:
        f.seek(HEADER_SKIP)
        head = f.read(1 << 20)
    ok = bool((dev[:1 << 20].cpu().numpy().tobytes() == head))

    unreg_t0 = time.perf_counter()
    cudart.cudaHostUnregister(base + HEADER_SKIP)
    unregister_s = time.perf_counter() - unreg_t0

    # --- control arm: PROT_READ-only registration should be rejected ---
    # Runs LAST: empirically the failed register poisons the next torch
    # CUDA check on this driver (observed: torch.empty raised
    # cudaErrorInvalidValue right after). All measurement is done by now.
    ro = mmap.mmap(fd, map_len, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ)
    # ctypes.from_buffer requires a writable buffer; torch.frombuffer is
    # zero-copy and read-only-tolerant, so use it to obtain the address.
    ro_t = torch.frombuffer(memoryview(ro), dtype=torch.uint8)
    ro_status = int(cudart.cudaHostRegister(ro_t.data_ptr(), map_len, 0))
    if ro_status == 0:  # unexpectedly succeeded -- clean up
        cudart.cudaHostUnregister(ro_t.data_ptr())
    del ro_t
    ro.close()

    med = statistics.median(h2d_ts)
    result = {
        "kind": "hostregister_mmap_probe",
        "bank": str(args.bank),
        "slab_bytes": slab,
        "slab_mib": slab / 2**20,
        "iters": args.iters,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "readonly_control": {
            "status": ro_status,
            "expected": "nonzero (cudaErrorInvalidValue=1) without ReadOnly flag",
        },
        "fault_pass_s": fault_s,
        "register_s": register_s,
        "register_gib_per_s": gib / register_s,
        "unregister_s": unregister_s,
        "h2d_registered_mmap": {
            "median_s": med, "min_s": min(h2d_ts), "max_s": max(h2d_ts),
            "gib_per_s_median": gib / med, "raw_s": h2d_ts,
        },
        "submit_wall_s_median": statistics.median(wall_ts),
        "async_dma_path": statistics.median(wall_ts) < med / 10,
        "first_mib_matches_file": ok,
        "reference_ceilings_gib_per_s": {
            "pinned_hostalloc_dma": 52.78, "pageable_mmap": 18.54,
            "source": "benchmarks/results/slab_h2d/slab_h2d.json",
        },
        "derived": {
            "forward_48_layers_s_at_this_rate": 48 * med,
            "full_bank_register_projection_s": (48 * gib) / (gib / register_s),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"readonly control: status={ro_status} "
          f"({'rejected as documented' if ro_status != 0 else 'ACCEPTED (?)'})")
    print(f"register {slab / 2**20:.1f} MiB: {register_s * 1e3:.1f} ms "
          f"({gib / register_s:.2f} GiB/s) | unregister {unregister_s * 1e3:.1f} ms")
    print(f"H2D from registered mmap: {gib / med:6.2f} GiB/s "
          f"(pinned ceiling 52.78, pageable 18.54)")
    print(f"submit wall {statistics.median(wall_ts) * 1e6:.0f} us -> "
          f"async DMA path: {result['async_dma_path']}")
    print(f"first MiB matches file: {ok}")
    print(f"48-layer forward at this rate: {48 * med:.2f} s | "
          f"full-bank register projection: "
          f"{result['derived']['full_bank_register_projection_s']:.1f} s one-time")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
