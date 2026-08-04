#!/usr/bin/env python3
"""Dual-GPU expert offload test (optimized) — runs under .venv (CUDA).

Measures real compute + transport latency for three configs:
  1. 5090 only (all experts local)
  2. 5090 + CPU fallback (cold experts on CPU)
  3. 5090 + B70 (warm experts on B70 via shared memory transport)
"""
import sys, os, struct, time, subprocess
import numpy as np
import torch

H = 2048; I = 512; TOPK = 8; N_LAYERS = 40; TOKENS = 8
POOL_5090 = 48; POOL_B70 = 32
p_b70 = 0.307
cuda_dev = torch.device("cuda:0")

def compute_cuda(x, eids, wts, wg, wu, wd):
    out = torch.zeros(H, dtype=torch.float32, device=cuda_dev)
    for j, eid in enumerate(eids):
        g = (x @ wg[eid]).float()
        u = (x @ wu[eid]).float()
        a = torch.nn.functional.silu(g) * u
        out += wts[j] * (a @ wd[eid].float()).float()
    return out

def main():
    print(f"=== Dual-GPU Expert Offload Test ===")
    print(f"H={H} I={I} topk={TOPK} layers={N_LAYERS} tokens={TOKENS}")
    print(f"B70 prob: {p_b70:.0%} (~{int(TOPK*p_b70)} of {TOPK}/layer)")
    print(f"[5090] {torch.cuda.get_device_name(0)}\n")

    wg50 = torch.randn(POOL_5090, H, I, dtype=torch.bfloat16, device=cuda_dev)
    wu50 = torch.randn(POOL_5090, H, I, dtype=torch.bfloat16, device=cuda_dev)
    wd50 = torch.randn(POOL_5090, I, H, dtype=torch.bfloat16, device=cuda_dev)

    # Shared memory
    from multiprocessing import shared_memory
    SHM_SIZE = 12380
    shm_name = "sb_dual2"
    try:
        old = shared_memory.SharedMemory(name=shm_name); old.close(); old.unlink()
    except FileNotFoundError: pass
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=SHM_SIZE)
    buf = shm.buf
    for i in range(SHM_SIZE): buf[i] = 0

    # Launch B70 worker
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xpu_py = os.path.join(root, ".venv-xpu", "bin", "python3")
    worker = os.path.join(root, "experiments", "b70_worker.py")
    print("[b70] launching...")
    b70 = subprocess.Popen([xpu_py, worker, "--shm-name", shm_name,
                            "--n-experts", str(POOL_B70)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(20):
        line = b70.stdout.readline()
        if line:
            print(f"[b70] {line.rstrip()}")
            if "ready" in line: break
        if b70.poll() is not None: print("[!] B70 died!"); return

    rng = np.random.RandomState(42)
    x = torch.randn(H, dtype=torch.bfloat16, device=cuda_dev)
    x_np_cache = x.float().cpu().numpy().astype(np.float16)  # pre-converted

    # Generate routes once (fair comparison across modes)
    routes = []
    for _ in range(TOKENS * N_LAYERS):
        eids = rng.randint(0, POOL_5090 + POOL_B70, size=TOPK)
        wts = rng.rand(TOPK).astype(np.float32); wts /= wts.sum()
        mask_b70 = eids >= POOL_5090
        routes.append((eids, wts, mask_b70))

    # === Mode 1: 5090 only ===
    print("\n--- Mode 1: 5090 only ---")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for eids, wts, _ in routes:
        _ = compute_cuda(x, eids % POOL_5090, wts, wg50, wu50, wd50)
    torch.cuda.synchronize()
    t1 = (time.perf_counter() - t0) / TOKENS
    print(f"  {t1*1e3:.2f} ms/tok | {1/t1:.1f} tok/s")

    # === Mode 2: 5090 + CPU ===
    print("\n--- Mode 2: 5090 + CPU fallback ---")
    wg_cpu = torch.randn(POOL_B70, H, I, dtype=torch.float32)
    wu_cpu = torch.randn(POOL_B70, H, I, dtype=torch.float32)
    wd_cpu = torch.randn(POOL_B70, I, H, dtype=torch.float32)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for eids, wts, mask_b70 in routes:
        e50 = eids[~mask_b70] % POOL_5090; w50 = wts[~mask_b70]
        e_cpu = (eids[mask_b70] - POOL_5090) % POOL_B70; w_cpu = wts[mask_b70]
        if len(e50) > 0: _ = compute_cuda(x, e50, w50, wg50, wu50, wd50)
        xc = x.float().cpu()
        for j, eid in enumerate(e_cpu):
            g = xc @ wg_cpu[eid]; u = xc @ wu_cpu[eid]
            _ = (torch.nn.functional.silu(g) * u) @ wd_cpu[eid] * w_cpu[j]
    torch.cuda.synchronize()
    t2 = (time.perf_counter() - t0) / TOKENS
    print(f"  {t2*1e3:.2f} ms/tok | {1/t2:.1f} tok/s")

    # === Mode 3: 5090 + B70 ===
    print("\n--- Mode 3: 5090 + B70 ---")
    seq = 200
    # Warmup
    buf[84:84+H*2] = x_np_cache.tobytes()
    struct.pack_into("<I", buf, 16, 1)
    struct.pack_into("<I", buf, 20, 0)
    buf[52:56] = np.float32(1.0).tobytes()
    struct.pack_into("<Q", buf, 0, seq)
    while struct.unpack_from("<Q", buf, 8)[0] != seq: pass
    seq += 1

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for eids, wts, mask_b70 in routes:
        e50 = eids[~mask_b70] % POOL_5090; w50 = wts[~mask_b70]
        e_b70 = (eids[mask_b70] - POOL_5090) % POOL_B70; w_b70 = wts[mask_b70]
        n_b70 = len(e_b70)

        # Fire to B70 first
        if n_b70 > 0:
            buf[84:84+H*2] = x_np_cache.tobytes()
            struct.pack_into("<I", buf, 16, n_b70)
            for j, eid in enumerate(e_b70):
                struct.pack_into("<I", buf, 20+j*4, int(eid))
            buf[52:52+n_b70*4] = np.asarray(w_b70, dtype=np.float32).tobytes()
            struct.pack_into("<Q", buf, 0, seq)

        # Overlap: compute 5090 experts while B70 works
        if len(e50) > 0:
            _ = compute_cuda(x, e50, w50, wg50, wu50, wd50)

        # Collect B70
        if n_b70 > 0:
            while struct.unpack_from("<Q", buf, 8)[0] != seq: pass
            seq += 1

    torch.cuda.synchronize()
    t3 = (time.perf_counter() - t0) / TOKENS
    print(f"  {t3*1e3:.2f} ms/tok | {1/t3:.1f} tok/s")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Config':<28} {'ms/tok':>8} {'tok/s':>8} {'vs CPU':>8}")
    print(f"{'-'*60}")
    print(f"{'5090 only':<28} {t1*1e3:>8.2f} {1/t1:>8.1f} {'':>8}")
    print(f"{'5090 + CPU fallback':<28} {t2*1e3:>8.2f} {1/t2:>8.1f} {t2/t2:>7.1f}x")
    print(f"{'5090 + B70':<28} {t3*1e3:>8.2f} {1/t3:>8.1f} {t2/t3:>7.1f}x")
    print(f"{'-'*60}")
    if t2 > t1:
        cpu_penalty = t2 - t1
        b70_recovery = max(0, cpu_penalty - (t3 - t1))
        print(f"CPU penalty: {cpu_penalty*1e3:.1f} ms/tok")
        print(f"B70 recovered: {b70_recovery*1e3:.1f} ms/tok ({b70_recovery/cpu_penalty*100:.0f}%)")
        print(f"B70 vs full-5090: {t3/t1*100:.0f}% speed")
    print(f"{'='*60}")

    # Cleanup
    struct.pack_into("<Q", buf, 0, 0xFFFF_FFFF_FFFF_FFFF)
    time.sleep(0.5)
    try: b70.terminate(); b70.wait(timeout=3)
    except: pass
    if b70.stdout:
        rest = b70.stdout.read()
        if rest: print(f"[b70] {rest.strip()}")
    shm.close(); shm.unlink()
    print("\n[ctrl] done")

if __name__ == "__main__":
    main()
