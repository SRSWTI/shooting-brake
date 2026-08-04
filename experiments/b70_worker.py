#!/usr/bin/env python3
"""B70 expert worker (optimized) — runs under .venv-xpu.

Pre-allocates all tensors. Minimizes Python overhead in the hot loop.
Reads activation from shared memory via numpy zero-copy, computes on B70,
writes partial back. Uses a simple spin-poll on the sequence counter.
"""
import sys, os, struct, time, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm-name", required=True)
    ap.add_argument("--n-experts", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=512)
    args = ap.parse_args()

    import torch
    dev = torch.device("xpu:0")
    H, I, E = args.hidden, args.inter, args.n_experts

    print(f"[b70] {torch.xpu.get_device_name(0)} | torch {torch.__version__}", flush=True)
    print(f"[b70] loading {E} experts...", flush=True)

    wg = torch.randn(E, H, I, dtype=torch.bfloat16, device=dev)
    wu = torch.randn(E, H, I, dtype=torch.bfloat16, device=dev)
    wd = torch.randn(E, I, H, dtype=torch.bfloat16, device=dev)

    # Pre-allocate activation tensor on B70 (reused every call)
    act_xpu = torch.empty(H, dtype=torch.float32, device=dev)
    partial = torch.zeros(H, dtype=torch.float32, device=dev)
    torch.xpu.synchronize()
    print("[b70] ready", flush=True)

    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=args.shm_name)
    buf = shm.buf
    SHUTDOWN = 0xFFFF_FFFF_FFFF_FFFF
    last_seq = 0
    n_calls = 0
    t_start = time.perf_counter()

    while True:
        seq_in = struct.unpack_from("<Q", buf, 0)[0]
        if seq_in == SHUTDOWN:
            break
        if seq_in == last_seq:
            continue  # tight spin, no sleep
        last_seq = seq_in
        n_calls += 1

        n = struct.unpack_from("<I", buf, 16)[0]
        eids = struct.unpack_from("<8I", buf, 20)
        wts = np.frombuffer(buf, dtype=np.float32, count=8, offset=52)

        # Zero-copy read: create numpy view over shared memory, copy to pre-allocated XPU tensor
        act_np = np.frombuffer(buf, dtype=np.float16, count=H, offset=84).astype(np.float32)
        act_xpu.copy_(torch.from_numpy(act_np))

        # Compute
        partial.zero_()
        for j in range(n):
            eid = eids[j]
            g = (act_xpu @ wg[eid]).float()
            u = (act_xpu @ wu[eid]).float()
            a = torch.nn.functional.silu(g) * u
            partial += wts[j] * (a @ wd[eid].float()).float()

        torch.xpu.synchronize()

        # Write partial back (single memcpy)
        out_np = partial.cpu().numpy()
        buf[4180:4180+H*4] = out_np.tobytes()
        struct.pack_into("<Q", buf, 8, seq_in)

    elapsed = time.perf_counter() - t_start
    print(f"[b70] done: {n_calls} calls in {elapsed:.2f}s ({elapsed/max(1,n_calls)*1e6:.0f} us/call)", flush=True)
    shm.close()

if __name__ == "__main__":
    main()
