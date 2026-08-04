#!/usr/bin/env python3
"""B70 ESIMD expert worker — uses PyTorch XPU with BF16 (simplest fast path).

Receives activation via shared memory, computes assigned experts on B70
using standard PyTorch BF16 matmul (45 us/expert as measured).
Weights pre-loaded from a file at startup.

Run under .venv-xpu.
"""
import sys, os, struct, time, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm-name", required=True)
    ap.add_argument("--weights-file", required=True, help="FP32 binary file with expert weights")
    ap.add_argument("--n-experts", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=512)
    args = ap.parse_args()

    import torch
    dev = torch.device("xpu:0")
    H, I, E = args.hidden, args.inter, args.n_experts
    print(f"[b70] {torch.xpu.get_device_name(0)} | H={H} I={I} E={E}", flush=True)

    # Load weights from file: [E, 3, max(I*H, H*I)] float32
    # Layout per expert: gate[I*H], up[I*H], down[H*I]
    print(f"[b70] loading weights from {args.weights_file}...", flush=True)
    wdata = np.fromfile(args.weights_file, dtype=np.float32)
    # Each expert: I*H + I*H + H*I = 2*I*H + H*I floats
    per_expert = 2*I*H + H*I
    assert len(wdata) >= E * per_expert, f"Expected {E*per_expert} floats, got {len(wdata)}"
    wdata = wdata[:E*per_expert].reshape(E, per_expert)

    # Create BF16 tensors on B70
    w_gate = torch.empty(E, I, H, dtype=torch.bfloat16, device=dev)
    w_up   = torch.empty(E, I, H, dtype=torch.bfloat16, device=dev)
    w_down = torch.empty(E, H, I, dtype=torch.bfloat16, device=dev)
    for e in range(E):
        off = e * per_expert
        g = torch.from_numpy(wdata[off:off+I*H].reshape(I, H).copy())
        u = torch.from_numpy(wdata[off+I*H:off+2*I*H].reshape(I, H).copy())
        d = torch.from_numpy(wdata[off+2*I*H:off+2*I*H+H*I].reshape(H, I).copy())
        w_gate[e] = g.to(dev).bfloat16()
        w_up[e] = u.to(dev).bfloat16()
        w_down[e] = d.to(dev).bfloat16()
    del wdata
    print(f"[b70] weights loaded ({E} experts)", flush=True)

    # Pre-allocate buffers
    x_buf = torch.empty(H, dtype=torch.bfloat16, device=dev)
    inter = torch.empty(I, dtype=torch.float32, device=dev)
    out_buf = torch.zeros(H, dtype=torch.float32, device=dev)

    print("[b70] ready", flush=True)

    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=args.shm_name)
    buf = shm.buf
    SHUTDOWN = 0xFFFF_FFFF_FFFF_FFFF
    last_seq = 0

    while True:
        seq_in = struct.unpack_from("<Q", buf, 0)[0]
        if seq_in == SHUTDOWN:
            break
        if seq_in == last_seq:
            continue
        last_seq = seq_in

        n = struct.unpack_from("<I", buf, 16)[0]
        eids = struct.unpack_from("<8I", buf, 20)
        wts = np.frombuffer(buf, dtype=np.float32, count=8, offset=52)
        act_np = np.frombuffer(buf, dtype=np.float32, count=H, offset=84).copy()
        x_buf.copy_(torch.from_numpy(act_np).to(dev).bfloat16())

        out_buf.zero_()
        for j in range(n):
            e = eids[j]
            if e >= E: continue
            # gate+up GEMV
            g = (x_buf.float() @ w_gate[e].float().t())
            u = (x_buf.float() @ w_up[e].float().t())
            # SiLU(g)*u
            a = (torch.sigmoid(g) * g * u)
            # down GEMV
            y = a @ w_down[e].float().t()
            out_buf += wts[j] * y

        torch.xpu.synchronize()
        out_np = out_buf.cpu().numpy()
        buf[4180:4180+H*4] = out_np.tobytes()
        struct.pack_into("<Q", buf, 8, seq_in)

    shm.close()
    print("[b70] shutdown", flush=True)

if __name__ == "__main__":
    main()
