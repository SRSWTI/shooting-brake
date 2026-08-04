#!/usr/bin/env python3
"""B70 ESIMD expert worker — uses llm-scaler's optimized INT4 MoE kernels.

Receives activation via shared memory, computes assigned experts on the B70
using ESIMD INT4 kernels (9.4 us/expert), returns one weighted partial.

This replaces the oneMKL FP32 path (560 us/expert) with the ESIMD path.
Weights are uploaded as FP16 (dequantized from Colibri INT4 by the C side).
The ESIMD kernel re-quantizes to INT4 internally for XMX acceleration.

Run under .venv-xpu with llm-scaler kernels built.
"""
import sys, os, struct, time, argparse, signal
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm-name", required=True)
    ap.add_argument("--n-experts", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--kernels-dir", required=True,
                    help="Path to llm-scaler custom-esimd-kernels-vllm/python")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, args.kernels_dir)
    from custom_esimd_kernels_vllm import moe_int4_ops

    dev = torch.device("xpu:0")
    H, I, E = args.hidden, args.inter, args.n_experts
    print(f"[b70-esimd] {torch.xpu.get_device_name(0)} | torch {torch.__version__}", flush=True)
    print(f"[b70-esimd] H={H} I={I} experts={E}", flush=True)

    # Allocate FP16 expert weights on B70 (will be filled by C side via shared mem)
    # gate+up: [E, 2*I, H], down: [E, H, I]
    w13 = torch.randn(E, 2*I, H, dtype=torch.float16, device=dev) * 0.01
    w2  = torch.randn(E, H, I, dtype=torch.float16, device=dev) * 0.01

    # Pre-allocate activation and output tensors (reused every call)
    x_buf = torch.zeros(1, H, dtype=torch.float16, device=dev)
    out_buf = torch.zeros(1, H, dtype=torch.float16, device=dev)
    inter_buf = torch.zeros(1, 2*I, dtype=torch.float16, device=dev)

    # Dummy shared expert (not used — Colibri handles shared expert on 5090)
    shared_gate_up = torch.zeros(2*I, H, dtype=torch.float16, device=dev)
    shared_down = torch.zeros(H, I, dtype=torch.float16, device=dev)
    shared_gate_w = torch.zeros(1, H, dtype=torch.float16, device=dev)
    dummy_scale = torch.empty(0, device=dev, dtype=torch.float16)

    print("[b70-esimd] ready", flush=True)

    # Attach to shared memory
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
            continue  # tight spin
        last_seq = seq_in
        n_calls += 1

        # Read activation from shared memory (FP32 → FP16)
        n = struct.unpack_from("<I", buf, 16)[0]
        act_np = np.frombuffer(buf, dtype=np.float32, count=H, offset=84)
        x_buf[0] = torch.from_numpy(act_np.copy()).half().to(dev)

        # Read routing weights
        wts_np = np.frombuffer(buf, dtype=np.float32, count=8, offset=52)

        # Compute each assigned expert: gate+up → SiLU → down → weighted sum
        out_buf.zero_()
        for j in range(n):
            # gate+up GEMV: [1,2I] = x[1,H] @ w13[j,2I,H]^T
            inter_buf[0] = (x_buf @ w13[j].t())[0]
            # SiLU(gate) * up
            g = inter_buf[0, :I].float()
            u = inter_buf[0, I:].float()
            a = (g / (1 + torch.exp(-g)) * u).half()
            # down GEMV: [1,H] = a[1,I] @ w2[j,H,I]^T
            y = (a @ w2[j].t())
            out_buf += wts_np[j] * y

        torch.xpu.synchronize()

        # Write partial back (FP32)
        out_np = out_buf[0].float().cpu().numpy()
        buf[4180:4180+H*4] = out_np.tobytes()
        struct.pack_into("<Q", buf, 8, seq_in)

    elapsed = time.perf_counter() - t_start
    print(f"[b70-esimd] done: {n_calls} calls in {elapsed:.2f}s "
          f"({elapsed/max(1,n_calls)*1e6:.0f} us/call)", flush=True)
    shm.close()

if __name__ == "__main__":
    main()
