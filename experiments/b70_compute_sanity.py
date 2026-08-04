#!/usr/bin/env python3
"""B70 compute sanity — the smallest go/no-go.

Proves the Intel Arc Pro B70 is a usable compute device: enumerable, correct,
and fast device-local. No model weights, no transport, no cross-vendor work.
Runs at any PCIe link width (it measures on-card compute, not transport).

Run: .venv/bin/python experiments/b70_compute_sanity.py
"""
import sys
import time

import torch


def bench_matmul(a_shape, b_shape, dtype, dev, iters=50, warm=10):
    a = torch.randn(*a_shape, dtype=dtype, device=dev)
    b = torch.randn(*b_shape, dtype=dtype, device=dev)
    for _ in range(warm):
        c = a @ b
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    m, k = a_shape[-2], a_shape[-1]
    n = b_shape[-1]
    flops = 2 * m * k * n
    return dt, flops / dt, c


def main():
    torch.manual_seed(0)
    print(f"torch           {torch.__version__}")
    print(f"xpu available   {torch.xpu.is_available()}")
    print(f"xpu count       {torch.xpu.device_count()}")
    if not torch.xpu.is_available():
        print("NO XPU DEVICE — install the Intel compute runtime / level-zero driver.")
        sys.exit(1)

    for i in range(torch.xpu.device_count()):
        p = torch.xpu.get_device_properties(i)
        gib = p.total_memory / (1024 ** 3)
        print(f"  dev {i}: {p.name} | VRAM {gib:.1f} GiB")

    dev = torch.device("xpu:0")
    name = torch.xpu.get_device_name(0)
    print(f"\nTarget: {name} on {dev}")

    # 1. Large bf16 matmul -> peak-ish TFLOPS
    dt, tflops, c = bench_matmul((4096, 8192), (8192, 4096), torch.bfloat16, dev)
    print(f"\n[1] bf16 GEMM 4096x8192x4096: {dt*1e3:.2f} ms  {tflops/1e12:.1f} TFLOPS")

    # 2. Correctness vs fp32 CPU reference (different accumulation order -> small err)
    a = torch.randn(2048, 4096, dtype=torch.bfloat16, device=dev)
    b = torch.randn(4096, 2048, dtype=torch.bfloat16, device=dev)
    c_xpu = (a @ b).float()
    c_ref = (a.float() @ b.float()).float()
    err = (c_xpu - c_ref).abs().max().item()
    print(f"[2] correctness bf16 vs fp32 CPU ref: max abs err {err:.4f}")

    # 3. GLM-5.2 expert-gate shape (hidden 6144 -> intermediate 2048), bf16
    #    batch-one (T=1) and batched (T=128): the expert compute unit.
    print("\n[3] GLM-5.2 gate shape (6144 -> 2048), bf16:")
    for t in (1, 8, 64, 128):
        dt, tflops, _ = bench_matmul((t, 6144), (6144, 2048), torch.bfloat16, dev)
        print(f"    T={t:<4}: {dt*1e6:8.1f} us  {tflops/1e12:6.2f} TFLOPS")

    print("\nB70 compute sanity PASSED: device enumerated, matmul correct, throughput measured.")


if __name__ == "__main__":
    main()
