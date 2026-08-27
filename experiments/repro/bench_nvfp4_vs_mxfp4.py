"""Same-harness timing of the Xe2 grouped MoE GEMM across B-dtypes.

Every variant goes through the identical `cutlass_grouped_gemm_xe2` entry on the
same device at the same M/N/K/E, so the numbers are directly comparable. Run
with the freshly built kernel library on PYTHONPATH.
"""
import sys
import time

import torch

from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2

DEVICE = "xpu"


def rows_uniform(num_experts, rows_per_expert):
    return torch.full((num_experts,), rows_per_expert, dtype=torch.int32,
                      device=DEVICE)


def build(variant, num_experts, n, k, total_m, dtype):
    a = torch.randn((total_m, k), dtype=dtype, device=DEVICE).contiguous()
    out = torch.empty((total_m, n), dtype=dtype, device=DEVICE)

    if variant == "nvfp4":
        group_size = 16
        w = torch.randint(0, 0xff, [num_experts, n, k // 2],
                          device=DEVICE).to(torch.uint8)
        s = (torch.rand((num_experts, n, k // group_size), device=DEVICE) * 0.5
             + 0.5).to(torch.float8_e4m3fn)
        w = w.view(torch.float4_e2m1fn_x2)
    elif variant == "mxfp4":
        group_size = 32
        w = torch.randint(0, 0xff, [num_experts, n, k // 2],
                          device=DEVICE).to(torch.uint8)
        s = torch.randint(0, 0x7f, (num_experts, n, k // group_size),
                          dtype=torch.uint8, device=DEVICE)
        w = w.view(torch.float4_e2m1fn_x2)
    elif variant == "bits16":
        w = torch.randn((num_experts, k, n), dtype=dtype,
                        device=DEVICE).contiguous()
        s = None
    else:
        raise ValueError(variant)

    return a, w, s, out


def bench(variant, num_experts, n, k, rows_per_expert, dtype, iters=20):
    total_m = num_experts * rows_per_expert
    a, w, s, out = build(variant, num_experts, n, k, total_m, dtype)
    rows = rows_uniform(num_experts, rows_per_expert)

    def once():
        cutlass_grouped_gemm_xe2(a, w, s, None, out, rows, n, k, num_experts)

    for _ in range(5):
        once()
    torch.xpu.synchronize()

    best = float("inf")
    for _ in range(3):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            once()
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)

    flop = 2.0 * total_m * n * k
    return best * 1e3, flop / best / 1e12


def main():
    num_experts = 85
    n, k = 2048, 3072
    dtype = torch.float16

    print(f"E={num_experts} n={n} k={k} dtype={dtype}, "
          f"uniform rows/expert, best of 3 x 20 iters\n")
    print(f"{'variant':<10} {'rows/exp':>9} {'total_m':>8} {'ms':>9} {'TFLOP/s':>9}")
    for rows_per_expert in (30, 120):
        for variant in ("nvfp4", "mxfp4", "bits16"):
            try:
                ms, tflops = bench(variant, num_experts, n, k, rows_per_expert,
                                   dtype)
                print(f"{variant:<10} {rows_per_expert:>9} "
                      f"{num_experts * rows_per_expert:>8} {ms:>9.4f} "
                      f"{tflops:>9.1f}")
            except Exception as exc:  # noqa: BLE001
                print(f"{variant:<10} {rows_per_expert:>9} "
                      f"{'':>8} {'FAILED':>9}  {type(exc).__name__}: {exc}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
