#!/usr/bin/env python3
"""Correctness gate for the CPU DDR5 expert tier.

Checks the native SwiGLU kernels and the routed-batch contract against a
PyTorch fp32 reference. The native side widens bf16 to fp32 and accumulates
in fp32, so the reference does exactly that (``.float()`` before matmul) and
the only expected divergence is accumulation order.

Run::

    make -C phase7 cpu
    .venv/bin/python phase7/cpu_expert_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "phase4" / "src"))

import nvfp4_testutil  # noqa: E402
from shooting_brake_vllm.cpu_expert_host import (  # noqa: E402
    CpuExpertError,
    CpuExpertHost,
    PackedPlane,
)

LAYERS = 4
EXPERTS = 8
HIDDEN = 256
INTER = 128
TOPK = 4

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def close(name: str, got: torch.Tensor, want: torch.Tensor,
          rtol: float = 1e-3, atol: float = 1e-3) -> None:
    ok = torch.allclose(got, want, rtol=rtol, atol=atol)
    err = (got - want).abs().max().item() if got.shape == want.shape else float("nan")
    scale = want.abs().max().item()
    check(name, ok, f"max|err|={err:.3e} (ref max={scale:.3e})")


def reference_ffn(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
                  down: torch.Tensor) -> torch.Tensor:
    """y = (silu(x @ gate^T) * (x @ up^T)) @ down^T, all in fp32."""
    return nvfp4_testutil.ffn(x, gate, up, down)


def make_expert(
    gen: torch.Generator,
) -> tuple[tuple, tuple]:
    """Packed planes plus their dequantized references.

    Both are needed: the arena stores packed NVFP4, while the checks compare
    against vLLM's dequantization of those same bytes.
    """
    return nvfp4_testutil.make_expert(HIDDEN, INTER, gen)


def main() -> int:
    gen = torch.Generator().manual_seed(20260807)
    lib = Path(__file__).resolve().parent / "libsb_cpu_expert.so"
    if not lib.is_file():
        print(f"missing {lib}; run: make -C phase7 cpu")
        return 1

    print("== arena + residency ==")
    host = CpuExpertHost(
        num_layers=LAYERS, num_experts=EXPERTS, hidden=HIDDEN,
        intermediate=INTER, max_experts=LAYERS * EXPERTS, num_threads=4,
        lib_path=lib,
    )
    check("host created", host.resident_count == 0,
          f"resident={host.resident_count}")
    check("arena reserved", host.arena_capacity_bytes > 0,
          f"{host.arena_capacity_bytes / 2**20:.1f} MiB")

    weights: dict[tuple[int, int], tuple] = {}
    # Packed planes kept alongside their dequantized references: the guards
    # below need well-formed planes to pair with a deliberately broken one.
    packed: dict[tuple[int, int], tuple] = {}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            planes, refs = make_expert(gen)
            weights[(layer, expert)] = refs
            packed[(layer, expert)] = planes
            host.load_expert(layer, expert, *planes)
    check("all experts loaded", host.resident_count == LAYERS * EXPERTS,
          f"resident={host.resident_count}")
    check("arena used <= capacity",
          host.arena_used_bytes <= host.arena_capacity_bytes,
          f"{host.arena_used_bytes / 2**20:.1f} / "
          f"{host.arena_capacity_bytes / 2**20:.1f} MiB")
    check("has_expert true for loaded", host.has_expert(2, 5))

    print("\n== single-expert FFN vs PyTorch reference ==")
    for M in (1, 3, 16):
        x = (torch.randn(M, HIDDEN, generator=gen) * 0.5).bfloat16()
        got = host.expert_forward(1, 3, x)
        want = reference_ffn(x, *weights[(1, 3)])
        close(f"expert_forward M={M}", got, want)

    print("\n== routed batch: weighted accumulation ==")
    # One token, one route: output must be exactly weight * FFN.
    x = (torch.randn(1, HIDDEN, generator=gen) * 0.5).bfloat16()
    ids = torch.tensor([[4, -1, -1, -1]], dtype=torch.int32)
    rw = torch.tensor([[0.75, 0.0, 0.0, 0.0]], dtype=torch.float32)
    got = host.moe_forward(0, x, ids, rw)
    want = 0.75 * reference_ffn(x, *weights[(0, 4)])
    close("single route, weight applied", got, want)

    # One token, four routes: sum of independently weighted experts.
    ids = torch.tensor([[0, 2, 5, 7]], dtype=torch.int32)
    rw = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    got = host.moe_forward(0, x, ids, rw)
    want = sum(
        rw[0, k].item() * reference_ffn(x, *weights[(0, int(ids[0, k]))])
        for k in range(TOPK)
    )
    close("four routes, summed", got, want)

    print("\n== bucketing: many tokens through shared experts ==")
    # Every token routes to expert 3 plus one distinct expert. The native
    # side buckets by expert and computes each weight set once, so this is
    # where a gather/scatter index error would surface.
    M = 12
    x = (torch.randn(M, HIDDEN, generator=gen) * 0.5).bfloat16()
    ids = torch.zeros(M, TOPK, dtype=torch.int32)
    rw = torch.zeros(M, TOPK, dtype=torch.float32)
    for m in range(M):
        ids[m] = torch.tensor([3, m % EXPERTS, -1, -1], dtype=torch.int32)
        rw[m] = torch.tensor([0.6, 0.4, 0.0, 0.0], dtype=torch.float32)
    got = host.moe_forward(2, x, ids, rw)
    want = torch.zeros(M, HIDDEN, dtype=torch.float32)
    for m in range(M):
        for k in range(TOPK):
            e = int(ids[m, k])
            if e < 0:
                continue
            want[m] += rw[m, k].item() * reference_ffn(
                x[m:m + 1], *weights[(2, e)]).squeeze(0)
    close("12 tokens, shared + distinct experts", got, want)

    print("\n== contract: skips and zeros ==")
    # A token with no routes must come back exactly zero, not stale scratch.
    ids = torch.full((2, TOPK), -1, dtype=torch.int32)
    rw = torch.zeros(2, TOPK, dtype=torch.float32)
    got = host.moe_forward(0, (torch.randn(2, HIDDEN, generator=gen)).bfloat16(),
                           ids, rw)
    check("all-skip batch returns zeros", bool((got == 0).all()),
          f"max|out|={got.abs().max().item():.3e}")

    # Output is zero-initialised per call, so a smaller follow-up batch
    # cannot inherit the previous one's values.
    before = host.skipped_routes
    check("no skipped routes on resident batches", before == 0,
          f"skipped={before}")

    print("\n== contract: non-resident route is counted, not silently dropped ==")
    sparse = CpuExpertHost(
        num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, intermediate=INTER,
        max_experts=1, num_threads=2, lib_path=lib,
    )
    w0 = make_expert(gen)
    sparse.load_expert(0, 0, *w0[0])
    x1 = (torch.randn(1, HIDDEN, generator=gen) * 0.5).bfloat16()
    ids = torch.tensor([[0, 6, -1, -1]], dtype=torch.int32)  # 6 not resident
    rw = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float32)
    got = sparse.moe_forward(0, x1, ids, rw)
    want = 0.5 * reference_ffn(x1, *w0[1])
    close("resident route still correct", got, want)
    check("non-resident route counted", sparse.skipped_routes == 1,
          f"skipped={sparse.skipped_routes}")
    sparse.close()

    print("\n== contract: input validation ==")
    for name, fn in [
        ("unpacked (full-width) weights rejected", lambda: host.load_expert(
            0, 0,
            PackedPlane(torch.zeros(INTER, HIDDEN, dtype=torch.uint8),
                        torch.zeros(INTER, HIDDEN // 16, dtype=torch.uint8),
                        1.0),
            *packed[(0, 0)][1:])),
        ("transposed gate rejected", lambda: host.load_expert(
            0, 0,
            PackedPlane(torch.zeros(HIDDEN, INTER // 2, dtype=torch.uint8),
                        torch.zeros(HIDDEN, INTER // 16, dtype=torch.uint8),
                        1.0),
            *packed[(0, 0)][1:])),
        ("fp32 block scales rejected", lambda: host.load_expert(
            0, 0,
            PackedPlane(torch.zeros(INTER, HIDDEN // 2, dtype=torch.uint8),
                        torch.zeros(INTER, HIDDEN // 16), 1.0),
            *packed[(0, 0)][1:])),
        ("fp32 hidden rejected", lambda: host.moe_forward(
            0, torch.zeros(1, HIDDEN), torch.zeros(1, TOPK, dtype=torch.int32),
            torch.zeros(1, TOPK))),
        ("mismatched ids/weights rejected", lambda: host.moe_forward(
            0, torch.zeros(1, HIDDEN, dtype=torch.bfloat16),
            torch.zeros(1, TOPK, dtype=torch.int32), torch.zeros(1, 2))),
    ]:
        try:
            fn()
            check(name, False, "no error raised")
        except CpuExpertError:
            check(name, True)

    print("\n== reload in place does not grow the arena ==")
    used_before = host.arena_used_bytes
    host.load_expert(0, 0, *packed[(0, 0)])
    check("reload reuses slot", host.arena_used_bytes == used_before,
          f"{used_before} -> {host.arena_used_bytes}")

    host.close()

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
