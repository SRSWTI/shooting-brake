"""Correctness gate for the packed-NVFP4 arena.

The arena stores weights exactly as the checkpoint carries them and
reconstructs ``e2m1(nibble) * e4m3(blockscale) * gscale``. Every part of that
has a plausible-but-wrong variant: nibble order within a byte, the sign
convention, e4m3's subnormal branch, and above all the direction the global
scale folds -- vLLM's own reference quantiser divides by it where
``dequantize_to_dtype`` multiplies. Each mistake yields believable numbers
rather than an error.

So the reference here is vLLM's ``dequantize_to_dtype`` applied to the *same
bytes* the arena was handed. Anything the two disagree on is a decode bug,
and no quantiser convention has to be guessed at: random packed bytes are
generated directly, which also exercises magnitudes and scale exponents a
real checkpoint might never produce.

Run: python phase7/cpu_packed_test.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "phase4/src")

from shooting_brake_vllm.cpu_expert_host import (  # noqa: E402
    CpuExpertHost,
    PackedPlane,
)

HIDDEN = 256
INTER = 128
NUM_LAYERS = 2
NUM_EXPERTS = 4
TOPK = 2
BLOCK = 16

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def random_plane(rows: int, cols: int, gen: torch.Generator) -> PackedPlane:
    """A packed plane with deliberately wide coverage.

    Nibbles are uniform over all 16 codes, so both signs, zero, the
    subnormal 0.5 and the top magnitude 6 all appear. Scale bytes avoid
    exponent 0 (whose subnormals are vanishingly small) and the NaN code, but
    otherwise range freely.
    """
    q = torch.randint(0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8)
    # e4m3 bytes with exponent in [4, 10] keeps scales in a sane magnitude
    # band while still varying mantissa and sign.
    exp = torch.randint(4, 11, (rows, cols // BLOCK), generator=gen)
    man = torch.randint(0, 8, (rows, cols // BLOCK), generator=gen)
    sgn = torch.zeros_like(exp)  # weight scales are non-negative in practice
    sf = ((sgn << 7) | (exp << 3) | man).to(torch.uint8)
    gscale = float(torch.empty(1).uniform_(0.5, 2.0, generator=gen).item())
    return PackedPlane(q.contiguous(), sf.contiguous(), gscale)


def reference(plane: PackedPlane, rows: int, cols: int) -> torch.Tensor:
    """Dequantize with vLLM, from the identical bytes the arena holds."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
        dequantize_to_dtype,
    )

    # swizzle=False selects vLLM's linear-scale path, which is a Triton
    # kernel and therefore needs device tensors. The arena stores linear
    # scales, so this is the matching reference; the checkpoint's swizzled
    # layout is undone at load time, not here.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dequantize_to_dtype(
        plane.q.to(dev),
        plane.sf.to(dev),
        torch.tensor(plane.gscale, dtype=torch.float32, device=dev),
        torch.float32,
        block_size=BLOCK,
        swizzle=False,
    ).reshape(rows, cols).cpu()


def main() -> int:
    gen = torch.Generator().manual_seed(19)
    host = CpuExpertHost(
        num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS, hidden=HIDDEN,
        intermediate=INTER, max_experts=NUM_LAYERS * NUM_EXPERTS,
    )

    print("== packing density")
    elems = HIDDEN * INTER
    want_bytes = 3 * (elems // 2 + elems // BLOCK)
    planes = {}
    for layer in range(NUM_LAYERS):
        for e in range(NUM_EXPERTS):
            g = random_plane(INTER, HIDDEN, gen)
            u = random_plane(INTER, HIDDEN, gen)
            d = random_plane(HIDDEN, INTER, gen)
            planes[(layer, e)] = (g, u, d)
            host.load_expert(layer, e, g, u, d)

    per_expert = host.arena_used_bytes / host.resident_count
    check("arena holds packed bytes, not dequantized",
          abs(per_expert - want_bytes) <= 64,
          f"{per_expert:.0f} B/expert, expected {want_bytes}")
    bf16_equiv = 3 * elems * 2
    check("3.5x smaller than bf16 storage",
          per_expert < bf16_equiv / 3.4,
          f"{bf16_equiv / per_expert:.2f}x reduction")

    print("== decode matches vLLM dequantize_to_dtype")
    # One expert's FFN, computed from the arena, against torch over vLLM's
    # own dequantization of the same bytes. A scale-direction or nibble-order
    # error cannot survive this.
    g, u, d = planes[(0, 0)]
    gate = reference(g, INTER, HIDDEN)
    up = reference(u, INTER, HIDDEN)
    down = reference(d, HIDDEN, INTER)

    for M in (1, 5, 33):
        x = (torch.randn(M, HIDDEN, generator=gen) * 0.5).bfloat16()
        xf = x.float()
        want = (torch.nn.functional.silu(xf @ gate.T) * (xf @ up.T)) @ down.T
        got = host.expert_forward(0, 0, x)
        err = (got - want).abs().max().item()
        scale = max(want.abs().max().item(), 1e-6)
        check(f"M={M} FFN matches reference", err / scale < 5e-3,
              f"max|err|={err:.3e} rel={err / scale:.3e}")

    print("== per-plane global scales survive the round trip")
    gs = host.expert_gscales(0, 0)
    check("gscales returned in gate/up/down order",
          all(abs(a - b) < 1e-6 for a, b in
              zip(gs, (g.gscale, u.gscale, d.gscale))),
          f"{gs} vs {(g.gscale, u.gscale, d.gscale)}")

    print("== routed batch")
    M = 12
    ids = torch.randint(0, NUM_EXPERTS, (M, TOPK), generator=gen, dtype=torch.int32)
    w = torch.rand(M, TOPK, generator=gen, dtype=torch.float32)
    x = (torch.randn(M, HIDDEN, generator=gen) * 0.5).bfloat16()
    got = host.moe_forward(1, x, ids, w)

    want = torch.zeros(M, HIDDEN, dtype=torch.float32)
    for e in range(NUM_EXPERTS):
        pg, pu, pd = planes[(1, e)]
        ge, ue, de = (reference(pg, INTER, HIDDEN), reference(pu, INTER, HIDDEN),
                      reference(pd, HIDDEN, INTER))
        for m in range(M):
            for k in range(TOPK):
                if int(ids[m, k]) != e:
                    continue
                xf = x[m: m + 1].float()
                y = (torch.nn.functional.silu(xf @ ge.T) * (xf @ ue.T)) @ de.T
                want[m] += float(w[m, k]) * y[0]

    err = (got - want).abs().max().item()
    check("routed batch matches per-route reference",
          err / max(want.abs().max().item(), 1e-6) < 5e-3,
          f"max|err|={err:.3e}")
    check("no routes skipped", host.skipped_routes == 0,
          f"skipped={host.skipped_routes}")

    print("== guards")
    try:
        bad = PackedPlane(
            torch.zeros(INTER, HIDDEN, dtype=torch.uint8),  # unpacked width
            torch.zeros(INTER, HIDDEN // BLOCK, dtype=torch.uint8), 1.0,
        )
        host.load_expert(0, 1, bad, bad, bad)
        check("rejects a plane that is not nibble-packed", False, "accepted")
    except Exception as exc:
        check("rejects a plane that is not nibble-packed", True,
              type(exc).__name__)

    host.close()
    print()
    if _failures:
        print(f"FAILED {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
