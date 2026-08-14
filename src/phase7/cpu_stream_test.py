"""Correctness gate for Phase 6 — streamed expert compute on CUDA.

Checks the streaming path against the CPU tier it stands in for. Both read
the same arena weights, so they must produce the same routed partial; only
where the arithmetic happens differs. A layout error in the block slicing,
a wrong gate/up split, or a ring slot overwritten while still being read all
show up here as a mismatch rather than as slightly worse output later.

The tolerance is loose on purpose: the reference accumulates in fp32 on CPU
while CUDA runs bf16 GEMMs with fp32 accumulate and a different reduction
order, so exact equality is not the property under test. Layout faults are
off by orders of magnitude, not by a few ULP.

Run: python src/phase7/cpu_stream_test.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src/phase4/src")

from shooting_brake_vllm.cpu_expert_host import (  # noqa: E402
    CpuExpertHost,
    PackedPlane,
)
from shooting_brake_vllm.cpu_stream import ExpertStreamer  # noqa: E402

HIDDEN = 512
INTER = 192
NUM_LAYERS = 3
NUM_EXPERTS = 16
TOPK = 4
BLOCK = 16

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def make_plane(rows: int, cols: int, gen: torch.Generator) -> PackedPlane:
    """A packed NVFP4 plane covering every nibble code and a scale band."""
    q = torch.randint(0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8)
    exp = torch.randint(4, 11, (rows, cols // BLOCK), generator=gen)
    man = torch.randint(0, 8, (rows, cols // BLOCK), generator=gen)
    sf = ((exp << 3) | man).to(torch.uint8)
    return PackedPlane(q.contiguous(), sf.contiguous(), 1.0)


def make_expert(gen: torch.Generator) -> tuple[PackedPlane, ...]:
    return (make_plane(INTER, HIDDEN, gen),
            make_plane(INTER, HIDDEN, gen),
            make_plane(HIDDEN, INTER, gen))


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable — Phase 6 streaming cannot be tested")
        return 1

    gen = torch.Generator().manual_seed(7)
    host = CpuExpertHost(
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden=HIDDEN,
        intermediate=INTER,
        max_experts=NUM_LAYERS * NUM_EXPERTS,
    )

    print("== arena load")
    weights: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
    for layer in range(NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            w = make_expert(gen)
            weights[(layer, expert)] = w
            host.load_expert(layer, expert, *w)
    check("all experts resident",
          host.resident_count == NUM_LAYERS * NUM_EXPERTS,
          f"{host.resident_count} experts, {host.arena_used_bytes / 2**20:.1f} MiB")

    print("== arena block view")
    # The block must alias the arena and expose the six packed sections in
    # order; if they were not one allocation this is where it breaks.
    blk = host.expert_block(0, 0)
    elems = HIDDEN * INTER
    nq, nsf = elems // 2, elems // BLOCK
    want_bytes = 3 * (nq + nsf)
    check("block is one contiguous packed extent", blk.numel() == want_bytes,
          f"{blk.numel()} bytes, expected {want_bytes}")
    g, u, d = weights[(0, 0)]
    for i, (name, plane, rows, cols) in enumerate((
        ("gate", g, INTER, HIDDEN),
        ("up", u, INTER, HIDDEN),
        ("down", d, HIDDEN, INTER),
    )):
        base = i * (nq + nsf)
        check(f"{name} nibbles match",
              torch.equal(blk[base:base + nq].view(rows, cols // 2), plane.q))
        check(f"{name} block scales match",
              torch.equal(
                  blk[base + nq:base + nq + nsf].view(rows, cols // BLOCK),
                  plane.sf))

    print("== arena pinning")
    pinned = host.pin_arena()
    check("arena registered with CUDA", pinned,
          "cudaHostRegister ok" if pinned else "unpinned — copies stay staged")

    print("== streamed vs CPU-tier partial")
    # Enough slots to force ring reuse, so a slot overwritten while still
    # being read would corrupt a later expert rather than going unnoticed.
    streamer = ExpertStreamer(host, HIDDEN, INTER, slots=2)

    for label, M in (("M=1 (decode shape)", 1), ("M=64", 64), ("M=257", 257)):
        ids = torch.randint(
            0, NUM_EXPERTS, (M, TOPK), generator=gen, dtype=torch.int32
        )
        w = torch.rand(M, TOPK, generator=gen, dtype=torch.float32)
        x = (torch.randn(M, HIDDEN, generator=gen) * 0.5).bfloat16()

        want = host.moe_forward(0, x, ids, w)
        got = streamer.forward(
            0, x.cuda(), ids.cuda().long(), w.cuda(),
        ).float().cpu()

        err = (got - want).abs().max().item()
        scale = max(want.abs().max().item(), 1e-6)
        check(f"{label} matches CPU tier", err / scale < 2e-2,
              f"max|err|={err:.3e} rel={err / scale:.3e}")

    print("== route selection")
    # -1 means the route belongs to another tier; it must contribute nothing.
    M = 32
    ids = torch.full((M, TOPK), -1, dtype=torch.int32)
    ids[:, 0] = 3
    w = torch.rand(M, TOPK, generator=gen, dtype=torch.float32)
    x = (torch.randn(M, HIDDEN, generator=gen) * 0.5).bfloat16()
    want = host.moe_forward(0, x, ids, w)
    got = streamer.forward(0, x.cuda(), ids.cuda().long(), w.cuda()).float().cpu()
    err = (got - want).abs().max().item()
    check("-1 routes skipped, single expert honoured",
          err / max(want.abs().max().item(), 1e-6) < 2e-2,
          f"max|err|={err:.3e}")

    all_none = torch.full((M, TOPK), -1, dtype=torch.int32)
    got = streamer.forward(
        0, x.cuda(), all_none.cuda().long(), w.cuda()
    ).float().cpu()
    check("no cold routes returns exactly zero",
          bool(torch.all(got == 0)), f"max={got.abs().max().item():.3e}")

    print("== per-layer isolation")
    # Same expert id in different layers holds different weights; a streamer
    # that ignored layer_idx would silently return the wrong expert.
    ids = torch.zeros(8, TOPK, dtype=torch.int32)
    ids[:, 1:] = -1
    w = torch.ones(8, TOPK, dtype=torch.float32)
    x = (torch.randn(8, HIDDEN, generator=gen) * 0.5).bfloat16()
    y0 = streamer.forward(0, x.cuda(), ids.cuda().long(), w.cuda()).float().cpu()
    y2 = streamer.forward(2, x.cuda(), ids.cuda().long(), w.cuda()).float().cpu()
    ref2 = host.moe_forward(2, x, ids, w)
    check("layer 2 differs from layer 0",
          not torch.allclose(y0, y2, atol=1e-3))
    check("layer 2 matches its own arena weights",
          (y2 - ref2).abs().max().item()
          / max(ref2.abs().max().item(), 1e-6) < 2e-2)

    print("== stats")
    st = streamer.stats
    check("streamed expert count recorded", st["experts_streamed"] > 0, str(st))

    host.close()
    print()
    if _failures:
        print(f"FAILED {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
