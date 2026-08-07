"""Locate the batch size where streaming cold experts beats computing them.

Both tiers read the same ~9 MiB of weights per expert, so at one token they
cost about the same and the choice barely matters. What separates them is how
that cost scales: weight traffic is fixed per expert, arithmetic grows with
the tokens routed to it. The CPU therefore starts bandwidth-bound and turns
compute-bound, while streaming stays transfer-bound and amortises across
every token in the batch.

This measures where the two curves cross, which is the only defensible way to
set SHOOTING_BRAKE_CPU_STREAM_T. Model shape and routing statistics both move
that point, so it is a per-deployment number rather than a constant.

Routing is sampled the way the real placement produces it -- top-k drawn from
the full expert range, with only the cold subset resident -- so the token
count each cold expert receives grows the way it actually does in serving,
rather than every expert getting the whole batch.

Run: python phase7/stream_crossover_bench.py
"""

from __future__ import annotations

import statistics
import sys
import time

import torch

sys.path.insert(0, "phase4/src")

from shooting_brake_vllm.cpu_expert_host import CpuExpertHost  # noqa: E402
from shooting_brake_vllm.cpu_stream import ExpertStreamer  # noqa: E402

# Qualified model shape (Qwen3.6-35B-A3B): 9.0 MiB per expert in bf16.
HIDDEN = 2048
INTER = 768
NUM_EXPERTS = 256
TOPK = 8

#: Cold experts per layer, matching allout:16:8:8.
CPU_PER_LAYER = 8

BATCHES = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048)
REPEATS = 7
WARMUP = 2


def build_routes(
    M: int, cold: list[int], gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k routing over the full expert range, cold routes kept.

    Routes landing on resident experts keep their id; everything else is -1,
    exactly as the partition hands it to either tier.
    """
    ids = torch.randint(
        0, NUM_EXPERTS, (M, TOPK), generator=gen, dtype=torch.int32
    )
    cold_set = torch.zeros(NUM_EXPERTS, dtype=torch.bool)
    cold_set[torch.tensor(cold)] = True
    ids = torch.where(cold_set[ids.long()], ids, torch.full_like(ids, -1))
    weights = torch.rand(M, TOPK, generator=gen, dtype=torch.float32)
    return ids, weights


def timed(fn, repeats: int = REPEATS, warmup: int = WARMUP) -> float:
    """Median wall time in microseconds; median resists a stray scheduler hit."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable — cannot measure the streaming side")
        return 1

    gen = torch.Generator().manual_seed(11)
    cold = list(range(NUM_EXPERTS - CPU_PER_LAYER, NUM_EXPERTS))

    host = CpuExpertHost(
        num_layers=1, num_experts=NUM_EXPERTS, hidden=HIDDEN,
        intermediate=INTER, max_experts=CPU_PER_LAYER,
    )
    print(f"loading {CPU_PER_LAYER} cold experts "
          f"({3 * HIDDEN * INTER * 2 / 2**20:.1f} MiB each)")
    for e in cold:
        host.load_expert(
            0, e,
            (torch.randn(INTER, HIDDEN, generator=gen) * 0.05).bfloat16(),
            (torch.randn(INTER, HIDDEN, generator=gen) * 0.05).bfloat16(),
            (torch.randn(HIDDEN, INTER, generator=gen) * 0.05).bfloat16(),
        )
    pinned = host.pin_arena()
    print(f"arena {host.arena_used_bytes / 2**20:.1f} MiB, "
          f"{'pinned (DMA)' if pinned else 'UNPINNED — staged copies'}")

    streamer = ExpertStreamer(host, HIDDEN, INTER)
    print(f"ring: {streamer.stats['slots']} slots x "
          f"{streamer.stats['slot_bytes'] / 2**20:.1f} MiB\n")

    print(f"{'M':>6} {'cold routes':>12} {'CPU cores':>12} {'streamed':>12} "
          f"{'speedup':>9}  winner")
    print("-" * 70)

    crossover = None
    for M in BATCHES:
        ids, weights = build_routes(M, cold, gen)
        n_cold = int((ids >= 0).sum())
        x = (torch.randn(M, HIDDEN, generator=gen) * 0.5).bfloat16()

        xc, idc, wc = x.cuda(), ids.cuda().long(), weights.cuda()

        cpu_us = timed(lambda: host.moe_forward(0, x, ids, weights))
        gpu_us = timed(lambda: streamer.forward(0, xc, idc, wc))

        speedup = cpu_us / gpu_us if gpu_us > 0 else float("inf")
        winner = "stream" if gpu_us < cpu_us else "cpu"
        if crossover is None and gpu_us < cpu_us:
            crossover = M
        print(f"{M:>6} {n_cold:>12} {cpu_us:>11.1f}u {gpu_us:>11.1f}u "
              f"{speedup:>8.2f}x  {winner}")

    print()
    if crossover is None:
        print("CPU cores won at every batch size measured; leave streaming off "
              "or raise SHOOTING_BRAKE_CPU_STREAM_T above the largest batch.")
    else:
        print(f"crossover at M={crossover} — set "
              f"SHOOTING_BRAKE_CPU_STREAM_T={crossover}")
    print("Note both paths read identical weights; the divergence is purely "
          "how arithmetic scales with tokens per expert.")

    host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
