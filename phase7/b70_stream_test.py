#!/usr/bin/env python3
"""Gates for B70 prefill weight streaming.

Streaming reuses the cold tier's DRAM arena for B70-owned experts, so the two
tiers now share one store keyed by ``(layer, expert)``. That sharing is what
makes a single streamer and a single ring serve both, and it is only safe
while the id sets are disjoint -- an overlap would have one tier's load
silently overwrite the other's, producing wrong weights rather than an error.
These tests pin that invariant across every policy, plus the branch predicate
that decides dispatch vs stream.

Numerical agreement between streamed weights and their source is already
covered by phase7/cpu_stream_test.py, which exercises the same
ExpertStreamer against dequantize_to_dtype; end-to-end agreement is covered
by phase7/prefill_probe.py via prompt logprobs. This file covers what those
two do not: the placement bookkeeping and the dispatch decision.

Run: python phase7/b70_stream_test.py    (no GPU, no B70, no model)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4" / "src"))

from shooting_brake_vllm.config import (  # noqa: E402
    QUALIFIED_ARCHITECTURE,
    QUALIFIED_EXPERTS,
    QUALIFIED_HIDDEN_SIZE,
    QUALIFIED_LAYERS,
    QUALIFIED_MODEL,
    QUALIFIED_MOE_INTERMEDIATE,
    QUALIFIED_TOP_K,
    QualifiedModel,
)
from shooting_brake_vllm.placement import (  # noqa: E402
    Device,
    build_for_qualified,
)

#: The frozen qualified shape, built directly rather than parsed out of a
#: live vLLM config so these gates need no engine, no GPU and no B70.
QUALIFIED = QualifiedModel(
    model=QUALIFIED_MODEL,
    architecture=QUALIFIED_ARCHITECTURE,
    hidden_size=QUALIFIED_HIDDEN_SIZE,
    num_layers=QUALIFIED_LAYERS,
    num_experts=QUALIFIED_EXPERTS,
    top_k=QUALIFIED_TOP_K,
    moe_intermediate_size=QUALIFIED_MOE_INTERMEDIATE,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


# Every policy that can put experts anywhere other than CUDA. all-out needs
# its env gate to build at all, which is itself part of the contract.
POLICIES = ["all-cuda", "split:128", "subset:16:8", "subset:8:32", "allout:16:8:8"]


def test_id_sets_partition() -> None:
    """cuda / b70 / cpu ids must partition the expert range, per layer.

    Disjoint because the arena is keyed by (layer, expert) and shared;
    exhaustive because a route whose expert belongs to no tier has nowhere
    to execute and would be silently dropped.
    """
    print("\nid-set partition")
    os.environ["SHOOTING_BRAKE_ALL_OUT"] = "1"
    for spec in POLICIES:
        pl = build_for_qualified(QUALIFIED, spec)
        bad = []
        for layer in range(pl.num_layers):
            b70 = set(pl.b70_expert_ids(layer))
            cpu = set(pl.cpu_expert_ids(layer))
            cuda = {
                e for e, o in enumerate(pl.owners[layer])
                if o.device is Device.CUDA
            }
            if b70 & cpu:
                bad.append(f"L{layer} b70&cpu={sorted(b70 & cpu)[:4]}")
            if b70 & cuda or cpu & cuda:
                bad.append(f"L{layer} overlaps cuda")
            if len(b70) + len(cpu) + len(cuda) != pl.num_experts:
                bad.append(f"L{layer} not exhaustive")
        check(f"{spec}: disjoint and exhaustive", not bad, str(bad[:2]))


def test_counts_agree() -> None:
    """Per-layer id lists must sum to the aggregate counts the arena is
    sized from. A mismatch means the reservation is too small and a later
    layer's load fails after earlier layers already committed."""
    print("\narena sizing")
    for spec in POLICIES:
        pl = build_for_qualified(QUALIFIED, spec)
        b70 = sum(len(pl.b70_expert_ids(i)) for i in range(pl.num_layers))
        cpu = sum(len(pl.cpu_expert_ids(i)) for i in range(pl.num_layers))
        check(f"{spec}: b70 ids == b70_count()", b70 == pl.b70_count(),
              f"{b70} != {pl.b70_count()}")
        check(f"{spec}: cpu ids == cpu_count()", cpu == pl.cpu_count(),
              f"{cpu} != {pl.cpu_count()}")


def test_b70_ids_match_capable_layers() -> None:
    """Only B70-capable layers may own B70 experts.

    Layers 32-39 are FP8 and CUDA-forced; streaming must not try to retain a
    host copy for a layer whose experts never left the 5090.
    """
    print("\nb70 ids respect capability")
    for spec in POLICIES:
        pl = build_for_qualified(QUALIFIED, spec)
        bad = [
            layer for layer in range(pl.num_layers)
            if pl.b70_expert_ids(layer) and not pl.is_b70_capable(layer)
        ]
        check(f"{spec}: no b70 ids on incapable layers", not bad, str(bad[:4]))


def test_flag_and_threshold() -> None:
    """The flag defaults off and the threshold is overridable.

    Default-off is the contract that keeps every measured hybrid and
    all-CUDA number valid: with the flag unset no host copy is retained, no
    arena space is reserved, and the dispatch path is unchanged.
    """
    print("\nflag and threshold")
    from shooting_brake_vllm import routed_experts as re_mod

    os.environ.pop("SHOOTING_BRAKE_B70_PREFILL_STREAM", None)
    check("flag defaults off", not re_mod.b70_prefill_stream_enabled())

    os.environ["SHOOTING_BRAKE_B70_PREFILL_STREAM"] = "1"
    check("flag reads 1", re_mod.b70_prefill_stream_enabled())
    os.environ["SHOOTING_BRAKE_B70_PREFILL_STREAM"] = "0"
    check("flag rejects 0", not re_mod.b70_prefill_stream_enabled())
    os.environ.pop("SHOOTING_BRAKE_B70_PREFILL_STREAM", None)

    os.environ.pop("SHOOTING_BRAKE_B70_STREAM_T", None)
    default = re_mod.b70_stream_threshold()
    check("threshold has a default", default > 0, str(default))
    # Above the analytic crossover (~311), where streaming's flat cost beats
    # dispatch's per-token cost with margin for routing that touches fewer
    # experts than the model assumes.
    check("default is above the analytic crossover", default >= 311, str(default))
    os.environ["SHOOTING_BRAKE_B70_STREAM_T"] = "77"
    check("threshold overridable", re_mod.b70_stream_threshold() == 77)
    os.environ.pop("SHOOTING_BRAKE_B70_STREAM_T", None)


def test_slot_ids_are_not_global_ids() -> None:
    """The provider's compact slots and the arena's global ids differ.

    This is the trap that actually bit: ``_b70_slot_map`` maps a global
    expert to a dense per-layer slot for the provider, while the DRAM arena
    is keyed by global id like ``_cpu_id_map``. Both are int tensors of the
    same shape carrying small non-negative numbers, so passing one where the
    other is expected raises nothing -- it silently addresses a different
    expert, and for a slot below the CUDA-resident count, one the arena was
    never asked to hold. The engine died on
    "expert (layer=16, expert=2) is not resident".

    Asserting the two disagree keeps any future refactor that unifies them
    from doing so silently.
    """
    print("\nid spaces are distinct")
    from shooting_brake_vllm.routed_experts import _build_b70_slot_map

    for spec in ("subset:16:8", "split:128"):
        pl = build_for_qualified(QUALIFIED, spec)
        slot_map = _build_b70_slot_map(pl)
        layer = next(
            (i for i in range(pl.num_layers) if pl.b70_expert_ids(i)), None
        )
        check(f"{spec}: has a B70 layer", layer is not None)
        if layer is None:
            continue
        globals_ = pl.b70_expert_ids(layer)
        slots = [int(slot_map[e]) for e in globals_]
        check(f"{spec}: slots are dense from 0",
              slots == list(range(len(globals_))), str(slots[:4]))
        check(f"{spec}: slots differ from global ids",
              slots != list(globals_),
              "slot map is an identity — the distinction has been lost")
        # Every global id the streamer may be handed must be one the arena
        # was told to load.
        resident = set(globals_)
        check(f"{spec}: slot values are not valid arena keys",
              not set(slots).issubset(resident) or slots == list(globals_),
              "slots happen to be resident ids; the gate cannot detect misuse")


def test_crossover_model_is_consistent() -> None:
    """The default threshold must sit at the measured tie point.

    The analytic model (dispatch 26.9 us/token/layer against a flat 0.41 GiB
    transfer) puts the crossover near 311 tokens. Measurement
    (benchmarks/stream_matrix.py) puts it just under 1024: the model assumes
    every route reaches a distinct expert so the whole bank always moves,
    and at moderate M the touched set is smaller than that.

    The default follows the measurement, and this gate keeps the two from
    drifting apart silently -- if either constant is re-measured, the
    documented reasoning has to be revisited with it.
    """
    print("\ncost model")
    from shooting_brake_vllm import routed_experts as re_mod

    us_per_token_layer = 26.9
    stream_ms = 0.41 * 1024 / 50.0          # GiB -> MiB / (GB/s) => ms
    analytic = stream_ms * 1000 / us_per_token_layer
    check("analytic crossover near 311", 250 < analytic < 400, f"{analytic:.0f}")
    # Measured: 0.45x at M=256 (dispatch clearly better), 1.05x at M=1024.
    check("default is the measured tie point",
          re_mod.b70_stream_threshold() == 1024,
          str(re_mod.b70_stream_threshold()))
    check("default is above the analytic crossover",
          re_mod.b70_stream_threshold() >= analytic,
          f"{re_mod.b70_stream_threshold()} vs {analytic:.0f}")


def main() -> int:
    print("B70 prefill streaming gates")
    test_id_sets_partition()
    test_counts_agree()
    test_b70_ids_match_capable_layers()
    test_flag_and_threshold()
    test_slot_ids_are_not_global_ids()
    test_crossover_model_is_consistent()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all gates pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
