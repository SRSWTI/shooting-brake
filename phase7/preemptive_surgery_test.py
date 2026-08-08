#!/usr/bin/env python3
"""Gates for pre-emptive VRAM surgery — the allocation-time expert subset.

Post-hoc surgery allocates every expert, loads them, then slices ownership
away. That is only viable while the whole bank transiently fits in VRAM: 13.5
GiB for the 35B, 59.5 GiB for the 122B against a 32 GiB card. Pre-emptive
allocation never creates the non-CUDA experts, so peak equals steady state.

These checks need no GPU and no model. They cover the parts that fail
*silently* — a wrong expert map computes the wrong experts rather than
raising, and a dropped weight-loader attribute leaves zeros in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4" / "src"))

from shooting_brake_vllm.placement import (  # noqa: E402
    AllOutPolicy,
    Device,
    LayerSubsetPolicy,
    build_placement,
)
from shooting_brake_vllm.routed_experts import (  # noqa: E402
    _reject_unsupported_preemptive_tiers,
)

CHECKS = 0


def check(cond: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(f"FAIL: {label}")
    print(f"  ok  {label}")


def build(policy, layers=40, experts=256, capable=32):
    return build_placement(
        policy,
        num_layers=layers,
        num_experts=experts,
        b70_capable=frozenset(range(capable)),
    )


def expert_map_for(placement, layer: int) -> torch.Tensor:
    """Mirror of the map the alloc hook builds, kept in one place."""
    ids = placement.cuda_expert_ids(layer)
    m = torch.full((placement.num_experts,), -1, dtype=torch.int32)
    for local, global_id in enumerate(ids):
        m[global_id] = local
    return m


def test_partition_is_exact() -> None:
    """Every expert has exactly one owner, at any cuda_per_layer."""
    print("\n[partition]")
    for n_cuda in (8, 16, 32, 48, 64, 80):
        p = build(AllOutPolicy(active_layers=32, cuda_per_layer=n_cuda,
                               cpu_per_layer=40))
        cuda = set(p.cuda_expert_ids(16))
        b70 = set(p.b70_expert_ids(16))
        cpu = set(p.cpu_expert_ids(16))
        check(len(cuda) == n_cuda, f"n_cuda={n_cuda}: cuda set has {n_cuda}")
        check(
            not (cuda & b70) and not (cuda & cpu) and not (b70 & cpu),
            f"n_cuda={n_cuda}: tiers are disjoint",
        )
        check(
            cuda | b70 | cpu == set(range(256)),
            f"n_cuda={n_cuda}: tiers cover every expert",
        )


def test_cuda_ids_are_sorted() -> None:
    """Local id == position in the ascending global id list.

    Post-hoc surgery builds its compact layout with `index_select` on a
    sorted id tensor. Pre-emptive allocation fills slots in iteration order.
    The two agree only while this list is ascending, and a mismatch would
    make the two strategies compute different experts from identical weights.
    """
    print("\n[ordering]")
    p = build(AllOutPolicy(active_layers=32, cuda_per_layer=80,
                           cpu_per_layer=40))
    ids = p.cuda_expert_ids(16)
    check(list(ids) == sorted(ids), "cuda_expert_ids is ascending")
    check(
        all(expert_map_for(p, 16)[g].item() == i for i, g in enumerate(ids)),
        "expert_map sends each global id to its position",
    )


def test_non_cuda_experts_are_masked() -> None:
    """Offloaded experts map to -1, which is what makes the loader skip."""
    print("\n[masking]")
    p = build(AllOutPolicy(active_layers=32, cuda_per_layer=80,
                           cpu_per_layer=40))
    m = expert_map_for(p, 16)
    offloaded = set(p.b70_expert_ids(16)) | set(p.cpu_expert_ids(16))
    check(all(m[e].item() == -1 for e in offloaded),
          "every B70/CPU expert maps to -1")
    check(int((m != -1).sum()) == 80, "exactly 80 entries survive")
    check(m.dtype == torch.int32, "map dtype matches vLLM's expert_map")


def test_all_cuda_layer_keeps_stock_path() -> None:
    """The FP8 tail owns no offloaded experts and must not be subsetted.

    Returning a full-length identity map there would be correct but would
    put an all-CUDA layer on the compact path for no reason; the hook is
    specified to return None so those layers stay bit-for-bit stock.
    """
    print("\n[fp8 tail]")
    p = build(AllOutPolicy(active_layers=32, cuda_per_layer=80,
                           cpu_per_layer=40))
    for layer in (32, 39):
        offload = p.layer_b70_count(layer) + p.layer_cpu_count(layer)
        check(offload == 0, f"layer {layer} owns nothing offloaded")
        check(len(p.cuda_expert_ids(layer)) == 256,
              f"layer {layer} keeps all 256 experts")


def test_subset_policy_matches_allout_cuda_set() -> None:
    """subset and allout at equal cuda_per_layer allocate the same slots.

    This is what lets a 35B parity run compare the two surgery strategies
    without also changing which experts CUDA owns.
    """
    print("\n[policy agreement]")
    a = build(LayerSubsetPolicy(active_layers=16, cuda_per_layer=8))
    b = build(AllOutPolicy(active_layers=16, cuda_per_layer=8,
                           cpu_per_layer=8))
    check(a.cuda_expert_ids(30) == b.cuda_expert_ids(30),
          "subset:16:8 and allout:16:8:8 agree on the CUDA set")


def test_input_scales_are_locally_indexed() -> None:
    """The loader must not index input scales by *global* expert id.

    vLLM bypasses its `-1` skip for a param only when
    `use_global_sf and "input_scale" in weight_name`
    (fused_moe/routed_experts.py). Our backend sets `use_global_sf`, so the
    whole question is the tensor name — and this checkpoint calls them
    `input_global_scale`, which does NOT contain `input_scale` as a
    contiguous substring. That is why the compact allocation is safe.

    If a future checkpoint or vLLM renames them to `input_scale`, the
    bypass starts firing, the loader writes `param.data[global_id]` into a
    compactly-sized tensor, and the load dies with an index error — or
    worse, silently writes the wrong row. This is the tripwire.
    """
    print("\n[input scale indexing]")
    import inspect

    from vllm.model_executor.layers.fused_moe import routed_experts as vllm_re
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A4Nvfp4MoEMethod,
    )

    loader_src = inspect.getsource(vllm_re)
    check('"input_scale" in weight_name' in loader_src,
          "vLLM still gates the global-id bypass on that substring")

    create_src = inspect.getsource(
        CompressedTensorsW4A4Nvfp4MoEMethod.create_weights
    )
    check('"w13_input_global_scale"' in create_src,
          "the NVFP4 method still names the param input_global_scale")
    check("input_scale" not in "input_global_scale",
          "so the bypass cannot fire and compact sizing is safe")


def test_placement_assigns_before_module_init() -> None:
    """A plain attribute may be set before `nn.Module.__init__`.

    The hook reads `shooting_brake_placement` during `create_weights`, which
    runs inside `super().__init__()`. The adapter therefore assigns it
    first. torch permits that for non-Parameter, non-Module values only.
    """
    print("\n[init ordering]")

    class Early(torch.nn.Module):
        def __init__(self) -> None:
            self.marker = "set-before-super"
            super().__init__()

    check(Early().marker == "set-before-super",
          "plain attribute survives a pre-super assignment")

    class TooEarly(torch.nn.Module):
        def __init__(self) -> None:
            self.p = torch.nn.Parameter(torch.zeros(1))
            super().__init__()

    try:
        TooEarly()
        raised = False
    except AttributeError:
        raised = True
    check(raised, "a Parameter still cannot be assigned pre-super")


def test_host_tiers_require_a_bank() -> None:
    """An all-out placement is admitted only when a bank exists.

    The arena used to `index_select` global expert ids out of the layer
    weights, which pre-emptive allocation leaves holding only the
    CUDA-owned ones — so the combination was refused outright. It is now
    served from the expert bank instead, verified byte-for-byte against
    the VRAM path across all 128 host experts at allout:16:8:8.

    What must still fail closed is a *missing* bank: the tier would load
    nothing, its routes would contribute zero, and the output would be
    plausible tokens rather than an error.
    """
    print("\n[tier guard]")
    subset = build(LayerSubsetPolicy(active_layers=16, cuda_per_layer=8))
    _reject_unsupported_preemptive_tiers(subset)
    check(True, "subset placement is accepted (no host tier, no bank needed)")

    allout = build(AllOutPolicy(active_layers=16, cuda_per_layer=8,
                                cpu_per_layer=8))
    _reject_unsupported_preemptive_tiers(allout)
    check(True, "all-out is accepted when the bank is present")

    import os

    prev = os.environ.get("SHOOTING_BRAKE_B70_BANK")
    os.environ["SHOOTING_BRAKE_B70_BANK"] = "/nonexistent/expert_bank.bin"
    try:
        _reject_unsupported_preemptive_tiers(allout)
        raised = False
    except RuntimeError:
        raised = True
    finally:
        if prev is None:
            del os.environ["SHOOTING_BRAKE_B70_BANK"]
        else:
            os.environ["SHOOTING_BRAKE_B70_BANK"] = prev
    check(raised, "all-out is refused when the bank is missing")


def test_owner_device_matches_accessor() -> None:
    """`cuda_expert_ids` agrees with the raw owner table it summarises."""
    print("\n[accessor consistency]")
    p = build(AllOutPolicy(active_layers=32, cuda_per_layer=64,
                           cpu_per_layer=52))
    raw = tuple(
        e for e in range(p.num_experts)
        if p.owners[16][e].device is Device.CUDA
    )
    check(raw == p.cuda_expert_ids(16),
          "accessor matches a direct scan of owners")


def main() -> int:
    for fn in (
        test_partition_is_exact,
        test_cuda_ids_are_sorted,
        test_non_cuda_experts_are_masked,
        test_all_cuda_layer_keeps_stock_path,
        test_subset_policy_matches_allout_cuda_set,
        test_input_scales_are_locally_indexed,
        test_placement_assigns_before_module_init,
        test_owner_device_matches_accessor,
        test_host_tiers_require_a_bank,
    ):
        fn()
    print(f"\nall {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
