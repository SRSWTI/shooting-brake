#!/usr/bin/env python3
"""Gate for the absolute-model-layer <-> compact-bank-row contract.

Why this exists
---------------
Laguna models carry a DENSE MLP at layer 0 and routed MoE only at layers
1..47, so a compact 47-row bank stores model layer 1 at row 0. The native
provider indexes rows directly (`b70_provider.cpp` treats the submitted
`layer` as a bank ordinal), so passing an absolute model index straight
through reads the NEIGHBOURING layer's experts.

That failure is silent. It does not crash, it does not produce NaNs, and the
bank's own size check cannot detect it -- the file is exactly the right
length either way. It produces plausible, wrong tokens. Every assertion here
exists to make that shift impossible to introduce accidentally.

The legacy Qwen contract (every layer routed, bank = leading prefix) must be
preserved bit-for-bit; the first block below is the regression guard for it.

Run:
  .venv/bin/python src/phase6/layer_topology_unit_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4" / "src"))

from shooting_brake_vllm.config import (  # noqa: E402
    SUPPORTED_MODELS,
    QualifiedModel,
)
from shooting_brake_vllm.placement import (  # noqa: E402
    Device,
    ExpertOwner,
    Placement,
    b70_bank_covers,
)

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(f"{'ok' if cond else 'FAIL'} {label}")
    if not cond:
        FAILURES.append(label)


def raises(label: str, exc: type[BaseException], fn) -> None:
    try:
        fn()
    except exc:
        print(f"ok {label}")
        return
    except BaseException as e:  # noqa: BLE001
        print(f"FAIL {label} (raised {type(e).__name__}: {e})")
        FAILURES.append(label)
        return
    print(f"FAIL {label} (no {exc.__name__})")
    FAILURES.append(label)


def qwen_like(bank_layers: int = 32) -> QualifiedModel:
    """The 35B shape: all 40 layers routed, bank is the leading 32."""
    return QualifiedModel(
        model="unsloth/Qwen3.6-35B-A3B-NVFP4",
        architecture="Qwen3_5MoeForConditionalGeneration",
        hidden_size=2048, num_layers=40, num_experts=256, top_k=8,
        moe_intermediate_size=512, bank_layers=bank_layers,
        bank_experts_per_layer=256,
    )


def laguna_like(bank_layers: int = 47) -> QualifiedModel:
    """The r20 shape: layer 0 dense, layers 1..47 routed and banked."""
    return QualifiedModel(
        model="srswti/axe-superveloce-jota-118b-r20-nvfp4",
        architecture="LagunaForCausalLM",
        hidden_size=3072, num_layers=48, num_experts=205, top_k=10,
        moe_intermediate_size=1024, bank_layers=bank_layers,
        bank_experts_per_layer=205,
        routed_layer_ids=tuple(range(1, 48)),
        bank_layer_ids=tuple(range(1, 48)),
    )


def laguna_placement(*, b70_in_dense_layer: bool = False) -> Placement:
    """48 rows of 205 experts: layer 0 all-CUDA, layers 1..47 split 54/151.

    Layer 0 still gets CUDA owners because Placement cannot yet express
    "this layer owns no routed module at all" -- that is a separate change.
    What this fixture pins is that a B70 owner never appears there, and
    that coverage notices when one does.
    """
    rows = []
    for layer in range(48):
        row: list[ExpertOwner] = []
        remote_slot = 0
        dense_row = layer == 0 and not b70_in_dense_layer
        for expert in range(205):
            if dense_row or expert < 54:
                row.append(ExpertOwner(Device.CUDA, expert))
            else:
                row.append(ExpertOwner(Device.B70, remote_slot, remote_slot % 2))
                remote_slot += 1
        rows.append(tuple(row))
    return Placement(
        generation=1, num_layers=48, num_experts=205, owners=tuple(rows),
        b70_capable_layers=frozenset(range(1, 48)), policy_name="laguna-test",
    )


def main() -> int:
    # ---- legacy contract is untouched -------------------------------------
    q = qwen_like()
    check("legacy: all layers routed", q.routed_layer_ids == tuple(range(40)))
    check("legacy: bank is leading prefix", q.bank_layer_ids == tuple(range(32)))
    check("legacy: capable layers unchanged",
          q.b70_capable_layers == frozenset(range(32)))
    check("legacy: bank row is identity",
          all(q.bank_row_for_model_layer(i) == i for i in range(32)))
    check("legacy: every layer is routed",
          all(q.is_routed_layer(i) for i in range(40)))
    check("legacy: non-banked routed layer has no row",
          32 not in q.b70_capable_layers and q.is_routed_layer(32))
    raises("legacy: FP8 layer 32 rejected as a bank row", KeyError,
           lambda: q.bank_row_for_model_layer(32))

    # ---- Laguna topology --------------------------------------------------
    g = laguna_like()
    check("laguna: 47 routed layers", len(g.routed_layer_ids) == 47)
    check("laguna: layer 0 not routed", not g.is_routed_layer(0))
    check("laguna: layer 1 routed", g.is_routed_layer(1))
    check("laguna: layer 47 routed", g.is_routed_layer(47))
    check("laguna: capable layers exclude 0",
          g.b70_capable_layers == frozenset(range(1, 48)))

    # The two boundaries that a prefix assumption gets wrong.
    check("laguna: model layer 1 -> bank row 0",
          g.bank_row_for_model_layer(1) == 0)
    check("laguna: model layer 47 -> bank row 46",
          g.bank_row_for_model_layer(47) == 46)
    check("laguna: shift is exactly one everywhere",
          all(g.bank_row_for_model_layer(L) == L - 1 for L in range(1, 48)))
    raises("laguna: dense layer 0 has no bank row", KeyError,
           lambda: g.bank_row_for_model_layer(0))
    raises("laguna: layer 48 out of range", KeyError,
           lambda: g.bank_row_for_model_layer(48))

    # ---- round trip -------------------------------------------------------
    check("laguna: row->layer->row round-trips",
          all(g.bank_row_for_model_layer(g.model_layer_for_bank_row(r)) == r
              for r in range(47)))
    check("laguna: row 0 is model layer 1", g.model_layer_for_bank_row(0) == 1)
    check("laguna: row 46 is model layer 47", g.model_layer_for_bank_row(46) == 47)
    raises("laguna: row 47 rejected", KeyError,
           lambda: g.model_layer_for_bank_row(47))
    raises("laguna: negative row rejected", KeyError,
           lambda: g.model_layer_for_bank_row(-1))

    # ---- fail-closed validation ------------------------------------------
    # A 48-row bank under a 47-sparse-layer model is the exact artifact that
    # would serve layer L+1's weights at layer L. It must not construct.
    raises("48-row bank against 47 banked layers is rejected", ValueError,
           lambda: laguna_like(bank_layers=48))
    raises("46-row bank against 47 banked layers is rejected", ValueError,
           lambda: laguna_like(bank_layers=46))
    raises("banked layer outside routed set is rejected", ValueError,
           lambda: QualifiedModel(
               model="x", architecture="LagunaForCausalLM", hidden_size=3072,
               num_layers=48, num_experts=205, top_k=10,
               moe_intermediate_size=1024, bank_layers=48,
               bank_experts_per_layer=205,
               routed_layer_ids=tuple(range(1, 48)),
               bank_layer_ids=tuple(range(0, 48)),  # includes dense layer 0
           ))
    raises("unsorted layer ids rejected", ValueError,
           lambda: QualifiedModel(
               model="x", architecture="LagunaForCausalLM", hidden_size=3072,
               num_layers=48, num_experts=205, top_k=10,
               moe_intermediate_size=1024, bank_layers=3,
               bank_experts_per_layer=205,
               routed_layer_ids=tuple(range(1, 48)),
               bank_layer_ids=(3, 1, 2),
           ))
    raises("out-of-range layer id rejected", ValueError,
           lambda: QualifiedModel(
               model="x", architecture="LagunaForCausalLM", hidden_size=3072,
               num_layers=48, num_experts=205, top_k=10,
               moe_intermediate_size=1024, bank_layers=1,
               bank_experts_per_layer=205,
               routed_layer_ids=(99,), bank_layer_ids=(99,),
           ))

    # ---- bank coverage over a real Placement ------------------------------
    p = laguna_placement()
    check("coverage: accepts a 1..47 placement given explicit layer ids",
          b70_bank_covers(
              p, bank_layers=47, bank_experts_per_layer=205,
              bank_layer_ids=tuple(range(1, 48))) is True)
    # This is the whole point of the change: the legacy count-derived set is
    # {0..46}, which does not contain sparse layer 47. Coverage must refuse
    # rather than quietly accept a placement it cannot serve.
    check("coverage: refuses the same placement under the prefix assumption",
          b70_bank_covers(
              p, bank_layers=47, bank_experts_per_layer=205) is False)
    check("coverage: refuses a B70 owner in the dense layer",
          b70_bank_covers(
              laguna_placement(b70_in_dense_layer=True),
              bank_layers=47, bank_experts_per_layer=205,
              bank_layer_ids=tuple(range(1, 48))) is False)
    # A 48-row bank spanning 0..47 covers the placement, but that artifact is
    # already rejected at QualifiedModel construction -- belt and braces.
    check("coverage: a 0..47 bank would cover it (rejected earlier instead)",
          b70_bank_covers(
              p, bank_layers=48, bank_experts_per_layer=205,
              bank_layer_ids=tuple(range(48))) is True)

    # ---- registry row -----------------------------------------------------
    spec = SUPPORTED_MODELS.get("srswti/axe-superveloce-jota-118b-r20-nvfp4")
    check("r20 spec is registered", spec is not None)
    if spec is not None:
        check("r20 arch is Laguna", spec.architecture == "LagunaForCausalLM")
        check("r20 model_type is laguna", spec.model_type == "laguna")
        check("r20 top_k is 10", spec.top_k == 10)
        check("r20 keeps the 99B's 205 experts", spec.num_experts == 205)
        check("r20 routed layers are 1..47",
              spec.routed_layer_ids == tuple(range(1, 48)))
        check("r20 bank layers are 1..47",
              spec.bank_layer_ids == tuple(range(1, 48)))
        check("r20 bank filename is distinct from the 99B",
              spec.default_bank_filename != "expert_bank_99b.bin")

    # legacy specs must not have acquired a topology by accident
    for name in ("unsloth/Qwen3.6-35B-A3B-NVFP4",
                 "srswti/axe-superveloce-99b-nvfp4"):
        s = SUPPORTED_MODELS[name]
        check(f"{name.split('/')[-1]}: no explicit topology",
              s.routed_layer_ids == () and s.bank_layer_ids == ())

    if FAILURES:
        print(f"\nlayer-topology unit-test FAIL ({len(FAILURES)}): {FAILURES}")
        return 1
    print("\nlayer-topology unit-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
