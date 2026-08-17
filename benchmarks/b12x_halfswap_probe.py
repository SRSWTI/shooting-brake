#!/usr/bin/env python3
"""Gate/up half-order probe: [gate;up] vs [up;gate] against the fp32 oracle.

rel_l2~1.0 with cosine~0.19 (magnitude right, direction wrong) points at a
structural permutation. SwiGLU is asymmetric -- silu(g)*u != silu(u)*g --
so if the kernel expects [up;gate] our [gate;up] bank produces exactly this
signature. Also probes the sf2 conversion orientation as arm 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
for p in ("benchmarks", "src/phase1", "src/phase4/src"):
    sys.path.insert(0, str(REPO / p))

from b12x_bank_poc import load_bank_layer, reference_expert_mats  # noqa: E402
from extract_experts_int4 import DEFAULT_NVFP4_MODEL_DIR  # noqa: E402
from shooting_brake_vllm.b12x_bank_format import read_b12x_bank_header  # noqa: E402

from flashinfer.fused_moe import B12xMoEWrapper  # noqa: E402
from vllm.utils.flashinfer import (  # noqa: E402
    flashinfer_convert_sf_to_mma_layout as conv,
)

bank = REPO / "src/phase1/expert_bank_int4.bin.b12x"
h = read_b12x_bank_header(bank)
e, k, i = h.experts_per_layer, h.hidden, h.moe_intermediate
v = load_bank_layer(bank, h, 0, "cuda")
ones = torch.ones(e, device="cuda", dtype=torch.float32)

wrap = B12xMoEWrapper(num_experts=e, top_k=8, hidden_size=k,
                      intermediate_size=i, use_cuda_graph=False,
                      max_num_tokens=64, num_local_experts=e,
                      activation="silu")
torch.manual_seed(11)
m, e_probe = 16, 12
x = torch.randn(m, k)
ids = torch.randint(0, e_probe, (m, 8), dtype=torch.int32)
w = torch.rand(m, 8)
w = w / w.sum(-1, keepdim=True)

mats = {s: reference_expert_mats(DEFAULT_NVFP4_MODEL_DIR, 0,
                                 s + h.expert_id_base, k, i)
        for s in range(e_probe)}
y_ref = torch.zeros(m, k)
for tok in range(m):
    for j in range(8):
        wg, wu, wd = mats[int(ids[tok, j])]
        act = F.silu(x[tok] @ wg) * (x[tok] @ wu)
        y_ref[tok] += float(w[tok, j]) * (act @ wd)


def swap_halves_sf(sf_bank: torch.Tensor) -> torch.Tensor:
    """Rebuild sf1 with gate/up halves swapped, back in MMA layout."""
    lin = torch.empty(e * 2 * i * (k // 16), dtype=torch.float8_e4m3fn,
                      device="cuda")
    view = conv(lin.reshape(e * 2 * i, k // 16), m=2 * i, k=k, num_groups=e)
    view.copy_(sf_bank)  # logical order into linear storage
    logical = lin.reshape(e, 2 * i, k // 16)
    swapped = torch.cat([logical[:, i:], logical[:, :i]], dim=1).contiguous()
    view2 = conv(swapped.reshape(e * 2 * i, k // 16), m=2 * i, k=k,
                 num_groups=e)
    return view2.contiguous()


def run(w1, sf1, sf2, label):
    y = wrap.run(x=x.cuda().bfloat16(), w1_weight=w1, w1_weight_sf=sf1,
                 w1_alpha=v["alpha1"], w2_alpha=v["alpha2"],
                 input_global_scale=ones, fc2_input_scale=ones,
                 w2_weight=v["w2"], w2_weight_sf=sf2,
                 token_selected_experts=ids.cuda(),
                 token_final_scales=w.cuda()).float().cpu()
    cos = float(F.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0))
    rl2 = float((y - y_ref).norm() / y_ref.norm())
    print(f"{label}: cosine={cos:.5f} rel_l2={rl2:.4f}")


run(v["w1"], v["sf1"], v["sf2"], "1 [gate;up] as banked      ")
w1_sw = torch.cat([v["w1"][:, i:], v["w1"][:, :i]], dim=1).contiguous()
run(w1_sw, swap_halves_sf(v["sf1"]), v["sf2"], "2 [up;gate] halves swapped ")
# arm 3: sf2 converted with transposed orientation (m=i, k=k) -- layout probe
sf2_alt = None
try:
    lin2 = torch.empty(e * k * (i // 16), dtype=torch.float8_e4m3fn,
                       device="cuda")
    view2 = conv(lin2.reshape(e * k, i // 16), m=k, k=i, num_groups=e)
    view2.copy_(v["sf2"])
    logical2 = lin2.reshape(e, k, i // 16)
    sf2_alt = conv(logical2.reshape(e * k, i // 16), m=k, k=i,
                   num_groups=e)  # direct view (non-materialized)
    run(v["w1"], v["sf1"], sf2_alt, "3 sf2 direct-view          ")
except Exception as ex:
    print("3 sf2 direct-view: RAISED", type(ex).__name__, str(ex)[:100])
