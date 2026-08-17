#!/usr/bin/env python3
"""Bisect which input feature zeroes b12x output with real bank data.

Arms (cumulative from synthetic-like to production-like):
  A: real w1/w2 + real sf, alpha=1, no input_global_scale   (synthetic-call shape)
  B: A + alpha=real (~1e-5), no input_global_scale          (dual-role alpha)
  C: A + alpha=real + input_global_scale=ones vector
  D: A + alpha=real + input_global_scale=scalar 1.0
  E: C + fc2_input_scale omitted (None)

Prints absmax/nonzero per arm. GPU must be free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "src/phase4/src"))

from b12x_bank_poc import load_bank_layer  # noqa: E402
from shooting_brake_vllm.b12x_bank_format import read_b12x_bank_header  # noqa: E402

from flashinfer.fused_moe import B12xMoEWrapper  # noqa: E402

bank = REPO / "src/phase1/expert_bank_int4.bin.b12x"
h = read_b12x_bank_header(bank)
e, k, i = h.experts_per_layer, h.hidden, h.moe_intermediate
v = load_bank_layer(bank, h, 0, "cuda")

wrap = B12xMoEWrapper(num_experts=e, top_k=8, hidden_size=k,
                      intermediate_size=i, use_cuda_graph=False,
                      max_num_tokens=64, num_local_experts=e,
                      activation="silu")
torch.manual_seed(7)
m = 16
x = (torch.randn(m, k) * 0.05).cuda().bfloat16()
ids = torch.randint(0, e, (m, 8), dtype=torch.int32, device="cuda")
w = torch.rand(m, 8, device="cuda")
w = w / w.sum(-1, keepdim=True)
ones_v = torch.ones(e, device="cuda")
ones_s = torch.ones(1, device="cuda")

def arm(label, **kw):
    try:
        y = wrap.run(x=x, w1_weight=v["w1"], w1_weight_sf=v["sf1"],
                     w2_weight=v["w2"], w2_weight_sf=v["sf2"],
                     token_selected_experts=ids, token_final_scales=w, **kw)
        torch.cuda.synchronize()
        print(f"{label}: absmax={float(y.abs().max()):.3e} "
              f"nonzero_frac={float((y != 0).float().mean()):.3f}")
    except Exception as ex:
        print(f"{label}: RAISED {type(ex).__name__}: {str(ex)[:120]}")

arm("A alpha=1, gs=default   ", w1_alpha=ones_v, w2_alpha=ones_v,
    fc2_input_scale=ones_v)
arm("B alpha=real, gs=default", w1_alpha=v["alpha1"], w2_alpha=v["alpha2"],
    fc2_input_scale=ones_v)
arm("C alpha=real, gs=ones[e]", w1_alpha=v["alpha1"], w2_alpha=v["alpha2"],
    input_global_scale=ones_v, fc2_input_scale=ones_v)
arm("D alpha=real, gs=scalar1", w1_alpha=v["alpha1"], w2_alpha=v["alpha2"],
    input_global_scale=ones_s, fc2_input_scale=ones_v)
arm("E alpha=real, gs=ones[e], fc2=None", w1_alpha=v["alpha1"],
    w2_alpha=v["alpha2"], input_global_scale=ones_v)
