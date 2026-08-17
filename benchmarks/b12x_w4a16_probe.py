#!/usr/bin/env python3
"""b12x W4A16 mode probe: fidelity + speed with checkpoint-native planes.

W4A4 measured dead: flashinfer's own validated quantizer reaches only
cosine ~0.82/layer on real weights (static e2m1 activation grid; matches
the FTZ literature). W4A16 keeps activations bf16 -- weight-only fp4 --
so the oracle cosine should be ~0.999 like every weight-only path.

Inputs per prepare_b12x_w4a16_weights (source_format="modelopt"):
  w1_fp4 [E, 2I, H/2] verbatim, w1_blockscale LINEAR [E, 2I, H/16] e4m3
  (gate/up ratio-baked -- resolves the per-projection scale_2 fusion that
  upstream vLLM merely warns about), w1_global_scale [E] = max(g_s2, u_s2);
  w2 planes verbatim + w2_global_scale [E] = d_s2.

Decision gate: cosine >= 0.99 AND ms/layer < 7.07 (Marlin incumbent).
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
for p in ("benchmarks", "src/phase1", "src/phase4/src"):
    sys.path.insert(0, str(REPO / p))

from b12x_bank_poc import reference_expert_mats  # noqa: E402
from build_b12x_bank import ShardCache  # noqa: E402
from extract_experts_int4 import (  # noqa: E402
    DEFAULT_NVFP4_MODEL_DIR,
    index_shards,
    tensor_key,
)

from flashinfer.fused_moe import B12xMoEWrapper  # noqa: E402
from flashinfer.fused_moe.prepare import prepare_b12x_w4a16_weights  # noqa: E402

E, K, I, TOP_K, BASE = 126, 3072, 1024, 8, 54

_, index = index_shards(DEFAULT_NVFP4_MODEL_DIR)
cache = ShardCache(DEFAULT_NVFP4_MODEL_DIR, index)

w1 = torch.empty(E, 2 * I, K // 2, dtype=torch.uint8)
w2 = torch.empty(E, K, I // 2, dtype=torch.uint8)
s1 = torch.empty(E, 2 * I, K // 16, dtype=torch.float32)
s2 = torch.empty(E, K, I // 16, dtype=torch.float32)
a1 = torch.empty(E, dtype=torch.float32)
a2 = torch.empty(E, dtype=torch.float32)
for slot, ex in enumerate(range(BASE, BASE + E)):
    def t(proj, suf):
        return cache.get(tensor_key(0, ex, proj, suf))

    w1[slot, :I] = t("gate_proj", "weight")
    w1[slot, I:] = t("up_proj", "weight")
    w2[slot] = t("down_proj", "weight")
    g2, u2 = float(t("gate_proj", "weight_scale_2")), float(t("up_proj", "weight_scale_2"))
    a = max(g2, u2)
    a1[slot], a2[slot] = a, float(t("down_proj", "weight_scale_2"))
    s1[slot, :I] = t("gate_proj", "weight_scale").float() * (g2 / a)
    s1[slot, I:] = t("up_proj", "weight_scale").float() * (u2 / a)
    s2[slot] = t("down_proj", "weight_scale").float()

dev = "cuda"
tensors = prepare_b12x_w4a16_weights(
    w1.to(dev), s1.to(torch.float8_e4m3fn).to(dev), a1.to(dev),
    w2.to(dev), s2.to(torch.float8_e4m3fn).to(dev), a2.to(dev),
    activation="silu", source_format="modelopt",
)

wrap = B12xMoEWrapper(num_experts=E, top_k=TOP_K, hidden_size=K,
                      intermediate_size=I, use_cuda_graph=False,
                      max_num_tokens=8192, num_local_experts=E,
                      activation="silu", quant_mode="w4a16",
                      source_format="modelopt")

def run(x, ids, w):
    return wrap.run(x=x, token_selected_experts=ids, token_final_scales=w,
                    **tensors)

# fidelity
torch.manual_seed(11)
m, e_probe = 16, 12
x = torch.randn(m, K)
ids = torch.randint(0, e_probe, (m, TOP_K), dtype=torch.int32)
w = torch.rand(m, TOP_K)
w = w / w.sum(-1, keepdim=True)
y = run(x.cuda().bfloat16(), ids.cuda(), w.cuda()).float().cpu()

mats = {s: reference_expert_mats(DEFAULT_NVFP4_MODEL_DIR, 0, s + BASE, K, I)
        for s in range(e_probe)}
y_ref = torch.zeros(m, K)
for tok in range(m):
    for j in range(TOP_K):
        wg, wu, wd = mats[int(ids[tok, j])]
        act = F.silu(x[tok] @ wg) * (x[tok] @ wu)
        y_ref[tok] += float(w[tok, j]) * (act @ wd)
cos = float(F.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0))
rel = float((y - y_ref).norm() / y_ref.norm())
print(f"W4A16 fidelity: cosine={cos:.6f} rel_l2={rel:.5f} "
      f"(gate: cosine >= 0.99)")

# speed at M=8192
xb = torch.randn(8192, K, device=dev, dtype=torch.bfloat16)
idb = torch.randint(0, E, (8192, TOP_K), device=dev, dtype=torch.int32)
wb = torch.rand(8192, TOP_K, device=dev)
run(xb, idb, wb)
torch.cuda.synchronize()
ev_a, ev_b = torch.cuda.Event(True), torch.cuda.Event(True)
ts = []
for _ in range(7):
    torch.cuda.synchronize()
    ev_a.record()
    run(xb, idb, wb)
    ev_b.record()
    torch.cuda.synchronize()
    ts.append(ev_a.elapsed_time(ev_b))
print(f"W4A16 speed: {statistics.median(ts):.2f} ms/layer @ M=8192 "
      f"(marlin 7.07, w4a4 2.70)")
