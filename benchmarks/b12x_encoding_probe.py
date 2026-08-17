#!/usr/bin/env python3
"""Definitive b12x encoding probe: their quantizer vs our bank, byte-level.

Steps:
  1. Dequantize checkpoint layer-0 remote experts to bf16 (repo oracle path,
     [gate;up] stacking) -- ground truth weights W.
  2. Re-quantize W with flashinfer's own prepare_b12x_nvfp4_weights (the
     upstream-validated w4a4 input builder).
  3. Run the kernel with THEIR tensors, compare vs fp32 oracle.
     - cosine >= 0.99: kernel + oracle + [gate;up] stacking all consistent
       -> fault isolated to OUR plane encoding; step 4 pinpoints it.
     - cosine ~0.2: stacking/oracle wrong -> try [up;gate].
  4. Byte-diff their w1_weight/w1_weight_sf vs our bank planes for expert 0:
     reveals nibble order and sf-layout deltas exactly.
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
from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_weights  # noqa: E402

bank = REPO / "src/phase1/expert_bank_int4.bin.b12x"
h = read_b12x_bank_header(bank)
e, k, i = h.experts_per_layer, h.hidden, h.moe_intermediate
E_PROBE = 8  # few experts keep dequant cheap; kernel still gets full-E tensors

# 1. ground-truth bf16 weights for the first E_PROBE experts (rest zeros)
w1_bf16 = torch.zeros(e, 2 * i, k, dtype=torch.bfloat16, device="cuda")
w2_bf16 = torch.zeros(e, k, i, dtype=torch.bfloat16, device="cuda")
mats = {}
for slot in range(E_PROBE):
    wg, wu, wd = reference_expert_mats(
        DEFAULT_NVFP4_MODEL_DIR, 0, slot + h.expert_id_base, k, i)
    mats[slot] = (wg, wu, wd)
    w1_bf16[slot, :i] = wg.T.bfloat16().cuda()   # [gate; up] stacking, [out,in]
    w1_bf16[slot, i:] = wu.T.bfloat16().cuda()
    w2_bf16[slot] = wd.T.bfloat16().cuda()

# 2. their quantizer
theirs = prepare_b12x_nvfp4_weights(
    w1_bf16, w2_bf16, num_local_experts=e, hidden_size=k,
    intermediate_size=i, activation="silu",
)

# 3. kernel with THEIR tensors vs oracle
wrap = B12xMoEWrapper(num_experts=e, top_k=8, hidden_size=k,
                      intermediate_size=i, use_cuda_graph=False,
                      max_num_tokens=64, num_local_experts=e,
                      activation="silu")
torch.manual_seed(11)
m = 16
x = torch.randn(m, k)
ids = torch.randint(0, E_PROBE, (m, 8), dtype=torch.int32)
w = torch.rand(m, 8)
w = w / w.sum(-1, keepdim=True)

y = wrap.run(x=x.cuda().bfloat16(), token_selected_experts=ids.cuda(),
             token_final_scales=w.cuda(), **{k_: v_.cuda() if torch.is_tensor(v_)
             else v_ for k_, v_ in theirs.items()}).float().cpu()

y_ref = torch.zeros(m, k)
for tok in range(m):
    for j in range(8):
        wg, wu, wd = mats[int(ids[tok, j])]
        act = F.silu(x[tok] @ wg) * (x[tok] @ wu)
        y_ref[tok] += float(w[tok, j]) * (act @ wd)

cos = float(F.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0))
print(f"THEIR quantizer vs oracle: cosine={cos:.5f} "
      f"rel_l2={float((y - y_ref).norm() / y_ref.norm()):.4f}")

# 4. byte-diff their encodings vs our bank (expert 0)
v = load_bank_layer(bank, h, 0, "cuda")
tw1 = theirs["w1_weight"]
print("their w1 shape/dtype:", tuple(tw1.shape), tw1.dtype,
      "| bank w1:", tuple(v["w1"].shape), v["w1"].dtype)
if tuple(tw1.shape) == tuple(v["w1"].shape):
    same = bool((tw1[0] == v["w1"][0]).all())
    frac = float((tw1[0] == v["w1"][0]).float().mean())
    swap = ((v["w1"][0] << 4) | (v["w1"][0] >> 4))
    frac_swapped = float((tw1[0] == swap).float().mean())
    print(f"w1[0] bytes equal: {same} (match frac {frac:.4f}, "
          f"nibble-swapped match frac {frac_swapped:.4f})")
tsf = theirs["w1_weight_sf"]
print("their sf1 shape/dtype/strides:", tuple(tsf.shape), tsf.dtype,
      tuple(tsf.stride()), "| bank sf1:", tuple(v["sf1"].shape),
      tuple(v["sf1"].stride()))
print("their alpha1:", theirs["w1_alpha"][:3].tolist(),
      "| their fc2:", theirs["fc2_input_scale"].tolist())
