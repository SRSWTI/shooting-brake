#!/usr/bin/env python3
"""Is FP8 on the wire good enough? Decided before any provider code is touched.

The wire carries two things per dispatch:

    5090 --H2D--> B70   activations, today bf16      (6 KB/token/layer)
    B70  --D2H--> 5090  routed partial, today fp16  (12 KB/token/layer)

FP8 would halve both. The question is not "is E4M3 exact" -- it is not -- but
"is E4M3's error comparable to the fp16 output wire we ALREADY shipped and
gated, or is it a different regime?"  fp16 output cleared a 120-prompt argmax
sweep at 90.0% against a 90.8% control ceiling (kill-bench Bench 26). That is
the yardstick, so it is measured here as an arm rather than assumed.

Real weights, real activation range: expert weights come from the checkpoint's
NVFP4 planes, and activations are drawn at the amax the checkpoint itself
calibrated (`input_global_scale`), not a guess.

Arms:
  ref     fp32 everywhere                                    (truth)
  bf16in  activations bf16                                   (today's H2D)
  f16out  partial fp16                                       (today's D2H, SHIPPED)
  f8in    activations E4M3 per-token                         (candidate)
  f8out   partial E4M3 per-token                             (candidate)
  f8both  both                                               (candidate)

and the case that actually worries me:

  2card   each card quantises its OWN partial with its OWN per-token scale,
          then the two are summed with the local+shared contribution.
          Cancellation in that sum can magnify relative error even when each
          partial looks clean alone. Never gate a partial in isolation.

Usage:
  .venv/bin/python benchmarks/fp8_wire_numerics.py --layer 3 --experts 16 --tokens 512
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # NVFP4 magnitude LUT
FP8_MAX = 448.0  # E4M3


def find_checkpoint(model: str) -> Path:
    pat = f"models--{model.replace('/', '--')}"
    hits = glob.glob(str(Path.home() / f".cache/huggingface/hub/{pat}/snapshots/*/"))
    if not hits:
        raise SystemExit(f"checkpoint not found for {model}")
    return Path(sorted(hits)[-1])


def unpack_nvfp4(packed: torch.Tensor, scale_e4m3: torch.Tensor,
                 global_scale: torch.Tensor, dev: str) -> torch.Tensor:
    """NVFP4 -> fp32. E2M1 nibbles x E4M3 block-16 scales / fp32 global scale.

    compressed-tensors convention: global_scale = 448*6/amax scales the block
    scales up into E4M3's range, so dequant divides it back out.
    """
    b = packed.to(dev)
    lo, hi = b & 0x0F, (b >> 4) & 0x0F
    lut = torch.tensor(E2M1, device=dev, dtype=torch.float32)

    def dec(n):
        # torch.where evaluates BOTH branches, so the magnitude lookup must be
        # masked to 0..7 before indexing -- indexing the 8-entry LUT with the
        # raw 0..15 nibble is an out-of-bounds launch failure, not a no-op.
        mag = lut[(n & 7).long()]
        return torch.where(n >= 8, -mag, mag)

    # low nibble is the earlier column (verified against the k-tile layout,
    # kill-bench Bench 25 -- getting this backwards is a silent 0.19 cosine)
    q = torch.stack([dec(lo), dec(hi)], dim=-1).reshape(b.shape[0], -1)
    s = scale_e4m3.to(dev).float() / global_scale.to(dev).float()
    return q * s.repeat_interleave(16, dim=1)[:, : q.shape[1]]


def q_e4m3_per_token(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through per-token E4M3, matching scaled_fp8_quant exactly:
    scale = maxabs/448 (min-clamped), q = clamp(x/scale), x_hat = q*scale."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.float() * scale


def q_e4m3_blocked(x: torch.Tensor, block: int) -> torch.Tensor:
    """Per-BLOCK E4M3 round-trip. Per-token gives one scale to all 3072
    coordinates, so small coordinates inside a token get crushed; a finer block
    keeps the scale local. This is what `w4a8_nvfp4` does (dynamic per-32-block
    activation quant) and why that kernel measured near-lossless where flat
    per-tensor W4A4 topped out at 0.82 cosine.

    Wire cost: block=32 over hidden 3072 is 96 fp32 scales = 384 B/token on top
    of a 3072 B payload, so 1.78x reduction against bf16 instead of 2.00x.
    """
    *lead, h = x.shape
    assert h % block == 0, f"hidden {h} not divisible by block {block}"
    xb = x.reshape(*lead, h // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    q = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).reshape(*lead, h)


def rt(x: torch.Tensor, dtype) -> torch.Tensor:
    return x.to(dtype).float()


def metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = got.flatten().double(), ref.flatten().double()
    cos = torch.dot(g, r) / (g.norm() * r.norm() + 1e-30)
    rel_l2 = (g - r).norm() / (r.norm() + 1e-30)
    # max relative error only over coordinates large enough to be meaningful;
    # tiny coordinates have terrible relative precision in ANY float format.
    # quantile() caps at 2^24 elements, so the threshold comes from a subsample.
    ab = r.abs()
    sub = ab[torch.randint(0, ab.numel(), (min(ab.numel(), 1 << 20),), device=ab.device)]
    big = ab > sub.quantile(0.5)
    mre = ((g - r).abs()[big] / ab[big]).max() if big.any() else torch.tensor(0.0)
    return {"cos": cos.item(), "rel_l2": rel_l2.item(), "max_rel": mre.item()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="srswti/axe-superveloce-jota-118b-r15-nvfp4")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    from safetensors import safe_open

    snap = find_checkpoint(a.model)
    wmap = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    dev = a.device
    torch.manual_seed(a.seed)

    pre = f"model.layers.{a.layer}.mlp.experts"
    handles: dict[str, object] = {}

    def T(name: str) -> torch.Tensor:
        shard = wmap[name]
        if shard not in handles:
            handles[shard] = safe_open(str(snap / shard), framework="pt")
        return handles[shard].get_tensor(name)

    # activation amax the checkpoint itself calibrated, so the test runs at the
    # range the model actually sees rather than a synthetic one
    igs = T(f"{pre}.0.gate_proj.input_global_scale").float().item()
    act_amax = FP8_MAX * 6.0 / igs
    print(f"checkpoint input_global_scale={igs:.4f} -> calibrated act amax={act_amax:.3f}")

    print(f"loading {a.experts} experts from layer {a.layer} ...", flush=True)
    W = []
    for e in range(a.experts):
        w = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            n = f"{pre}.{e}.{proj}"
            w[proj] = unpack_nvfp4(T(f"{n}.weight_packed"), T(f"{n}.weight_scale"),
                                   T(f"{n}.weight_global_scale"), dev)
        W.append(w)
    H = W[0]["gate_proj"].shape[1]
    I = W[0]["gate_proj"].shape[0]
    wstd = W[0]["gate_proj"].std().item()
    print(f"hidden={H} intermediate={I} dequant weight std={wstd:.5f}")
    if not (0.001 < wstd < 0.5):
        print(f"!! weight std {wstd:.5f} is implausible for a trained MLP -- "
              f"dequant convention is likely wrong, ABORTING rather than "
              f"reporting numbers from garbage weights")
        return 2

    # activations at the calibrated range. std chosen so amax over the token
    # lands near act_amax, matching what calibration observed.
    x = torch.randn(a.tokens, H, device=dev) * (act_amax / 4.0)
    print(f"activations: {a.tokens}x{H}, amax={x.abs().max().item():.3f}\n")

    def expert_fwd(xin: torch.Tensor, w: dict) -> torch.Tensor:
        g = xin @ w["gate_proj"].T
        u = xin @ w["up_proj"].T
        return (torch.nn.functional.silu(g) * u) @ w["down_proj"].T

    # ---- single-expert arms -------------------------------------------------
    ref = torch.stack([expert_fwd(x, w) for w in W])
    arms = {
        "bf16in  (today H2D)": torch.stack([expert_fwd(rt(x, torch.bfloat16), w) for w in W]),
        "f16out  (today D2H, SHIPPED)": rt(ref, torch.float16),
        "f8in    (candidate)": torch.stack([expert_fwd(q_e4m3_per_token(x), w) for w in W]),
        "f8out   (candidate)": q_e4m3_per_token(ref.reshape(-1, ref.shape[-1])).reshape(ref.shape),
        "f8both  (candidate)": q_e4m3_per_token(
            torch.stack([expert_fwd(q_e4m3_per_token(x), w) for w in W]).reshape(-1, ref.shape[-1])
        ).reshape(ref.shape),
    }
    # granularity sweep: per-token is one scale for all 3072 coordinates, which
    # is the coarsest thing FP8 can do. Finer blocks cost scale bytes and buy
    # precision -- this finds the knee.
    for blk in (512, 128, 32, 16):
        arms[f"f8out/b{blk:<3} (candidate)"] = q_e4m3_blocked(ref, blk)
        arms[f"f8both/b{blk:<2} (candidate)"] = q_e4m3_blocked(
            torch.stack([expert_fwd(q_e4m3_blocked(x, blk), w) for w in W]), blk)

    print(f"{'arm':<30} {'cosine':>12} {'rel L2':>10} {'max rel':>9}")
    print("-" * 64)
    rows = {}
    for k, v in arms.items():
        m = metrics(v, ref)
        rows[k.split()[0]] = m
        print(f"{k:<30} {m['cos']:>12.8f} {m['rel_l2']:>10.2e} {m['max_rel']:>9.2e}")

    # ---- the case that actually worries me: two-card independent scales ----
    # top_k routes split across two cards; each card sums its own experts and
    # quantises that partial with ITS OWN per-token scale. Then both are added
    # to the local+shared contribution. Cancellation lives here.
    half = a.experts // 2
    p0 = ref[:half].sum(0)
    p1 = ref[half:].sum(0)
    local = expert_fwd(x, W[0]) * 0.5  # stand-in for local+shared contribution
    combined_ref = p0 + p1 + local

    print(f"\n{'combined output (2 cards + local)':<30} {'cosine':>12} {'rel L2':>10} {'max rel':>9}")
    print("-" * 64)
    combos = {
        "f16 partials (SHIPPED)": rt(p0, torch.float16) + rt(p1, torch.float16) + local,
        "f8 partials, shared scale": q_e4m3_per_token(p0) + q_e4m3_per_token(p1) + local,
        "f8 partials, indep scales": q_e4m3_per_token(p0 * 1.0) + q_e4m3_per_token(p1 * 1.0) + local,
    }
    for k, v in combos.items():
        m = metrics(v, combined_ref)
        rows["combined:" + k.split()[0]] = m
        print(f"{k:<30} {m['cos']:>12.8f} {m['rel_l2']:>10.2e} {m['max_rel']:>9.2e}")

    # cancellation stress: how much does error amplify when the sum is small
    # relative to its terms? This is the amplification factor to fear.
    term = (p0.abs() + p1.abs() + local.abs())
    amp = (term / combined_ref.abs().clamp_min(1e-9)).median().item()
    print(f"\ncancellation amplification (median |terms|/|sum|): {amp:.2f}x")

    # ---- verdict -----------------------------------------------------------
    ship = rows["f16out"]["rel_l2"]
    print("\n" + "=" * 64)
    print(f"the SHIPPED fp16 output wire sits at rel L2 = {ship:.2e}")
    for k in ("f8in", "f8out", "f8both"):
        r = rows[k]["rel_l2"] / ship
        verdict = "SAME REGIME" if r < 4 else ("MARGINAL" if r < 20 else "DIFFERENT REGIME")
        print(f"  {k:<8} is {r:>7.1f}x the shipped wire's error  -> {verdict}")
    print("=" * 64)
    print("\nThis is a numerics screen, not an acceptance gate. Passing here only\n"
          "earns the right to build the flag; shipping still needs the 120-prompt\n"
          "argmax sweep on combined logprobs against a live baseline.")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(
            {"model": a.model, "layer": a.layer, "experts": a.experts,
             "tokens": a.tokens, "hidden": H, "intermediate": I,
             "input_global_scale": igs, "act_amax": act_amax,
             "weight_std": wstd, "cancellation_amp": amp, "arms": rows}, indent=2))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
