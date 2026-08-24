#!/usr/bin/env python3
"""Phase-1 gate for SHOOTING_BRAKE_B70_PIPELINE: prove the refactor changed nothing.

The per-chunk buffer refactor claims that at nchunks=1 the provider makes the
byte-identical kernel call it made before, and that at nchunks>1 (still
sequential, one queue) the maths is unchanged because each token's routes live
inside one chunk and chunks write disjoint output rows.

Claims are cheap. This captures provider OUTPUT BYTES for a fixed input set and
compares builds:

  capture:  .venv/bin/python benchmarks/pipeline_identity_gate.py capture \
                --lib /tmp/sb_old.so --out /tmp/gate_old_run1.npz
  compare:  .venv/bin/python benchmarks/pipeline_identity_gate.py compare \
                /tmp/gate_old_run1.npz /tmp/gate_new_run1.npz

Protocol (the envelope matters -- the grouped kernel's final scatter uses
atomic float adds, so run-to-run nondeterminism is a PRE-EXISTING property,
measured on 2026-08-22, not something this refactor may hide behind):

  A1, A2   old .so twice        -> the baseline's own run-to-run envelope
  B1       new .so, flag unset  -> must match A within the A1-A2 envelope;
                                   if A1 == A2 bitwise, B1 must be bitwise
  B2       new .so, PIPELINE=1  -> must equal B1 bitwise (same code path)
  C4       new .so, PIPELINE=4  -> chunk-view indexing check: still one queue,
                                   still sequential, so any drift beyond the
                                   A-envelope is a buffer-indexing bug

Single card (Gen4, 0000:15:00.0). The refactor is per-provider; one provider
exercises every changed line.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "src/phase1/expert_bank_jota_118b_r15.bin"
SELECTOR = "0000:15:00.0"
TOP_K = 10
MAX_BATCH = 2048
MS = (1, 7, 33, 256, 2048)  # GEMV path, tail shapes, grouped path, full tile
SEED = 42


def capture(lib: str, out: Path, pipeline: str | None) -> int:
    # Env must be set before the .so is loaded; the provider reads it in load().
    os.environ["SHOOTING_BRAKE_B70_GROUPED"] = "1"
    os.environ["SHOOTING_BRAKE_B70_OUT_FP16"] = "1"
    if pipeline is None:
        os.environ.pop("SHOOTING_BRAKE_B70_PIPELINE", None)
    else:
        os.environ["SHOOTING_BRAKE_B70_PIPELINE"] = pipeline
    os.environ.setdefault("SYCL_UR_USE_LEVEL_ZERO_V2", "0")  # V2 segfaults, Bench 23

    sys.path.insert(0, str(REPO / "src/phase4/src"))
    from shooting_brake_vllm.b70_binding import B70ProviderClient
    from shooting_brake_vllm.config import read_bank_header

    hdr = read_bank_header(str(BANK))
    layers = sorted({0, hdr.layers // 2, hdr.layers - 1})
    hidden = hdr.hidden_size

    client = B70ProviderClient(lib)
    # None means "all experts per layer" (52 GiB -- rejected by placement).
    # Production runs ~85 residents per card; any fixed valid subset works for
    # an identity gate, so take the first 85 expert ids.
    client.load(BANK, top_k=TOP_K, max_batch=MAX_BATCH, device_selector=SELECTOR,
                resident_experts=np.arange(85, dtype=np.int32))
    slots = 85  # compact per-layer slots == the resident set passed above

    rng = np.random.default_rng(SEED)
    blobs: dict[str, np.ndarray] = {}
    for layer in layers:
        for M in MS:
            # Inputs are a function of (SEED, layer, M) only -- independent of
            # build, flag, and iteration order.
            r = np.random.default_rng([SEED, layer, M])
            h = (r.standard_normal((M, hidden)) * 0.35).astype(np.float16)
            ids = np.stack(
                [r.choice(slots, size=TOP_K, replace=False) for _ in range(M)]
            ).astype(np.int32)
            w = r.random((M, TOP_K)).astype(np.float32)
            w /= w.sum(axis=1, keepdims=True)
            seq = client.issue(layer, h, ids, w)
            outb = client.take(seq, M)
            blobs[f"L{layer}_M{M}"] = outb
    _ = rng
    client.shutdown() if hasattr(client, "shutdown") else None

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **blobs)
    man = hashlib.sha256()
    for k in sorted(blobs):
        man.update(k.encode())
        man.update(blobs[k].tobytes())
    print(f"[gate] lib={lib} pipeline={pipeline or 'unset'} cells={len(blobs)} "
          f"manifest={man.hexdigest()[:16]}")
    print(f"[gate] wrote {out}")
    return 0


def compare(a_path: Path, b_path: Path) -> int:
    a, b = np.load(a_path), np.load(b_path)
    if sorted(a.files) != sorted(b.files):
        print(f"FAIL: cell sets differ: {sorted(a.files)} vs {sorted(b.files)}")
        return 1
    worst = 0.0
    exact = 0
    for k in sorted(a.files):
        xa, xb = a[k], b[k]
        if xa.tobytes() == xb.tobytes():
            exact += 1
            print(f"  {k:>12}  BITWISE IDENTICAL")
            continue
        d = np.abs(xa.astype(np.float64) - xb.astype(np.float64))
        rel = d.max() / max(np.abs(xa).max(), 1e-12)
        worst = max(worst, rel)
        print(f"  {k:>12}  differs: max|d|={d.max():.3e} rel={rel:.3e} "
              f"nbytes_diff={(xa != xb).sum()}")
    n = len(a.files)
    print(f"\n{exact}/{n} cells bitwise identical; worst rel delta {worst:.3e}")
    if exact == n:
        print("VERDICT: BITWISE IDENTICAL")
        return 0
    # Caller judges non-bitwise against the baseline's own A1-vs-A2 envelope.
    print("VERDICT: NOT bitwise -- compare against the baseline envelope")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--lib", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--pipeline", default=None)
    d = sub.add_parser("compare")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "capture":
        return capture(args.lib, args.out, args.pipeline)
    return compare(args.a, args.b)


if __name__ == "__main__":
    raise SystemExit(main())
