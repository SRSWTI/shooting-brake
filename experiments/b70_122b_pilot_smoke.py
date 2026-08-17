#!/usr/bin/env python3
"""122B pilot-bank smoke: real Qwen-GPTQ planes through the B70 kernel.

Loads the 2-layer/256-expert pilot bank (built by extract_experts_int4.py
from Qwen/Qwen3.5-122B-A10B-GPTQ-Int4, bit-exact-validated against shards),
fires production-shaped dispatches on the idle B70, and gates the kernel
output against a CPU fp32 oracle dequantized from the same bank planes --
the same three-level gate discipline the 88B earned its 2.15e-09 with.

  ZE_AFFINITY_MASK=0 SHOOTING_BRAKE_B70_INT4=1 .venv/bin/python \
      experiments/b70_122b_pilot_smoke.py
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phase1"))
from extract_experts_int4 import dequantize_int4  # noqa: E402

BANK = Path("/tmp/sb_122b_pilot_int4.bin")
LIB = ROOT / "src/phase7/libsb_b70_provider.so"
HIDDEN, INTER, TOPK, MAX_BATCH = 3072, 1024, 8, 256
E = 256
GENERATION = 9


def read_bank_planes(layer: int, expert: int):
    """Six planes of one expert straight from the bank file (mmap)."""
    import mmap

    with open(BANK, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    # header prefix: <8s26I2Q (128 bytes)
    import struct

    fields = struct.unpack_from("<8s26I2Q", mm, 0)
    magic, _version, data_offset = fields[0], fields[1], fields[2]
    assert magic == b"SBINT401", magic
    plane = fields[15:27]  # 6 x (offset, size) pairs start at index 15
    expert_stride, layer_stride = fields[27], fields[28]
    base = data_offset + layer * layer_stride + expert * expert_stride
    out = {}
    names = ("gate_q", "gate_s", "up_q", "up_s", "down_q", "down_s")
    for i, name in enumerate(names):
        off, size = plane[2 * i], plane[2 * i + 1]
        buf = bytes(mm[base + off: base + off + size])
        if name.endswith("_q"):
            out[name] = torch.frombuffer(bytearray(buf), dtype=torch.int32)
        else:
            out[name] = torch.frombuffer(bytearray(buf), dtype=torch.float16)
    mm.close()
    # reshape to AutoGPTQ [K/8, N] / [K/128, N]
    out["gate_q"] = out["gate_q"].view(HIDDEN // 8, INTER)
    out["up_q"] = out["up_q"].view(HIDDEN // 8, INTER)
    out["down_q"] = out["down_q"].view(INTER // 8, HIDDEN)
    out["gate_s"] = out["gate_s"].view(HIDDEN // 128, INTER)
    out["up_s"] = out["up_s"].view(HIDDEN // 128, INTER)
    out["down_s"] = out["down_s"].view(INTER // 128, HIDDEN)
    return out


def oracle(layer, x, ids, wts):
    """fp32 reference from the SAME bank planes the B70 loaded."""
    M = x.shape[0]
    out = torch.zeros(M, HIDDEN, dtype=torch.float64)
    xr = x.double()
    for m in range(M):
        for j in range(ids.shape[1]):
            e = int(ids[m, j])
            if e < 0:
                continue
            p = read_bank_planes(layer, e)
            gate_w = dequantize_int4(p["gate_q"], p["gate_s"]).double()  # [K,N]
            up_w = dequantize_int4(p["up_q"], p["up_s"]).double()
            down_w = dequantize_int4(p["down_q"], p["down_s"]).double()
            g = xr[m] @ gate_w
            u = xr[m] @ up_w
            act = F.silu(g) * u
            out[m] += float(wts[m, j]) * (act @ down_w)
    return out.float()


def main():
    assert BANK.exists(), "run the pilot extraction first"
    lib = ctypes.CDLL(str(LIB))
    lib.sb_b70_create.restype = ctypes.c_void_p
    lib.sb_b70_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_uint64, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_size_t]
    lib.sb_b70_issue.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                 ctypes.c_uint64, ctypes.c_size_t,
                                 ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_size_t]
    lib.sb_b70_take.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                ctypes.c_uint64, ctypes.c_void_p,
                                ctypes.c_size_t]
    lib.sb_b70_num_resident.restype = ctypes.c_size_t
    lib.sb_b70_num_resident.argtypes = [ctypes.c_void_p]

    p = lib.sb_b70_create()
    t0 = time.perf_counter()
    rc = lib.sb_b70_load(p, str(BANK).encode(), GENERATION, None, 0, MAX_BATCH)
    assert rc == 0, f"load rc={rc}"
    resident = lib.sb_b70_num_resident(p)
    print(f"pilot bank loaded in {time.perf_counter()-t0:.1f}s, "
          f"resident experts/layer: {resident}")
    assert resident % E == 0, resident  # reports total across layers here

    rng = np.random.default_rng(3)
    M = 3
    hidden = (rng.standard_normal((M, HIDDEN)) * 0.05).astype(np.float16)
    ids = np.full((M, TOPK), -1, dtype=np.int32)
    wts = np.zeros((M, TOPK), dtype=np.float32)
    picks = rng.choice(E, size=(M, 6), replace=False)
    for m in range(M):
        for j in range(6):
            ids[m, j] = picks[m, j]
            wts[m, j] = 1.0 / 6.0

    worst = 0.0
    for layer in range(2):
        out = np.zeros((M, HIDDEN), dtype=np.float32)
        rc = lib.sb_b70_issue(p, GENERATION, layer + 1, layer,
                              hidden.ctypes.data, ids.ctypes.data,
                              wts.ctypes.data, M)
        assert rc == 0, f"issue layer {layer} rc={rc}"
        rc = lib.sb_b70_take(p, GENERATION, layer + 1, out.ctypes.data,
                             M * HIDDEN)
        assert rc == 0, f"take layer {layer} rc={rc}"

        ref = oracle(layer, torch.from_numpy(hidden), torch.from_numpy(ids),
                     torch.from_numpy(wts)).numpy()
        scale = np.abs(ref).max()
        delta = np.abs(out - ref).max()
        cos = float(F.cosine_similarity(torch.from_numpy(out).flatten(),
                                        torch.from_numpy(ref).flatten(), dim=0))
        worst = max(worst, delta / max(scale, 1e-9))
        print(f"layer {layer}: |ref|max={scale:.4f} max|delta|={delta:.3e} "
              f"rel={delta/max(scale,1e-9):.3e} cosine={cos:.9f}")

    lib.sb_b70_shutdown(p)
    lib.sb_b70_destroy(p)
    assert worst < 1e-3, f"kernel-vs-oracle rel delta too large: {worst}"
    print(f"PASS: 122B pilot bank serves correctly on the B70 "
          f"(worst rel delta {worst:.3e})")


if __name__ == "__main__":
    main()
