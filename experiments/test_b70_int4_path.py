#!/usr/bin/env python3
"""Validate Colibri S4 repack and B70 K-major SLM ESIMD execution.

Exercises eight distinct resident slots in non-sorted route order, unequal FP16
routing weights, all three projections, and final route accumulation.
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TIER_LIB = Path("/tmp/libb70_tier_test.so")
MOE_LIB = ROOT / "colibri-variants/colibri-qwen36/c/libb70_moe.so"
H, I, GS, EXPERTS = 2048, 512, 64, 8
UNSHUFFLE = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.uint32)


def pack_s4(values: np.ndarray) -> np.ndarray:
    low = values[:, 0::2].astype(np.uint8) & 0x0F
    high = values[:, 1::2].astype(np.uint8) & 0x0F
    return np.ascontiguousarray(low | (high << 4))


def decode_ipex(
    words: np.ndarray, scale_bits: np.ndarray, rows: int, columns: int
) -> tuple[np.ndarray, np.ndarray]:
    words = words.reshape(columns // 8, rows)
    scales = scale_bits.view(np.float16).reshape(columns // GS, rows)
    values = np.empty((rows, columns), dtype=np.int8)
    for packed in range(columns // 8):
        for element in range(8):
            nibble = (
                (words[packed] >> (UNSHUFFLE[element] * 4)) & 0x0F
            ).astype(np.int16)
            values[:, packed * 8 + element] = (nibble - 8).astype(np.int8)
    return values, scales.T.copy()


def load_abis():
    tier = ctypes.CDLL(str(TIER_LIB))
    convert = tier.b70_convert_expert_s4
    convert.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 4
    convert.restype = ctypes.c_int

    moe = ctypes.CDLL(str(MOE_LIB))
    moe.b70_moe_init.argtypes = [ctypes.c_int] * 5
    moe.b70_moe_init.restype = ctypes.c_int
    moe.b70_moe_upload.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4
    moe.b70_moe_upload.restype = ctypes.c_int
    moe.b70_moe_dispatch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
    moe.b70_moe_dispatch.restype = ctypes.c_int
    moe.b70_moe_shutdown.argtypes = []
    return convert, moe


def dispatch_latency(moe, activation, output, ids, weights, routes, iterations=200):
    for _ in range(10):
        assert moe.b70_moe_dispatch(
            activation.ctypes.data, output.ctypes.data, ids.ctypes.data,
            weights.ctypes.data, routes
        ) == 0
    start = time.perf_counter()
    for _ in range(iterations):
        assert moe.b70_moe_dispatch(
            activation.ctypes.data, output.ctypes.data, ids.ctypes.data,
            weights.ctypes.data, routes
        ) == 0
    return (time.perf_counter() - start) * 1e6 / iterations


def main() -> None:
    rng = np.random.default_rng(20260803)
    convert, moe = load_abis()

    gu_words = np.empty((H // 8) * (2 * I), dtype=np.uint32)
    gu_scale_bits = np.empty((H // GS) * (2 * I), dtype=np.uint16)
    down_words = np.empty((I // 8) * H, dtype=np.uint32)
    down_scale_bits = np.empty((I // GS) * H, dtype=np.uint16)

    references = []
    assert moe.b70_moe_init(EXPERTS, H, I, EXPERTS, GS) == 0
    try:
        for expert in range(EXPERTS):
            gate_q = rng.integers(-8, 8, (I, H), dtype=np.int8)
            up_q = rng.integers(-8, 8, (I, H), dtype=np.int8)
            down_q = rng.integers(-8, 8, (H, I), dtype=np.int8)
            gate = pack_s4(gate_q)
            up = pack_s4(up_q)
            down = pack_s4(down_q)
            gate_scales = rng.uniform(0.001, 0.009, (I, H // GS)).astype(np.float32)
            up_scales = rng.uniform(0.002, 0.011, (I, H // GS)).astype(np.float32)
            down_scales = rng.uniform(0.0015, 0.008, (H, I // GS)).astype(np.float32)

            assert convert(
                gate.ctypes.data, up.ctypes.data, down.ctypes.data,
                gate_scales.ctypes.data, up_scales.ctypes.data,
                down_scales.ctypes.data, H, I, GS,
                gu_words.ctypes.data, gu_scale_bits.ctypes.data,
                down_words.ctypes.data, down_scale_bits.ctypes.data,
            ) == 1

            if expert == 0:
                decoded_gu, decoded_gu_scales = decode_ipex(
                    gu_words, gu_scale_bits, 2 * I, H
                )
                decoded_down, decoded_down_scales = decode_ipex(
                    down_words, down_scale_bits, H, I
                )
                np.testing.assert_array_equal(decoded_gu[:I], gate_q)
                np.testing.assert_array_equal(decoded_gu[I:], up_q)
                np.testing.assert_array_equal(decoded_down, down_q)
                np.testing.assert_array_equal(
                    decoded_gu_scales[:I], gate_scales.astype(np.float16)
                )
                np.testing.assert_array_equal(
                    decoded_gu_scales[I:], up_scales.astype(np.float16)
                )
                np.testing.assert_array_equal(
                    decoded_down_scales, down_scales.astype(np.float16)
                )

            assert moe.b70_moe_upload(
                expert, gu_words.ctypes.data, gu_scale_bits.ctypes.data,
                down_words.ctypes.data, down_scale_bits.ctypes.data
            ) == 0
            references.append((
                gate_q,
                up_q,
                down_q,
                gate_scales.astype(np.float16),
                up_scales.astype(np.float16),
                down_scales.astype(np.float16),
            ))

        print("Converter: exact values/layout for K-major GS64; FP16 scales verified")

        activation = rng.normal(0.0, 0.1, H).astype(np.float32)
        output = np.empty(H, dtype=np.float32)
        route_ids = np.array([7, 2, 5, 0, 6, 1, 4, 3], dtype=np.int32)
        route_weights = np.array(
            [0.23, 0.04, 0.17, 0.09, 0.14, 0.07, 0.18, 0.08],
            dtype=np.float32,
        )

        x = activation.astype(np.float16).astype(np.float32)
        route_outputs = []
        for expert, route_weight in zip(route_ids, route_weights):
            gate_q, up_q, down_q, gate_s, up_s, down_s = references[int(expert)]
            gate_w = gate_q.astype(np.float32) * np.repeat(
                gate_s.astype(np.float32), GS, axis=1
            )
            up_w = up_q.astype(np.float32) * np.repeat(
                up_s.astype(np.float32), GS, axis=1
            )
            down_w = down_q.astype(np.float32) * np.repeat(
                down_s.astype(np.float32), GS, axis=1
            )
            gate_value = gate_w @ x
            up_value = up_w @ x
            intermediate = (
                gate_value / (1.0 + np.exp(-gate_value)) * up_value
            ).astype(np.float16).astype(np.float32)
            routed = (
                (down_w @ intermediate)
                * np.float16(route_weight).astype(np.float32)
            ).astype(np.float16).astype(np.float32)
            route_outputs.append(routed)
        reference = np.sum(route_outputs, axis=0, dtype=np.float32).astype(np.float16).astype(np.float32)

        eight_us = dispatch_latency(
            moe, activation, output, route_ids, route_weights, EXPERTS
        )
        cosine = float(
            np.dot(reference, output)
            / (np.linalg.norm(reference) * np.linalg.norm(output))
        )
        max_abs = float(np.max(np.abs(reference - output)))
        mean_abs = float(np.mean(np.abs(reference - output)))
        print(
            f"Eight-route B70 vs CPU: cosine={cosine:.8f} "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
        )
        print(f"Eight-route dispatch: {eight_us:.1f} us ({eight_us / 8:.1f} us/expert)")
        assert cosine > 0.999
        assert mean_abs < 0.01

        one_us = dispatch_latency(
            moe, activation, output, route_ids, route_weights, 1
        )
        print(f"One-route dispatch: {one_us:.1f} us")
    finally:
        moe.b70_moe_shutdown()

    print("B70 INT4 converter + multi-route ESIMD path PASSED")


if __name__ == "__main__":
    main()
