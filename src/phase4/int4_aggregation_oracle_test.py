#!/usr/bin/env python3
"""Focused CPU tests for the offline int4 aggregation oracle."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.phase4.int4_aggregation_oracle import _dequant, _error


class Int4AggregationOracleTest(unittest.TestCase):
    def test_k_major_nibbles_zero_point_and_negative_scales(self) -> None:
        # Column 0 nibbles are 0..7; column 1 nibbles are 8..15.
        packed = np.array(
            [[sum(k << (4 * k) for k in range(8)),
              sum((k + 8) << (4 * k) for k in range(8))]],
            dtype=np.uint32,
        ).view(np.int32)
        scales = np.array([[-2.0, 0.5]], dtype=np.float16)
        got = _dequant(packed, scales, group_size=8)
        expected = torch.tensor(
            [[(k - 8) * -2.0, k * 0.5] for k in range(8)],
            dtype=torch.float32,
        )
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)

    def test_error_reports_exact_sum(self) -> None:
        value = torch.tensor([[1.0, -2.0, 3.0]])
        metrics = _error(value, value.clone())
        self.assertEqual(metrics["max_abs"], 0.0)
        self.assertEqual(metrics["mean_abs"], 0.0)
        self.assertEqual(metrics["rel_l2"], 0.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
