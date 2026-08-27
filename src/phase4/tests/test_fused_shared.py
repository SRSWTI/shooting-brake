from __future__ import annotations

from types import SimpleNamespace

import torch

from shooting_brake_vllm.fused_shared import (
    _make_output_rescale,
    _merge_weight_and_block_scale,
)


def test_merged_weight_and_per_half_global_scale_match_two_matmuls() -> None:
    """FP32-emulate packed values + K-block scales; no vLLM/GPU required."""
    generator = torch.Generator().manual_seed(20260825)
    tokens, hidden, intermediate, group_size = 3, 8, 4, 2

    x = torch.randn(tokens, hidden, generator=generator)
    gate_codes = torch.randn(intermediate, hidden, generator=generator)
    up_codes = torch.randn(intermediate, hidden, generator=generator)
    gate_block_scale = torch.rand(
        intermediate, hidden // group_size, generator=generator
    )
    up_block_scale = torch.rand(
        intermediate, hidden // group_size, generator=generator
    )

    gate = SimpleNamespace(weight=gate_codes, weight_scale=gate_block_scale)
    up = SimpleNamespace(weight=up_codes, weight_scale=up_block_scale)
    merged_codes, merged_block_scale = _merge_weight_and_block_scale(gate, up)

    gate_alpha = torch.tensor(0.25, dtype=torch.float32)
    up_alpha = torch.tensor(0.75, dtype=torch.float32)
    output_rescale = _make_output_rescale(
        gate_alpha, up_alpha, intermediate
    )
    assert output_rescale is not None

    gate_weight = gate_codes * gate_block_scale.repeat_interleave(
        group_size, dim=1
    )
    up_weight = up_codes * up_block_scale.repeat_interleave(group_size, dim=1)
    separate = torch.cat(
        (
            x @ gate_weight.T * gate_alpha,
            x @ up_weight.T * up_alpha,
        ),
        dim=-1,
    )

    merged_weight = merged_codes * merged_block_scale.repeat_interleave(
        group_size, dim=1
    )
    fused = (x @ merged_weight.T * gate_alpha) * output_rescale

    torch.testing.assert_close(merged_codes[:intermediate], gate_codes)
    torch.testing.assert_close(merged_codes[intermediate:], up_codes)
    torch.testing.assert_close(
        merged_block_scale[:intermediate], gate_block_scale
    )
    torch.testing.assert_close(
        merged_block_scale[intermediate:], up_block_scale
    )
    torch.testing.assert_close(fused, separate)
