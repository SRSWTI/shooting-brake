"""Flag-gated NVFP4 gate/up fusion for Laguna shared experts.

This module deliberately stays outside vLLM.  :func:`install` monkey-patches the
inference-only ``LagunaMLP.forward`` only when called by the plugin registration
hook; importing this module alone changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class _CannotFuse(RuntimeError):
    """The loaded linear layout cannot be consumed by the fused CUTLASS path."""


@dataclass(frozen=True)
class _MergedNVFP4:
    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_global_scale_inv: torch.Tensor
    alpha: torch.Tensor
    output_rescale: torch.Tensor | None
    half_width: int
    weights_padding_cols: int


def _merge_weight_and_block_scale(
    gate: Any, up: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate output rows; FP4 packing is only along the K dimension."""
    if gate.weight.ndim != 2 or up.weight.ndim != 2:
        raise _CannotFuse("gate/up weights must be matrices")
    if gate.weight.shape != up.weight.shape:
        raise _CannotFuse(
            f"gate/up packed weight shapes differ: {gate.weight.shape} vs "
            f"{up.weight.shape}"
        )
    if gate.weight_scale.ndim != 2 or up.weight_scale.ndim != 2:
        raise _CannotFuse("gate/up block scales must be matrices")
    if gate.weight_scale.shape != up.weight_scale.shape:
        raise _CannotFuse(
            f"gate/up block-scale shapes differ: {gate.weight_scale.shape} vs "
            f"{up.weight_scale.shape}"
        )
    return (
        torch.cat((gate.weight, up.weight), dim=0).contiguous(),
        torch.cat((gate.weight_scale, up.weight_scale), dim=0).contiguous(),
    )


def _make_output_rescale(
    gate_alpha: torch.Tensor,
    up_alpha: torch.Tensor,
    half_width: int,
) -> torch.Tensor | None:
    """Return factors that make one-alpha GEMM algebra match two GEMMs.

    CUTLASS accepts one scalar ``alpha`` for the merged GEMM.  We use the gate
    alpha as that scalar, leave the gate half unchanged, and multiply the up
    half by ``up_alpha / gate_alpha``.  Equal alphas need no extra kernel.
    """
    if torch.equal(gate_alpha, up_alpha):
        return None
    if gate_alpha.numel() != 1 or up_alpha.numel() != 1:
        raise _CannotFuse("gate/up alpha tensors must be scalar")
    if not bool(torch.isfinite(gate_alpha).all()) or not bool(
        torch.isfinite(up_alpha).all()
    ):
        raise _CannotFuse("gate/up alpha tensors must be finite")
    if not bool(torch.ne(gate_alpha, 0).all()):
        raise _CannotFuse("gate alpha must be nonzero")

    ratio = (up_alpha / gate_alpha).to(dtype=torch.float32)
    return torch.cat(
        (
            torch.ones(
                half_width,
                dtype=torch.float32,
                device=gate_alpha.device,
            ),
            ratio.expand(half_width),
        )
    ).contiguous()


def _build_merged_nvfp4(module: Any) -> _MergedNVFP4:
    gate = module.gate_proj
    up = module.up_proj

    # Installed vLLM 0.27.1 schema evidence:
    # compressed_tensors_w4a4_nvfp4.py:52-91 registers ``weight_packed``
    # (uint8 [N,K/2]), ``weight_global_scale``, ``weight_scale``
    # (FP8-E4M3 [N,K/16]), and ``input_global_scale``.  Lines 95-138 rename
    # weight_packed -> weight, invert the checkpoint global scales, retain the
    # checkpoint input divisor as ``input_global_scale_inv``, and precompute
    # ``alpha = input_global_scale * weight_global_scale``.
    # ``weight_scale_2``/``input_scale`` are ModelOpt's on-load aliases
    # (modelopt.py:1164-1190), normalized to the same runtime global names at
    # modelopt.py:1207-1222.  Laguna's compressed-tensors scheme does not
    # expose those aliases, so this patch consumes only the standardized
    # post-load attributes above.
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A4Fp4,
    )
    from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
        CutlassNvFp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        FlashInferCuteDslNvFp4LinearKernel,
        FlashInferCutlassNvFp4LinearKernel,
    )

    cutlass_layout_kernels = (
        CutlassNvFp4LinearKernel,
        FlashInferCuteDslNvFp4LinearKernel,
        FlashInferCutlassNvFp4LinearKernel,
    )
    for name, linear in (("gate", gate), ("up", up)):
        scheme = getattr(linear, "scheme", None)
        if not isinstance(scheme, CompressedTensorsW4A4Fp4) or scheme.use_a16:
            raise _CannotFuse(f"{name}_proj is not compressed-tensors W4A4 NVFP4")
        if not isinstance(scheme.kernel, cutlass_layout_kernels):
            raise _CannotFuse(
                f"{name}_proj kernel {type(scheme.kernel).__name__} does not use "
                "the CUTLASS swizzled weight layout"
            )
        if getattr(linear, "bias", None) is not None:
            raise _CannotFuse(f"{name}_proj has a bias")
        if linear.weight.dtype != torch.uint8:
            raise _CannotFuse(f"{name}_proj packed weight is not uint8")
        if linear.weight_scale.dtype != torch.float8_e4m3fn:
            raise _CannotFuse(f"{name}_proj block scale is not FP8-E4M3")

    gate_input_scale = gate.input_global_scale_inv
    up_input_scale = up.input_global_scale_inv
    if not torch.equal(gate_input_scale, up_input_scale):
        # A single scaled_fp4_quant call can encode only one tensor-global
        # divisor (vllm/_custom_ops.py:1523-1598).
        raise _CannotFuse("gate/up input_global_scale_inv values differ")

    if gate.output_size_per_partition != up.output_size_per_partition:
        raise _CannotFuse("gate/up logical output widths differ")
    half_width = int(gate.output_size_per_partition)

    gate_padding = int(getattr(gate, "weights_padding_cols", 0))
    up_padding = int(getattr(up, "weights_padding_cols", 0))
    if gate_padding != up_padding:
        raise _CannotFuse("gate/up CUTLASS K padding differs")

    # cutlass.py:35-43 swizzles scales and pads weights during post-load.
    # swizzle_blockscale (nvfp4_utils.py:37-53) pads N to 128 before laying
    # out 128-row tiles.  Concatenating two already-swizzled tensors is valid
    # only at a complete tile boundary.  Refuse padded N rather than placing
    # the up rows after the gate tensor's internal zero padding.
    if gate.weight.shape[0] != half_width or up.weight.shape[0] != half_width:
        raise _CannotFuse("gate/up packed weights have padded output rows")
    if (
        gate.weight_scale.shape[0] != half_width
        or up.weight_scale.shape[0] != half_width
    ):
        raise _CannotFuse(
            "gate/up swizzled block scales contain N-padding; row concat is unsafe"
        )

    weight, weight_scale = _merge_weight_and_block_scale(gate, up)
    output_rescale = _make_output_rescale(gate.alpha, up.alpha, half_width)
    return _MergedNVFP4(
        weight=weight,
        weight_scale=weight_scale,
        input_global_scale_inv=gate_input_scale,
        alpha=gate.alpha,
        output_rescale=output_rescale,
        half_width=half_width,
        weights_padding_cols=gate_padding,
    )


def install() -> None:
    """Install the Laguna MLP monkey-patch; the caller owns flag gating."""
    from vllm import _custom_ops as ops
    from vllm.logger import init_logger
    from vllm.model_executor.models.laguna import LagunaMLP

    original_forward = LagunaMLP.forward
    if getattr(original_forward, "_shooting_brake_fused_shared", False):
        return

    logger = init_logger(__name__)
    # activation.py:117-150 wraps torch.ops._C.silu_and_mul in a CustomOp,
    # but CustomOp construction asserts a live vLLM config context and this
    # install runs at plugin registration (boot failure 2026-08-25:
    # "CustomOp was instantiated at module import time"). Call the raw
    # kernel directly instead -- same op the CustomOp dispatches to. The
    # only fused SiLU+mul+NVFP4 quant in _custom_ops.py:1663-1725 is
    # expert-aware, so the dense shared expert keeps quantization inside
    # down_proj.

    def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
        d = gate_up.shape[-1] // 2
        out = torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype,
                          device=gate_up.device)
        torch.ops._C.silu_and_mul(out, gate_up)
        return out

    def disarm(module: Any, reason: BaseException) -> None:
        module._shooting_brake_fused_shared_disarmed = True
        module._shooting_brake_fused_shared_cache = None
        prefix = getattr(module.gate_proj, "prefix", type(module).__name__)
        logger.warning(
            "Shooting Brake: fused shared expert permanently disabled for %s: %s",
            prefix,
            reason,
        )

    def fused_forward(module: Any, x: torch.Tensor) -> torch.Tensor:
        if not getattr(module, "_shooting_brake_fused_seen", False):
            module._shooting_brake_fused_seen = True
            logger.info(
                "Shooting Brake: fused_forward ENTERED for %s (training=%s, "
                "grad=%s)",
                getattr(module.gate_proj, "prefix", type(module).__name__),
                module.training, torch.is_grad_enabled(),
            )
            print("[fused-shared] ENTERED "
                  f"{getattr(module.gate_proj, 'prefix', '?')}", flush=True)
        # Preserve training/autograd behavior exactly; this optimization is
        # an inference dispatch path, not a replacement MLP implementation.
        if module.training or torch.is_grad_enabled():
            return original_forward(module, x)
        if getattr(module, "_shooting_brake_fused_shared_disarmed", False):
            return original_forward(module, x)

        try:
            cache = getattr(module, "_shooting_brake_fused_shared_cache", None)
            if cache is None:
                cache = _build_merged_nvfp4(module)
                module._shooting_brake_fused_shared_cache = cache
                logger.info(
                    "Shooting Brake: fused shared expert ARMED for %s "
                    "(half_width=%d, rescale=%s)",
                    getattr(module.gate_proj, "prefix", "?"),
                    cache.half_width,
                    cache.output_rescale is not None,
                )
                print("[fused-shared] ARMED "
                      f"{getattr(module.gate_proj, 'prefix', '?')}",
                      flush=True)

            # Mirrors CutlassNvFp4LinearKernel.apply_weights exactly
            # (cutlass.py:51-77), except N is gate||up and x is quantized once.
            x_fp4, x_blockscale = ops.scaled_fp4_quant(
                x,
                cache.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend="cutlass",
                padded_n=x.shape[-1] + cache.weights_padding_cols * 2,
            )
            gate_up = ops.cutlass_scaled_fp4_mm(
                x_fp4,
                cache.weight,
                x_blockscale,
                cache.weight_scale,
                cache.alpha,
                x.dtype,
            )
            if cache.output_rescale is not None:
                gate_up.mul_(cache.output_rescale)
            activated = silu_and_mul(gate_up)
            activated = activated.view(*x.shape[:-1], cache.half_width)
            output, _ = module.down_proj(activated)
            return output
        except Exception as exc:  # Safety contract: fail closed per module.
            disarm(module, exc)
            return original_forward(module, x)

    fused_forward._shooting_brake_fused_shared = True  # type: ignore[attr-defined]
    fused_forward._shooting_brake_original = original_forward  # type: ignore[attr-defined]
    LagunaMLP.forward = fused_forward


__all__ = ["install"]
