"""Laguna-faithful DFlash draft registered into SpecForge at process startup."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel,
    Qwen3RotaryEmbedding,
)

import specforge.modeling.draft.dflash as dflash
from specforge.modeling.draft.dflash import DFlashDraftModel, Qwen3DFlashDecoderLayer
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels
from specforge.modeling.draft.registry import register_draft

TARGET_LAYER_IDS = [1, 10, 19, 29, 38, 47]


def _validate_laguna_config(config) -> None:
    expected = {
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "head_dim": 128,
        "num_attention_heads": 72,
        "num_key_value_heads": 8,
        "num_hidden_layers": 6,
        "vocab_size": 100352,
        "draft_vocab_size": 100352,
        "sliding_window": 512,
        "gating": "per-head",
        "attention_bias": False,
        "block_size": 16,
        "num_target_layers": 48,
    }
    for name, value in expected.items():
        if getattr(config, name, None) != value:
            raise ValueError(
                f"Laguna training config {name}={getattr(config, name, None)!r}, "
                f"expected {value!r}"
            )
    if list(getattr(config, "layer_types", ())) != ["sliding_attention"] * 6:
        raise ValueError("Laguna DFlash requires six uniform sliding-attention layers")
    dflash_config = getattr(config, "dflash_config", None) or {}
    expected_dflash = {
        "block_size": 16,
        "mask_token_id": 12,
        "num_target_layers": 48,
        "target_layer_ids": TARGET_LAYER_IDS,
        "causal": True,
    }
    if dflash_config != expected_dflash:
        raise ValueError(
            f"Laguna training dflash_config={dflash_config!r}, "
            f"expected {expected_dflash!r}"
        )


class LagunaDFlashAttention(dflash.Qwen3DFlashAttention):
    """DFlash attention with Laguna's learned softplus per-head output gate."""

    def __init__(self, config, layer_idx: int, kernels: DFlashKernels) -> None:
        super().__init__(config, layer_idx, kernels)
        gating = getattr(config, "gating", None)
        self.gating = bool(gating)
        self.gate_per_head = gating is True or gating == "per-head"
        if self.gating:
            gate_width = (
                config.num_attention_heads
                if self.gate_per_head
                else config.num_attention_heads * self.head_dim
            )
            self.g_proj = nn.Linear(config.hidden_size, gate_width, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch, query_length = hidden_states.shape[:-1]
        context_length = target_hidden.shape[1]
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, query_length, -1, self.head_dim)
        ).transpose(1, 2)
        key = torch.cat(
            (self.k_proj(target_hidden), self.k_proj(hidden_states)), dim=1
        ).view(batch, context_length + query_length, -1, self.head_dim)
        value = torch.cat(
            (self.v_proj(target_hidden), self.v_proj(hidden_states)), dim=1
        ).view(batch, context_length + query_length, -1, self.head_dim)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeddings
        query, key = dflash.apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs
            )

        valid_queries = None
        if self.config._attn_implementation == "flex_attention":
            kernel_options = dict(kwargs.pop("kernel_options", None) or {})
            backend = dflash.flex_attention_backend()
            if backend is not None:
                kernel_options["BACKEND"] = backend
            output = dflash.compile_friendly_flex_attention(
                query,
                key,
                value,
                block_mask=attention_mask,
                enable_gqa=True,
                scale=self.scaling,
                kernel_options=kernel_options or None,
            ).transpose(1, 2)
            attention_weights = None
        else:
            attention_function = dflash.eager_attention_forward
            if self.config._attn_implementation == "eager":
                attention_mask, valid_queries = dflash._prepare_dflash_eager_mask(
                    attention_mask, query.dtype
                )
            else:
                attention_function = dflash.ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
            output, attention_weights = attention_function(
                self,
                query,
                key,
                value,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
            if valid_queries is not None and attention_weights is not None:
                attention_weights = attention_weights.masked_fill(~valid_queries, 0)

        output = output.reshape(batch, query_length, -1)
        if self.gating:
            gate = F.softplus(self.g_proj(hidden_states).float()).to(output.dtype)
            if self.gate_per_head:
                shape = output.shape
                output = (
                    output.view(
                        *shape[:-1], self.config.num_attention_heads, self.head_dim
                    )
                    * gate.unsqueeze(-1)
                ).view(shape)
            else:
                output = output * gate
        output = self.o_proj(output)
        if valid_queries is not None:
            output = output.masked_fill(~valid_queries.any(dim=1), 0)
        return output, attention_weights


class LagunaDFlashDecoderLayer(Qwen3DFlashDecoderLayer):
    """Laguna dense decoder layer, including per-layer context normalization."""

    def __init__(self, config, layer_idx: int, kernels: DFlashKernels) -> None:
        # Build the shared dense MLP and norms once, then install Laguna attention.
        super().__init__(config, layer_idx, kernels)
        self.self_attn = LagunaDFlashAttention(config, layer_idx, kernels)

    def forward(
        self,
        target_hidden=None,
        hidden_states=None,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        normalized_hidden = self.input_layernorm(hidden_states)
        # vLLM's DFlashLaguna context precomputation applies this same layer's
        # input norm before its K/V projection.
        normalized_target = self.input_layernorm(target_hidden)
        hidden_states = self.self_attn(
            hidden_states=normalized_hidden,
            target_hidden=normalized_target,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@register_draft
class LagunaDFlashDraftModel(DFlashDraftModel):
    """SpecForge DFlash backbone matching poolside's native checkpoint exactly."""

    _no_split_modules = ["LagunaDFlashDecoderLayer"]

    def __init__(
        self,
        config,
        dflash_kernels: Optional[DFlashKernels] = None,
    ) -> None:
        _validate_laguna_config(config)
        # DFlashDraftModel constructs Qwen layers and calls post_init itself.
        # Replacing those layers afterward doubles peak construction memory and
        # leaves the replacements outside HF initialization, so initialize the
        # common backbone directly and construct only the Laguna layers.
        Qwen3PreTrainedModel.__init__(self, config)
        self.config = config
        self.layer_types, self.sliding_window = dflash.resolve_dflash_attention_layout(
            config
        )
        kernels = dflash_kernels or DEFAULT_DFLASH_KERNELS
        self.layers = nn.ModuleList(
            [
                LagunaDFlashDecoderLayer(config, layer_index, kernels)
                for layer_index in range(config.num_hidden_layers)
            ]
        )

        dflash_config = getattr(config, "dflash_config", None) or {}
        self.target_layer_ids = list(dflash_config["target_layer_ids"])
        self.aux_hidden_norms = nn.ModuleList(
            [
                kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
                for _ in self.target_layer_ids
            ]
        )
        self.norm = kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.block_size = config.block_size
        self.mask_token_id = dflash_config["mask_token_id"]
        self.projector_type = dflash_config.get("projector_type")
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)
        self._init_draft_head(config, dflash_config)
        self.register_load_state_dict_pre_hook(
            dflash.normalize_draft_head_checkpoint_keys
        )
        self.post_init()

    def forward(self, *args, target_hidden=None, **kwargs):
        if target_hidden is None:
            raise ValueError("Laguna DFlash requires target_hidden")
        feature_count = len(self.target_layer_ids)
        expected_width = feature_count * self.config.hidden_size
        if target_hidden.shape[-1] != expected_width:
            raise ValueError(
                f"target_hidden width {target_hidden.shape[-1]} != {expected_width}"
            )
        slices = target_hidden.unflatten(
            -1, (feature_count, self.config.hidden_size)
        )
        normalized = torch.cat(
            [
                norm(slices[..., index, :])
                for index, norm in enumerate(self.aux_hidden_norms)
            ],
            dim=-1,
        )
        return DFlashDraftModel.forward(
            self, *args, target_hidden=normalized, **kwargs
        )
