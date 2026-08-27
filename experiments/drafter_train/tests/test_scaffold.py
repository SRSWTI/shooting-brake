from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parents[1]
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(REPO_ROOT / "vendor" / "SpecForge"))

import gate  # noqa: E402
from export_laguna import validate_config  # noqa: E402
from prepare_warmstart import (  # noqa: E402
    build_config,
    expected_native_keys,
    expected_training_keys,
)
from specforge.config import load_config  # noqa: E402


def poolside_config() -> dict:
    return {
        "architectures": ["DFlashLagunaForCausalLM"],
        "model_type": "laguna",
        "attention_bias": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "max_position_embeddings": 1048576,
        "num_attention_heads": 72,
        "num_hidden_layers": 6,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "sliding_window": 512,
        "vocab_size": 100352,
        "layer_types": ["sliding_attention"] * 6,
        "rope_theta": 500000.0,
        "gating": "per-head",
        "draft_vocab_size": 100352,
        "torch_dtype": "bfloat16",
        "eagle_aux_hidden_state_layer_ids": [2, 11, 20, 30, 39, 48],
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 12,
            "num_target_layers": 48,
            "target_layer_ids": [1, 10, 19, 29, 38, 47],
            "causal": True,
        },
    }


def quality_payload(logprob_shift: float = 0.0) -> dict:
    quality = []
    for index in range(gate.QUALITY_COUNT):
        quality.append(
            {
                "index": index,
                "top_logprobs": {
                    " alpha": math.log(0.7) + logprob_shift,
                    " beta": math.log(0.2),
                },
            }
        )
    return {
        "schema": "drafter_quality_sweep_v1",
        "model": "test",
        "top_logprobs": gate.QUALITY_TOP_LOGPROBS,
        "quality": quality,
    }


def ladder_rows(*, accepted: int = 60, drafted: int = 100, tpot_ms: float = 6.0) -> list[dict]:
    return [
        {
            "label": gate.LADDER_LABEL,
            "ctx": context,
            "prompt_tokens": context,
            "out_tokens": 512,
            "ttft_s": 1.0,
            "tpot_ms": tpot_ms,
            "chunk_gap_p50_ms": 1.0,
            "decode_tok_s": 1000.0 / tpot_ms,
            "wall_s": 4.0,
            "drafted": drafted,
            "accepted": accepted,
            "acceptance_pct": round(100.0 * accepted / drafted, 1),
        }
        for context in gate.LADDER_CONTEXTS
    ]


def test_poolside_config_maps_to_registered_training_contract() -> None:
    native = poolside_config()
    validate_config(native)
    training = build_config(native)

    assert training["architectures"] == ["LagunaDFlashDraftModel"]
    assert training["model_type"] == "qwen3"
    assert training["block_size"] == 16
    assert training["num_target_layers"] == 48
    assert "eagle_aux_hidden_state_layer_ids" not in training
    assert len(expected_native_keys()) == 69
    assert len(expected_training_keys()) == 81
    for layer in range(6):
        prefix = f"layers.{layer}.self_attn."
        assert prefix + "qkv_proj.weight" in expected_native_keys()
        assert prefix + "qkv_proj.weight" not in expected_training_keys()
        assert prefix + "q_proj.weight" in expected_training_keys()
        assert prefix + "k_proj.weight" in expected_training_keys()
        assert prefix + "v_proj.weight" in expected_training_keys()


def test_pilot_is_a_typed_single_gpu_offline_warmstart_run() -> None:
    config = load_config(str(MODULE_DIR / "pilot.yaml"))

    assert config.mode == "offline"
    assert config.training.strategy == "dflash"
    # Warm start is mandatory, but it is selected by train.sh, not baked
    # into pilot.yaml: SpecForge overrides are string-typed, so a value
    # baked here could never be unset by a "=null" override and would
    # collide with training.resume_from on a resume (the two are mutually
    # exclusive). pilot.yaml therefore leaves BOTH unset and train.sh sets
    # exactly one -- the invariant is asserted in the test below.
    assert config.model.draft_checkpoint_path is None
    assert config.training.resume_from is None
    assert config.deployment.trainer.nnodes == 1
    assert config.deployment.trainer.nproc_per_node == 1
    assert config.training.fsdp_sharding == "NO_SHARD"
    assert config.training.batch_size == 1
    assert config.training.accumulation_steps == 4

def test_train_sh_sets_exactly_one_of_warmstart_or_resume() -> None:
    """Fresh runs MUST warm start; resumes MUST NOT re-apply it."""
    script = (MODULE_DIR / "train.sh").read_text()

    # Anchor on the branch banners, not on "else": train.sh contains
    # several if/else blocks before this one.
    _, _, after_resume = script.partition("[3/4] resuming")
    resume_branch, _, warm_branch = after_resume.partition("[3/4] training")
    assert resume_branch, "resume branch banner not found in train.sh"
    assert warm_branch, "warm-start branch banner not found in train.sh"

    assert "training.resume_from=" in resume_branch
    assert "model.draft_checkpoint_path=" not in resume_branch

    warm_branch = warm_branch.partition("\nfi")[0]
    assert "model.draft_checkpoint_path=" in warm_branch
    assert "training.resume_from=" not in warm_branch


def test_quality_gate_compares_distributions_not_generated_text(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(quality_payload()) + "\n")
    candidate.write_text(json.dumps(quality_payload()) + "\n")

    comparison = gate.compare(baseline, candidate)

    assert comparison["metric"] == "top_logprob_jensen_shannon_divergence_nats"
    assert comparison["quality_pct"] == 100.0
    assert comparison["max_jsd"] == 0.0


def test_acceptance_thresholds_pass_at_inclusive_boundaries(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    ladder = tmp_path / "ladder.json"
    output = tmp_path / "summary.json"
    baseline.write_text(json.dumps(quality_payload()) + "\n")
    candidate.write_text(json.dumps(quality_payload()) + "\n")
    ladder.write_text(json.dumps(ladder_rows()) + "\n")

    assert gate.summarize(baseline, candidate, ladder, output) == 0
    summary = json.loads(output.read_text())
    assert summary["passed"] is True
    assert summary["aggregate_acceptance_pct"] == 60.0
    assert summary["worst_effective_decode_ms_per_token"] == 6.0
    assert all(summary["gates"].values())


def test_shell_entrypoints_parse() -> None:
    for script in ("train.sh", "acceptance.sh"):
        result = subprocess.run(
            ["bash", "-n", str(MODULE_DIR / script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
