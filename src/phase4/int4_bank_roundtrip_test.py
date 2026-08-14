#!/usr/bin/env python3
"""Real extractor-to-plugin round trip for the canonical int4 bank ABI."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shooting_brake_vllm.config import (
    QUALIFIED_ARCHITECTURE,
    QUALIFIED_MODEL_TYPE,
    read_bank_header,
    require_qualified_config,
)
from shooting_brake_vllm.expert_bank import Int4ExpertBank
from shooting_brake_vllm.int4_bank_format import (
    MAGIC,
    VERSION,
    read_int4_bank_header,
)

ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR = ROOT / "src" / "phase1" / "extract_experts_int4.py"
EXPERT_BYTES = 4_866_048
LAYER_BYTES = 8 * EXPERT_BYTES
EXPECTED_PLANE_SIZES = (
    1_572_864,
    49_152,
    1_572_864,
    49_152,
    1_572_864,
    49_152,
)

MODEL = "srswti/axe-superveloce-88b-nvfp4a16"


def fake_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=MODEL,
            hf_config=SimpleNamespace(
                architectures=[QUALIFIED_ARCHITECTURE]
            ),
            hf_text_config=SimpleNamespace(
                model_type=QUALIFIED_MODEL_TYPE,
                hidden_size=3072,
                num_hidden_layers=48,
                num_experts=180,
                num_experts_per_tok=8,
                moe_intermediate_size=1024,
            ),
            language_model_only=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enable_eplb=False,
        ),
    )


class WriterReaderRoundTripTest(unittest.TestCase):
    def test_real_builder_and_plugin_reader_share_every_header_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "tiny-int4-bank.bin"
            command = [
                sys.executable,
                str(EXTRACTOR),
                "--layers", "2",
                "--experts", "8",
                "--out", str(bank_path),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"extractor failed:\n{completed.stdout}",
            )
            self.assertTrue(bank_path.is_file(), completed.stdout)
            self.assertEqual(bank_path.read_bytes()[:8], MAGIC)

            canonical = read_int4_bank_header(bank_path)
            self.assertEqual(VERSION, 2)
            self.assertEqual(canonical.num_layers, 2)
            self.assertEqual(canonical.source_num_layers, 48)
            self.assertEqual(canonical.experts_per_layer, 8)
            self.assertEqual(canonical.source_experts_per_layer, 180)
            self.assertEqual(canonical.source_expert_ids, tuple(range(8)))
            self.assertEqual(canonical.resident_set_shared_across_layers, 1)
            self.assertEqual(canonical.hidden, 3072)
            self.assertEqual(canonical.moe_intermediate, 1024)
            self.assertEqual(canonical.group_size, 128)
            self.assertEqual(canonical.bits, 4)
            self.assertEqual(canonical.zero_point, 8)
            self.assertEqual(canonical.data_offset, 4096)
            self.assertEqual(canonical.plane_sizes, EXPECTED_PLANE_SIZES)
            expected_offsets = []
            offset = 0
            for size in EXPECTED_PLANE_SIZES:
                expected_offsets.append(offset)
                offset += size
            self.assertEqual(canonical.plane_offsets, tuple(expected_offsets))
            self.assertEqual(canonical.expert_stride_bytes, EXPERT_BYTES)
            self.assertEqual(canonical.layer_stride_bytes, LAYER_BYTES)
            self.assertEqual(
                bank_path.stat().st_size,
                canonical.data_offset + 2 * LAYER_BYTES,
            )

            config_header = read_bank_header(str(bank_path))
            self.assertEqual(config_header.version, canonical.version)
            self.assertEqual(config_header.layers, canonical.num_layers)
            self.assertEqual(
                config_header.source_layers, canonical.source_num_layers
            )
            self.assertEqual(
                config_header.experts_per_layer, canonical.experts_per_layer
            )
            self.assertEqual(
                config_header.source_experts_per_layer,
                canonical.source_experts_per_layer,
            )
            self.assertEqual(
                config_header.source_expert_ids, canonical.source_expert_ids
            )
            self.assertEqual(config_header.hidden_size, canonical.hidden)
            self.assertEqual(
                config_header.moe_intermediate_size,
                canonical.moe_intermediate,
            )
            self.assertEqual(config_header.group_size, canonical.group_size)
            self.assertEqual(config_header.bits, canonical.bits)
            self.assertEqual(config_header.zero_point, canonical.zero_point)
            self.assertEqual(config_header.data_offset, canonical.data_offset)
            self.assertEqual(
                config_header.expert_bytes, canonical.expert_stride_bytes
            )
            self.assertEqual(
                config_header.resident_set_shared_across_layers,
                canonical.resident_set_shared_across_layers,
            )

            with patch.dict(
                os.environ,
                {
                    "SHOOTING_BRAKE_B70_BANK": str(bank_path),
                    "SHOOTING_BRAKE_MODEL": MODEL,
                },
                clear=True,
            ):
                qualified = require_qualified_config(fake_vllm_config())
            self.assertEqual(qualified.bank_layers, 2)
            self.assertEqual(qualified.bank_experts_per_layer, 8)
            self.assertEqual(qualified.bank_source_expert_ids, tuple(range(8)))
            self.assertEqual(qualified.num_experts, 180)
            self.assertEqual(
                qualified.b70_capable_layers, frozenset((0, 1))
            )

            reader = Int4ExpertBank(bank_path)
            self.assertEqual(reader.header, canonical)
            planes = reader.expert(1, 7)
            self.assertEqual(planes.gate_qweight.shape, (384, 1024))
            self.assertEqual(planes.gate_scales.shape, (24, 1024))
            self.assertEqual(planes.up_qweight.shape, (384, 1024))
            self.assertEqual(planes.up_scales.shape, (24, 1024))
            self.assertEqual(planes.down_qweight.shape, (128, 3072))
            self.assertEqual(planes.down_scales.shape, (8, 3072))
            with self.assertRaisesRegex(IndexError, "not resident"):
                reader.expert(0, 8)
            reader.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
