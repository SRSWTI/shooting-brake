#!/usr/bin/env python3
"""CPU-only admission and compaction tests for the 88B int4 hybrid path.

The synthetic bank files contain only the canonical 4 KiB variable header.
No checkpoint or bank payload is created or read.
"""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from shooting_brake_vllm.config import (
    QUALIFIED_ARCHITECTURE,
    QUALIFIED_MODEL_TYPE,
    QualificationError,
    require_qualified_config,
    validate_int4_layer_ownership,
)
from shooting_brake_vllm.int4_bank_format import Int4BankHeader
from shooting_brake_vllm.partition import (
    DispatchBufferGeometry,
    PartitionError,
    build_cuda_expert_maps,
    compact_cuda_routes,
    validate_dispatch_buffer_shapes,
)
from shooting_brake_vllm.placement import Device, build_for_qualified

MODEL = "srswti/axe-superveloce-88b-nvfp4a16"
CUDA_IDS = tuple(range(54))
B70_IDS = tuple(range(54, 180))
PLANE_SIZES = (1_572_864, 49_152, 1_572_864, 49_152, 1_572_864, 49_152)


def fake_vllm_config(*, language_model_only: bool = True) -> SimpleNamespace:
    """Admission-shape stand-in; it does not prove real VllmConfig APIs."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=MODEL,
            hf_config=SimpleNamespace(architectures=[QUALIFIED_ARCHITECTURE]),
            hf_text_config=SimpleNamespace(
                model_type=QUALIFIED_MODEL_TYPE,
                hidden_size=3072,
                num_hidden_layers=48,
                num_experts=180,
                num_experts_per_tok=8,
                moe_intermediate_size=1024,
            ),
            multimodal_config=SimpleNamespace(
                language_model_only=language_model_only,
                limit_per_prompt={"image": 0 if language_model_only else 1},
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enable_eplb=False,
        ),
    )


def canonical_header(source_expert_ids: tuple[int, ...] = B70_IDS) -> Int4BankHeader:
    offsets: list[int] = []
    offset = 0
    for size in PLANE_SIZES:
        offsets.append(offset)
        offset += size
    return Int4BankHeader(
        num_layers=48,
        source_num_layers=48,
        source_experts_per_layer=180,
        hidden=3072,
        moe_intermediate=1024,
        group_size=128,
        bits=4,
        zero_point=8,
        plane_offsets=tuple(offsets),
        plane_sizes=PLANE_SIZES,
        expert_stride_bytes=offset,
        layer_stride_bytes=len(source_expert_ids) * offset,
        source_expert_ids=source_expert_ids,
    )


def write_header(path: Path, source_expert_ids: tuple[int, ...] = B70_IDS) -> None:
    path.write_bytes(canonical_header(source_expert_ids).to_bytes())


def hybrid_environment(bank: Path) -> dict[str, str]:
    return {
        "SHOOTING_BRAKE_PHASE4": "all-cuda",
        "SHOOTING_BRAKE_MODEL": MODEL,
        "SHOOTING_BRAKE_HYBRID": "1",
        "SHOOTING_BRAKE_B70_INT4": "1",
        "SHOOTING_BRAKE_PLACEMENT": "split:54",
        "SHOOTING_BRAKE_B70_BANK": str(bank),
    }


class Int4HybridPlacementTest(unittest.TestCase):
    def qualify(self, bank: Path):
        with patch.dict(os.environ, hybrid_environment(bank), clear=True):
            return require_qualified_config(fake_vllm_config())

    def test_step1_placement_is_54_cuda_126_b70_on_every_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "header-only.bin"
            write_header(bank)
            qualified = self.qualify(bank)

        placement = build_for_qualified(qualified, policy_name="split:54")
        self.assertEqual(placement.num_layers, 48)
        self.assertEqual(placement.num_experts, 180)
        self.assertEqual(placement.remote_device_indices(), (0,))
        for layer in range(48):
            with self.subTest(layer=layer):
                self.assertEqual(placement.cuda_expert_ids(layer), CUDA_IDS)
                self.assertEqual(placement.b70_expert_ids(layer), B70_IDS)
                self.assertEqual(placement.layer_b70_count(layer), 126)
                self.assertTrue(all(
                    placement.owners[layer][expert].device is Device.CUDA
                    for expert in CUDA_IDS
                ))
                self.assertTrue(all(
                    placement.owners[layer][expert].device is Device.B70
                    and placement.owners[layer][expert].device_index == 0
                    for expert in B70_IDS
                ))
        self.assertEqual(placement.cuda_count(), 48 * 54)
        self.assertEqual(placement.b70_count(), 48 * 126)

    def test_cuda_compaction_round_trips_and_remote_ids_become_zero_weight_dummy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "header-only.bin"
            write_header(bank)
            qualified = self.qualify(bank)

        placement = build_for_qualified(qualified, policy_name="split:54")
        global_to_local, local_to_global = build_cuda_expert_maps(placement, 17)
        self.assertTrue(torch.equal(
            global_to_local[local_to_global], torch.arange(54, dtype=torch.long)
        ))
        self.assertTrue(torch.equal(
            local_to_global[global_to_local[:54]], torch.arange(54, dtype=torch.long)
        ))
        self.assertTrue(torch.equal(
            global_to_local[54:], torch.full((126,), -1, dtype=torch.long)
        ))

        global_ids = torch.tensor([[0, 53, 54, 179]], dtype=torch.long)
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        local_ids, cuda_weights, cuda_mask = compact_cuda_routes(
            global_ids, weights, global_to_local
        )
        self.assertEqual(local_ids.tolist(), [[0, 53, 0, 0]])
        self.assertEqual(cuda_mask.tolist(), [[True, True, False, False]])
        self.assertTrue(torch.equal(
            cuda_weights, torch.tensor([[0.1, 0.2, 0.0, 0.0]])
        ))
        # The compact CUDA tensor has only slots 0..53. Remote global IDs 54
        # and 179 reach it only as dummy slot 0 with exactly zero weight.
        self.assertFalse(bool(cuda_weights[~cuda_mask].count_nonzero()))
        self.assertLess(int(local_ids.max()), 54)

    def test_dispatch_buffers_use_3072_hidden_geometry(self) -> None:
        geometry = DispatchBufferGeometry(max_batch=128, hidden_size=3072, top_k=8)
        self.assertEqual(geometry.hidden_shape, (128, 3072))
        self.assertEqual(geometry.route_shape, (128, 8))
        validate_dispatch_buffer_shapes(
            geometry,
            pinned_hidden=torch.empty(128, 3072, dtype=torch.float16),
            pinned_ids=torch.empty(128, 8, dtype=torch.int32),
            pinned_weights=torch.empty(128, 8, dtype=torch.float32),
            pinned_output=torch.empty(128, 3072, dtype=torch.float32),
        )
        with self.assertRaisesRegex(PartitionError, "pinned_hidden=.*2048"):
            validate_dispatch_buffer_shapes(
                geometry,
                pinned_hidden=torch.empty(128, 2048, dtype=torch.float16),
                pinned_ids=torch.empty(128, 8, dtype=torch.int32),
                pinned_weights=torch.empty(128, 8, dtype=torch.float32),
                pinned_output=torch.empty(128, 3072, dtype=torch.float32),
            )

    def test_language_model_only_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "header-only.bin"
            write_header(bank)
            with patch.dict(os.environ, hybrid_environment(bank), clear=True):
                with self.assertRaisesRegex(QualificationError, "language_model_only=True"):
                    require_qualified_config(
                        fake_vllm_config(language_model_only=False)
                    )


class Int4HybridRejectionTest(unittest.TestCase):
    def assert_admission_rejected(self, data: bytes, message: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bad-header-only.bin"
            bank.write_bytes(data)
            with patch.dict(os.environ, hybrid_environment(bank), clear=True):
                with self.assertRaisesRegex(QualificationError, message):
                    require_qualified_config(fake_vllm_config())

    def test_bank_resident_ids_must_equal_placement(self) -> None:
        # Same valid count and all IDs are in range, but it is the wrong set.
        wrong_ids = tuple(range(53, 179))
        self.assert_admission_rejected(
            canonical_header(wrong_ids).to_bytes(),
            "B70 ownership does not match.*placement.*bank",
        )

    def test_coverage_gap_is_rejected(self) -> None:
        with self.assertRaisesRegex(QualificationError, "coverage.*gap|missing"):
            validate_int4_layer_ownership(
                layer=0,
                num_experts=180,
                cuda_expert_ids=range(53),
                b70_expert_ids=B70_IDS,
                bank_source_expert_ids=B70_IDS,
            )

    def test_cuda_b70_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(QualificationError, "overlap|disjoint"):
            validate_int4_layer_ownership(
                layer=0,
                num_experts=180,
                cuda_expert_ids=range(55),
                b70_expert_ids=B70_IDS,
                bank_source_expert_ids=B70_IDS,
            )

    def test_version_one_is_rejected(self) -> None:
        data = bytearray(canonical_header().to_bytes())
        struct.pack_into("<I", data, 8, 1)
        self.assert_admission_rejected(bytes(data), "version 1")

    def test_wrong_zero_point_is_rejected(self) -> None:
        data = bytearray(canonical_header().to_bytes())
        # Header field 12 is zero_point: 8-byte magic + eleven preceding u32s.
        struct.pack_into("<I", data, 8 + 11 * 4, 7)
        self.assert_admission_rejected(bytes(data), "zero_point=7")


if __name__ == "__main__":
    unittest.main(verbosity=2)
