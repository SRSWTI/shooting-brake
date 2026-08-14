#!/usr/bin/env python3
"""Focused CPU-only tests for split checkpoints and multi-device placement."""

from __future__ import annotations

import inspect
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch

from shooting_brake_vllm.config import (
    QUALIFIED_ARCHITECTURE,
    QUALIFIED_MODEL_TYPE,
    SUPPORTED_MODELS,
    QualificationError,
    QualifiedModel,
    phase4_enabled,
    require_qualified_config,
    read_bank_header,
    validate_int4_layer_ownership,
)
from shooting_brake_vllm.expert_bank import Int4ExpertBank
from shooting_brake_vllm.int4_bank_format import Int4BankHeader
from shooting_brake_vllm.partition import (
    DispatchBufferGeometry,
    PartitionError,
    build_cuda_expert_maps,
    compact_cuda_routes,
    validate_dispatch_buffer_shapes,
    validate_cuda_dummy_slot_placement,
)
from shooting_brake_vllm.placement import (
    Device,
    DeviceCapacity,
    DeviceTarget,
    ExpertGroup,
    ExpertGroupPolicy,
    FractionalRemotePolicy,
    Placement,
    PlacementError,
    SplitPolicy,
    b70_bank_covers,
    build_for_qualified,
    build_placement,
)
import shooting_brake_vllm.routed_experts as routed_experts_module
from shooting_brake_vllm.routed_experts import install_preemptive_alloc_hook
from shooting_brake_vllm.telemetry import (
    _collect_eager_partition,
    _provider_health_stats,
)


MODEL = "srswti/axe-superveloce-88b-nvfp4a16"
INT4_MODEL = "srswti/axe-superveloce-88b-int4"
INT4_EXPERT_BYTES = 4_866_048  # 4.640625 MiB, qzeros omitted.


def fake_vllm_config(*, language_model_only: bool = True) -> SimpleNamespace:
    """Admission-shape stand-in, not a vLLM lifecycle/API compatibility test.

    Constructing a real VllmConfig resolves the remote HF model and initializes
    runtime subsystems. Real Placement and HybridRoutedExperts objects cover
    the APIs this focused CPU suite can construct without that side effect.
    """
    text = SimpleNamespace(
        model_type=QUALIFIED_MODEL_TYPE,
        hidden_size=3072,
        num_hidden_layers=48,
        num_experts=180,
        num_experts_per_tok=8,
        moe_intermediate_size=1024,
    )
    model = SimpleNamespace(
        model=MODEL,
        hf_config=SimpleNamespace(architectures=[QUALIFIED_ARCHITECTURE]),
        hf_text_config=text,
        multimodal_config=SimpleNamespace(
            language_model_only=language_model_only,
            limit_per_prompt={"image": 0 if language_model_only else 1},
        ),
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_eplb=False,
    )
    return SimpleNamespace(model_config=model, parallel_config=parallel)


def int4_header(
    source_expert_ids: tuple[int, ...],
    *,
    num_layers: int = 48,
) -> Int4BankHeader:
    hidden, intermediate, group_size = 3072, 1024, 128
    sizes = (
        hidden // 8 * intermediate * 4,
        hidden // group_size * intermediate * 2,
        hidden // 8 * intermediate * 4,
        hidden // group_size * intermediate * 2,
        intermediate // 8 * hidden * 4,
        intermediate // group_size * hidden * 2,
    )
    offsets: list[int] = []
    offset = 0
    for size in sizes:
        offsets.append(offset)
        offset += size
    return Int4BankHeader(
        num_layers=num_layers,
        source_num_layers=48,
        source_experts_per_layer=180,
        hidden=hidden,
        moe_intermediate=intermediate,
        group_size=group_size,
        bits=4,
        zero_point=8,
        plane_offsets=tuple(offsets),
        plane_sizes=sizes,
        expert_stride_bytes=offset,
        layer_stride_bytes=offset * len(source_expert_ids),
        source_expert_ids=source_expert_ids,
    )


def write_int4_header(
    path: Path,
    source_expert_ids: tuple[int, ...],
    *,
    version: int = 2,
) -> None:
    data = bytearray(int4_header(source_expert_ids).to_bytes())
    struct.pack_into("<I", data, 8, version)
    path.write_bytes(data)


class SplitCheckpointConfigTest(unittest.TestCase):
    def test_model_registry_and_qualification_without_weights(self) -> None:
        spec = SUPPORTED_MODELS[MODEL]
        self.assertEqual(spec.routed_experts_model, INT4_MODEL)
        self.assertEqual(spec.routed_expert_format, "gptq-int4-group128")
        self.assertEqual(
            (spec.hidden_size, spec.num_layers, spec.num_experts, spec.top_k,
             spec.moe_intermediate_size),
            (3072, 48, 180, 8, 1024),
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "SHOOTING_BRAKE_B70_BANK": f"{directory}/absent.bin",
                "SHOOTING_BRAKE_PHASE4": "all-cuda",
                "SHOOTING_BRAKE_MODEL": MODEL,
            },
            clear=True,
        ):
            self.assertTrue(phase4_enabled())
            qualified = require_qualified_config(fake_vllm_config())
        self.assertEqual(qualified.model, MODEL)
        self.assertEqual(qualified.routed_experts_model, INT4_MODEL)
        self.assertEqual(qualified.routed_expert_format, "gptq-int4-group128")
        self.assertEqual(qualified.bank_layers, 0)
        self.assertEqual(qualified.bank_experts_per_layer, 180)
        self.assertEqual(qualified.b70_capable_layers, frozenset())

    def test_legacy_bank_defaults_reject_88b_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "bank_experts_per_layer=256"):
            QualifiedModel(
                model=MODEL,
                architecture=QUALIFIED_ARCHITECTURE,
                hidden_size=3072,
                num_layers=48,
                num_experts=180,
                top_k=8,
                moe_intermediate_size=1024,
            )

    def test_language_model_only_is_required_for_88b(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"SHOOTING_BRAKE_B70_BANK": f"{directory}/absent.bin"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                QualificationError, "language_model_only=True",
            ):
                require_qualified_config(
                    fake_vllm_config(language_model_only=False)
                )

    def test_language_model_only_flag_is_an_effective_signal(self) -> None:
        config = fake_vllm_config(language_model_only=False)
        config.model_config.multimodal_config = SimpleNamespace(
            language_model_only=True,
            limit_per_prompt={"image": 1},
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"SHOOTING_BRAKE_B70_BANK": f"{directory}/absent.bin"},
            clear=True,
        ):
            qualified = require_qualified_config(config)
        self.assertEqual(qualified.model, MODEL)

    def test_absent_multimodal_config_is_effectively_text_only(self) -> None:
        config = fake_vllm_config(language_model_only=False)
        config.model_config.multimodal_config = None
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"SHOOTING_BRAKE_B70_BANK": f"{directory}/absent.bin"},
            clear=True,
        ):
            qualified = require_qualified_config(config)
        self.assertEqual(qualified.model, MODEL)

    def test_zero_modality_limits_do_not_prove_tower_is_excluded(self) -> None:
        config = fake_vllm_config(language_model_only=False)
        config.model_config.multimodal_config = SimpleNamespace(
            language_model_only=False,
            limit_per_prompt={"image": 0, "video": 0},
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"SHOOTING_BRAKE_B70_BANK": f"{directory}/absent.bin"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                QualificationError, "language_model_only=True",
            ):
                require_qualified_config(config)


class Step1HybridContractTest(unittest.TestCase):
    RESIDENT = tuple(range(54, 180))

    def test_preemptive_hook_covers_active_modelopt_and_legacy_class(
        self,
    ) -> None:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (
            CompressedTensorsW4A4Nvfp4MoEMethod,
        )
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4FusedMoE,
        )

        install_preemptive_alloc_hook()
        expected_parameters = (
            "self",
            "layer",
            "num_experts",
            "hidden_size",
            "intermediate_size_per_partition",
            "params_dtype",
            "extra_weight_attrs",
        )
        for method in (
            CompressedTensorsW4A4Nvfp4MoEMethod,
            ModelOptNvFp4FusedMoE,
        ):
            self.assertTrue(
                getattr(method, "_shooting_brake_alloc_hook", False)
            )
            self.assertEqual(
                tuple(inspect.signature(method.create_weights).parameters),
                expected_parameters,
            )
        self.assertIsNot(
            CompressedTensorsW4A4Nvfp4MoEMethod.create_weights,
            ModelOptNvFp4FusedMoE.create_weights,
        )

    def qualify(self, bank: Path, *, placement: str = "split:54") -> QualifiedModel:
        with patch.dict(
            os.environ,
            {
                "SHOOTING_BRAKE_B70_BANK": str(bank),
                "SHOOTING_BRAKE_HYBRID": "1",
                "SHOOTING_BRAKE_B70_INT4": "1",
                "SHOOTING_BRAKE_PLACEMENT": placement,
                "SHOOTING_BRAKE_MODEL": MODEL,
            },
            clear=True,
        ):
            return require_qualified_config(fake_vllm_config())

    def test_step1_bank_and_placement_pass_all_48_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "step1.bin"
            write_int4_header(bank, self.RESIDENT)
            qualified = self.qualify(bank)

        self.assertEqual(qualified.bank_layers, 48)
        self.assertEqual(qualified.bank_experts_per_layer, 126)
        self.assertEqual(qualified.bank_source_expert_ids, self.RESIDENT)
        placement = build_for_qualified(qualified, "split:54")
        self.assertEqual(placement.cuda_count(), 48 * 54)
        self.assertEqual(placement.b70_count(), 48 * 126)
        self.assertEqual(placement.remote_device_indices(), (0,))
        for layer in range(48):
            self.assertEqual(placement.cuda_expert_ids(layer), tuple(range(54)))
            self.assertEqual(
                placement.b70_expert_ids(layer), tuple(range(54, 180)),
            )

    def test_wrong_resident_set_rejects_with_both_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "wrong-residents.bin"
            write_int4_header(bank, tuple(range(126)))
            with self.assertRaisesRegex(
                QualificationError,
                r"B70 ownership does not match.*placement size=126.*bank size=126",
            ):
                self.qualify(bank)

    def test_version_one_rejects_before_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "version1.bin"
            write_int4_header(bank, self.RESIDENT, version=1)
            with self.assertRaisesRegex(
                QualificationError, "unsupported int4 bank version 1",
            ):
                self.qualify(bank)

    def test_wrong_magic_rejects_before_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "wrong-magic.bin"
            data = bytearray(int4_header(self.RESIDENT).to_bytes())
            data[:8] = b"NOTINT4!"
            bank.write_bytes(data)
            with self.assertRaisesRegex(
                QualificationError, "not a supported Shooting Brake bank",
            ):
                self.qualify(bank)

    def test_gap_and_overlap_are_loud(self) -> None:
        with self.assertRaisesRegex(QualificationError, "coverage gap"):
            validate_int4_layer_ownership(
                layer=0,
                num_experts=180,
                cuda_expert_ids=range(53),
                b70_expert_ids=self.RESIDENT,
                bank_source_expert_ids=self.RESIDENT,
            )
        with self.assertRaisesRegex(QualificationError, "overlaps"):
            validate_int4_layer_ownership(
                layer=0,
                num_experts=180,
                cuda_expert_ids=range(55),
                b70_expert_ids=self.RESIDENT,
                bank_source_expert_ids=self.RESIDENT,
            )

    def test_cuda_compaction_round_trips_and_masks_b70_routes(self) -> None:
        placement = build_placement(
            SplitPolicy(54),
            num_layers=48,
            num_experts=180,
            b70_capable=frozenset(range(48)),
        )
        global_to_local, local_to_global = build_cuda_expert_maps(placement, 0)
        self.assertEqual(local_to_global.tolist(), list(range(54)))
        self.assertEqual(global_to_local[:54].tolist(), list(range(54)))
        self.assertTrue(torch.equal(
            global_to_local[54:], torch.full((126,), -1, dtype=torch.long),
        ))
        self.assertTrue(torch.equal(
            global_to_local[local_to_global],
            torch.arange(54, dtype=torch.long),
        ))

        global_ids = torch.tensor([[0, 53, 54, 179]])
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        local_ids, cuda_weights, cuda_mask = compact_cuda_routes(
            global_ids, weights, global_to_local,
        )
        self.assertEqual(local_ids.tolist(), [[0, 53, 0, 0]])
        self.assertEqual(cuda_mask.tolist(), [[True, True, False, False]])
        self.assertTrue(torch.equal(
            cuda_weights, torch.tensor([[0.1, 0.2, 0.0, 0.0]]),
        ))
        self.assertTrue(bool((local_ids < 54).all()))

    def test_zero_cuda_experts_rejects_dummy_slot_path(self) -> None:
        no_cuda = build_placement(
            FractionalRemotePolicy((0,), cuda_fraction=0.0),
            num_layers=48,
            num_experts=180,
            b70_capable=frozenset(range(48)),
        )
        with self.assertRaisesRegex(
            PartitionError, "zero CUDA experts.*dummy local slot 0",
        ):
            validate_cuda_dummy_slot_placement(no_cuda)

    def test_dispatch_buffers_resolve_to_hidden_3072(self) -> None:
        geometry = DispatchBufferGeometry(
            max_batch=128, hidden_size=3072, top_k=8,
        )
        self.assertEqual(geometry.hidden_shape, (128, 3072))
        self.assertEqual(geometry.route_shape, (128, 8))
        validate_dispatch_buffer_shapes(
            geometry,
            pinned_hidden=torch.empty(128, 3072),
            pinned_ids=torch.empty(128, 8),
            pinned_weights=torch.empty(128, 8),
            pinned_output=torch.empty(128, 3072),
        )
        with self.assertRaisesRegex(
            Exception, r"pinned_output=\(128, 2048\).*expected \(128, 3072\)",
        ):
            validate_dispatch_buffer_shapes(
                geometry,
                pinned_hidden=torch.empty(128, 3072),
                pinned_ids=torch.empty(128, 8),
                pinned_weights=torch.empty(128, 8),
                pinned_output=torch.empty(128, 2048),
            )

    def test_sync_dispatch_rejects_cuda_capture_before_host_work(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "synchronous B70 dispatch.*enforce_eager=True",
        ):
            routed_experts_module._validate_dispatch_capture_mode(
                graph_mode=False, stream_capturing=True,
            )
        routed_experts_module._validate_dispatch_capture_mode(
            graph_mode=False, stream_capturing=False,
        )
        routed_experts_module._validate_dispatch_capture_mode(
            graph_mode=True, stream_capturing=True,
        )

    def test_b70_cuda_slot_map_exists_before_first_forward(self) -> None:
        layer = routed_experts_module.HybridRoutedExperts.__new__(
            routed_experts_module.HybridRoutedExperts
        )
        torch.nn.Module.__init__(layer)
        layer._hybrid_active = True
        layer._b70_slot_map = torch.tensor(
            [-1] * 54 + list(range(126)), dtype=torch.int32,
        ).numpy()
        layer._initialize_b70_slot_map_cuda(device="cpu")
        self.assertEqual(layer._b70_slot_map_cuda.shape, (180,))
        self.assertEqual(layer._b70_slot_map_cuda[:54].tolist(), [-1] * 54)
        self.assertEqual(
            layer._b70_slot_map_cuda[54:].tolist(), list(range(126))
        )

    def test_preemptive_validator_uses_real_placement_owner_map(self) -> None:
        placement = build_placement(
            SplitPolicy(54),
            num_layers=48,
            num_experts=180,
            b70_capable=frozenset(range(48)),
        )
        method_type = type("ModelOptNvFp4FusedMoE", (), {})
        method = method_type()
        method.use_global_sf = True
        compact = torch.empty(54, 1)
        layer = SimpleNamespace(
            shooting_brake_placement=placement,
            layer_name="model.layers.0.mlp.experts",
            quant_method=method,
            w13_weight=compact,
            w2_weight=compact,
            w13_weight_scale=compact,
            w2_weight_scale=compact,
            w13_input_scale=torch.empty(180, 1),
            w2_input_scale=torch.empty(180),
        )
        prior = dict(routed_experts_module._preemptive_alloc_invocations)
        try:
            routed_experts_module._preemptive_alloc_invocations.clear()
            routed_experts_module._preemptive_alloc_invocations.update({
                layer_idx: (
                    f"model.layers.{layer_idx}.mlp.experts",
                    "ModelOptNvFp4FusedMoE",
                )
                for layer_idx in range(48)
            })
            with patch.dict(
                os.environ,
                {"SHOOTING_BRAKE_PREEMPTIVE_SURGERY": "1"},
                clear=False,
            ):
                routed_experts_module._validate_preemptive_allocations(layer)
        finally:
            routed_experts_module._preemptive_alloc_invocations.clear()
            routed_experts_module._preemptive_alloc_invocations.update(prior)

    def test_sparse_bank_coverage_uses_source_ids_not_prefix_count(self) -> None:
        sparse = build_placement(
            ExpertGroupPolicy((
                ExpertGroup(0, 1, DeviceTarget(Device.B70, 0)),
                ExpertGroup(1, 4, DeviceTarget(Device.CUDA)),
                ExpertGroup(4, 5, DeviceTarget(Device.B70, 0)),
                ExpertGroup(5, 9, DeviceTarget(Device.CUDA)),
                ExpertGroup(9, 10, DeviceTarget(Device.B70, 0)),
                ExpertGroup(10, 12, DeviceTarget(Device.CUDA)),
            )),
            num_layers=1,
            num_experts=12,
            b70_capable=frozenset({0}),
        )
        self.assertTrue(b70_bank_covers(
            sparse,
            bank_layers=1,
            bank_experts_per_layer=3,
            bank_source_expert_ids=(0, 4, 9),
        ))
        self.assertFalse(b70_bank_covers(
            sparse,
            bank_layers=1,
            bank_experts_per_layer=3,
            bank_source_expert_ids=(0, 1, 8),
        ))


class Int4ExpertBankTest(unittest.TestCase):
    def test_canonical_header_and_explicit_resident_ids(self) -> None:
        hidden, intermediate, group_size = 3072, 1024, 128
        qweight_bytes = hidden // 8 * intermediate * 4
        scales_bytes = hidden // group_size * intermediate * 2
        down_qweight_bytes = intermediate // 8 * hidden * 4
        down_scales_bytes = intermediate // group_size * hidden * 2
        sizes = (
            qweight_bytes, scales_bytes,
            qweight_bytes, scales_bytes,
            down_qweight_bytes, down_scales_bytes,
        )
        offsets: list[int] = []
        offset = 0
        for size in sizes:
            offsets.append(offset)
            offset += size
        header = Int4BankHeader(
            num_layers=1,
            source_num_layers=48,
            source_experts_per_layer=180,
            hidden=hidden,
            moe_intermediate=intermediate,
            group_size=group_size,
            bits=4,
            zero_point=8,
            plane_offsets=tuple(offsets),
            plane_sizes=sizes,
            expert_stride_bytes=offset,
            layer_stride_bytes=offset,
            source_expert_ids=(2,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.bin"
            path.write_bytes(header.to_bytes() + bytes(header.layer_stride_bytes))
            parsed = read_bank_header(str(path))
            self.assertEqual(parsed.format, "gptq-int4-group128")
            self.assertEqual(parsed.experts_per_layer, 1)
            self.assertEqual(parsed.logical_experts_per_layer, 180)
            self.assertEqual(parsed.source_expert_ids, (2,))
            self.assertEqual(parsed.data_offset, 4096)

            bank = Int4ExpertBank(path)
            planes = bank.expert(0, 2)
            self.assertEqual(planes.gate_qweight.shape, (384, 1024))
            self.assertEqual(planes.gate_scales.shape, (24, 1024))
            self.assertEqual(planes.down_qweight.shape, (128, 3072))
            with self.assertRaisesRegex(IndexError, "not resident"):
                bank.expert(0, 1)
            bank.close()


class EagerTelemetryTest(unittest.TestCase):
    def test_partition_liveness_reports_expected_remote_steps(self) -> None:
        placement = build_placement(
            SplitPolicy(54),
            num_layers=2,
            num_experts=180,
            b70_capable=frozenset(range(2)),
        )
        layers = [
            SimpleNamespace(
                _layer_idx=layer_idx,
                shooting_brake_placement=placement,
                shooting_brake_partition_stats={
                    "steps": 5,
                    "remote_steps": 5 if layer_idx == 0 else 4,
                    "remote_routes": 27 if layer_idx == 0 else 25,
                },
            )
            for layer_idx in range(2)
        ]
        self.assertEqual(
            _collect_eager_partition(layers),
            {
                "steps": 10,
                "remote_steps": 9,
                "expected_remote_steps": 10,
                "expected_provider_dispatches": 10,
                "all_cuda_route_steps": 1,
                "remote_routes": 52,
            },
        )
    def test_provider_health_reports_raw_baseline_and_delta(self) -> None:
        health = SimpleNamespace(
            generation=9,
            dispatches=37,
            last_error="",
        )
        self.assertEqual(
            _provider_health_stats(health, (9, 12)),
            {
                "available": True,
                "generation_raw": 9,
                "generation_baseline": 9,
                "dispatches_raw": 37,
                "dispatches_baseline": 12,
                "dispatches_delta": 25,
                "last_error": "",
            },
        )

    def test_provider_health_rejects_counter_regression(self) -> None:
        health = SimpleNamespace(
            generation=9,
            dispatches=11,
            last_error="",
        )
        stats = _provider_health_stats(health, (9, 12))
        self.assertFalse(stats["available"])
        self.assertIsNone(stats["dispatches_delta"])
        self.assertIn("counter reset", stats["reason"])

    def test_provider_health_rejects_generation_change(self) -> None:
        health = SimpleNamespace(
            generation=10,
            dispatches=37,
            last_error="",
        )
        stats = _provider_health_stats(health, (9, 12))
        self.assertFalse(stats["available"])
        self.assertIsNone(stats["dispatches_delta"])
        self.assertIn("generation changed", stats["reason"])



class MultiDevicePlacementTest(unittest.TestCase):
    def build(self, policy, *, capacities=(), experts: int = 180) -> Placement:
        return build_placement(
            policy,
            num_layers=48,
            num_experts=experts,
            b70_capable=frozenset(range(48)),
            device_capacities=capacities,
        )

    def test_explicit_groups_are_complete_disjoint_and_indexed(self) -> None:
        placement = self.build(ExpertGroupPolicy((
            ExpertGroup(0, 60, DeviceTarget(Device.B70, 0)),
            ExpertGroup(60, 120, DeviceTarget(Device.B70, 1)),
            ExpertGroup(120, 180, DeviceTarget(Device.CUDA)),
        )))
        row = placement.owners[0]
        self.assertEqual(
            [row[index].target for index in (0, 59, 60, 119, 120, 179)],
            [
                DeviceTarget(Device.B70, 0), DeviceTarget(Device.B70, 0),
                DeviceTarget(Device.B70, 1), DeviceTarget(Device.B70, 1),
                DeviceTarget(Device.CUDA), DeviceTarget(Device.CUDA),
            ],
        )
        self.assertEqual(placement.remote_device_indices(), (0, 1))
        self.assertEqual(placement.count_target(DeviceTarget(Device.B70, 0)), 48 * 60)
        self.assertEqual(placement.count_target(DeviceTarget(Device.B70, 1)), 48 * 60)
        self.assertEqual(placement.cuda_count(), 48 * 60)
        self.assertEqual(Placement.from_manifest(placement.to_manifest()), placement)

    def test_group_gap_is_rejected(self) -> None:
        with self.assertRaisesRegex(PlacementError, "do not cover"):
            self.build(ExpertGroupPolicy((
                ExpertGroup(0, 60, DeviceTarget(Device.B70, 0)),
                ExpertGroup(61, 180, DeviceTarget(Device.CUDA)),
            )))

    def test_group_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(PlacementError, "more than one group"):
            self.build(ExpertGroupPolicy((
                ExpertGroup(0, 100, DeviceTarget(Device.B70, 0)),
                ExpertGroup(99, 180, DeviceTarget(Device.B70, 1)),
            )))

    def test_180_experts_balance_across_two_and_three_devices(self) -> None:
        for device_count, expected_per_device in ((2, 90), (3, 60)):
            with self.subTest(device_count=device_count):
                placement = self.build(FractionalRemotePolicy(
                    tuple(range(device_count)), cuda_fraction=0.0,
                ))
                self.assertEqual(placement.cuda_count(), 0)
                self.assertEqual(placement.remote_device_indices(), tuple(range(device_count)))
                for index in range(device_count):
                    self.assertEqual(
                        placement.count_target(DeviceTarget(Device.B70, index)),
                        48 * expected_per_device,
                    )

    def test_fractional_policy_keeps_half_on_cuda_without_divisibility_assumption(self) -> None:
        placement = self.build(FractionalRemotePolicy((0, 1), cuda_fraction=0.5))
        self.assertEqual(placement.cuda_count(), 48 * 90)
        self.assertEqual(placement.count_target(DeviceTarget(Device.B70, 0)), 48 * 45)
        self.assertEqual(placement.count_target(DeviceTarget(Device.B70, 1)), 48 * 45)

        odd = self.build(
            FractionalRemotePolicy((0, 1), cuda_fraction=0.0), experts=181
        )
        self.assertEqual(odd.count_target(DeviceTarget(Device.B70, 0)), 48 * 91)
        self.assertEqual(odd.count_target(DeviceTarget(Device.B70, 1)), 48 * 90)

    def test_device_vram_overcommit_is_rejected(self) -> None:
        target = DeviceTarget(Device.B70, 0)
        resident_experts = 48 * 60
        limit = DeviceCapacity(
            target=target,
            capacity_bytes=resident_experts * INT4_EXPERT_BYTES - 1,
            bytes_per_expert=INT4_EXPERT_BYTES,
        )
        with self.assertRaisesRegex(PlacementError, "but capacity is"):
            self.build(
                ExpertGroupPolicy((
                    ExpertGroup(0, 60, target),
                    ExpertGroup(60, 180, DeviceTarget(Device.CUDA)),
                )),
                capacities=(limit,),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
