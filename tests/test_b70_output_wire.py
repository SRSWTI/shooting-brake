from __future__ import annotations

import importlib
import importlib.util
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_PACKAGE = Path(__file__).parents[1] / "src" / "phase4" / "src" / "shooting_brake_vllm"
_HIDDEN = 256
_INTERMEDIATE = 256
_WIRE_FLAG = "SHOOTING_BRAKE_B70_OUT_FP16"


def _write_unit_bank(path: Path, down_scale: float = 1.0) -> None:
    plane_sizes = (
        2 * _INTERMEDIATE * (_HIDDEN // 2),
        2 * _INTERMEDIATE * (_HIDDEN // 16),
        _HIDDEN * (_INTERMEDIATE // 2),
        _HIDDEN * (_INTERMEDIATE // 16),
    )
    with path.open("wb") as stream:
        stream.write(
            struct.pack("<8sIIIIIQQQQ", b"SBEXP001", 1, 1, _HIDDEN, _INTERMEDIATE, 0, *plane_sizes)
        )
        for value, size in zip((0x22, 0x38, 0x22, 0x38), plane_sizes):
            stream.write(bytes([value]) * size)
        stream.write(struct.pack("<ff", 1.0 / _HIDDEN, down_scale / _INTERMEDIATE))


@pytest.fixture(scope="module")
def b70_binding():
    if "SB_TEST_B70_LIBRARY" not in os.environ:
        pytest.skip("set SB_TEST_B70_LIBRARY to exercise the actual B70 provider")

    # Load the real package under a test-local name so the binding's relative
    # config import works without importing an installed vLLM plugin instead.
    spec = importlib.util.spec_from_file_location(
        "_b70_output_wire_package",
        _PACKAGE / "__init__.py",
        submodule_search_locations=[str(_PACKAGE)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f"{spec.name}.b70_binding")


@pytest.fixture(params=[None, "0", "1", "01"], ids=["unset", "zero", "one", "zero-one"])
def wire_provider(request, b70_binding, tmp_path):
    flag = request.param
    settings = {
        _WIRE_FLAG: flag,
        "SHOOTING_BRAKE_B70_GROUPED": "0",
        "SYCL_UR_USE_LEVEL_ZERO_V2": "0",
    }
    previous = {key: os.environ.get(key) for key in settings}
    provider = None
    try:
        for key, value in settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        bank = tmp_path / "expert_bank.bin"
        _write_unit_bank(bank)

        # An explicitly configured missing library or a provider/load failure
        # must fail, not turn a requested hardware run into a skip.
        provider = b70_binding.B70ProviderClient(os.environ["SB_TEST_B70_LIBRARY"])
        provider.load(
            bank,
            top_k=1,
            max_batch=1,
            device_selector=os.environ.get("SB_TEST_B70_SELECTOR", ""),
        )
        yield provider, flag
    finally:
        try:
            if provider is not None:
                provider.shutdown()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def test_b70_output_wire_matches_actual_result(wire_provider):
    provider, flag = wire_provider
    hidden = np.ones((1, _HIDDEN), dtype=np.float16)
    ids = np.zeros((1, 1), dtype=np.int32)
    weights = np.ones((1, 1), dtype=np.float32)

    sequence = provider.issue(layer=0, hidden_fp16=hidden, ids=ids, weights=weights)
    output = provider.take(sequence, M=1)

    assert output.shape == (1, _HIDDEN)
    assert output.dtype == np.dtype(np.float16 if flag == "1" else np.float32)
    assert np.isfinite(output).all()
    # Unit FP4 weights and reciprocal global scales give gate=up=1 and
    # down=mean(SiLU(1) * 1), hence every output element is sigmoid(1).
    expected = 1.0 / (1.0 + math.exp(-1.0))
    np.testing.assert_allclose(output, expected, rtol=0.0, atol=2e-3)


@pytest.mark.parametrize("wire", ["0", "1"], ids=["fp32", "fp16"])
@pytest.mark.parametrize("pipeline", ["1", "2"])
def test_grouped_swiglu_preserves_range_and_reused_scratch(b70_binding, tmp_path, wire, pipeline):
    settings = {
        _WIRE_FLAG: wire,
        "SHOOTING_BRAKE_B70_GROUPED": "1",
        "SHOOTING_BRAKE_B70_PIPELINE": pipeline,
        "SB_GROUPED_BACKEND": "native",
        "SB_GROUPED_SCATTER": "atomic",
        "SYCL_UR_USE_LEVEL_ZERO_V2": "0",
    }
    previous = {key: os.environ.get(key) for key in settings}
    provider = None
    try:
        os.environ.update(settings)
        bank = tmp_path / "grouped_range_bank.bin"
        _write_unit_bank(bank, down_scale=1.0 / 16)
        provider = b70_binding.B70ProviderClient(os.environ["SB_TEST_B70_LIBRARY"])
        provider.load(
            bank,
            top_k=1,
            max_batch=64,
            device_selector=os.environ.get("SB_TEST_B70_SELECTOR", ""),
        )
        # 255**2 fits FP16, 256**2 does not, and 512**2 needs a larger
        # scaling exponent. The final down-scaled outputs all fit FP16.
        values = np.resize(np.array([0, 1, 255, 256, 512, -512], dtype=np.float16), 64)
        hidden = np.repeat(values[:, None], _HIDDEN, axis=1)
        ids = np.zeros((64, 1), dtype=np.int32)
        weights = np.linspace(0.25, 1.0, 64, dtype=np.float32)[:, None]
        weights[4] = 0.0
        output = provider.dispatch(0, hidden, ids, weights)
        values64 = values.astype(np.float64)
        expected = values64 * values64 / (1.0 + np.exp(-values64)) / 16
        expected = np.repeat((expected * weights[:, 0])[:, None], _HIDDEN, axis=1)
        assert np.isfinite(output).all()
        np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-5)

        # Route scales must not leak into the next dispatch through slot_w.
        hidden.fill(1.0)
        output = provider.dispatch(0, hidden, ids, weights)
        expected = np.repeat(weights / (16 * (1 + math.exp(-1))), _HIDDEN, axis=1)
        assert np.isfinite(output).all()
        np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-5)

        # Empty experts skip the workgroup reduction uniformly and return zero.
        ids.fill(-1)
        output = provider.dispatch(0, hidden, ids, weights)
        np.testing.assert_array_equal(output, np.zeros_like(output))
    finally:
        try:
            if provider is not None:
                provider.shutdown()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
