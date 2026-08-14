from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_BENCHMARKS = Path(__file__).parents[1] / "benchmarks"

_LOCALITY_SPEC = importlib.util.spec_from_file_location(
    "route_locality", _BENCHMARKS / "route_locality.py"
)
assert _LOCALITY_SPEC is not None and _LOCALITY_SPEC.loader is not None
route_locality = importlib.util.module_from_spec(_LOCALITY_SPEC)
sys.modules[_LOCALITY_SPEC.name] = route_locality
_LOCALITY_SPEC.loader.exec_module(route_locality)

_TOPOLOGY_SPEC = importlib.util.spec_from_file_location(
    "route_topology", _BENCHMARKS / "route_topology.py"
)
assert _TOPOLOGY_SPEC is not None and _TOPOLOGY_SPEC.loader is not None
route_topology = importlib.util.module_from_spec(_TOPOLOGY_SPEC)
sys.modules[_TOPOLOGY_SPEC.name] = route_topology
_TOPOLOGY_SPEC.loader.exec_module(route_topology)


def _make_trace(tmp_path, mode: str, *, tokens: int, layers: int = 1):
    path = tmp_path / f"{mode}.sbrt"
    records = route_topology._synthetic_records(
        mode=mode,
        tokens=tokens,
        layers=layers,
        top_k=8,
        cuda_experts=12,
        split_boundary=96,
        num_experts=180,
        seed=0xB70,
    )
    route_locality.write_trace(
        path,
        records,
        num_layers=layers,
        num_experts=180,
        top_k=8,
    )
    return route_locality.read_trace(path)


def _critical_mean(summary):
    return summary["critical_routes"]["mean"]


def test_uniform_generator_reproduces_binomial_expected_max(tmp_path):
    header, records = _make_trace(tmp_path, "uniform", tokens=50_000)
    contiguous, replicated, _baseline = route_topology.simulate_topologies(header, records)

    analytic = sum(
        math.comb(8, count) * max(count, 8 - count) for count in range(9)
    ) / 2**8
    assert analytic == 5.09375
    assert abs(_critical_mean(contiguous) - analytic) < 0.03
    assert _critical_mean(replicated) == 4.0
    assert math.isclose(replicated["mean_critical_kernel_us"], 32.6)


def test_clustered_generator_separates_topologies(tmp_path):
    header, records = _make_trace(tmp_path, "id-clustered", tokens=2_000)
    contiguous, replicated, _baseline = route_topology.simulate_topologies(header, records)

    assert _critical_mean(contiguous) == 8.0
    assert contiguous["any_card_zero_fraction"] == 1.0
    assert _critical_mean(replicated) == 4.0
    assert replicated["any_card_zero_fraction"] == 0.0


def test_single_row_selection_excludes_multirow_forwards(tmp_path):
    path = route_locality.write_trace(
        tmp_path / "mixed.sbrt",
        [
            (0, 0, 0, range(8)),
            (0, 0, 1, range(8, 16)),
            (1, 0, 0, range(16, 24)),
            (0, 1, 0, range(24, 32)),
            (0, 1, 1, range(32, 40)),
            (1, 1, 0, range(40, 48)),
        ],
        num_layers=2,
        num_experts=180,
        top_k=8,
    )
    _header, records = route_locality.read_trace(path)
    selected, kept_groups, excluded_groups = route_topology.select_single_row_steps(records)

    assert kept_groups == 2
    assert excluded_groups == 2
    assert len(selected) == 2
    assert np.all(selected["step"] == 1)


def test_last_step_runs_select_request_tail(tmp_path):
    path = route_locality.write_trace(
        tmp_path / "runs.sbrt",
        (
            (step, 0, 0, range(8))
            for step in (0, 1, 3, 4, 7)
        ),
        num_layers=1,
        num_experts=180,
        top_k=8,
    )
    _header, records = route_locality.read_trace(path)
    selected, bounds = route_topology.select_last_step_runs(records, 2)

    assert bounds == [(3, 4), (7, 7)]
    assert list(map(int, selected["step"])) == [3, 4, 7]


def test_boundary_sweep_marks_balanced_uniform_cut(tmp_path):
    header, records = _make_trace(tmp_path, "uniform", tokens=5_000)
    sweep = route_topology.sweep_boundaries(header, records, cuda_experts=12)
    selected = next(row for row in sweep if row["boundary"] == 96)

    assert selected["a_experts"] == 84
    assert selected["b_experts"] == 84
    assert abs(float(selected["mean_max"]) - 5.09375) < 0.08
    assert len(sweep) == 167
