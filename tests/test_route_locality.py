from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_ANALYZER = Path(__file__).parents[1] / "benchmarks" / "route_locality.py"
_SPEC = importlib.util.spec_from_file_location("route_locality", _ANALYZER)
assert _SPEC is not None and _SPEC.loader is not None
route_locality = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = route_locality
_SPEC.loader.exec_module(route_locality)


NUM_LAYERS = 2
NUM_EXPERTS = 256
TOP_K = 8
ROWS = 4
STEPS = 1_500


def _high_reuse_records():
    for step in range(STEPS):
        for layer in range(NUM_LAYERS):
            experts = tuple(range(layer * TOP_K, (layer + 1) * TOP_K))
            for row in range(ROWS):
                yield step, layer, row, experts


def _uniform_records(seed: int):
    rng = random.Random(seed)
    population = range(NUM_EXPERTS)
    for step in range(STEPS):
        for layer in range(NUM_LAYERS):
            for row in range(ROWS):
                yield step, layer, row, rng.sample(population, TOP_K)


def _analyze(path):
    header, records = route_locality.read_trace(path)
    return route_locality.analyze_trace(header, records)


def test_analyzer_separates_reuse_from_uniform_chance(tmp_path):
    high_path = route_locality.write_trace(
        tmp_path / "high.sbrt",
        _high_reuse_records(),
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    )
    random_path = route_locality.write_trace(
        tmp_path / "random.sbrt",
        _uniform_records(0xB70),
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    )

    high = _analyze(high_path)
    uniform = _analyze(random_path)
    chance = uniform["baseline"]

    assert high["aggregate"]["jaccard"] > 0.99
    assert high["aggregate"]["jaccard"] > uniform["aggregate"]["jaccard"] + 0.95
    assert abs(uniform["aggregate"]["jaccard"] - chance["jaccard"]) < 0.006

    for window in route_locality.WINDOWS:
        random_working_set = uniform["aggregate"]["working_set"][window]
        assert abs(random_working_set["mean_distinct"] - chance["working_set"][window]) < 1.0
        high_working_set = high["aggregate"]["working_set"][window]
        assert high_working_set["mean_distinct"] == TOP_K

    for capacity in route_locality.CACHE_CAPACITIES:
        random_lru = uniform["aggregate"]["lru"][capacity]
        assert abs(random_lru["hit_rate"] - chance["lru_hit_rate"][capacity]) < 0.02
        assert random_lru["round_trips_saved"] == random_lru["hits"]
        assert random_lru["expert_payloads_saved"] == random_lru["hits"]

    assert high["aggregate"]["lru"][8]["hit_rate"] > 0.99
    report = route_locality.format_report(uniform)
    assert "Consecutive-token top-k Jaccard" in report
    assert "Distinct-expert working set" in report
    assert "LRU expert-weight cache" in report
