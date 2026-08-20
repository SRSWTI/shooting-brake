#!/usr/bin/env python3
"""Dual-B70 standalone doorbell gate — the real-silicon test before any boot.

Exercises the exact production shape on BOTH cards at once, with no vLLM:

  1. Two providers, two SYCL contexts in one process, selected by PCI BDF.
  2. Two native pollers, one pinned core each.
  3. One CUDA graph that rings BOTH doorbells before waiting on EITHER —
     the production issue-all-then-take-all shape.
  4. Independent per-card partials summed and checked against a CPU
     oracle, per-card references added.
  5. Sentinel cross-card isolation: a fixture that routes ONLY to card 0
     must leave card 1's output identically zero, and vice versa.
  6. 200 alternating replays — three replays cannot catch a flag-ordering
     race; this loop can.
  7. Concurrent-dispatch latency: dual-graph replay time vs each solo
     graph, on one host clock.

Two bank formats, auto-detected by magic:

  * SBINT401 (int4): one bank file PER CARD, each holding exactly that
    card's experts (the 122B dev0/dev1 split banks). Slot maps derive
    from each bank's source-ID header; the CPU oracle dequantizes GPTQ
    int4.
  * SBEXP001 (NVFP4): ONE monolithic bank holding every expert; each
    card takes a disjoint RESIDENT LIST at load (--bank0 == --bank1).
    This is the 99B shape — one extraction, no per-split rebuilds. The
    CPU oracle dequantizes through vLLM's own ``dequantize_to_dtype``
    (global scale applied explicitly as the bank's stored multiplier, so
    the multiplier-vs-divisor convention cannot silently flip).

Run (from repo root, venv active):

  int4 (122B split banks):
    .venv/bin/python experiments/b70_dual_card_smoke.py \
        --bank0 src/phase1/expert_bank_int4_122b_dev0.bin \
        --bank1 src/phase1/expert_bank_int4_122b_dev1.bin

  nvfp4 (99B monolithic bank, resident split 103/102):
    .venv/bin/python experiments/b70_dual_card_smoke.py \
        --bank0 src/phase1/expert_bank_99b.bin \
        --bank1 src/phase1/expert_bank_99b.bin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "phase4" / "src"))
sys.path.insert(0, str(ROOT / "src" / "phase4"))
sys.path.insert(0, str(ROOT / "src" / "phase7"))

from shooting_brake_vllm.b70_binding import B70ProviderClient  # noqa: E402
from shooting_brake_vllm.b70_poller import B70Poller  # noqa: E402
from shooting_brake_vllm.expert_bank import (  # noqa: E402
    ExpertBank,
    Int4ExpertBank,
)
from shooting_brake_vllm.stream_signal import (  # noqa: E402
    alloc_host_mapped_flag,
    wait_flag,
    write_flag,
)
from int4_aggregation_oracle import _cpu_b70_partial  # noqa: E402

TOPK = 8


def detect_format(path: Path) -> str:
    with path.open("rb") as f:
        magic = f.read(8)
    if magic == b"SBINT401":
        return "int4"
    if magic == b"SBEXP001":
        return "nvfp4"
    raise SystemExit(f"{path}: unsupported bank magic {magic!r}")


# --- NVFP4 CPU oracle ------------------------------------------------------

def _dequant_nvfp4(packed: np.ndarray, sf: np.ndarray, inv_global: float,
                   rows: int, cols: int) -> torch.Tensor:
    """fp32 weight via vLLM's own dequant, global applied EXPLICITLY.

    ``dequantize_to_dtype`` is called with global scale 1.0 (pure
    e2m1 x e4m3), then the bank's stored reciprocal multiplier is applied
    by hand — the one convention the extractor documents
    (weight = e2m1 * block_scale * (1/global)). This sidesteps the
    quantizer-vs-dequantizer fold-direction trap entirely.
    """
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
        dequantize_to_dtype,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.from_numpy(np.ascontiguousarray(packed)).to(dev)
    s = torch.from_numpy(np.ascontiguousarray(sf)).to(dev)
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    raw = dequantize_to_dtype(
        q, s, one, torch.float32, block_size=16, swizzle=False,
    ).reshape(rows, cols).cpu()
    return raw * inv_global


def _cpu_nvfp4_partial(
    bank: ExpertBank,
    resident: frozenset[int],
    layer: int,
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted NVFP4 expert partial for the routes this card owns."""
    x = x.float()
    weights = weights.float()
    result = torch.zeros(x.shape[0], bank.hidden, dtype=torch.float32)
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    inter, hidden = bank.intermediate, bank.hidden
    for row in range(ids.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[row, route])
            if expert not in resident:
                continue
            if expert not in cache:
                p = bank.expert(layer, expert)
                cache[expert] = (
                    _dequant_nvfp4(p.gate, p.gate_sf, p.w13_inv_global,
                                   inter, hidden),
                    _dequant_nvfp4(p.up, p.up_sf, p.w13_inv_global,
                                   inter, hidden),
                    _dequant_nvfp4(p.down, p.down_sf, p.w2_inv_global,
                                   hidden, inter),
                )
            gate, up, down = cache[expert]
            xr = x[row]
            act = torch.nn.functional.silu(gate @ xr) * (up @ xr)
            result[row].add_(down @ act, alpha=float(weights[row, route]))
    return result


class Lane:
    """One card's complete doorbell lane, standalone-script edition."""

    def __init__(self, name: str, bank_path: Path, bdf: str, pin_cpu: int,
                 lib_path: str, fmt: str,
                 resident: tuple[int, ...] | None) -> None:
        self.name = name
        self.fmt = fmt
        if fmt == "int4":
            self.bank = Int4ExpertBank(bank_path)
            self.source_ids = tuple(self.bank.source_expert_ids)
            load_resident = None  # SBINT401 adopts the bank-defined set
        else:
            self.bank = ExpertBank(bank_path)
            assert resident is not None
            self.source_ids = resident
            load_resident = np.array(resident, dtype=np.int32)
        self.hidden = self.bank.hidden
        if (self.hidden, self.bank.intermediate) != (3072, 1024):
            raise RuntimeError(
                f"{name}: unexpected bank geometry "
                f"{self.hidden}/{self.bank.intermediate}"
            )
        self.resident = frozenset(self.source_ids)
        # global id -> this card's compact slot (resident-list order).
        self.slot_of = {expert: slot for slot, expert in
                        enumerate(self.source_ids)}

        print(f"[{name}] loading {bank_path.name} "
              f"({len(self.source_ids)} experts/layer, {fmt}) on {bdf} ...",
              flush=True)
        t0 = time.perf_counter()
        self.provider = B70ProviderClient(lib_path)
        self.provider.load(bank_path, top_k=TOPK, generation=1,
                           resident_experts=load_resident,
                           device_selector=bdf)
        print(f"[{name}] loaded in {time.perf_counter() - t0:.1f}s",
              flush=True)

        self.pinned_hidden = torch.empty(
            1, self.hidden, dtype=torch.float16, pin_memory=True)
        self.pinned_ids = torch.full(
            (1, TOPK), -1, dtype=torch.int32, pin_memory=True)
        self.pinned_weights = torch.zeros(
            1, TOPK, dtype=torch.float32, pin_memory=True)
        self.pinned_output = torch.empty(
            1, self.hidden, dtype=torch.float32, pin_memory=True)
        self.dev_fp32 = torch.empty(1, self.hidden, dtype=torch.float32,
                                    device="cuda")
        self.signal_host, self.signal_dev = alloc_host_mapped_flag(0)
        self.completion_host, self.completion_dev = alloc_host_mapped_flag(0)

        self.poller = B70Poller(self.provider, pin_cpu=pin_cpu)
        self.poller.register_layer(
            layer_idx=0,
            signal_host=self.signal_host,
            completion_host=self.completion_host,
            pinned_hidden=self.pinned_hidden,
            pinned_ids=self.pinned_ids,
            pinned_weights=self.pinned_weights,
            pinned_output=self.pinned_output,
        )
        self.poller.start()

    def cpu_partial(self, layer: int, x: torch.Tensor, gids: torch.Tensor,
                    weights: torch.Tensor) -> torch.Tensor:
        if self.fmt == "int4":
            return _cpu_b70_partial(self.bank, layer, x.to(torch.float16),
                                    gids, weights)
        return _cpu_nvfp4_partial(self.bank, self.resident, layer,
                                  x.to(torch.float16), gids, weights)

    def compact(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Global router ids -> this card's slots, -1 for foreign routes."""
        out = torch.full_like(global_ids, -1)
        for row in range(global_ids.shape[0]):
            for col in range(global_ids.shape[1]):
                out[row, col] = self.slot_of.get(int(global_ids[row, col]),
                                                 -1)
        return out

    def issue_graph(self, x_static: torch.Tensor, ids_static: torch.Tensor,
                    weights_static: torch.Tensor) -> None:
        """Production _b70_issue_graph shape: 3 D2H copies + signal write."""
        self.pinned_hidden[:1].copy_(x_static.to(torch.float16),
                                     non_blocking=True)
        self.pinned_ids[:1].copy_(ids_static, non_blocking=True)
        self.pinned_weights[:1].copy_(weights_static, non_blocking=True)
        write_flag(self.signal_dev, 1)

    def take_graph(self) -> None:
        """Production _b70_take_graph shape: wait + H2D copy + reset."""
        wait_flag(self.completion_dev, 1)
        self.dev_fp32[:1].copy_(self.pinned_output[:1], non_blocking=True)
        write_flag(self.completion_dev, 0)

    def close(self) -> None:
        self.poller.stop()
        self.provider.shutdown()
        self.bank.close()


def fixture(seed: int, lanes: tuple[Lane, Lane],
            routes_per_card: tuple[int, int], hidden: int) -> dict:
    """Deterministic input + routes straddling the cards as requested."""
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(
        (rng.standard_normal(hidden) * 0.5).astype(np.float32)
    ).reshape(1, -1)
    ids: list[int] = []
    n0, n1 = routes_per_card
    assert n0 + n1 == TOPK
    picks0 = rng.choice(len(lanes[0].source_ids), size=n0, replace=False)
    picks1 = rng.choice(len(lanes[1].source_ids), size=n1, replace=False)
    ids.extend(lanes[0].source_ids[i] for i in picks0)
    ids.extend(lanes[1].source_ids[i] for i in picks1)
    global_ids = torch.tensor(ids, dtype=torch.int32).reshape(1, -1)
    weights = torch.from_numpy(
        rng.uniform(0.02, 0.25, size=TOPK).astype(np.float32)
    ).reshape(1, -1)
    return {"x": x, "global_ids": global_ids, "weights": weights}


def peak_relative(reference: torch.Tensor, actual: torch.Tensor) -> float:
    denominator = float(reference.abs().max())
    if denominator == 0.0:
        return float(actual.abs().max())
    return float((actual.float() - reference.float()).abs().max()) \
        / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank0", type=Path,
                        default=ROOT / "src/phase1/expert_bank_int4_122b_dev0.bin")
    parser.add_argument("--bank1", type=Path,
                        default=ROOT / "src/phase1/expert_bank_int4_122b_dev1.bin")
    parser.add_argument("--bdf0", default="0000:15:00.0",
                        help="card for bank0 (default: the Gen4 x4 card)")
    parser.add_argument("--bdf1", default="0000:11:00.0",
                        help="card for bank1 (default: the Gen3 x4 card)")
    parser.add_argument("--pin0", type=int, default=5)
    parser.add_argument("--pin1", type=int, default=6)
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--latency-reps", type=int, default=300)
    parser.add_argument("--max-peak-relative", type=float, default=5e-5)
    parser.add_argument("--out", type=Path, default=None,
                        help="report path (default derives from format)")
    args = parser.parse_args()

    if os.environ.get("ZE_AFFINITY_MASK"):
        raise SystemExit(
            "ZE_AFFINITY_MASK is set — it would hide a card. Unset it; this "
            "gate selects devices by PCI BDF."
        )
    lib_path = os.environ.get(
        "SHOOTING_BRAKE_B70_LIB", str(ROOT / "src/phase7/libsb_b70_provider.so"))

    fmt0 = detect_format(args.bank0)
    fmt1 = detect_format(args.bank1)
    if fmt0 != fmt1:
        raise SystemExit(f"mixed bank formats: {fmt0} vs {fmt1}")
    fmt = fmt0
    if fmt == "int4":
        # The provider demands the explicit SBINT401 opt-in production sets.
        os.environ.setdefault("SHOOTING_BRAKE_B70_INT4", "1")
        resident0 = resident1 = None
    else:
        # Monolithic bank, disjoint resident halves in FractionalRemote
        # order: card0 takes the low contiguous run, card1 the rest.
        header_experts = ExpertBank(args.bank0).experts_per_layer
        half = (header_experts + 1) // 2
        resident0 = tuple(range(0, half))
        resident1 = tuple(range(half, header_experts))

    if args.out is None:
        args.out = (ROOT / "benchmarks/results/b70_gemv_audit/"
                           f"dual_card_smoke_{fmt}.json")

    torch.cuda.init()
    lane0 = Lane("dev0", args.bank0, args.bdf0, args.pin0, lib_path,
                 fmt, resident0)
    lane1 = Lane("dev1", args.bank1, args.bdf1, args.pin1, lib_path,
                 fmt, resident1)
    lanes = (lane0, lane1)
    hidden = lane0.hidden
    overlap = set(lane0.source_ids) & set(lane1.source_ids)
    if overlap:
        raise RuntimeError(f"cards overlap on {len(overlap)} experts")

    report: dict = {"config": {
        "format": fmt,
        "bank0": str(args.bank0), "bank1": str(args.bank1),
        "bdf0": args.bdf0, "bdf1": args.bdf1,
        "experts_per_card": [len(lane0.source_ids), len(lane1.source_ids)],
        "pin_cpus": [args.pin0, args.pin1],
        "cycles": args.cycles, "latency_reps": args.latency_reps,
    }}

    # Fixtures: A/B straddle both cards; C is card0-only, D is card1-only
    # (the sentinel-isolation cases).
    fixtures = {
        "A": fixture(11, lanes, (4, 4), hidden),
        "B": fixture(23, lanes, (5, 3), hidden),
        "C": fixture(37, lanes, (8, 0), hidden),
        "D": fixture(41, lanes, (0, 8), hidden),
    }
    references = {}
    for name, f in fixtures.items():
        ref0 = lane0.cpu_partial(0, f["x"], f["global_ids"], f["weights"])
        ref1 = lane1.cpu_partial(0, f["x"], f["global_ids"], f["weights"])
        references[name] = {"dev0": ref0, "dev1": ref1, "sum": ref0 + ref1}

    # Static graph tensors. Per-lane compact ids are recomputed per fixture
    # on the host and copied into static CUDA tensors between replays,
    # exactly like production's per-batch-size captured graphs.
    x_static = torch.empty(1, hidden, dtype=torch.float32, device="cuda")
    ids0_static = torch.empty(1, TOPK, dtype=torch.int32, device="cuda")
    ids1_static = torch.empty(1, TOPK, dtype=torch.int32, device="cuda")
    weights_static = torch.empty(1, TOPK, dtype=torch.float32, device="cuda")
    combined_static = torch.empty(1, hidden, dtype=torch.float32,
                                  device="cuda")

    def load_fixture(f: dict) -> None:
        x_static.copy_(f["x"])
        ids0_static.copy_(lane0.compact(f["global_ids"]))
        ids1_static.copy_(lane1.compact(f["global_ids"]))
        weights_static.copy_(f["weights"])
        torch.cuda.synchronize()

    # Warm both doorbells once, eagerly, before any capture (first dispatch
    # allocates provider-side state that must not happen mid-capture).
    load_fixture(fixtures["A"])
    for lane, ids_static in ((lane0, ids0_static), (lane1, ids1_static)):
        lane.issue_graph(x_static, ids_static, weights_static)
        lane.take_graph()
    torch.cuda.synchronize()

    # THE production shape: both signals written before either wait.
    dual_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(dual_graph):
        lane0.issue_graph(x_static, ids0_static, weights_static)
        lane1.issue_graph(x_static, ids1_static, weights_static)
        lane0.take_graph()
        lane1.take_graph()
        torch.add(lane0.dev_fp32, lane1.dev_fp32, out=combined_static)
    torch.cuda.synchronize()

    solo_graphs = {}
    for name, lane, ids_static in (("dev0", lane0, ids0_static),
                                   ("dev1", lane1, ids1_static)):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            lane.issue_graph(x_static, ids_static, weights_static)
            lane.take_graph()
        torch.cuda.synchronize()
        solo_graphs[name] = g

    # ---- correctness: A -> B -> A, then the two isolation sentinels ------
    failures = []
    correctness = {}
    for step, name in enumerate(("A", "B", "A", "C", "D")):
        f = fixtures[name]
        load_fixture(f)
        # Poison outputs so "card wrote zeros" is distinguishable from
        # "card never wrote".
        lane0.pinned_output.fill_(float("nan"))
        lane1.pinned_output.fill_(float("nan"))
        dual_graph.replay()
        torch.cuda.synchronize()

        out0 = lane0.pinned_output[:1].clone()
        out1 = lane1.pinned_output[:1].clone()
        combined = combined_static.detach().cpu().clone()
        rel0 = peak_relative(references[name]["dev0"], out0)
        rel1 = peak_relative(references[name]["dev1"], out1)
        rel_sum = peak_relative(references[name]["sum"], combined)
        entry = {"fixture": name, "peak_rel_dev0": rel0,
                 "peak_rel_dev1": rel1, "peak_rel_sum": rel_sum}
        if name == "C":
            entry["dev1_exactly_zero"] = bool((out1 == 0).all())
        if name == "D":
            entry["dev0_exactly_zero"] = bool((out0 == 0).all())
        correctness[f"step{step}_{name}"] = entry
        for key, rel in (("dev0", rel0), ("dev1", rel1), ("sum", rel_sum)):
            if rel > args.max_peak_relative or rel != rel:
                failures.append(f"step {step} {name}: {key} rel {rel:.2e}")
        if name == "C" and not entry["dev1_exactly_zero"]:
            failures.append("C: card1 output not zero — cross-card leak")
        if name == "D" and not entry["dev0_exactly_zero"]:
            failures.append("D: card0 output not zero — cross-card leak")
    report["correctness"] = correctness

    # ---- stress: alternating replays, flag-ordering race hunt ------------
    lane0.poller.reset()
    lane1.poller.reset()
    stress_names = ("A", "B")
    worst = 0.0
    for cycle in range(args.cycles):
        name = stress_names[cycle % 2]
        load_fixture(fixtures[name])
        dual_graph.replay()
        torch.cuda.synchronize()
        worst = max(worst, peak_relative(
            references[name]["sum"], combined_static.detach().cpu()))
    report["stress"] = {
        "cycles": args.cycles,
        "worst_peak_rel_sum": worst,
        "errors_dev0": lane0.poller.error_count,
        "errors_dev1": lane1.poller.error_count,
        "service_mean_us_dev0": lane0.poller.service_mean_us,
        "service_mean_us_dev1": lane1.poller.service_mean_us,
    }
    if worst > args.max_peak_relative:
        failures.append(f"stress worst rel {worst:.2e}")
    if lane0.poller.error_count or lane1.poller.error_count:
        failures.append("poller errors nonzero")

    # ---- concurrent-dispatch latency, one host clock ----------------------
    load_fixture(fixtures["A"])
    timing = {}
    for label, graph in (("dual", dual_graph),
                         ("solo_dev0", solo_graphs["dev0"]),
                         ("solo_dev1", solo_graphs["dev1"])):
        for _ in range(20):  # warm
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.latency_reps):
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        arr = np.array(samples)
        timing[label] = {
            "mean_us": float(arr.mean()),
            "p50_us": float(np.percentile(arr, 50)),
            "p90_us": float(np.percentile(arr, 90)),
            "p99_us": float(np.percentile(arr, 99)),
        }
    t_dual = timing["dual"]["mean_us"]
    t0_solo = timing["solo_dev0"]["mean_us"]
    t1_solo = timing["solo_dev1"]["mean_us"]
    timing["serialization"] = {
        # 1.0 => perfectly parallel (dual == max of solos);
        # ~2x max => fully serialized.
        "dual_over_max_solo": t_dual / max(t0_solo, t1_solo),
        "dual_over_sum_solo": t_dual / (t0_solo + t1_solo),
    }
    report["latency"] = timing

    report["verdict"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n{report['verdict']} -> {args.out}")

    lane0.close()
    lane1.close()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
