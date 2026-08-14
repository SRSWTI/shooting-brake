"""Per-(layer, expert) routing frequency counters.

Placement in this project has always been positional: ``LayerSubsetPolicy``
keeps experts ``0..cuda_per_layer-1`` on CUDA and sends the rest to the B70,
which is an ordering by expert *index* and not by expert *use*. The README
argues the opposite -- that a small set of hot experts handles most tokens and
the rest idle -- but nothing in the codebase has ever counted, so the claim has
never been checked against this model. This module counts.

Design constraints, in the order they bind:

* **Zero cost when off.** The counter is only constructed when
  ``SHOOTING_BRAKE_ROUTE_STATS=1``; :func:`get_counter` returns ``None``
  otherwise and the call site skips entirely. No tensor is allocated, no
  branch is taken on the hot path beyond one ``is None`` check.
* **CUDA-graph safe.** Accumulation is a device-side ``scatter_add_`` into a
  statically shaped tensor. It reads no value back, so it introduces no host
  sync and nothing that invalidates capture. This matters because the counters
  are most useful on the graph-captured decode path.
* **Placement-independent.** Routing is decided by the router on the 5090
  before anything is dispatched anywhere, so the histogram does not depend on
  where experts live. One calibration run in all-CUDA mode -- the fastest
  configuration, and the one with no B70 round trips -- produces a profile
  valid for every placement and every budget. That is why the hook sits above
  the all-CUDA passthrough rather than inside the hybrid branch.

The analysis follows the StreamMoE-style summary used by the reference
implementation in ``experiments--/misc/lucebox``
(``server/src/common/moe_hybrid_routing_stats.cpp``): per layer, how many
distinct experts fired, the cumulative share held by the top 10/30/60, and how
many experts are needed to cover 50/80/90% of activations. ``n80`` is the
number that matters for placement -- it is the per-layer hot set size that a
budget has to hold to capture most of the traffic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_ENV = "SHOOTING_BRAKE_ROUTE_STATS"
_OUT_ENV = "SHOOTING_BRAKE_ROUTE_STATS_OUT"


class RouteCounter:
    """Device-resident ``[n_layer, n_expert]`` activation counts.

    One instance is shared by every layer; :meth:`observe` is called with the
    global (pre-remap) ``topk_ids`` so the table describes the model's routing
    rather than any particular placement's view of it.
    """

    def __init__(self, num_layers: int, num_experts: int, device: torch.device):
        self.num_layers = num_layers
        self.num_experts = num_experts
        # int64: a long calibration run at 8 top-k over millions of tokens
        # will overflow int32 per-expert on a hot expert.
        #
        # inference_mode(False): construction happens lazily on the first
        # forward, which vLLM runs under torch.inference_mode(). A tensor
        # created there is an "inference tensor", and reset_worker_stats --
        # a collective_rpc handler running between steps, outside inference
        # mode -- cannot zero_() one. Allocating with inference mode
        # suspended yields normal tensors, writable from both contexts.
        with torch.inference_mode(False):
            self.counts = torch.zeros(
                num_layers, num_experts, dtype=torch.int64, device=device,
            )
            self.tokens = torch.zeros(
                num_layers, dtype=torch.int64, device=device,
            )
        self.top_k = 0

    def observe(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Accumulate one forward's routes for ``layer_idx``.

        ``topk_ids`` is ``[num_tokens, top_k]`` of global expert ids. Both
        updates are device-side and shape-static, so this is safe to call
        under CUDA graph capture.
        """
        if layer_idx >= self.num_layers:
            return
        if not self.top_k:
            self.top_k = int(topk_ids.shape[1])
        flat = topk_ids.reshape(-1).to(torch.int64)
        self.counts[layer_idx].scatter_add_(
            0, flat, torch.ones_like(flat),
        )
        self.tokens[layer_idx] += topk_ids.shape[0]

    # -- readout ---------------------------------------------------------
    # Everything below syncs, and is called once at the end of calibration.

    def to_cpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.counts.cpu(), self.tokens.cpu()

    def save_csv(self, path: str | Path) -> Path:
        counts, tokens = self.to_cpu()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# route histogram: n_layer={self.num_layers} "
            f"n_expert={self.num_experts} top_k={self.top_k}",
            "layer,expert,count",
        ]
        for il in range(self.num_layers):
            row = counts[il]
            if not int(row.sum()):
                continue
            for ie in range(self.num_experts):
                c = int(row[ie])
                if c:
                    lines.append(f"{il},{ie},{c}")
        out.write_text("\n".join(lines) + "\n")
        return out


def load_csv(path: str | Path) -> tuple[torch.Tensor, int]:
    """Read a histogram written by :meth:`RouteCounter.save_csv`.

    Returns the ``[n_layer, n_expert]`` count table and ``top_k``.
    """
    text = Path(path).read_text().splitlines()
    n_layer = n_expert = top_k = 0
    for line in text:
        if line.startswith("#"):
            for tok in line.replace("#", " ").split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    if k == "n_layer":
                        n_layer = int(v)
                    elif k == "n_expert":
                        n_expert = int(v)
                    elif k == "top_k":
                        top_k = int(v)
            break
    if not (n_layer and n_expert):
        raise ValueError(f"{path}: missing or malformed header")
    counts = torch.zeros(n_layer, n_expert, dtype=torch.int64)
    for line in text:
        if line.startswith("#") or line.startswith("layer,"):
            continue
        if not line.strip():
            continue
        il, ie, c = line.split(",")
        counts[int(il), int(ie)] = int(c)
    return counts, top_k


def analyze(counts: torch.Tensor, top_k: int) -> dict:
    """StreamMoE-style skew summary.

    ``n50``/``n80``/``n90`` are the number of experts needed to cover that
    percentage of a layer's activations. Against a uniform router they would
    sit at 50/80/90% of ``n_expert``; well below that is the skew the README
    assumes and the knapsack would exploit.
    """
    n_layer, n_expert = counts.shape
    layers = []
    for il in range(n_layer):
        row = counts[il]
        total = int(row.sum())
        if not total:
            layers.append(None)
            continue
        srt, _ = torch.sort(row, descending=True)
        cum = torch.cumsum(srt, 0)
        unique = int((srt > 0).sum())

        def cover(pct: int) -> int:
            need = (total * pct + 99) // 100
            idx = int(torch.searchsorted(cum, torch.tensor(need)))
            return min(idx + 1, n_expert)

        def top(n: int) -> float:
            n = min(n, n_expert)
            return 100.0 * float(cum[n - 1]) / total

        layers.append({
            "layer": il,
            "unique": unique,
            "total": total,
            "top10_pct": top(10),
            "top30_pct": top(30),
            "top60_pct": top(60),
            "n50": cover(50),
            "n80": cover(80),
            "n90": cover(90),
        })

    live = [row for row in layers if row]
    summary = {
        "n_layer": n_layer,
        "n_expert": n_expert,
        "top_k": top_k,
        "active_layers": len(live),
        "layers": layers,
    }
    if live:
        summary["mean_n80"] = sum(r["n80"] for r in live) / len(live)
        summary["mean_top10_pct"] = sum(r["top10_pct"] for r in live) / len(live)
        # The honest null hypothesis: a uniform router needs 80% of experts to
        # cover 80% of traffic. Skew is how far below that the real n80 sits.
        summary["uniform_n80"] = 0.8 * n_expert
        summary["skew_ratio"] = summary["uniform_n80"] / summary["mean_n80"]
    return summary


def format_analysis(summary: dict) -> str:
    lines = [
        "=== Expert frequency analysis ===",
        f"{summary['n_layer']} layers x {summary['n_expert']} experts, "
        f"top-{summary['top_k']} routing "
        f"({summary['active_layers']} layers observed)",
        "",
        f"{'layer':>5} {'unique':>7} {'top10':>7} {'top30':>7} {'top60':>7} "
        f"{'n50':>5} {'n80':>5} {'n90':>5}",
    ]
    for row in summary["layers"]:
        if not row:
            continue
        lines.append(
            f"{row['layer']:>5} {row['unique']:>7} "
            f"{row['top10_pct']:>6.1f}% {row['top30_pct']:>6.1f}% "
            f"{row['top60_pct']:>6.1f}% "
            f"{row['n50']:>5} {row['n80']:>5} {row['n90']:>5}"
        )
    if "mean_n80" in summary:
        lines += [
            "",
            "--- summary ---",
            f"experts covering 80% of routes: {summary['mean_n80']:.1f} per layer "
            f"(uniform router would need {summary['uniform_n80']:.0f})",
            f"skew ratio: {summary['skew_ratio']:.2f}x "
            f"(1.0 = no skew, higher = hotter tail)",
            f"top 10 experts hold {summary['mean_top10_pct']:.1f}% of routes "
            f"(uniform: {100.0 * 10 / summary['n_expert']:.1f}%)",
        ]
    return "\n".join(lines)


# -- process-wide singleton ----------------------------------------------

_counter: RouteCounter | None = None
_checked = False


def enabled() -> bool:
    return os.environ.get(_ENV) == "1"


def get_counter(
    num_layers: int, num_experts: int, device: torch.device,
) -> RouteCounter | None:
    """The shared counter, or ``None`` when the flag is off.

    Construction is lazy so that an off run allocates nothing at all.
    """
    global _counter, _checked
    if not enabled():
        return None
    if _counter is None:
        _counter = RouteCounter(num_layers, num_experts, device)
        if not _checked:
            _checked = True
            logger.warning(
                "Shooting Brake: route statistics enabled (%d layers x %d "
                "experts). This adds a device-side scatter_add per layer.",
                num_layers, num_experts,
            )
    return _counter


def peek() -> RouteCounter | None:
    """The counter if one exists, without creating it."""
    return _counter


def dump(path: str | Path | None = None) -> Path | None:
    """Write the histogram, defaulting to ``SHOOTING_BRAKE_ROUTE_STATS_OUT``."""
    if _counter is None:
        return None
    target = path or os.environ.get(_OUT_ENV) or "route_stats.csv"
    out = _counter.save_csv(target)
    logger.warning("Shooting Brake: route histogram written to %s", out)
    return out
