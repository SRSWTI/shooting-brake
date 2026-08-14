#!/usr/bin/env python3
"""Capture and analyze RTX 5090 decode profiles for Shooting Brake.

This tool deliberately keeps the two timing questions separate:

* ``measure`` drives an ordinary live server with profiling disabled.  Run it
  once against the real B70 provider and once against the immediate-zero dummy
  provider.  ``compare`` reports the end-to-end critical-path bound.
* ``capture`` brackets a short request with vLLM's opt-in torch-profiler HTTP
  endpoints.  ``analyze`` attributes CUDA *execution* events.  Stream waits and
  idle gaps are reported separately and never charged as 5090 compute.
* ``roofline`` prints the compulsory-byte arithmetic known before a trace.

The live server must have been launched with vLLM's profiler config for
``capture``; ``measure`` requires no profiler endpoint.  See
``benchmarks/CUDA_DECODE_PROFILE.md`` for the exact run order and controls.
No command in this file starts a model server or selects a GPU implicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "srswti/axe-superveloce-88b-nvfp4a16"
DEFAULT_PROMPT = (
    "Measure the decode critical path. Return a deterministic sequence of "
    "short words and do not stop until the requested token limit."
)
BANDWIDTH_GB_S = 1792.0
LAYERS = 48
GDN_LAYERS = 36
FULL_ATTENTION_LAYERS = 12
HIDDEN = 3072
VOCAB = 248_320
TOP_K = 8
EXPERTS = 180
CUDA_EXPERTS = 54
MOE_INTERMEDIATE = 1024

# CUDA driver/runtime calls which enqueue synchronization rather than execute
# arithmetic. They are intentionally not CUDA-compute categories.
_WAIT_RE = re.compile(
    r"(?:cuStreamWaitValue|cudaStreamWait|cudaEventSynchronize|"
    r"cudaStreamSynchronize|cudaDeviceSynchronize)", re.IGNORECASE
)
_WRITE_SIGNAL_RE = re.compile(r"cuStreamWriteValue", re.IGNORECASE)

_CATEGORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lm_head", re.compile(r"(?:lm[_ .-]?head|vocab_parallel|logits_processor)", re.I)),
    ("sampling", re.compile(r"(?:sampl|multinomial|top[_ -]?[kp]|argmax|rejection)", re.I)),
    ("moe_marlin", re.compile(r"(?:marlin.*moe|moe.*marlin|fused[_ .-]?moe|grouped.*gemm)", re.I)),
    ("gdn_attention", re.compile(r"(?:gated[_ .-]?delta|deltanet|delta[_ .-]?rule|causal[_ .-]?conv|fla[_ .-])", re.I)),
    ("full_attention", re.compile(r"(?:flashinfer|flash[_ .-]?attn|paged[_ .-]?attention|batch[_ .-]?decode|attention[_ .-]?kernel)", re.I)),
    ("norms", re.compile(r"(?:rms[_ .-]?norm|layer[_ .-]?norm)", re.I)),
    ("router", re.compile(r"(?:router|routing|moe[_ .-]?align|topk[_ .-]?softmax)", re.I)),
    ("residual_combine", re.compile(r"(?:residual|combine[_ .-]?partial)", re.I)),
    ("memory_transfer", re.compile(r"(?:memcpy|memset)", re.I)),
)

_APPLICABLE_LAYERS = {
    "gdn_attention": GDN_LAYERS,
    "full_attention": FULL_ATTENTION_LAYERS,
    "qkv_o_projections": LAYERS,
    "moe_marlin": LAYERS,
    "norms": LAYERS,
    "router": LAYERS,
    "residual_combine": LAYERS,
    "memory_transfer": LAYERS,
    "other": LAYERS,
    "unattributed_gemm": LAYERS,
    "lm_head": 1,
    "sampling": 1,
}


@dataclass(frozen=True)
class RooflineRow:
    category: str
    bytes_per_token: float
    minimum_us: float
    basis: str
    confidence: str


@dataclass(frozen=True)
class DeviceEvent:
    name: str
    category: str
    duration_us: float
    timestamp_us: float
    pid: int | str | None
    tid: int | str | None


def _minimum_us(byte_count: float, bandwidth_gb_s: float) -> float:
    # GB/s is decimal, matching NVIDIA's advertised 1792 GB/s.
    return byte_count / (bandwidth_gb_s * 1e9) * 1e6


def roofline_rows(
    *,
    context_tokens: int = 256,
    cuda_routes_per_layer: float = TOP_K * CUDA_EXPERTS / EXPERTS,
    bandwidth_gb_s: float = BANDWIDTH_GB_S,
) -> list[RooflineRow]:
    """Return compulsory logical-byte lower bounds for one M=1 token.

    These are not claimed DRAM transactions. They count checkpoint bytes and
    persistent state/cache traffic whose geometry is known. Cache hits,
    swizzled runtime padding, fusion, intermediate spills, and write allocation
    can move the actual HBM traffic in either direction; Nsight Compute is the
    tool that measures that traffic.
    """
    if context_tokens < 0:
        raise ValueError("context_tokens must be non-negative")
    if not 0.0 <= cuda_routes_per_layer <= TOP_K:
        raise ValueError(f"cuda_routes_per_layer must be in [0, {TOP_K}]")
    if bandwidth_gb_s <= 0:
        raise ValueError("bandwidth_gb_s must be positive")

    # GDN checkpoint bytes per layer. FP8 matrices: qkv [12288,3072],
    # z [8192,3072], out [3072,8192]. BF16 companions: a and b [64,3072],
    # conv [12288,1,4], A_log [64], dt_bias [64], and norm [128]. This is
    # 2.983 GiB/token across 36 layers and reproduces the source checkpoint.
    gdn_fp8 = (12_288 * HIDDEN) + (8_192 * HIDDEN) + (HIDDEN * 8_192)
    gdn_bf16 = 2 * ((64 * HIDDEN) + (12_288 * 4) + 64 + 128)
    gdn_projection_bytes = GDN_LAYERS * (gdn_fp8 + gdn_bf16)

    # Full attention FP8 q/k/v/o matrices. q includes the output gate:
    # [16384,3072], k/v [512,3072], o [3072,8192].
    full_projection_per_layer = (
        (16_384 * HIDDEN) + 2 * (512 * HIDDEN) + (HIDDEN * 8_192)
    )
    full_projection_bytes = FULL_ATTENTION_LAYERS * full_projection_per_layer

    # Persistent GDN state, compulsory per decode step if the kernel updates the
    # full state: SSM [64,128,128] fp32 read+write; conv state conservatively
    # counts the three retained [12288] BF16 rows read+write.
    ssm_state_rw = 2 * 64 * 128 * 128 * 4
    conv_state_rw = 2 * 3 * 12_288 * 2
    gdn_state_bytes = GDN_LAYERS * (ssm_state_rw + conv_state_rw)

    # FP8 KV: K+V = 2 heads * 256 dims * 2 tensors = 1024 B/context token.
    # Read the existing context and write the new entry in each full layer.
    kv_bytes = FULL_ATTENTION_LAYERS * (context_tokens + 1) * 1_024

    router_bytes = LAYERS * EXPERTS * HIDDEN * 2  # BF16 gate weights.

    # One NVFP4 expert: packed gate/up/down weights plus E4M3 block scales and
    # three fp32 globals, taken from the checkpoint shapes. Include the shared
    # expert (always on CUDA) and the selected CUDA-routed experts, not all 54
    # resident experts. Top-k IDs are unique within a row.
    packed_per_expert = 3 * (HIDDEN * MOE_INTERMEDIATE // 2)
    scales_per_expert = 3 * (HIDDEN * MOE_INTERMEDIATE // 16)
    globals_per_expert = 3 * 4
    expert_bytes = packed_per_expert + scales_per_expert + globals_per_expert
    moe_bytes = LAYERS * (1.0 + cuda_routes_per_layer) * expert_bytes

    # Each RMSNorm logically reads input + BF16 weight and writes output. Two
    # per layer plus the final norm. This excludes a fused residual read whose
    # actual traffic must come from hardware counters.
    norm_bytes = (2 * LAYERS + 1) * (HIDDEN * 2 * 3)

    lm_head_bytes = (VOCAB * (HIDDEN // 2)) + (VOCAB * (HIDDEN // 16)) + 4
    sampling_bytes = VOCAB * 4  # conservative fp32-logit read for greedy M=1.

    specs = (
        ("gdn_projections", gdn_projection_bytes,
         "36 layers: FP8 qkv/z/out checkpoint matrices plus BF16 a/b/conv/state parameters", "exact checkpoint geometry"),
        ("full_attention_projections", full_projection_bytes,
         "12 layers: FP8 q/k/v/o checkpoint matrices", "exact checkpoint geometry"),
        ("gdn_recurrent_state", gdn_state_bytes,
         "SSM fp32 and retained convolution-state logical read+write", "logical lower bound"),
        ("full_attention_kv", kv_bytes,
         f"12 layers: FP8 K+V for {context_tokens} cached tokens plus one write", "context-parameterized lower bound"),
        ("router", router_bytes,
         "48 BF16 [180,3072] gate matrices", "exact checkpoint geometry"),
        ("cuda_moe_marlin", moe_bytes,
         f"shared expert + {cuda_routes_per_layer:.3f} selected CUDA experts/layer; packed NVFP4+scales", "route-count parameterized lower bound"),
        ("norms", norm_bytes,
         "two per layer plus final; input+weight+output logical bytes", "logical lower bound"),
        ("lm_head", lm_head_bytes,
         "NVFP4 packed [248320,3072] plus E4M3 scales and fp32 global", "exact checkpoint geometry"),
        ("sampling", sampling_bytes,
         "one fp32 read of 248320 logits", "implementation-dependent lower bound"),
    )
    return [
        RooflineRow(name, float(byte_count), _minimum_us(byte_count, bandwidth_gb_s), basis, confidence)
        for name, byte_count, basis, confidence in specs
    ]


def _http_json(method: str, url: str, body: dict[str, Any] | None, timeout: float) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, method=method, data=data)
    request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"{method} {url}: HTTP {response.status}")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url}: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url}: {exc.reason}") from exc
    if not payload:
        return None
    return json.loads(payload)


def _parse_sse(lines: Iterable[bytes]) -> tuple[list[float], dict[str, int]]:
    arrivals: list[float] = []
    usage: dict[str, int] = {}
    for raw in lines:
        line = raw.decode("utf-8", errors="strict").strip()
        if not line.startswith("data: "):
            continue
        value = line[6:]
        if value == "[DONE]":
            break
        chunk = json.loads(value)
        if chunk.get("usage"):
            usage.update({k: int(v) for k, v in chunk["usage"].items() if v is not None})
        choices = chunk.get("choices") or []
        if choices and choices[0].get("text"):
            arrivals.append(time.perf_counter())
    return arrivals, usage


def completion_request(
    *, target: str, model: str, prompt: str, tokens: int, timeout: float
) -> dict[str, Any]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = target.rstrip("/") + "/v1/completions"
    request = Request(url, method="POST", data=json.dumps(body).encode("utf-8"))
    request.add_header("Content-Type", "application/json")
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            arrivals, usage = _parse_sse(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url}: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {url}: {exc.reason}") from exc
    end = time.perf_counter()
    if len(arrivals) != tokens:
        raise RuntimeError(
            f"server emitted {len(arrivals)} streamed token chunks, expected exactly {tokens}; "
            "ignore_eos may be unsupported or chunks may aggregate tokens"
        )
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None and completion_tokens != tokens:
        raise RuntimeError(
            f"usage.completion_tokens={completion_tokens}, expected {tokens}"
        )
    itls = [(b - a) * 1e6 for a, b in zip(arrivals, arrivals[1:])]
    return {
        "request_start_unix_ns": time.time_ns() - int((end - start) * 1e9),
        "request_end_unix_ns": time.time_ns(),
        "wall_us": (end - start) * 1e6,
        "ttft_us": (arrivals[0] - start) * 1e6,
        "decode_intervals_us": itls,
        "decode_interval_mean_us": statistics.fmean(itls) if itls else None,
        "decode_interval_median_us": statistics.median(itls) if itls else None,
        "decode_interval_min_us": min(itls) if itls else None,
        "decode_interval_max_us": max(itls) if itls else None,
        "streamed_chunks": len(arrivals),
        "usage": usage,
    }


def _trace_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {
        p.resolve() for p in root.rglob("*")
        if p.is_file() and (p.name.endswith(".pt.trace.json") or p.name.endswith(".pt.trace.json.gz"))
    }


def run_measure(args: argparse.Namespace, *, use_profiler: bool) -> dict[str, Any]:
    trace_dir = Path(args.trace_dir).resolve() if use_profiler else None
    before = _trace_files(trace_dir) if trace_dir else set()

    warmup = completion_request(
        target=args.target, model=args.model, prompt=args.prompt,
        tokens=args.warmup_tokens, timeout=args.timeout,
    )
    profile_started = False
    try:
        if use_profiler:
            _http_json("POST", args.target.rstrip("/") + "/start_profile", None, args.timeout)
            profile_started = True
        measured = completion_request(
            target=args.target, model=args.model, prompt=args.prompt,
            tokens=args.decode_steps, timeout=args.timeout,
        )
    finally:
        if profile_started:
            _http_json("POST", args.target.rstrip("/") + "/stop_profile", None, args.timeout)

    new_traces: list[str] = []
    if trace_dir:
        deadline = time.monotonic() + args.trace_wait
        while time.monotonic() < deadline:
            current = _trace_files(trace_dir)
            if current - before:
                new_traces = sorted(str(path) for path in current - before)
                break
            time.sleep(0.25)
        if not new_traces:
            raise RuntimeError(
                f"profiler stopped but no new .pt.trace.json[.gz] appeared under {trace_dir} "
                f"within {args.trace_wait}s"
            )

    result = {
        "schema": 1,
        "kind": "torch_profile_capture" if use_profiler else "profile_off_measurement",
        "label": args.label,
        "target": args.target,
        "model": args.model,
        "prompt": args.prompt,
        "requested_decode_steps": args.decode_steps,
        "profiler_enabled": use_profiler,
        "warmup": warmup,
        "measured": measured,
        "trace_files": new_traces,
    }
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError(f"{path}: missing traceEvents list")
    return [event for event in events if isinstance(event, dict)]


def _external_id(event: dict[str, Any]) -> str | int | None:
    args = event.get("args") or {}
    for key in ("External id", "external id", "External Id", "correlation"):
        if key in args:
            return args[key]
    return None


def _is_device_execution(event: dict[str, Any]) -> bool:
    if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
        return False
    name = str(event.get("name", ""))
    if _WAIT_RE.search(name) or _WRITE_SIGNAL_RE.search(name):
        return False
    cat = str(event.get("cat", "")).lower()
    args = event.get("args") or {}
    device_type = str(args.get("Device Type", args.get("device_type", ""))).lower()
    # Kineto uses cat=kernel for CUDA kernels and cat=gpu_memcpy/gpu_memset
    # for device engines. CPU aten/cuda_runtime calls must not pass this gate.
    return (
        cat in {"kernel", "gpu_memcpy", "gpu_memset", "cuda_kernel"}
        or "gpu_memcpy" in cat
        or "gpu_memset" in cat
        or device_type in {"cuda", "gpu"}
    )


def _shape_numbers(value: Any) -> set[int]:
    numbers: set[int] = set()
    if isinstance(value, bool):
        return numbers
    if isinstance(value, int):
        numbers.add(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            numbers.update(_shape_numbers(item))
    return numbers


def classify_event(event: dict[str, Any], correlated_cpu: dict[str | int, list[dict[str, Any]]]) -> str:
    name = str(event.get("name", ""))
    ext = _external_id(event)
    cpu_events = correlated_cpu.get(ext, []) if ext is not None else []
    context_names = " ".join([name, *(str(item.get("name", "")) for item in cpu_events)])
    shape_values: list[Any] = []
    for item in cpu_events:
        args = item.get("args") or {}
        for key in ("Input Dims", "Concrete Inputs", "Input type", "input_shapes"):
            if key in args:
                shape_values.append(args[key])
    dims = _shape_numbers(shape_values)

    # Shape evidence outranks generic GEMM names. These dimensions are unique
    # enough to make lm_head/router attribution loud rather than heuristic.
    if VOCAB in dims:
        return "lm_head"
    if EXPERTS in dims and HIDDEN in dims:
        return "router"
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(context_names):
            return category
    if HIDDEN in dims and dims.intersection({512, 8_192, 12_288, 16_384}):
        return "qkv_o_projections"
    if re.search(r"(?:gemm|matmul|cutlass|cublas|nvfp4|fp8)", context_names, re.I):
        # Do not turn an unknown GEMM into a projection merely because that is
        # plausible. It remains explicit debt in the report.
        return "unattributed_gemm"
    return "other"


def analyze_traces(paths: Sequence[Path], decode_steps: int) -> dict[str, Any]:
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    all_events: list[dict[str, Any]] = []
    for path in paths:
        all_events.extend(_load_trace(path))

    correlated_cpu: dict[str | int, list[dict[str, Any]]] = defaultdict(list)
    wait_api: list[dict[str, Any]] = []
    for event in all_events:
        ext = _external_id(event)
        if ext is not None and not _is_device_execution(event):
            correlated_cpu[ext].append(event)
        if event.get("ph") == "X" and _WAIT_RE.search(str(event.get("name", ""))):
            wait_api.append(event)

    device: list[DeviceEvent] = []
    for event in all_events:
        if not _is_device_execution(event):
            continue
        category = classify_event(event, correlated_cpu)
        device.append(DeviceEvent(
            name=str(event.get("name", "")),
            category=category,
            duration_us=float(event["dur"]),
            timestamp_us=float(event.get("ts", 0.0)),
            pid=event.get("pid"),
            tid=event.get("tid"),
        ))

    by_category: dict[str, list[DeviceEvent]] = defaultdict(list)
    for event in device:
        by_category[event.category].append(event)
    categories: dict[str, Any] = {}
    for category in sorted(set(_APPLICABLE_LAYERS) | set(by_category)):
        events = by_category.get(category, [])
        total = sum(item.duration_us for item in events)
        applicable = _APPLICABLE_LAYERS.get(category, LAYERS)
        categories[category] = {
            "event_count": len(events),
            "device_time_total_us": total,
            "device_time_per_decode_step_us": total / decode_steps,
            "device_time_per_applicable_layer_us": total / (decode_steps * applicable),
            "applicable_layers_per_step": applicable,
        }

    # Union execution intervals independently per (process, stream/thread).
    # This is a lower bound on wall-span busy time and avoids double-counting
    # same-lane overlap. Category totals remain summed engine time by design.
    lane_intervals: dict[tuple[Any, Any], list[tuple[float, float]]] = defaultdict(list)
    for item in device:
        lane_intervals[(item.pid, item.tid)].append(
            (item.timestamp_us, item.timestamp_us + item.duration_us)
        )
    union_us = 0.0
    for intervals in lane_intervals.values():
        end = -math.inf
        for start, stop in sorted(intervals):
            if start >= end:
                union_us += stop - start
                end = stop
            elif stop > end:
                union_us += stop - end
                end = stop

    wait_names: dict[str, dict[str, float | int]] = {}
    grouped_wait: dict[str, list[float]] = defaultdict(list)
    for event in wait_api:
        grouped_wait[str(event.get("name", ""))].append(float(event.get("dur", 0.0)))
    for name, values in sorted(grouped_wait.items()):
        wait_names[name] = {"count": len(values), "cpu_api_duration_total_us": sum(values)}

    return {
        "schema": 1,
        "trace_files": [str(path) for path in paths],
        "decode_steps": decode_steps,
        "device_event_count": len(device),
        "device_execution_sum_us": sum(item.duration_us for item in device),
        "device_lane_union_us": union_us,
        "categories": categories,
        "wait_api_calls_excluded_from_compute": wait_names,
        "wait_accounting": (
            "Only CUDA kernel/memcpy/memset execution events are counted. "
            "cuStreamWaitValue32 and synchronization API calls are excluded; "
            "stream-idle gaps are not assigned to any compute category. The CPU "
            "API duration is enqueue/host time, not the parked-stream duration."
        ),
        "unattributed_event_names": sorted({
            item.name for item in device
            if item.category in {"other", "unattributed_gemm"}
        }),
    }


def compare_runs(real_path: Path, dummy_path: Path) -> dict[str, Any]:
    real = json.loads(real_path.read_text())
    dummy = json.loads(dummy_path.read_text())
    for label, payload in (("real", real), ("dummy", dummy)):
        if payload.get("kind") != "profile_off_measurement" or payload.get("profiler_enabled"):
            raise ValueError(f"{label} input is not a profile-off measurement")
        if not payload.get("label"):
            raise ValueError(f"{label} input has no arm label")
    if real.get("model") != dummy.get("model"):
        raise ValueError("real and dummy runs used different models")
    if real.get("requested_decode_steps") != dummy.get("requested_decode_steps"):
        raise ValueError("real and dummy runs used different decode step counts")

    r = real["measured"]["decode_interval_mean_us"]
    d = dummy["measured"]["decode_interval_mean_us"]
    if r is None or d is None:
        raise ValueError("each arm needs at least two decode tokens")
    return {
        "schema": 1,
        "real_label": real["label"],
        "dummy_label": dummy["label"],
        "decode_steps": real["requested_decode_steps"],
        "real_mean_itl_us": r,
        "dummy_mean_itl_us": d,
        "real_minus_dummy_mean_itl_us": r - d,
        "dummy_fraction_of_real": d / r,
        "interpretation": (
            "The dummy arm is the measured end-to-end CUDA/protocol control. "
            "The difference is the exposed real-provider penalty for this workload; "
            "it is not the B70 service time because real B70 work overlaps CUDA MoE."
        ),
    }


def _print_roofline(args: argparse.Namespace) -> None:
    rows = roofline_rows(
        context_tokens=args.context_tokens,
        cuda_routes_per_layer=args.cuda_routes_per_layer,
        bandwidth_gb_s=args.bandwidth_gb_s,
    )
    payload = {
        "bandwidth_gb_s_decimal": args.bandwidth_gb_s,
        "context_tokens": args.context_tokens,
        "cuda_routes_per_layer": args.cuda_routes_per_layer,
        "rows": [asdict(row) for row in rows],
        "total": {
            "bytes_per_token": sum(row.bytes_per_token for row in rows),
            "minimum_us": sum(row.minimum_us for row in rows),
        },
        "caveat": (
            "Logical compulsory-byte lower bounds, not measured DRAM traffic. "
            "Attainment requires a measured category duration and must be reported "
            "as minimum_us / measured_us."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _common_request_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--label", required=True, help="auditable arm name, e.g. real-b70 or dummy-zero")
    parser.add_argument("--out", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    roof = sub.add_parser("roofline", help="print per-category byte lower bounds")
    roof.add_argument("--context-tokens", type=int, default=256)
    roof.add_argument(
        "--cuda-routes-per-layer", type=float,
        default=TOP_K * CUDA_EXPERTS / EXPERTS,
        help="mean distinct routed CUDA experts; split:54 uniform baseline is 2.4",
    )
    roof.add_argument("--bandwidth-gb-s", type=float, default=BANDWIDTH_GB_S)

    measure = sub.add_parser("measure", help="profile-off production/dummy timing arm")
    _common_request_args(measure)

    capture = sub.add_parser("capture", help="capture a short vLLM torch-profiler trace")
    _common_request_args(capture)
    capture.add_argument("--trace-dir", required=True, help="same absolute directory configured on vLLM")
    capture.add_argument("--trace-wait", type=float, default=30.0)

    analyze = sub.add_parser("analyze", help="categorize CUDA execution in Kineto traces")
    analyze.add_argument("traces", nargs="+", type=Path)
    analyze.add_argument("--decode-steps", type=int, required=True)
    analyze.add_argument("--out", type=Path)

    compare = sub.add_parser("compare", help="compare real and immediate-zero profile-off arms")
    compare.add_argument("--real", type=Path, required=True)
    compare.add_argument("--dummy", type=Path, required=True)
    compare.add_argument("--out", type=Path)
    return parser


def _emit(payload: dict[str, Any], destination: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination is None:
        sys.stdout.write(text)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        sys.stdout.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "roofline":
        _print_roofline(args)
        return 0
    if args.command == "measure":
        _emit(run_measure(args, use_profiler=False), None)
        return 0
    if args.command == "capture":
        _emit(run_measure(args, use_profiler=True), None)
        return 0
    if args.command == "analyze":
        _emit(analyze_traces(args.traces, args.decode_steps), args.out)
        return 0
    if args.command == "compare":
        _emit(compare_runs(args.real, args.dummy), args.out)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
