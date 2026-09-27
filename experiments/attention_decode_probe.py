"""Isolated paged decode: installed vLLM FA2 versus FlashInfer kernels.

Uses Jota-compatible GQA, an independent float64 attention reference, and
CUDA-graph timing. This excludes projections, routing, transport, scheduling,
and HTTP. Repeated KV inputs may be cache-hot; results are not serving ITL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import flashinfer
import torch
import vllm
from vllm.vllm_flash_attn import flash_attn_varlen_func


def reference_attention(q, k, v, batch, sequence, kv_heads, window_left):
    head_dim = q.shape[-1]
    groups = q.shape[-2] // kv_heads
    keys = k.reshape(batch, -1, kv_heads, head_dim)[:, :sequence]
    values = v.reshape(batch, -1, kv_heads, head_dim)[:, :sequence]
    if window_left >= 0:
        first = max(0, sequence - window_left - 1)
        keys = keys[:, first:]
        values = values[:, first:]
    queries = q.double().reshape(batch, kv_heads, groups, head_dim)
    keys = keys.double().permute(0, 2, 3, 1)
    values = values.double().permute(0, 2, 1, 3)
    scores = torch.matmul(queries, keys) / math.sqrt(head_dim)
    return torch.matmul(scores.softmax(dim=-1), values).reshape_as(q)


def check_output(actual, expected):
    if not torch.isfinite(actual).all().item():
        raise AssertionError("attention output is nonfinite")
    error = actual.double() - expected
    relative_rmse = (error.norm() / expected.norm().clamp_min(1e-30)).item()
    max_error = error.abs().max().item()
    try:
        torch.testing.assert_close(actual.double(), expected, rtol=0.01, atol=1e-5)
    except AssertionError as exc:
        raise AssertionError(
            f"relative_rmse={relative_rmse}, max_absolute_error={max_error}: {exc}"
        ) from exc
    if relative_rmse > 0.005:
        raise AssertionError(f"attention relative RMS error {relative_rmse} exceeds 0.005")
    return {"relative_rmse": relative_rmse, "max_absolute_error": max_error}


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=32768)
    parser.add_argument("--query-heads", type=int, default=48)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--window-left", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("vllm-fa2", "flashinfer", "flashinfer-tc"),
        default=("vllm-fa2", "flashinfer-tc"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite attention benchmark evidence")
    if (
        min(
            args.sequence,
            args.query_heads,
            args.kv_heads,
            args.head_dim,
            args.batch,
            args.page_size,
            args.iterations,
            args.repeats,
        )
        <= 0
    ):
        raise ValueError("attention dimensions and timing counts must be positive")
    if args.query_heads % args.kv_heads or args.warmups < 0 or args.window_left < -1:
        raise ValueError("invalid GQA, warmup count, or window")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = torch.bfloat16
    pages_per_sequence = (args.sequence + args.page_size - 1) // args.page_size
    total_pages = args.batch * pages_per_sequence
    cache_shape = (total_pages, args.page_size, args.kv_heads, args.head_dim)
    k = torch.randn(cache_shape, dtype=dtype, device=device, generator=generator)
    v = torch.randn(cache_shape, dtype=dtype, device=device, generator=generator)
    q = torch.randn(
        (args.batch, args.query_heads, args.head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    expected = reference_attention(
        q, k, v, args.batch, args.sequence, args.kv_heads, args.window_left
    )
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        args.batch, pages_per_sequence
    )
    query_indptr = torch.arange(args.batch + 1, dtype=torch.int32, device=device)
    kv_indptr = query_indptr * pages_per_sequence
    kv_indices = block_table.flatten()
    kv_lengths = torch.full((args.batch,), args.sequence, dtype=torch.int32, device=device)
    last_page_lengths = torch.full(
        (args.batch,),
        (args.sequence - 1) % args.page_size + 1,
        dtype=torch.int32,
        device=device,
    )
    scale = args.head_dim**-0.5
    outputs = {name: torch.empty_like(q) for name in args.backends}

    def run_fa2():
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=outputs["vllm-fa2"],
            cu_seqlens_q=query_indptr,
            max_seqlen_q=1,
            seqused_k=kv_lengths,
            max_seqlen_k=args.sequence,
            block_table=block_table,
            softmax_scale=scale,
            causal=True,
            window_size=(args.window_left, 0),
            fa_version=2,
            num_splits=0,
        )

    # The CUDA-core decoder rejects GQA group size 6 on the installed stack.
    # Keep that unsupported candidate separate; do not alter Jota's geometry.
    functions = {"vllm-fa2": run_fa2} if "vllm-fa2" in args.backends else {}
    wrappers = {}
    for name, tensor_cores in (("flashinfer", False), ("flashinfer-tc", True)):
        if name not in args.backends:
            continue
        workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=tensor_cores,
            backend="fa2",
            paged_kv_indptr_buffer=torch.empty_like(kv_indptr),
            paged_kv_indices_buffer=torch.empty_like(kv_indices),
            paged_kv_last_page_len_buffer=torch.empty_like(last_page_lengths),
        )
        wrapper.plan(
            kv_indptr,
            kv_indices,
            last_page_lengths,
            args.query_heads,
            args.kv_heads,
            args.head_dim,
            args.page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            sm_scale=scale,
            window_left=args.window_left,
        )
        wrappers[name] = wrapper
        functions[name] = lambda wrapper=wrapper, name=name: wrapper.run(
            q, (k, v), out=outputs[name], window_left=args.window_left
        )

    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "versions": {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "flashinfer": flashinfer.__version__,
        },
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "BF16 paged decode; float64 reference; repeated same-KV CUDA graph; no serving or projection costs",
        "backends": {},
    }
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    for name, function in functions.items():
        with torch.cuda.stream(capture_stream):
            for _ in range(args.warmups + 1):
                function()
        torch.cuda.current_stream().wait_stream(capture_stream)
        try:
            initial_check = check_output(outputs[name], expected)
        except AssertionError as exc:
            result["backends"][name] = {"passed": False, "error": str(exc)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(name, str(exc), flush=True)
            continue
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(args.iterations):
                function()
        torch.cuda.current_stream().wait_stream(capture_stream)
        timings = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000 / args.iterations)
        final_check = check_output(outputs[name], expected)
        result["backends"][name] = {
            "passed": True,
            "initial_check": initial_check,
            "post_timing_check": final_check,
            "microseconds_per_call": timings,
            "median_microseconds": sorted(timings)[len(timings) // 2],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(name, json.dumps(result["backends"][name]), flush=True)
        del graph
    return 0 if all(item["passed"] for item in result["backends"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
