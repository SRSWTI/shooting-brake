"""Real-server checks for token accounting and native trace completeness.

Set SB_TEST_SERVER to an isolated loopback CampaignWorkerExtension server.
No transport, model, CUDA, or native provider is mocked.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pytest
import requests

from benchmarks.slo_split_matrix import one_request
from experiments.profile_workloads import rpc


@pytest.fixture(scope="module")
def server():
    target = os.environ.get("SB_TEST_SERVER")
    if not target:
        pytest.skip("set SB_TEST_SERVER to exercise the real isolated server")
    parsed = urlparse(target)
    assert parsed.hostname in ("127.0.0.1", "localhost", "::1")
    assert parsed.port != 8017, "do not run these tests against production"
    response = requests.get(target + "/v1/models", timeout=10)
    response.raise_for_status()
    return target, response.json()["data"][0]["id"]


def test_token_usage_and_first_delivery_event(server):
    async def exercise():
        first = asyncio.Event()
        async with aiohttp.ClientSession() as session:
            result = await one_request(
                session,
                *server,
                "Explain why caches improve repeated reads.",
                16,
                60,
                first_token_event=first,
                ignore_eos=True,
            )
        assert result["ok"], result
        assert first.is_set()
        assert result["tokens"] == result["usage"]["completion_tokens"] == 16
        assert sum(event["new_tokens"] for event in result["events"]) == result["tokens"]
        deliveries = [event for event in result["events"] if event["new_tokens"]]
        assert result["start_monotonic_ns"] <= deliveries[0]["monotonic_ns"]
        assert deliveries[-1]["monotonic_ns"] <= result["end_monotonic_ns"]
        assert result["ttft_s"] == pytest.approx(
            (deliveries[0]["monotonic_ns"] - result["start_monotonic_ns"]) / 1e9
        )
        assert result["tpot_ms"] == pytest.approx(result["e2e_s"] * 1000 / result["tokens"])
        if any(event["new_tokens"] > 1 for event in deliveries):
            assert result["itl_p50_ms"] is None and result["itl_p99_ms"] is None

    asyncio.run(exercise())


def test_http_failure_is_not_a_successful_latency_sample(server):
    async def exercise():
        async with aiohttp.ClientSession() as session:
            result = await one_request(
                session,
                server[0],
                "missing-campaign-checkpoint",
                "Hello",
                8,
                30,
            )
        assert not result["ok"]
        assert result["http_status"] == 404
        assert result["tokens"] == 0
        assert "ttft_s" not in result
        assert result["end_monotonic_ns"] >= result["start_monotonic_ns"]

    asyncio.run(exercise())


def test_trace_rows_cover_actual_autoregressive_work(server, tmp_path):
    async def exercise():
        async with aiohttp.ClientSession() as session:
            await rpc(
                session,
                server[0],
                "campaign_start",
                label="trace-regression",
                output_root=str(tmp_path),
                tracing="0",
            )
            try:
                result = await one_request(
                    session,
                    *server,
                    "Explain why a computer uses a cache.",
                    8,
                    60,
                    ignore_eos=True,
                )
            finally:
                stopped = await rpc(session, server[0], "campaign_stop")
        assert result["ok"], result
        assert stopped["passed_trace_coverage"], stopped
        trace = json.loads(Path(stopped["destination"]).read_text())
        assert trace["devices"]
        expected_rows = result["prompt_tokens"] + result["tokens"] - 1
        for device in trace["devices"].values():
            assert device["native_errors"] == 0
            assert len(device["entries"]) == device["dispatches"]
            by_layer = defaultdict(int)
            for entry in device["entries"]:
                assert entry["t0_ns"] <= entry["t1_ns"]
                by_layer[entry["layer"]] += entry["M"]
            assert len(by_layer) == trace["after"]["hybrid_layers"]
            assert set(by_layer.values()) == {expected_rows}
        assert all(
            sample["within_host_bounds"] for sample in trace["clock_start"] + trace["clock_stop"]
        )

    asyncio.run(exercise())
