from __future__ import annotations

import ctypes
import importlib.util
import time
from pathlib import Path

import pytest


def _wait_for_event(event, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not event.query() and time.monotonic() < deadline:
        time.sleep(0.001)
    return event.query()


@pytest.fixture
def cuda_signals():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for real stream-memory operations")
    source = Path(__file__).parents[1] / "src/phase4/src/shooting_brake_vllm/stream_signal.py"
    spec = importlib.util.spec_from_file_location("sb_test_stream_signal", source)
    assert spec is not None and spec.loader is not None
    signals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signals)
    signals._cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
    signals._cudart.cudaFreeHost.restype = ctypes.c_int
    return torch, signals


@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph-replay"])
def test_wait_requires_equality_on_current_stream(cuda_signals, capture):
    torch, signals = cuda_signals
    host, device = signals.alloc_host_mapped_flag(2)
    stream = torch.cuda.Stream()
    marker = torch.zeros(1, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        marker.add_(0)
    stream.synchronize()
    finished = torch.cuda.Event()
    try:
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                signals.wait_flag(device, 1)
                marker.add_(1)
            with torch.cuda.stream(stream):
                graph.replay()
        else:
            with torch.cuda.stream(stream):
                signals.wait_flag(device, 1)
                marker.add_(1)
        finished.record(stream)
        try:
            # GEQ would release on 2. EQ must keep the following work parked.
            time.sleep(0.05)
            assert not finished.query()
        finally:
            # Always release the GPU waiter, including on an assertion failure.
            signals.write_flag_host(host, 1)
        assert _wait_for_event(finished), "stream did not resume on exact target"
        assert marker.item() == 1
        if capture:
            # The same graph must consume a fresh signal on each replay.
            signals.write_flag_host(host, 2)
            with torch.cuda.stream(stream):
                graph.replay()
            finished.record(stream)
            try:
                time.sleep(0.05)
                assert not finished.query()
            finally:
                signals.write_flag_host(host, 1)
            assert _wait_for_event(finished), "second replay did not complete"
            assert marker.item() == 2
    finally:
        signals.write_flag_host(host, 1)
        stream.synchronize()
        assert signals._cudart.cudaFreeHost(host) == 0
