#!/usr/bin/env python3
"""Gate for the CPU tier's flag-driven dispatch (all-out mode, phase 3).

Exercises the native poller through the same handshake the CUDA stream uses,
driving the flags from Python instead of from a graph:

    write signal = M  ->  poller computes  ->  poller sets completion = 1

The result must match a direct ``moe_forward`` call, and the completion flag
must be raised even when the dispatch fails — the CUDA side parks in
``cuStreamWaitValue32``, which has no timeout, so a missed flag wedges the
device permanently.

This uses ordinary host memory for the flags rather than CUDA host-mapped
allocations; the poller only ever reads and writes them as plain ``uint32``,
so the mechanism is identical.

Run::

    make -C phase7 cpu
    .venv/bin/python phase7/cpu_poller_test.py
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "phase4" / "src"))

import nvfp4_testutil  # noqa: E402
from shooting_brake_vllm.cpu_expert_host import (  # noqa: E402
    CpuExpertHost,
    CpuExpertPoller,
)

LAYER = 0
EXPERTS = 8
HIDDEN = 256
INTER = 128
TOPK = 4
MAX_BATCH = 8

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


class Flag:
    """A 4-byte host flag, standing in for a CUDA host-mapped allocation."""

    def __init__(self) -> None:
        self._buf = ctypes.c_uint32(0)

    @property
    def addr(self) -> int:
        return ctypes.addressof(self._buf)

    def get(self) -> int:
        return ctypes.c_uint32.from_address(self.addr).value

    def set(self, v: int) -> None:
        ctypes.c_uint32.from_address(self.addr).value = v


def wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.0002)
    return False


def main() -> int:
    lib = Path(__file__).resolve().parent / "libsb_cpu_expert.so"
    if not lib.is_file():
        print(f"missing {lib}; run: make -C phase7 cpu")
        return 1

    gen = torch.Generator().manual_seed(20260807)

    host = CpuExpertHost(
        num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, intermediate=INTER,
        max_experts=EXPERTS, num_threads=4, lib_path=lib,
    )
    for e in range(EXPERTS):
        host.load_expert(
            LAYER, e,
            nvfp4_testutil.make_plane(INTER, HIDDEN, gen),
            nvfp4_testutil.make_plane(INTER, HIDDEN, gen),
            nvfp4_testutil.make_plane(HIDDEN, INTER, gen),
        )

    # Buffers the poller holds raw pointers into. These must stay referenced
    # for the poller's lifetime; a collected tensor is a use-after-free on a
    # background thread.
    pinned_hidden = torch.zeros(MAX_BATCH, HIDDEN, dtype=torch.bfloat16)
    pinned_ids = torch.full((MAX_BATCH, TOPK), -1, dtype=torch.int32)
    pinned_weights = torch.zeros(MAX_BATCH, TOPK, dtype=torch.float32)
    pinned_output = torch.zeros(MAX_BATCH, HIDDEN, dtype=torch.float32)

    signal, completion = Flag(), Flag()

    print("== poller lifecycle ==")
    poller = CpuExpertPoller(host)
    poller.register_layer(
        layer_idx=LAYER, signal_host=signal.addr,
        completion_host=completion.addr, pinned_hidden=pinned_hidden,
        pinned_ids=pinned_ids, pinned_weights=pinned_weights,
        pinned_output=pinned_output,
    )
    poller.start()
    check("poller started", poller.started)
    check("one layer registered", poller.layers == 1)
    check("idle poller does no work", poller.dispatch_count == 0)

    print("\n== dispatch handshake matches direct compute ==")
    for trial, M in enumerate((1, 4, 8)):
        x = (torch.randn(M, HIDDEN, generator=gen) * 0.5).bfloat16()
        ids = torch.randint(0, EXPERTS, (M, TOPK), generator=gen,
                            dtype=torch.int32)
        w = torch.rand(M, TOPK, generator=gen, dtype=torch.float32)

        want = host.moe_forward(LAYER, x, ids, w)

        # Stage exactly as the CUDA stream would, then raise the signal.
        pinned_hidden[:M].copy_(x)
        pinned_ids[:M].copy_(ids)
        pinned_weights[:M].copy_(w)
        pinned_output.zero_()
        completion.set(0)
        signal.set(M)

        served = wait_for(lambda: completion.get() == 1)
        check(f"M={M} completion raised", served)
        if not served:
            continue
        check(f"M={M} signal cleared by poller", signal.get() == 0)
        got = pinned_output[:M].clone()
        ok = torch.allclose(got, want, rtol=1e-3, atol=1e-3)
        check(f"M={M} result matches direct moe_forward", ok,
              f"max|err|={(got - want).abs().max().item():.3e}")
        check(f"M={M} dispatch counted", poller.dispatch_count == trial + 1,
              f"count={poller.dispatch_count}")

    print("\n== rows beyond M are untouched ==")
    # The signal's value bounds the work: staging a full buffer but signalling
    # M=1 must compute exactly one row, which is what makes one graph per
    # batch size correct.
    pinned_output.zero_()
    x = (torch.randn(MAX_BATCH, HIDDEN, generator=gen) * 0.5).bfloat16()
    pinned_hidden.copy_(x)
    pinned_ids.fill_(0)
    pinned_weights.fill_(1.0)
    completion.set(0)
    signal.set(1)
    if wait_for(lambda: completion.get() == 1):
        touched = (pinned_output[0].abs().sum() > 0).item()
        untouched = bool((pinned_output[1:] == 0).all())
        check("M=1 computed row 0", bool(touched))
        check("M=1 left rows 1..7 zero", untouched)
    else:
        check("M=1 completion raised", False)

    print("\n== failure still releases the waiter ==")
    # A layer index the host rejects makes moe_forward fail. The completion
    # flag must still rise: cuStreamWaitValue32 cannot time out, so a poller
    # that returns early on error would hang the GPU forever.
    bad_signal, bad_completion = Flag(), Flag()
    bad_poller = CpuExpertPoller(host)
    bad_poller.register_layer(
        layer_idx=99,  # out of range for num_layers=1
        signal_host=bad_signal.addr, completion_host=bad_completion.addr,
        pinned_hidden=pinned_hidden, pinned_ids=pinned_ids,
        pinned_weights=pinned_weights, pinned_output=pinned_output,
    )
    bad_poller.start()
    bad_completion.set(0)
    bad_signal.set(1)
    released = wait_for(lambda: bad_completion.get() == 1)
    check("failed dispatch raises completion", released)
    check("failure counted", bad_poller.error_count == 1,
          f"errors={bad_poller.error_count}")
    bad_poller.stop()
    bad_poller.close()

    print("\n== telemetry ==")
    check("service time recorded", poller.service_mean_us > 0.0,
          f"{poller.service_mean_us:.1f} us/dispatch")
    check("no errors on the good poller", poller.error_count == 0,
          f"errors={poller.error_count}")
    check("no skipped routes", host.skipped_routes == 0,
          f"skipped={host.skipped_routes}")

    poller.stop()
    check("poller stopped", not poller.started)
    poller.close()
    host.close()

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
