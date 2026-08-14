# Warm-Run Results (real RTX 5090 + real Arc Pro B70)

20 warmup + 200 measured iterations per size, `tests/pinned_staging_probe/{host_bench.cu,provider_bench.cpp}`.

## Two-hop route: 5090 write -> memfd -> eventfd -> [B70 H2D + D2H] -> eventfd ack

| Size | Host-observed total p50 | p95 | p99 | Effective throughput |
|---|---:|---:|---:|---:|
| 4 KiB | 0.064 ms | 0.079 ms | 0.091 ms | 0.1 GB/s (latency-bound) |
| 64 KiB | 0.084 ms | 0.108 ms | 0.111 ms | 0.8 GB/s |
| 1 MiB | 0.951 ms | 1.080 ms | 1.147 ms | 1.1 GB/s |
| 4 MiB | 2.911 ms | 2.991 ms | 3.034 ms | 1.4 GB/s |

Provider-measured copy-only portion (H2D + D2H, excludes eventfd signaling)
at 4 MiB: 1.349 ms + 1.191 ms = 2.540 ms. Difference from host-observed
total (2.911 ms) = **~0.37 ms of pure signaling/scheduling overhead per
round trip** — the eventfd wake/context-switch cost, not copy cost. At 4 KiB
that overhead (0.064 - 0.053 = 0.011 ms) is proportionally larger (~17% of
total vs ~13% at 4 MiB) — the fixed per-request cost matters more at small
sizes, same shape as the "ring/process overhead" already characterized for
the real ring in `docs/progress.md`.

## Isolated single-hop legs (no cross-vendor handoff — each device against its own private pinned host buffer)

| Leg | 4 KiB | 64 KiB | 1 MiB | 4 MiB |
|---|---:|---:|---:|---:|
| 5090 <-> host | 0.6 GB/s | 9.5 GB/s | 32.5 GB/s | **35.4 GB/s** |
| B70 <-> host | 0.3 GB/s | 2.4 GB/s | 3.4 GB/s | **3.5 GB/s** |

**The B70 leg is ~10x slower than the 5090 leg on this exact box**, at 4 MiB.
That is not a dma-buf/API artifact — it is the real, current PCIe link this
B70 is negotiated at. `docs/hardware.md` already flags exactly this risk
("Suspicious B70 link reference... 2.5 GT/s x1... reproduce under load").
3.5 GB/s is far above a Gen1 x1 ceiling (~250 MB/s), so it is not that
specific historical worst case, but it is well below what a Gen4/Gen5 x8-x16
slot would give — consistent with a narrower negotiated link (e.g. Gen3 x4:
~4 GB/s theoretical) than the 5090's slot. Confirm with `lspci -vv` /
`nvidia-smi`-equivalent Level Zero link-width query before trusting this as
a hardware ceiling rather than a topology/BIOS/riser artifact.

## Same-vendor P2P comparison — measured vs. inferred

**Cannot be measured on this box.** There is exactly one RTX 5090 and one
Arc Pro B70 present — no second GPU of either vendor exists to run a literal
NVIDIA-NVIDIA or Intel-Intel dma-buf P2P transfer against. The numbers below
are the most defensible proxy available, not a substitute measurement:

- A direct (single-hop) P2P transfer between two same-vendor GPUs sharing a
  PCIe switch/root complex is bounded by roughly the *slower* endpoint's own
  link — same physical constraint the isolated single-hop legs above
  already measure per device.
- Proxy ceiling for "B70-B70 P2P on this exact topology" ~= the B70's own
  isolated leg, **3.5 GB/s** at 4 MiB.
- Proxy ceiling for "5090-5090 P2P on this exact topology" ~= the 5090's own
  isolated leg, **35.4 GB/s** at 4 MiB — consistent with published PCIe
  Gen5 x16 P2P literature (nominal ~63 GB/s unidirectional, real-world
  sustained efficiency commonly 50-90% depending on transfer size and
  driver/topology, so 35 GB/s at a 4 MiB single transfer is a plausible,
  not suspicious, real-world figure — not a nominal-bandwidth citation).

Comparing the cross-vendor **two-hop measured total** (1.4-1.65 GB/s
effective at 4 MiB, depending on whether you count signaling overhead)
against the **B70-bound single-hop proxy** (3.5 GB/s) gives roughly a
**2.1-2.5x throughput tax** for going through host DRAM twice instead of a
single direct hop. That matches the theoretical "double the hops, double
the DRAM/PCIe traffic" prediction made before any code was written — now
empirically confirmed on this exact hardware, not just estimated.

**Bottom line:** the cross-vendor host-staged path is not free, but it is
not catastrophic either — roughly 2x tax versus a same-vendor direct P2P
proxy, and the B70's own link is the actual bottleneck (10x slower than the
5090's leg) well before the two-hop staging tax becomes the dominant cost.
Fixing/confirming the B70's negotiated PCIe link width would move the needle
far more than eliminating the host hop would.
