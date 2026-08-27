# Summary

In `examples/12_xe20_moe_gemm_cute_interface`, the `GFLOPS` figure the example
prints is strongly dependent on how many rows each expert receives — about 5.6x
between the built-in row table and a uniform 30-rows-per-expert fill, at
identical `--n`/`--k` and identical expert count.

The cause looks like arithmetic intensity rather than anything wrong with the
kernel: across a 30 → 250 rows/expert sweep the measured kernel time barely
moves (2.08 → 2.36 ms) while the FLOP count moves 8.3x. At MoE fills the leg is
bound by streaming expert weights, so FLOP/s ends up reporting the fill, not the
pipeline.

Since the fill is currently a fixed in-source table, a run's number is not
self-describing. I'd like to suggest making rows-per-expert and expert count
first-class options so the operating point is visible and sweepable. I have a
small PR ready that does exactly that and preserves current defaults and CI
invocations — happy to open it if you're interested.

To be clear about scope: this is about the *reported operating point*, not the
design. `moe_tile_scheduler.hpp` handles the variable-row case well, and the
measurements below actually demonstrate that (see the skew note at the end).

# Measurements

Arc Pro B70 (Battlemage G31), driver `1.15.39122+12`, NEO `26.27.39122.12`,
Level Zero loader `1.28.6`, oneAPI 2026.1, kernel `7.0.0-30-generic`,
Ubuntu 26.04. Built with `-DCUTLASS_ENABLE_SYCL=ON`, JIT `spir64`,
`SYCL_UR_USE_LEVEL_ZERO_V2=0`, single idle card, nothing else on the GPU.

Fixed at the CI shape `--n=2880 --k=2880 --num_layers=24 --verify=0`,
32 experts. Values are the median over the 24 layers of the `N=5760, K=2880`
leg, with min/max across layers:

| rows/expert | TFLOP/s (median) | min | max | kernel ms (median) |
|---|---|---|---|---|
| built-in table | 82.1 | 73.8 | 92.6 | 3.2510 |
| uniform 250 | 112.7 | 95.3 | 113.9 | 2.3550 |
| uniform 120 | 59.5 | 26.7 | 59.8 | 2.1419 |
| uniform 70 | 35.7 | 29.6 | 36.0 | 2.0821 |
| uniform 30 | 14.7 | 12.0 | 15.4 | 2.1663 |

Same picture on the `N=2880, K=2880` leg: 74.3 / 102.9 / 51.6 / 30.8 / 13.2
TFLOP/s for the same five fills.

The flat time column is the interesting part. From uniform 30 to uniform 250 the
FLOP count rises 8.3x, the reported rate rises 7.7x, and the kernel time rises
1.09x. That is consistent with the leg being weight-traffic-bound at these
fills: the same expert weights are streamed regardless of how many rows use
them, so `2*sum_e(M_e*N*K) / time` mostly measures `sum_e M_e`.

# Steps to reproduce

Today this needs a source edit, which is the thing I'm suggesting be fixed. With
the flags from the PR:

```
# built-in table (current default, unchanged)
./12_xe20_moe_gemm_cute_interface --n=2880 --k=2880 --num_layers=24 --verify=0

# uniform fill sweep at the same shape and expert count
for r in 30 70 120 250; do
  ./12_xe20_moe_gemm_cute_interface --n=2880 --k=2880 --num_layers=24 --verify=0 \
      --rows_per_expert=$r
done
```

# Why the low-fill end is the interesting one for MoE serving

The built-in table in `12_xe20_moe_gemm_cute_interface.cpp:433-496` is a fixed
24x32 table. Its statistics, computed directly from the source literal
(768 entries):

- mean 250.8 rows/expert, median 70.0
- min 0, max 1953
- 98 zeros (12.8% of entries)
- 40.6% of entries are <= 30 rows
- 53 entries are >= 1000 rows
- per-layer totals 7529..8048, per-layer means tightly clustered at 235.3..251.5

So the table is realistically skewed. But because the reported rate is an
aggregate over all experts, the fat tail dominates it: the 53 entries at >= 1000
rows carry the number, while the 312 entries at <= 30 rows are where a routed
MoE actually spends its time.

For reference, a routed MoE decode/prefill step in our workload (top_k=10 over
218 experts per layer, 85 experts resident per card) produces a per-expert row
distribution of 15..995 rows with 3728 rows total across 85 experts, i.e. a mean
of ~44 rows/expert. That is the regime in the 14-60 TFLOP/s rows of the table
above, not the 82-113 rows.

# One extra data point that reflects well on the scheduler

At essentially equal mean fill, the skewed table and a uniform fill are not the
same number:

- built-in table, mean 250.8 rows/expert → 82.1 TFLOP/s
- uniform 250 rows/expert → 112.7 TFLOP/s

So handling the realistic skewed distribution costs about 27% relative to a
perfectly balanced fill of the same mean. That is a reasonable price for
1953-row and 0-row experts coexisting in one launch, and it means the default
table is already the honest case rather than the flattering one — the flattering
configuration would have been the uniform one.

# Suggested change

Add `--rows_per_expert=<int>` (0 = use the built-in table) and `--experts=<int>`
to `Options`, print the effective expert count / fill mode / mean rows-per-expert
alongside the existing output, and leave the defaults exactly as they are so the
`--n=2880 --k=2880 --num_layers=24` CI invocation in
`examples/12_xe20_moe_gemm_cute_interface/CMakeLists.txt:29-43` is unaffected.

I have this implemented against `2db1b7c9` and verified that with no new flags
the example prints `Mean rows/expert : 250.824`, matching the table literal.
Glad to send it as a PR.
