## Feature

`examples/12_xe20_moe_gemm_cute_interface` gains two options:

- `--experts=<int>` — number of experts per layer (default 32, unchanged)
- `--rows_per_expert=<int>` — give every expert this many rows, replacing the
  built-in measured table (default 0 = use the built-in table, unchanged)

It also prints the effective configuration, so a result is self-describing:

```
MoE fill configuration
  Experts          : 32
  Layers           : 24
  Rows/expert      : built-in measured table
  Mean rows/expert : 250.824
```

Follow-up to #855.

## Use Case

The example previously fixed `num_experts` at 32 and drew per-expert row counts
from a fixed 24x32 in-source table, so the `GFLOPS` it prints could only be
obtained at one fill unless you edited the source.

That matters because at fixed `--n`/`--k` the printed rate is dominated by fill.
On Arc Pro B70 (Battlemage G31), 32 experts, `--n=2880 --k=2880 --num_layers=24
--verify=0`, median over 24 layers of the `N=5760, K=2880` leg:

| rows/expert | TFLOP/s (median) | kernel ms (median) |
|---|---|---|
| built-in table | 82.1 | 3.2510 |
| uniform 250 | 112.7 | 2.3550 |
| uniform 120 | 59.5 | 2.1419 |
| uniform 70 | 35.7 | 2.0821 |
| uniform 30 | 14.7 | 2.1663 |

From uniform 30 to uniform 250 the FLOP count rises 8.3x and the reported rate
rises 7.7x, while kernel time rises 1.09x — the leg is bound by streaming expert
weights at these fills, so the rate largely reports `sum_e M_e`. Routed MoE
inference operates near the low end of that table (top_k over tens to hundreds
of experts), which is exactly the region that previously required a source edit
to reach.

## API

Additive only. Both options are parsed with the existing `cutlass::CommandLine`
idiom next to `--n`/`--k`/`--num_layers`/`--verify`, and validated with the same
`error = true` pattern used by `examples/10_bmg_grouped_gemm_mixed_dtype`:

- `--experts` must be positive
- `--rows_per_expert` must be non-negative
- `--experts` above 32 requires `--rows_per_expert`, since the built-in table
  only has 32 columns; the diagnostic says so explicitly

`main()` gained the standard `if (options.error) { ... return -1; }` block that
sibling examples already use.

The per-layer row array became a `std::vector<int>` because the expert count is
now a runtime value; the literal table is untouched and is now a
`static const int [max_layers][kDefaultTableExperts]` that seeds the vector when
`--rows_per_expert` is not supplied.

## Example

```
# built-in table (current default, byte-for-byte the previous behaviour)
./12_xe20_moe_gemm_cute_interface --n=2880 --k=2880 --num_layers=24 --verify=0

# sweep fill at fixed shape and expert count
for r in 30 70 120 250; do
  ./12_xe20_moe_gemm_cute_interface --n=2880 --k=2880 --num_layers=24 \
      --verify=0 --rows_per_expert=$r
done

# a wider expert count, which requires an explicit uniform fill
./12_xe20_moe_gemm_cute_interface --n=2048 --k=3072 --experts=85 --rows_per_expert=30
```

## Testing

Built and run on Arc Pro B70 (Battlemage G31), driver `1.15.39122+12`, NEO
`26.27.39122.12`, oneAPI 2026.1, JIT `spir64`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`.

- Builds clean via `cmake --build build-sycl --target 12_xe20_moe_gemm_cute_interface`.
- **Default behaviour preserved.** With no new flags the example reports
  `Mean rows/expert : 250.824`. I independently brace-parsed the 24x32 table
  literal out of the source (768 entries, every inner group verified length 32)
  and its mean is 250.8 — so the vector path reproduces the table exactly. The
  CMake test arguments `--n=2880 --k=2880 --num_layers=24` are unaffected and
  need no change.
- **Validation exercised.** `--experts=85` without `--rows_per_expert` prints the
  diagnostic and aborts with `-1`:
  ```
  ERROR: --experts=85 exceeds the 32 experts covered by the built-in row table.
         Pass --rows_per_expert=<int> to supply a uniform fill instead.
  Aborting execution.
  ```
- **Sweep exercised.** The five-fill table above, plus `--experts=85
  --rows_per_expert={30,120,230}` at `--n=2048 --k=3072`, all run to completion.

Not tested: PVC, and `verify=1` at `--experts` > 32 (the correctness reference
path was not touched, but I only exercised it at the default expert count).

## ToDo

None that I'm aware of. If you'd rather have the flag spelled `--fill`, or want
the CMake test arguments extended to cover a second fill point in CI, say so and
I'll adjust.
