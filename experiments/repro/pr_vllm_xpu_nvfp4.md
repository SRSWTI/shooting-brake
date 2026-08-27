## Purpose

The Xe2 grouped MoE GEMM supports MXFP4 (E2M1 weights, E8M0 scales at group 32)
but not NVFP4 (E2M1 weights, **E4M3** scales at group **16**), which is the
4-bit format NVIDIA-ecosystem checkpoints actually ship — ModelOpt and
compressed-tensors both emit it. Today running such a checkpoint on Xe2 requires
requantizing to MXFP4, and e8m0/block-32 is a lossier scale representation than
e4m3/block-16, so that conversion costs accuracy.

This adds NVFP4 as a B-dtype so those checkpoints run as-is.

NVFP4 cannot reuse the MXFP4 path, because it differs on both axes:

- **Scale decode.** MXFP4's E8M0 is exponent-only and is decoded by shifting the
  byte into the float exponent field (`bits << 23`). E4M3 carries a mantissa, so
  that shift produces garbage. This goes through `cutlass::float_e4m3_t`.
- **Group size.** 16 rather than 32. The scale reload in `xe_gemm_4bits` is gated
  on `k_tile * tile_k % group_size == 0`, so a `tile_k` of 32 would span two
  16-element scale groups and apply only the first group's scale to all 32
  values — wrong results, no error.

What is in the change:

- `B_DTYPE::NVFP4`, appended so existing enumerator values are unchanged, plus
  the E4M3 decode branch. Scale indexing is identical to the MX path
  (`[N, K/group_size]` per expert, row-major), so no layout change is required.
- A `tile_k=16` policy ladder (`w4a16_policy_*_k16`) mirroring the existing
  w4a16 tiers, so `tile_k == group_size` and the existing reload gate stays
  correct without touching the mainloop.
- NVFP4 is told apart from MXFP4 by the **scale dtype**. As a side effect this
  stops an E4M3 scale tensor from being silently decoded as E8M0, which is what
  happens today.
- A `static_assert` that `tile_k` does not exceed the block-scale group size, so
  pairing a group size with too large a K tile is a build error rather than
  wrong numbers.

The per-expert FP32 global scale that NVFP4 checkpoints also carry is
deliberately left to the caller. It is constant per expert, so it factors out of
the dot product and is cheaper to apply to that expert's output rows than per
weight.

## Test Plan

Added `test_xe_grouped_gemm_nvfp4` to `tests/fused_moe/test_grouped_gemm.py`,
mirroring `test_xe_grouped_gemm_mxfp4` — same `FUSED_MOE_MNK_FACTORS`, both
activation dtypes, bias on and off — against an fp32 dequant reference.

Two deliberate additions to coverage:

- **E=85 as well as E=16.** Existing grouped-GEMM coverage runs at 16 experts. A
  routed MoE layer commonly shards more than that onto one device, and at these
  token counts E=85 also exercises **empty experts** (`init_rows_for_experts`
  leaves most experts with zero rows at `m=1`/`m=4`), which is the case the
  persistent scheduler has to walk past.
- A negative test of the new `static_assert`, done by hand (below).

```
# correctness
pytest tests/fused_moe/test_grouped_gemm.py -k nvfp4
# regression on the paths this touches
pytest tests/fused_moe/test_grouped_gemm.py -k mxfp4
pytest tests/fused_moe/test_grouped_gemm.py -k "int4 or mxfp8 or block_fp8"
```

Hardware: Intel Arc Pro B70 (Battlemage G31), driver `1.15.39122+12`, NEO
`26.27.39122.12`, oneAPI 2026.1, torch 2.13.0+xpu, JIT spir64,
`SYCL_UR_USE_LEVEL_ZERO_V2=0`.

## Test Result

Build: `make -C build/temp grouped_gemm_xe_2` succeeds for `bmg-g31-a0`, no new
warnings attributable to this change.

```
NVFP4   (new)                          32 passed, 112 deselected in 52.10s
MXFP4   (pre-existing, regression)     16 passed, 128 deselected in 10.52s
int4 / mxfp8 / block_fp8 (regression)  32 passed, 112 deselected in 18.82s
```

The `static_assert` was verified to fire, by temporarily pointing the NVFP4
ladder at `w4a16_policy` (`tile_k=32`) and rebuilding:

```
csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp:302:7: error: static assertion failed due to
requirement 'cute::size(cute::tuple<cute::C<128>, cute::C<256>, cute::C<32>>{}) <= group_size':
tile_k must not exceed the block-scale group size; pair a group_size of 16 with a
tile_k=16 policy (see w4a16_policy_*_k16)
```

### Performance, and a caveat I want to be upfront about

Timed through the same `cutlass_grouped_gemm_xe2` entry, same device, same
M/N/K/E, `E=85`, `n=2048`, `k=3072`, fp16 activations, uniform fill, best of
3 x 20 iterations:

| variant | rows/expert | total_m | ms | TFLOP/s |
|---|---|---|---|---|
| nvfp4 | 30 | 2550 | 1.1986 | 26.8 |
| mxfp4 | 30 | 2550 | 0.6267 | 51.2 |
| bits16 | 30 | 2550 | 2.2931 | 14.0 |
| nvfp4 | 120 | 10200 | 4.8562 | 26.4 |
| mxfp4 | 120 | 10200 | 2.0949 | 61.3 |
| bits16 | 120 | 10200 | 2.2234 | 57.7 |

**NVFP4 is 1.9x-2.3x slower than MXFP4 here, and at 120 rows/expert it is slower
than 16-bit.** That is not a measurement artifact and I do not want to bury it.
The cause is structural: correctness forces `tile_k == 16`, which halves the K
work per mainloop iteration relative to the `tile_k=32` policies every other
4-bit variant uses. The policy comment says the same thing.

So the value of this PR is capability, not throughput: it lets an NVFP4
checkpoint run without a lossy requantization step. It is not a faster 4-bit
path, and anyone who can requantize to MXFP4 without caring about the accuracy
delta should keep using MXFP4.

The obvious follow-up, which I have deliberately kept out of this PR to keep it
single-purpose, is to let one `tile_k=32` tile carry two scale groups so NVFP4
can use the wider tiles. That means touching the mainloop's scale handling
rather than just adding a dtype, and it should be its own change with its own
measurements. Happy to take direction on whether you want that, and whether
you would rather land it as one change instead — in which case I will close this
and come back with both.

## (Optional) Documentation Update

None. If you would like the group-size constraint written up somewhere more
discoverable than the `static_assert` message and the policy comment, point me at
the right file and I will add it.
