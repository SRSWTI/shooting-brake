## Summary

Fixes #5908.

Grouped matmul takes scales and zero points as memory objects at execution time,
but no API communicates their layout to the library. `quant_entry_t::get_md()`
derives the expected descriptor from the attribute alone and always produces a
dense descriptor with `abx` stride order
(`src/common/primitive_attr_quant.hpp:135-136`), and the grouped kernels index
those buffers with that layout baked in
(`src/gpu/intel/matmul/grouped_micro_gemm_m_axis.cl:437`,
`src/gpu/intel/matmul/ref_grouped_gemm.cl:170-173`).

The consequence is that passing a differently-laid-out scale memory is accepted,
runs, and returns wrong results with a success status. The documented requirement
exists (`doc/functional_api/primitives/matmul.md:383-389`, "Tensors must use
dense memory descriptors"), but nothing enforced it.

This validates the quantization execution arguments and returns
`dnnl_invalid_arguments` instead.

## Design notes

**The number of dimensions is deliberately not constrained.** Describing a
quantization tensor without its broadcast dimensions is legitimate and handled by
the kernels — a `[G, N]` descriptor for per-expert per-N weights scales of a
`[G, K, N]` weights tensor, for instance, which is what benchdnn produces for
`wei:5` because `md2dims(..., extend_by_ones=false, ...)` drops masked-out dims.
Comparing against the descriptor `get_md()` synthesizes would therefore have
rejected a large number of currently-working configurations. Instead the check
derives its reference strides from the supplied descriptor's own rank via
`memory_desc_matches_tag(mdw, get_abx_tag(mdw.ndims()))`, which also skips unit
dimensions whose strides are ambiguous.

**The check is at execution time, not primitive-descriptor creation time.** This
is forced: `dnnl_primitive_attr_set_scales()` carries no descriptor
(`include/oneapi/dnnl/dnnl.h:463-489`), so the layout genuinely is not knowable
at PD creation. The check lives in the common layer and is called from the three
grouped implementations that index scales as dense `abx`.

**Scope.** GPU micro (`grouped_gemm:micro:m_axis`), GPU reference, and CPU
reference grouped kernels. `src/cpu/x64/zen64/matmul/zen_grouped_matmul.cpp`
rejects scales outright and needs no check. Zero points are covered as well as
scales, since they are synthesized the same way.

## Test Plan / Results

Intel Arc Pro B70 (Battlemage G31), driver `1.15.39122+12`, NEO
`26.27.39122.12`, oneAPI 2026.1, Ubuntu 26.04. Configured with
`-DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=INTEL -DDNNL_BUILD_TESTS=ON
-DDNNL_EXPERIMENTAL=ON`.

### The defect, before and after

A standalone program building a grouped `f16 : f4_e2m1 : f16` matmul with
`f8_e4m3` group-16 weight scales, comparing against a scalar f64 reference, with
the scale memory bound two ways over identical logical dims `{E, K/16, N}`:

| scale md | before | after |
|---|---|---|
| dense `abc` (canonical), f32 dst | PASS, max rel err **0.000e+00** | PASS, max rel err **0.000e+00** |
| dense `abc` (canonical), f16 dst | PASS, max rel err **4.781e-04** | PASS, max rel err **4.781e-04** |
| strided `acb` | **accepted, max rel err 9.530e+01, status success** | **rejected, `dnnl_invalid_arguments`** |

The new diagnostic, verbatim under `ONEDNN_VERBOSE=all`:

```
onednn_verbose,v1,primitive,exec:check,matmul,weights scales memory descriptor must be dense with `abx` strides, got `acb`,src/common/matmul.cpp:291
```

The returned status is `dnnl_invalid_arguments` regardless of verbose level.

### Regression

`benchdnn --matmul --engine=gpu --batch=inputs/matmul/test_matmul_grouped_ci`:

```
before:  tests:621 passed:619 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:2
after:   tests:621 passed:619 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:2
```

**No previously-passing configuration now fails.**

The 2 failures are **pre-existing and unrelated** — they involve no scales at
all, so this check cannot reach them:

```
--dt=bf16:bf16:bf16 --grouped=0:4:8+8+8+8  --wtag=acb --attr-post-ops=swish:1+mul:f16:1 32x64:4x64x32
--dt=bf16:bf16:bf16 --grouped=0:4:8+0+16+8 --wtag=acb --attr-post-ops=swish:1+mul:f16:1 32x64:4x64x32
```

I measured them with the check compiled out to confirm they predate this change.
They look like a separate correctness problem in the grouped kernel (407/1024 and
460/1024 elements wrong); I have not investigated further and can file it
separately if useful.

A production-shaped NVFP4 case also still passes:

```
$ benchdnn --matmul --engine=gpu --mode=C --dt=f16:f4_e2m1:f16 --wtag=abc \
    --grouped=0:85:30+30+...(85 groups) --attr-scales=wei:7:f8_e4m3:16x1 \
    2550x3072:85x3072x2048
0:PASSED (5727 ms) __REPRO: ... --impl=grouped_gemm:micro:m_axis ...
tests:1 passed:1 ... failed:0
```

### New coverage

`TestGroupedMatmulScaleLayout` in `tests/gtests/test_iface_grouped.cpp`:

```
$ ./test_iface_grouped --engine=gpu --gtest_filter='*ScaleLayout*'
[       OK ] iface_grouped_test_t.TestGroupedMatmulScaleLayout (30 ms)
[  PASSED  ] 1 test.
```

It skips on a CPU SYCL engine, with the reason in the source: that stream submits
the primitive as a host task and always reports success, so an execution status
cannot be observed there. It runs and passes on GPU.

### Independent check for false positives

Separately from the above, the example added in #5909 uses a dense canonical
`{E, K/16, N}` `abc` scale descriptor. Built and run against this branch's
library it is unaffected:

```
Implementation     : grouped_gemm:micro:m_axis
Max relative error : 0
Result             : PASSED
```

`clang-format -style=file` applied to the touched files.

## Note

I would rather the library honored a supplied outer stride than rejected it —
that would let a producer holding `[E, N, K/group]` scales bind them zero-copy
instead of repacking. That is a larger change to the kernels
(`grouped_micro_gemm_m_axis.cl:437` hard-codes the per-expert extent while
already reading an inner stride via `ldweiq`), so this PR only converts the
silent wrong answer into an error. Happy to look at honoring strides as a
follow-up if that is a direction you would take.
