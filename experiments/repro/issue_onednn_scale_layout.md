# Summary

Grouped matmul requires weight scales to be a dense canonical `[E, K/group, N]`
`abc` tensor. That is documented. What is not handled is the case where the
caller's scale memory has a different layout: the primitive accepts it, runs, and
returns silently wrong results rather than reporting an error.

The optimized `grouped_gemm:micro:m_axis` implementation does read one stride
from the scales execution memory descriptor, but hard-codes the per-expert
(outer) stride as dense. So a non-canonical scale tensor is partially honored,
which is exactly what makes the failure quiet instead of loud.

Concretely: binding an `acb`-tagged scale memory over logical `[E, K/16, N]`
produces a max relative error of **9.53e+01** against a scalar reference, with no
warning and no error status. The same run with a dense `abc` scale tensor is
bit-exact.

I'd like either a validation error or honored strides. Details and repro below.

# Version

- oneDNN at `d22de940f3` (vendored), built with
  `-DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=INTEL -DDNNL_BUILD_TESTS=ON -DDNNL_EXPERIMENTAL=ON`
- `libdnnl.so.3.14`
- `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=1`

# Environment

- Intel Arc Pro B70 (`Battlemage G31`, `8086:e223`), driver `1.15.39122+12`
- NEO / compute-runtime `26.27.39122.12`, Level Zero loader `libze1 1.28.6`
- oneAPI 2026.1, kernel `7.0.0-30-generic`, Ubuntu 26.04 LTS
- `SYCL_UR_USE_LEVEL_ZERO_V2=0`

# Steps to reproduce

First, for context, the working configuration. This passes and dispatches to the
optimized grouped kernel:

```
ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=1 ./benchdnn --matmul --engine=gpu --mode=C \
  --dt=f16:f4_e2m1:f16 --wtag=abc \
  --grouped=0:85:30+30+...+30 \
  --attr-scales=wei:7:f8_e4m3:16x1 \
  2550x3072:85x3072x2048
```

```
0:PASSED (10752 ms) __REPRO: --engine=gpu --matmul --impl=grouped_gemm:micro:m_axis ...
| grouped_gemm:micro:m_axis : 1 (100%)                     |
tests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0
```

Now the problem. In a small SYCL program, build the same grouped f4_e2m1 matmul
(f16 src, f16/f32 dst, `attr.set_scales(DNNL_ARG_WEIGHTS, (1<<0)|(1<<1)|(1<<2),
{16,1}, f8_e4m3)`), and construct the scales execution memory two ways over the
same logical dims:

```cpp
auto scl_md = memory::desc({E, K / 16, N}, dt::f8_e4m3,
                           scale_acb ? tag::acb : tag::abc);
```

Run both against a scalar f64 dequant reference.

# Observed behavior

```
device: Intel(R) Arc(TM) Pro B70 Graphics
impl: grouped_gemm:micro:m_axis | low-first err 0.000e+00, high-first err 1.277e+02
canonical scales, f32 dst          PASS  (max rel err 0.000e+00)
impl (f16 dst): grouped_gemm:micro:m_axis
canonical scales, f16 dst          PASS  (max rel err 4.781e-04)
strided acb scale md honored: no (repack at load) (err 9.530e+01)
offsets as upper bound             PASS  (max rel err 0.000e+00)
padded tail untouched              PASS  (max rel err 0.000e+00)
```

The `acb` case returns `dnnl_success` at every step — primitive descriptor
creation, primitive creation, and execute. The only symptom is a wrong answer.

Where this comes from, as far as I can trace it:

- There is no API through which a caller can declare a scale layout.
  `dnnl_primitive_attr_set_scales(attr, arg, mask, group_ndims, group_dims,
  data_type)` (`include/oneapi/dnnl/dnnl.h:463-489`) carries no descriptor, and
  `quant_entry_t::get_md()` synthesizes a canonical one:
  `CHECK(memory_desc_init_by_tag(out_md, ndims, quant_dims, data_type_, get_abx_tag(ndims)));`
  (`src/common/primitive_attr_quant.hpp:135-136`).
- The requirement is documented at
  `doc/functional_api/primitives/matmul.md:383-389` — "Tensors must use dense
  memory descriptors" — and for K-grouped weights, "Scale tensor:
  [num_groups, K/32, N] - dense 3D tensor / Layout: standard abc layout."
- But the optimized path *does* consult strides, partially:
  `src/gpu/intel/matmul/grouped_micro_gemm.cpp:852-875` takes
  `ldweiq = ...strides[1]` from the actual execution memory descriptor and
  passes it to the kernel (`:895-918`), while
  `src/gpu/intel/matmul/grouped_micro_gemm_m_axis.cl:437` advances experts with
  a hard-coded dense extent:
  `wei_attr_scales += batch * n * (k / WEI_GROUP_SIZE);`
- The reference path ignores strides entirely:
  `src/gpu/intel/matmul/ref_grouped_gemm.cl:170-173` indexes
  `wei_scales[group_id * wei_scale_ngroups_k * N + (...) * N + n]`.
- I could not find any density or stride validation for these descriptors. Not
  in `src/gpu/intel/matmul/ref_grouped_gemm.hpp:112-159` (which does validate
  masks, dtypes and group divisibility), not in
  `src/common/primitive_exec_types.cpp:78-151`, not in
  `src/common/primitive_iface.cpp:244-275`. If one exists and I missed it,
  please point me at it and disregard this part.

# Expected behavior

Either of the first two would resolve it; the third would at least make it
diagnosable:

1. **Reject it.** Validate the scales execution memory descriptor against the
   canonical descriptor `quant_entry_t::get_md()` derives, and return
   `dnnl_invalid_arguments` on mismatch. Cheapest fix, and it converts silently
   wrong numerics into an actionable error.
2. **Honor it.** Take the outer/expert stride from the descriptor in
   `grouped_micro_gemm_m_axis.cl:437` the way the inner stride is already taken
   via `ldweiq`, and do the same in the reference kernel.
3. **Document the sharper version.** The current wording says scale tensors must
   be dense. It would help to say explicitly that strides on the scales
   *execution* memory are not consulted for the outer dimension, so a
   non-canonical layout will produce incorrect results rather than an error.

# Why this comes up in practice

NVFP4 MoE checkpoints in the NVIDIA ecosystem (ModelOpt, compressed-tensors)
commonly ship per-expert scale planes as `[E, N, K/16]` — N-major with the
K-group index contiguous — because that is the layout the matching packed
`[E, N, K/2]` weight tensor wants. oneDNN wants `[E, K/16, N]`.

A dense `acb` descriptor over logical `[E, K/16, N]` is physically exactly the
`[E, N, K/16]` bank, so this looked like it should be a zero-copy bind, which is
why I tried it. It is not, and it fails quietly.

The workaround is a one-time transpose into canonical `abc`, which is fine
functionally but costs device memory: for 85 experts x 47 layers at our shapes
that is an extra +2.35 GB per card held purely as a second copy of the scales.
If (2) were possible, that copy goes away.

For reference, the canonical path performs well — the two grouped legs of one MoE
layer (85 experts, 30 rows/expert; `2550x3072:85x3072x2048` and
`2550x1024:85x1024x3072`) total **0.970 ms** with `dst=f16` and **1.013 ms** with
`dst=f32`, both on `grouped_gemm:micro:m_axis`. No complaint about the
implementation at all; only about what happens when the scale layout does not
match what it assumes.
