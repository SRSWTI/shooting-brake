# Summary

Grouped matmul on GPU returns wrong results when a `binary_mul` post-op is
combined with `bf16` operands and `acb`-tagged weights. Roughly 80% of output
elements are wrong. The primitive reports success.

Switching any one of three things makes it correct again — the weights tag, the
post-op operand data type, or the main data type — which is what makes me think
this is a specific bad path rather than a general post-op problem.

Found while regression-testing an unrelated change; these two cases are already
failing in `test_matmul_grouped_ci` at current main, so CI would show it on a GPU
runner.

# Version

- oneDNN at `d22de940f3`, `libdnnl.so.3.14`
- Configured with `-DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=INTEL -DDNNL_BUILD_TESTS=ON -DDNNL_EXPERIMENTAL=ON`
- `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=1`

# Environment

- Intel Arc Pro B70 (`Battlemage G31`, `8086:e223`), driver `1.15.39122+12`
- NEO / compute-runtime `26.27.39122.12`, Level Zero loader `libze1 1.28.6`
- oneAPI 2026.1, kernel `7.0.0-30-generic`, Ubuntu 26.04 LTS
- `SYCL_UR_USE_LEVEL_ZERO_V2=0`
- Implementation selected in every case below: `grouped_gemm:micro:m_axis`

# Steps to reproduce

These two are already in the shipped test batch
(`tests/benchdnn/inputs/matmul/test_matmul_grouped_ci` →
`harness_matmul_grouped_2dby3d`):

```
benchdnn --matmul --engine=gpu --mode=C --dt=bf16:bf16:bf16 \
  --grouped=0:4:8+8+8+8 --wtag=acb --attr-post-ops=swish:1+mul:f16:1 \
  32x64:4x64x32

benchdnn --matmul --engine=gpu --mode=C --dt=bf16:bf16:bf16 \
  --grouped=0:4:8+0+16+8 --wtag=acb --attr-post-ops=swish:1+mul:f16:1 \
  32x64:4x64x32
```

Minimal form — the `swish` is not needed:

```
benchdnn --matmul --engine=gpu --mode=C --dt=bf16:bf16:bf16 \
  --grouped=0:4:8+8+8+8 --wtag=acb --attr-post-ops=mul:f16:1 \
  32x64:4x64x32
```

# Observed behavior

```
0:FAILED (errors:813 total:1024) __REPRO: --engine=gpu --matmul \
  --impl=grouped_gemm:micro:m_axis --dt=bf16:bf16:bf16 --grouped=0:4:8+8+8+8 \
  --wtag=acb --attr-post-ops=mul:f16:1 32x64:4x64x32
```

Isolation matrix, all at `--dt=bf16:bf16:bf16 --grouped=0:4:8+8+8+8
32x64:4x64x32` unless noted:

| configuration | result |
|---|---|
| `--wtag=acb --attr-post-ops=mul:f16:1` | **FAILED, errors 813/1024** |
| `--wtag=acb --attr-post-ops=mul:bf16:1` | **FAILED, errors 813/1024** |
| `--wtag=acb --attr-post-ops=mul:f16:0` | **FAILED, errors 751/1024** |
| `--wtag=acb --attr-post-ops=mul:f16:1`, `--dt=bf16:bf16:f32` | **FAILED, errors 824/1024** |
| `--wtag=acb --attr-post-ops=swish:1+mul:f16:1` | **FAILED, errors 407/1024** |
| `--wtag=abc --attr-post-ops=mul:f16:1` | PASSED |
| `--wtag=acb --attr-post-ops=mul:f32:1` | PASSED |
| `--wtag=acb --attr-post-ops=mul:f16:1`, `--dt=f16:f16:f16` | PASSED |
| `--wtag=acb --attr-post-ops=swish:1` | PASSED |
| `--wtag=acb`, no post-ops | PASSED |

So, reading the passing rows:

- the eltwise post-op alone is fine, so this is specific to the binary post-op
- an `f32` binary operand is fine; `f16` and `bf16` operands both fail
- `abc`-tagged weights are fine; only `acb` fails
- `f16` main data type is fine; only `bf16` fails
- a non-16-bit destination does not help (`bf16:bf16:f32` still fails)

`add:f16:1` and `div:f16:1` report `UNIMPLEMENTED` on this path, so `mul` is the
only binary post-op I could exercise.

# Expected behavior

Same results as the `--wtag=abc` case, or the same results as with an `f32`
binary operand. Since the reference and the `abc` path agree, the `acb` +
16-bit-operand combination appears to be the outlier.

# Additional information

The error counts being ~80% rather than 100% suggests some tiles are computed
correctly, which would be consistent with the binary operand being addressed with
the wrong stride or element size rather than a wholesale wrong pointer. I have
not gone into the kernel to confirm that, so treat it as a guess.

Happy to run further isolation on this hardware if that is useful — different
group counts, `--stag`/`--dtag` variants, or a narrowed shape.
