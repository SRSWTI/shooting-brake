## Summary

Adds `examples/matmul_grouped_nvfp4.cpp`, a grouped matmul example using
`f4_e2m1` weights with `f8_e4m3` block scales over groups of 16 along K.

`examples/matmul_grouped.cpp` covers grouped encoding with `f8_e4m3` operands and
`f32` scales, and there is currently no example anywhere under `examples/` that
uses `f4_e2m1` at all. The recipe itself is already supported and benchdnn-tested
(`tests/benchdnn/inputs/matmul/harness_matmul_grouped_2dby3d` enumerates both
MXFP4 and NVFP4), so this is only an examples gap.

## What the example shows

- Grouped encoding combined with 4-bit weights: `f16` src, `f4_e2m1` weights as
  logical `[num_experts, K, N]` in `acb`, `f32` dst.
- `f8_e4m3` weight scales with `set_scales(DNNL_ARG_WEIGHTS, 0b111, {16, 1},
  f8_e4m3)`, i.e. one scale per 16 elements of K per output column.
- An uneven token distribution — four experts receiving 6, 40, 0 and 18 tokens —
  so the zero-token expert is exercised rather than assumed to work.
- Why the scale tensor has to be a dense canonical `[num_experts, K/group, N]`
  `abc` tensor. Scales are described by mask and group size only, so a producer
  holding its scales as `[num_experts, N, K/group]` has to repack rather than
  alias them with a transposed descriptor. This is documented at
  `doc/functional_api/primitives/matmul.md:383-389`, and the example comment
  points at the consequence rather than restating the rule.

The example verifies its own result against a host reference that decodes the
E2M1 nibbles and the E4M3 scale bytes and accumulates in f32, and throws if the
relative error exceeds `1e-5`. An example that prints a plausible wrong answer
seemed worse than no example, particularly for a quantized recipe where a layout
mistake is silent.

To keep that reference honest without writing a general E4M3 decoder, the scale
bytes are drawn from four encodings whose values are exact (`0x30` = 0.5,
`0x38` = 1.0, `0x3C` = 1.5, `0x40` = 2.0), and the src values are small multiples
of 1/4, which are exact in f16.

Registered alongside `matmul_grouped.cpp` in the existing
`DNNL_EXPERIMENTAL_GROUPED_MEMORY` removal list in `examples/CMakeLists.txt`, so
builds with grouped memory disabled are unaffected.

## Testing

Built and run on Intel Arc Pro B70 (Battlemage G31), driver `1.15.39122+12`,
NEO `26.27.39122.12`, oneAPI 2026.1, Ubuntu 26.04, kernel `7.0.0-30-generic`.
oneDNN configured with `-DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=INTEL
-DDNNL_BUILD_TESTS=ON -DDNNL_EXPERIMENTAL=ON`.

```
$ cmake --build build --target matmul-grouped-nvfp4-cpp
[793/793] Linking CXX executable examples/matmul-grouped-nvfp4-cpp

$ ./build/examples/matmul-grouped-nvfp4-cpp gpu
Experts            : 4
Tokens per expert  : 6, 40, 0, 18 (total 64)
K, N               : 64, 32
Weight scales      : f8_e4m3, group 16 along K

Implementation     : grouped_gemm:micro:m_axis
Max relative error : 0
Result             : PASSED
Example passed on GPU.

$ ./build/examples/matmul-grouped-nvfp4-cpp cpu
Implementation     : ref_grouped:any
Max relative error : 0
Result             : PASSED
Example passed on CPU.
```

Bit-exact against the host reference on both engines. `clang-format -style=file`
applied (`.clang-format` at the repository root).

## Notes

While writing this I hit the failure mode the comment warns about: binding the
same logical scale dims with `format_tag::acb` instead of `abc` is accepted and
returns wrong results with no error status. I filed that separately as #5908 —
it is not addressed by this PR, and this example uses the correct dense layout.

Happy to rename the file, trim the verification, or move the E2M1/E4M3 decode
helpers if you would rather they lived somewhere shared.
