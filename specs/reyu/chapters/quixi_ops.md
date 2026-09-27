# Embedded QuixiCore-XPU operation families

This branch describes the callable operation layer inside the embedded QuixiCore-XPU backend. Read it after the repository HLD and before the runtime/graph chapter. The executable abstraction is [`quixi_ops.ryu`](../models/quixi_ops.ryu); the exact file correspondence is [`quixi_ops.json`](../maps/quixi_ops.json). Source, rather than comments or generated catalogs, is authoritative.

## 1. Breadth-first responsibility and boundaries

The subsystem exposes a framework-neutral C++ ABI: a caller supplies a SYCL queue, raw USM/device pointers, shapes, a storage dtype, a requested implementation `Variant`, and usually a `blocking` flag. [`ops.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/ops.hpp) contains 52 source-defined functions across activation, attention, normalization, matmul, quantization, MoE, sampling, serving, utility, optimizer, linear-attention, state-space and collective families. Dispatch translation units choose a kernel and optionally wait; semantic kernels live below `kernels/<family>/<operation>/variants/`.

The operation layer does **not** own Intel-device selection, queue construction, command-graph lifetime, PyTorch tensor lifetime, or vLLM scheduling. Those belong to the sibling runtime branch. With `blocking=false`, these void APIs discard the returned event; ordering, pointer lifetime and eventual synchronization remain caller/runtime obligations. Most blocking branches call `event::wait()`, not `wait_and_throw()`. The collective and some FP8 vendor paths are synchronously implemented exceptions.

Three inventories must remain separate:

| Layer | Meaning | Current source fact |
|---|---|---|
| [`contract/operations.hpp`](../../../src/QuixiCore-XPU/include/quixicore/contract/operations.hpp) | generated canonical naming taxonomy | 303 descriptors: 229 `observed`, 74 `planned`; no function pointers or backend capability |
| [`.quixicore/kernels.yaml`](../../../src/QuixiCore-XPU/.quixicore/kernels.yaml) | XPU maturity/variant manifest | 43 operation entries: 37 implemented, 6 experimental; 12 partial families and one capability-gated collective family |
| [`xpu/ops.hpp`](../../../src/QuixiCore-XPU/include/quixicore/xpu/ops.hpp) plus `src/dispatch` | actual source-level native ABI | 52 declarations, each defined in dispatch source |

Only 30 names join exactly from the canonical table to the native ABI. `observed` therefore does not mean XPU-callable. Twelve manifest/API names lack an exact canonical ID, and ten ABI helpers/operations lack an exact manifest key. `nvfp4_moe` is a manifest aggregate for fused and split API symbols. Canonical `layer_norm`, `linear_attention`, `qgemv`, `qgemm`/`int8_gemm`, and mean-pooling names resemble specialized ABI names but are aliases, not exact identity. Canonical `qk_norm_rope` is classified under norms while the XPU manifest classifies it under attention; canonical `nvfp4_gemm` is matmul while the manifest classifies it as quantization.

[`quant-formats.yaml`](../../../src/QuixiCore-XPU/.quixicore/quant-formats.yaml) is another independent maturity axis: GGUF is implemented; MXFP4, NVFP4, FP8, groupwise int4 and int8 are experimental; MXFP8, MXFP6, generic FP4, AWQ and BitNet are planned. An implemented operation over a format does not promote the entire format family.

## 2. Operation-family map

| Family | Public operations and responsibility | Dispatch behavior |
|---|---|---|
| Activations | GELU exact/tanh, GELU backward, SiLU, four GLU modes, fused tanh-GEGLU-to-f16, row softmax | GELU resolves variants; softmax `best` chooses SYCL for f32 and vendor for 16-bit; the rest ignore `Variant` and use SYCL |
| Attention | online SDPA/GQA, SDPA plus f16 context, symmetric SWA, NeoX RoPE, fused Q/K RMSNorm-scale-RoPE | native SYCL only; requested variants ignored |
| Norms | RMSNorm, fused residual-add RMSNorm, residual/post-norm/next-norm-to-f16, LayerNorm | RMS operations native; LayerNorm `best` is native and explicit vendor may use oneDNN |
| Matmul | row-major dense GEMM | `best` prefers compiled vendor, otherwise native 16x16 tiled baseline |
| Quantization | int4 producer/GEMV/W4A16, activation-int8/qGEMM, FP8, MXFP4, NVFP4 and 16 GGUF decoders | mostly native; FP8 has shape-dependent vendor requirements and codec errors |
| MoE | top-k route, AutoGPTQ int4 split, NVFP4 fused/split/grouped and baked-handle helpers | native; output is cleared before routed accumulation; grouped is never auto-profitable |
| Sampling | argmax, categorical and top-k sampling | native deterministic counter RNG; no range guards |
| Serving | embedding, KV scatter/gather, RMS-mean-L2 pooling | native indexed copy/pooling |
| Utils | dropout, cross entropy, Walsh-Hadamard | native |
| Optimizers | in-place AdamW | native, bias corrections computed by dispatch |
| Linear attention / SSM | normalized non-causal linear attention, fixed Qwen GDN decode, selective scan | native; GDN mutates state for valid unique slots |
| Collectives | host-facing float32 sum all-reduce over all GPUs on the largest one-platform device set | capability-discovered, synchronous, no oneCCL route |

All native numerical switches recognize f32, f16 and bf16 unless stated otherwise. Many unknown-dtype branches return a default event and perform no work; they do not throw. This differs from pooling, which throws for unsupported dtype/dimension, and from vendor conversions that can default an invalid dtype to f32. The source generally trusts pointer, shape, divisibility, alignment, enum and index preconditions.

## 3. Activations, attention, norms and dense matmul

### Activations

Native GELU computes in fp32 and casts once to storage: exact `0.5*x*(1+erf(x/sqrt(2)))` or tanh `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`. [`gelu.sycl.cpp`](../../../src/QuixiCore-XPU/kernels/activations/gelu/variants/xpu_sycl/gelu.sycl.cpp) uses 16-byte vectors and a scalar tail; [`gelu.onednn.cpp`](../../../src/QuixiCore-XPU/kernels/activations/gelu/variants/xpu_onednn/gelu.onednn.cpp) wraps caller USM directly. Backward implements the matching analytic derivative in [`gelu_backward.sycl.cpp`](../../../src/QuixiCore-XPU/kernels/activations/gelu_backward/variants/xpu_sycl/gelu_backward.sycl.cpp).

GLU input is `[rows,2*d]`, gate half then value half. The modes compute `activation(gate)*value`: SiLU, exact-erf GELU, ReLU or sigmoid. `glu_gelu_f16` deliberately uses tanh GELU and always stores f16. The model's `ReGLU` action uses an exact bounded integer instance of this implemented gate/value formula rather than a generic pass/fail toggle.

Softmax performs row max subtraction, fp32 exponential sum and normalization. Neither native nor oneDNN paths validate zero dimensions, NaNs, pointers or alignment. Common [`vec_map.hpp`](../../../src/QuixiCore-XPU/kernels/common/vec_map.hpp) defines vector widths and fp32 sigmoid/SiLU/GELU/ReLU helpers.

### Attention and position

Dense attention consumes Q `[heads,seq_q,d]`, K/V `[kv_heads,seq_k,d]`; GQA maps query head `h` to `h/(heads/kv_heads)`. It streams scores `dot(Q,K)/sqrt(d)` through the online-softmax recurrence `(m,l,acc)` and never materializes a score matrix. Causal range is end-aligned by `seq_k-seq_q`. Private arrays impose `d<=128`, but no guard enforces it; non-divisible/zero head counts and `seq_q>seq_k` are also unguarded. `attention_f16ctx` duplicates the algorithm and unconditionally emits both storage-dtype O and f16 O.

SWA uses the same recurrence but a **symmetric**, not causal, band centered at `qi+(seq_k-seq_q)`. `window==0` is dense. Its half-open bounds make an odd window contain `window` keys away from edges and an even nonzero window contain `window+1`; this is a source-level wording contradiction worth preserving. Its fixed arrays require `d<=256` without validation.

NeoX RoPE rotates `(i,i+d/2)` by `(pos0+t)*base^(-2i/d)`. Odd dimensions leave the last component unwritten. Fused QK norm/RoPE first applies per-head RMS normalization, learned weights, query-only scale, then the same rotation in place and optionally emits f16 copies.

### Norms and matmul

RMSNorm and LayerNorm use fp32 reductions with storage-dtype output. LayerNorm uses `E[x^2]-E[x]^2` without a nonnegative clamp. Fused RMS paths have load-bearing rounding details: fused add stores the residual in T and rounds a normalized value before multiplying weight; `rms_residual_next` computes its second scale from an unrounded updated residual but rereads the rounded stored residual for f16 output. A CPU oracle must reproduce those boundaries rather than compare to an algebraically simplified composition.

LayerNorm is the only assigned oneDNN family whose implementation catches `dnnl::error`; it optionally logs and submits native SYCL. Other vendor construction/execution failures propagate. Dense native GEMM computes row-major `C[M,N]=A[M,K]B[K,N]` with zero-padded 16x16 tiles and fp32 accumulation; the vendor path wraps row-major USM and overwrites C. The model's `Dense2` action is one exact output cell and exposes `best` selecting vendor only when compiled.

## 4. Quantized operation mechanics

### Groupwise signed int4 and int8

Int4 quantization uses `[N,K]`, scale `max(abs(group))/7` or 1 for an all-zero group, round/clamp to `[-8,7]`, fp16 scales, and two's-complement nibbles with even K in the low nibble. GEMV sign-extends the nibble and computes `sum(q*scale*x)` in fp32. W4A16 consumes the same bytes/scales and runs `[M,K] * dequant([N,K])^T` with 8x16x16 joint-matrix tiles. W4A16 accepts only f16/bf16; f32 returns an empty event. The model checks a bounded quantization clamp, a two-element dequantized GEMV, and the f32 silent-no-work branch.

Activation int8 chooses a per-row scale `max(abs(x))/127` or 1 and emits only `[-127,127]`. qGEMM accumulates signed-byte products in int32, then multiplies per-row activation and per-column weight scales. The oneDNN path expresses the latter as weight scales and a broadcast `[M,1]` binary post-op; oneDNN errors fall back to native.

### MXFP4 and NVFP4

MXFP4 stores two E2M1 values per byte and one E8M0 power-of-two scale per 32 values. The magnitude table is `{0,.5,1,1.5,2,3,4,6}` and dequantization is `E2M1*2^(scale-127)`.

NVFP4 stores packed E2M1 plus one E4M3 scale per 16 values and one fp32 global scale. Bit relocation decodes E2M1 and E4M3 with factors `2^-14` and `2^-8`; multiplying the global scale by `2^22` compensates exactly. Public NVFP4 GEMM deliberately launches one optimized GEMV per activation row because the checked-in decode-once M-tiled research kernel regressed at measured shapes. It returns only the last row event; on an out-of-order queue, waiting only that event would not explicitly depend on earlier independent rows.

### FP8

Native FP8xFP8 is only the `M==1` decode-GEMV path when the request is not explicit vendor. Every other `fp8_gemm` route requires the oneDNN build—even explicit `Variant::sycl` at `M>1`—and throws when oneDNN is absent or reports unsupported FP8 matmul. E4M3 relocation needs `2^8` per operand; native FP8xFP8 folds `2^16` into the final scale. E5M2 relocates directly.

FP8 W8A16 uses checkpoint-native weight `[N,K]`, native f32 accumulation and scalar/per-channel scale. Explicit vendor or `best` with `M>1` tries oneDNN; absence/failure falls back to native. FP8 encode/decode are synchronous oneDNN reorders, ignore `blocking`, and throw without the vendor build. The model's named scenarios preserve the surprising `M>1 + explicit sycl + no vendor = error` branch.

### GGUF

[`gguf_gemv.sycl.cpp`](../../../src/QuixiCore-XPU/kernels/quantization/gguf_gemv/variants/xpu_sycl/gguf_gemv.sycl.cpp) implements type codes 0–15: q8_0, q4_0, q6_K, q4_K, q5_K, q2_K, q3_K, IQ4_NL, q4_1, q5_0, q5_1, IQ4_XS, IQ2_XXS, IQ2_XS, IQ3_XXS and IQ1_S. Ordinary q4/q5 formats combine fp16 scales/minima with packed low/high bits; K-quants use 256-value blocks and packed subscales/minima; i-quants use the checked-in nonlinear grids/sign tables from [`gguf_iq_tables.hpp`](../../../src/QuixiCore-XPU/kernels/quantization/gguf_gemv/gguf_iq_tables.hpp). Tables used by the kernel are uploaded lazily into a process-global, never-freed device cache that is not keyed by context/device. Unknown nonzero type codes silently take q4_0. `gguf_kernel.hpp` documents only codes 0 and 1 and is stale relative to the 16-code implementation.

Packing divisibility is generally trusted, not checked: producer-compatible int4 group boundaries, MX/NVFP4 K multiples, FP8 W8A16 16-value chunks, and GGUF 32- or 256-value blocks are required for complete results. Malformed dimensions can truncate tails or corrupt row strides.

### Declared but not implemented

[`turboquant_kernel.hpp`](../../../src/QuixiCore-XPU/kernels/quantization/turboquant/turboquant_kernel.hpp) declares internal TurboQuant encode/decode contracts but has no assigned definition, public `ops.hpp` entry or manifest operation. The generated canonical IDs `turboquant` and `paged_attention_turboquant` are catalog observations, not executable exports. The model reaches an explicit unavailable-export witness.

## 5. Routing, MoE and recurrent state

Top-k routing repeatedly scans experts in ascending ID order, selects only on strict greater-than, and therefore resolves finite ties to the lowest remaining ID. It softmax-normalizes only selected logits. The fixed local arrays allow at most 16, but `k` is unchecked; zero, too-large, empty/all-NaN and exhausted expert domains can read invalid storage or repeat expert 0. The model's top-1 tie scenario encodes the actual lower-ID rule and exact selected weight 1.

Every routed expert computes gate/up dot products, `SiLU(gate)*up`, a down projection and a weighted addition into fp32 output. Invalid expert IDs are skipped. Public dispatch first clears output. AutoGPTQ int4 uses `(nibble-8)*scale`, no qzeros pointer, and scratch `[M*top_k,I]` containing post-SwiGLU values. NVFP4 fused materializes one route in local memory; split stores raw `[route,2I]` projections in caller scratch; both use relaxed fp32 atomic additions, so route summation order is not fixed.

Grouped NVFP4 counts valid routes, performs a padded expert-exclusive scan, scatters route IDs, runs f16/bf16 DPAS gate/up and down tiles, and atomically scatters results back. Workspace regions are caller-owned and 256-byte rounded. f32 activation falls back to fused rather than narrowing. The public profitability function returns false unconditionally after the recorded comparison lost, overriding the header's approximate crossover comment; explicit calls remain reachable. Baked Level-Zero handle extraction supports I 512/1024/2048 and catches all failures as `false`.

Linear attention forms per-head `KV=sum K^T V` and `z=sum K`, then `O=(Q KV)/(Q dot z+1e-6)` using all positions (non-causal). Fixed local storage requires `dim<=64` without enforcement. Selective scan starts h at zero on every call and runs, per channel/state, `h=exp(delta*A)h+delta*B*u`, `y=C dot h+D*u`; a fixed register array requires state <=16 without a guard. The model abstracts the exponential coefficient as an integer `decay` while preserving this exact transition/data dependency.

Qwen GDN decode has fixed Q/K 16x128, V/Z 32x128, convolution width 4 and projected widths. A valid unique state slot shifts the 3-tap convolution state, applies SiLU, normalizes q/k, computes decay/beta and mutates `[32,128,128]` SSM state. Invalid slots leave both states untouched, zero mixed/core output and still copy projected z. A bad state or dt-bias dtype can occur after the convolution submission, so failure is not transactional. The model includes the invalid-slot preservation witness.

## 6. Sampling, serving, utilities, optimizer and collectives

[`rng.hpp`](../../../src/QuixiCore-XPU/kernels/common/rng.hpp) supplies a stateless PCG hash; `uniform01(seed,index)` uses the upper 24 bits, so dropout is keyed by element and sampling by row. Argmax uses lowest index on finite ties. Categorical sampling follows ascending-token inverse CDF. Top-k repeatedly selects descending finite logits with lower-index ties into fixed arrays of 64, then samples selected values. Temperature, k and vocab are unvalidated.

Embedding and KV gather/scatter are bitwise row copies, not numerical conversion. Negative indices skip the write—including gather, which leaves output untouched. Positive indices are unchecked; duplicate scatter destinations race. Fused pooling permits only dimensions 256/512/768/1024, applies per-token RMSNorm, sequence mean and final L2 normalization in fp32, yields zero for an empty sequence, and throws for unsupported dim/dtype. Offset monotonicity/bounds are only documented.

Dropout implements inverted scaling, cross entropy uses stable log-sum-exp, and Hadamard performs the unnormalized butterfly in fixed 2048-float local memory. Probability, targets, power-of-two and size limits are not guarded. AdamW updates p/m/v in place with dispatch-computed bias corrections; step zero and hyperparameter domains are trusted.

`all_reduce_sum` chooses the platform with the most GPU devices, returns 0 without writing output if none exist, stages host slices, serially reduces on GPU 0, copies to host and submits peer broadcasts. Broadcast events are discarded before peer buffers are freed; explicit synchronization covers reduction and host result, not those broadcast/free pairs. Exceptions propagate and partial allocation has no cleanup guard.

## 7. Executable model scenarios and properties

[`quixi_ops.ryu`](../models/quixi_ops.ryu) first exposes the whole branch as a family/operation submission, then deepens representative functional transitions:

* `regluSuccessTest` checks the exact ReGLU gate/value formula and that this native-only op ignores a vendor request.
* `denseBestVendorTest` checks one output-cell dot product and `best` vendor resolution.
* `int4ClampTest` and `int4GemvTest` check bounded signed-int4 clamp/dequant/dot structure.
* `routeTieUsesLowerExpertTest` checks source-order tie selection and top-1 normalization.
* `boundedNegativeScatterTest` checks negative-index no-write behavior; `negativeScatterSkipsTest` separately demonstrates the model's bounded input guard.
* `selectiveScanCellTest` preserves recurrence then output ordering.
* `invalidGdnSlotPreservesStateTest` checks untouched recurrence, zero core and passed-through z.
* `fp8BatchNeedsVendorTest` and `fp8DecodeNativeTest` distinguish vendor-required batch from native M=1.
* `w4a16F32NoWorkTest` captures the empty-event/no-work dtype branch.
* `groupedNotAutoSelectedTest` captures unconditional unprofitability.
* `generatedTurboQuantIsNotImplementedTest` prevents generated catalog presence from becoming a false implementation claim.

`inv` relates queue phase to pending outcome, successful completion to finished phase, int4 bounds, route range/weight, grouped non-selection and missing TurboQuant. `successWitness`, `tieBreakWitness`, `skippedIndexWitness`, `unsupportedDtypeWitness`, `missingVendorWitness`, `invalidStateWitness` and `unavailableExportWitness` are reachable observations. The model is bounded and atomic at one kernel launch/completion; it does not model fp16/bf16 rounding, floating transcendental error, work-group scheduling, atomics order, pointer aliasing or device exceptions.

## 8. Exact assigned-source roles

The map lists every assigned file individually. The source grouping below explains their roles without treating presence as behavioral proof:

* `src/dispatch/{activations,attention,collectives,linear_attention,matmul,moe,norms,optimizers,quantization,sampling,serving,ssm,utils,variants}.cpp` are public ABI dispatch, variant routing, waits, output clears and explicit errors.
* Every assigned `*_kernel.hpp` is an internal declaration/layout contract. `turboquant_kernel.hpp` is declaration-only and unwired; `gguf_kernel.hpp` is incomplete relative to its implementation.
* Every assigned `variants/xpu_sycl/*.sycl.cpp` is executable native kernel behavior. Optional assigned `xpu_onednn/*.onednn.cpp` files implement GELU, softmax, LayerNorm, dense GEMM, int8 qGEMM and FP8 vendor paths.
* `common/vec_map.hpp` and `common/rng.hpp` are shared functional dependencies. `nvfp4_dequant.hpp` and `gguf_iq_tables.hpp` are format/decode dependencies rather than public exports.
* `kernels.yaml` and `quant-formats.yaml` are declarative maturity inventories; generated `operations.hpp` is a canonical naming artifact. None dispatches a kernel.

## 9. CPU comparisons and limits

No GPU numerical test is claimed here. Useful parent-run comparisons are: scalar exact/tanh GELU and gradients; materialized dense/causal/SWA attention; RoPE; norm formulas with the implementation's rounding points; row-major GEMM; int4 producer-to-consumer round trip; int8 scaled dot; exhaustive E2M1/E4M3/FP8 decode; all 16 GGUF blocks against ggml; route tie/invalid-ID handling; independent per-route versus grouped NVFP4; PCG masks and inverse-CDF sampling; indexed copies with sentinels; selective-scan/GDN state transitions; and host sums over the selected GPU count. These are proposed differential checks, not executed evidence. oneDNN, SYCL, Level Zero device capabilities and external CPU references remain dependencies.
