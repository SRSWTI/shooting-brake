# Banks and authenticated references

This branch explains how Shooting Brake turns checkpoint tensors into provider- and prefill-native banks, and how it establishes numerical reference provenance. Read it breadth first: responsibility and boundaries, then format distinctions, then builders and validators, and finally the Phase-3 oracle. The executable abstractions are [`banks.ryu`](../models/banks.ryu) and [`reference.ryu`](../models/reference.ryu). Source is authoritative; the models deliberately do not pretend that finite symbolic state proves tensor numerics.

## 1. Responsibility and actors

At system level, these files provide immutable inputs to model admission, provider load and prefill streaming:

```text
frozen Phase-0 claims and workloads
                |
        checkpoint snapshots
         /              \
 BF16 source       NVFP4 / GPTQ source
         |             /       |       \
         |      SBEXP001   SBINT401   SBB12X01
         |                    |
         |                 SBMARL01
         |                    |
         +--- authenticated Phase-3 fixture/oracles
                              |
                    provider wire comparison
```

The actors have intentionally different authority:

* **Phase 0 records evidence and intended compatibility.** [`freeze.yaml`](../../../src/phase0/freeze.yaml) and [`capability_manifest.yaml`](../../../src/phase0/capability_manifest.yaml) pin checkpoint IDs, geometry, hashes and a historical provider contract. They are documents/configuration, not code that authenticates a file at runtime.
* **Phase 1 builders own extraction and transformation.** They derive or validate geometry, stream bounded working sets, serialize a format and apply format-specific checks.
* **Phase 4 format modules own the canonical Python ABIs for SBINT401, SBMARL01 and SBB12X01.** Phase-1 builders import them rather than defining a second header convention. `BankSource` owns the host mapping/pinning choice used by the sidecar consumers.
* **Phase 1 validators provide focused byte and small numerical checks.** They are useful but do not bind as much identity or cover as much of the semantic matrix as Phase 3.
* **Phase 3 authenticates provenance and supplies an independent oracle.** It binds exact snapshots, the full bank and fixture hashes, selected bank bytes, deterministic inputs and two float64 output families. Its provider test then compares the real wire result with a source-defined error budget.

These files do **not** own model placement, runtime routing, provider device kernels or the vLLM scheduler. A valid bank is necessary but not sufficient for a valid serving configuration; admission and placement additionally decide which layer/expert identities may consume it.

## 2. State and operation map

| State boundary | Source operation | Observable success | Fail-closed cases represented in the model |
|---|---|---|---|
| Evidence recorded | Phase-0 YAML/JSON and benchmark outputs | Values are available for comparison | No authentication is implied merely by presence |
| Format selected | builder CLI and canonical format module | One ABI controls later field interpretation | Unknown magic/version, unsupported quantization family |
| Geometry accepted | `discover_shape`, `Int4BankHeader.validate`, `MarlinBankHeader.validate`, B12x geometry creation | dimensions, planes, strides and resident identities are coherent | non-divisible shapes, gaps, duplicate/out-of-range IDs, wrong plane layout |
| Payload written | per-layer builder loop | format-specific packed planes and scale convention emitted | missing tensor, wrong shape/dtype, gate/up mismatch, nontrivial `g_idx`, foreign zero point, B12x underflow assumption violation |
| Artifact validated | size/header checks, byte comparisons or fresh repack | payload agrees with its declared/source contract | truncated/wrong-size file, mismatching checkpoint bytes, transformed sidecar mismatch |
| Published | rename or completion of direct output | consumer can open the destination | builders differ in crash publication semantics; see below |
| Reference identity authenticated | `generate_reference.py` and `ReferenceFixture::open` | fixed hashes, schema, IDs, offsets, padding and finite values agree | any identity, layout, digest or finite-value mismatch |
| Oracle comparison accepted | `compute_artifact_metrics`, `output_budget`, provider harness | quality gates and per-element runtime budgets pass | excessive source-vs-NVFP4 error, stale identity, failed dispatch, poisoned output modification |

[`banks.ryu`](../models/banks.ryu) makes format, geometry, source identity, payload, resident map, scale contract, bit comparison and publication mode explicit. [`reference.ryu`](../models/reference.ryu) prevents Phase-0 evidence from becoming trusted oracle state before Phase-3 authentication and byte audit.

## 3. Four formats that must not be conflated

### 3.1 SBEXP001: legacy/full NVFP4 provider bank

[`extract_experts.py`](../../../src/phase1/extract_experts.py) owns this 60-byte packed header (`<8sIIIIIQQQQ`, magic `SBEXP001`). The header describes bank row count, experts per row, hidden/intermediate sizes, one reserved field and four plane byte counts. It carries **no source layer IDs and no source expert-ID list**. Therefore identity is partly external: extraction requires one contiguous NVFP4 source-layer run, and independent validation reconstructs `bank row -> sorted source NVFP4 layer`. A run starting above zero is permitted, but the operator should pin the intended first model layer when validating.

Each expert record is exactly:

1. packed E2M1 gate then up weights (`w13`, checkpoint order `[gate; up]`),
2. raw E4M3 gate then up block scales,
3. packed E2M1 down weights,
4. raw E4M3 down block scales,
5. two little-endian fp32 multipliers, `1 / gate_global` and `1 / down_global`.

Gate and up must have exactly equal global scales because one fused multiplier represents both. The source notes that vLLM may reorder its own VRAM copy to `[up; gate]`; that runtime order is not the bank order. Hidden and intermediate dimensions must be divisible by the NVFP4 block size 16. MTP shards are excluded so speculative-head tensors cannot masquerade as an extra model layer. The builder bounds checkpoint tensor residency to one layer, verifies the final byte count, writes a temporary file and uses `os.replace`.

The frozen 35B geometry is H=2048, I=512, 256 experts and 32 NVFP4 layers. The eight later expert layers are FP8 and remain outside this artifact. Per expert, the record is 1,769,480 bytes; the frozen full bank is 14,495,580,220 bytes with SHA-256 `0ce6377b…e8db`.

### 3.2 SBINT401 v2: sparse-resident AutoGPTQ bank

[`int4_bank_format.py`](../../../src/phase4/src/shooting_brake_vllm/int4_bank_format.py) is canonical. Its 128-byte fixed prefix is followed by an explicit ordered `int32` source-expert list and zero padding to 4096 bytes. Version 2 requires one strictly increasing, unique resident set shared across all resident layers. It records both resident and source layer/expert counts, so compact slot identity is not inferred from a prefix. Six contiguous 4096-aligned per-expert planes are stored:

`gate_qweight`, `gate_scales`, `up_qweight`, `up_scales`, `down_qweight`, `down_scales`.

The admitted contract is symmetric int4, group size 128, effective zero point 8. Qweight is copied verbatim as int32-packed nibbles and scales as fp16. The extractor accepts two uniform qzero encodings—AutoGPTQ v1 `0x77777777` (stored zero-point-minus-one) and GPTQModel v2 `0x88888888`—because both represent effective zero point 8. A tensor containing anything else, including a mixture, is rejected. If a `g_idx` tensor exists, one sample for each projection must be the trivial group ramp; otherwise copying qweight verbatim would preserve the wrong K ordering.

[`extract_experts_int4.py`](../../../src/phase1/extract_experts_int4.py) accepts an all-layer or layer-prefix selection plus a count/range/sorted list of resident source experts. It validates the complete checkpoint layer/expert key domain, the quantization family (`auto_round:auto_gptq` or non-desc-act GPTQ), dimensions, plane shapes/dtypes, zero points and `g_idx`. It can stream a full dry run to a page-touching counted `/dev/null`, optionally record packed-zero statistics, build through a temporary file and rename, sample exact bank-vs-shard bytes/dequantization, and report a separate NVFP4 cross-format comparison. The cross-format comparison is diagnostic evidence; it does not weaken the int4 byte contract.

The phase4 round-trip harness demonstrates a tiny real extraction with H=3072, I=1024: data offset 4096, six expected plane sizes, 4,866,048 bytes per expert, explicit IDs `0..7`, and correct plugin reader/admission views. It also confirms a nonresident expert lookup is rejected.

### 3.3 SBMARL01: transformed Marlin prefill sidecar

[`marlin_bank_format.py`](../../../src/phase4/src/shooting_brake_vllm/marlin_bank_format.py) defines a version-1 128-byte prefix plus strictly ascending source IDs and padding to 4096. It is derived from SBINT401 but is not interchangeable with it. Each layer is a contiguous arena:

* `m13 [E, K/16, 4I] int32`, fused gate/up Marlin qweight,
* `m2 [E, I/16, 2K] int32`, down Marlin qweight,
* `s13 [E, K/128, 2I]`, permuted gate/up scales,
* `s2 [E, I/128, K]`, permuted down scales.

The scale element type is itself ABI state: fp16 or bf16 must match the serving activation dtype, because the source records that a mismatch silently produced zeros. Header validation re-derives plane geometry and layer stride, requires int4/zp8, validates scale dtype and ascending IDs, and `read_marlin_bank_header` checks exact file size.

[`build_marlin_bank.py`](../../../src/phase1/build_marlin_bank.py) mmap-reads each strided SBINT401 layer, copies one layer to a reused device slab, repacks into one reused device arena, copies through one reused host buffer and `madvise(DONTNEED)`s consumed input pages. It guards anonymous RSS at 4 GiB. Gate and up are repacked separately into the fused plane; this is source-documented as bit-exact because I=1024 aligns with Marlin's N tiling. Selected output layers are compared byte-for-byte with a fresh repack.

Publication differs from the other builders: the current implementation opens the final path with `O_TRUNC`, writes it in place and `fsync`s. It does **not** use a temporary path plus atomic rename. The Reyu model retains that distinction rather than idealizing all formats as crash-atomically published.

### 3.4 SBB12X01: native FP4/FlashInfer B12x sidecar

[`b12x_bank_format.py`](../../../src/phase4/src/shooting_brake_vllm/b12x_bank_format.py) defines magic `SBB12X01`, version 1 and a 200-byte header padded to a 4096-byte data offset. The leading module description says “four planes,” but the live header and geometry contain **six**: `w1`, `w2`, `sf1`, `sf2`, `alpha1`, `alpha2`. The six-plane code is authoritative.

`w1` contains packed FP4 gate then up bytes and `w2` contains down bytes. `sf1` and `sf2` are E4M3 scale-factor tensors converted into the stored MMA-swizzled shapes. The two fp32 alpha planes are essential. A naive `block_scale * weight_scale_2` bake flushes most ModelOpt products below E4M3's minimum subnormal. [`build_b12x_bank.py`](../../../src/phase1/build_b12x_bank.py) instead stores gate/up block scales multiplied only by their ratio to `max(gate_scale_2, up_scale_2)`, places that maximum in `alpha1`, leaves down block scales verbatim and places the down global factor in `alpha2`. The kernel applies alpha pre-activation. The builder rejects a layer when more than 0.5% of nonzero ratio-baked `sf1` values still fall below the E4M3 floor.

The builder is intentionally specialized: 48 layers, H=3072, I=1024, group 16, and source experts 54 through 179 (126 residents) in provider order. It learns the actual swizzled shapes from a conversion of layer 0, writes every plane with 4096 padding to a temporary file, renames, and can recompute and bit-compare a requested layer prefix.

The B12x reader is weaker than the int4 and Marlin readers: `read_b12x_bank_header` preads 200 bytes and `parse_header` checks magic/version, but does not itself re-derive all geometry or check the file's implied size. Consumers/build validation must provide the remaining assurance. This is a source-visible asymmetry, not something the model hides.

## 4. Host mapping is a separate contract

[`bank_source.py`](../../../src/phase4/src/shooting_brake_vllm/bank_source.py) exposes Marlin/B12x plane data either as pageable `np.memmap` or as a post-fork private writable mmap registered with CUDA. Registered construction maps only the header-plus-data extent, advises `WILLNEED`, creates a byte tensor over the aligned data region, and calls `cudaHostRegister`. A registration error is fatal: the source explicitly refuses pageable fallback because the failed runtime call may poison the next CUDA check. Construction must occur in the worker after fork. This mapping choice changes transfer behavior and lifetime, not the bank's logical plane geometry.

## 5. Build and validation workflow

### NVFP4 extraction and exactness

`extract_experts.py` discovers a single expert key root and a contiguous set of packed layers, checks shape divisibility, groups tensor reads by shard and serializes every expert in deterministic layer/expert order. `validate_expert_bank.py` independently rediscovers the root and sparse layers, verifies exact header-implied file size and row count, checks first/last corner records plus deterministic interior samples, and reports the first mismatching plane. It additionally probes neighboring layers to diagnose a ±1 row shift. Because SBEXP001 lacks source layer IDs, `--expect-first-model-layer` is an important operator assertion.

`generate_golden.py` is a small legacy fixture: it decodes one layer-0/expert-0 E2M1/E4M3 expert, uses a deterministic fp16 input and writes one fp32 output. `validate_reference.py` checks expert-0 bank bytes and reciprocal multipliers, reconstructs the official compressed-tensors weights and compares that one golden with `atol=2e-6` and RMSE `5e-7`. This focused result is useful for local development, but it is not the richer authenticated Phase-3 fixture.

### Int4 extraction and round trip

The int4 extractor validates before and while writing. Header equality and exact file size are checked against a freshly derived `BankShape`. Selected qweight and scale bytes must equal the original shards, and dequantized bytes must agree exactly because the same canonical `(nibble - 8) * scale` interpretation is used. The phase4 round-trip harness crosses the ownership boundary: real builder, canonical format reader, admission header, `Int4ExpertBank` tensor views and nonresident lookup.

### Transformed sidecars

Marlin validation recomputes requested layers through the actual vLLM/CUDA repack routines and requires every output byte to match. B12x validation recomputes requested layers and compares the six source-derived planes. These comparisons establish transformation identity for sampled layers; they do not establish arbitrary-kernel numerical equivalence.

## 6. Phase 0 evidence versus Phase 3 authentication

Phase 0 freezes a useful history:

* BF16 source snapshot `995ad96e…b0`, NVFP4 snapshot `739af1e7…5d`, NVFP4 manifest hash and SBEXP001 bank hash/size;
* H=2048, I=512, 256 experts, top-k 8, 32 banked NVFP4 layers and 8 FP8 layers;
* CUDA/B70 software and device versions;
* an original logical provider protocol v1 and fixed benchmark workloads.

The logical [`provider_protocol.yaml`](../../../src/phase0/provider_protocol.yaml) describes a pinned-memory ring, per-route recovery mask and timeout recovery intent. It is historical design evidence, not the later executable fixed wire ABI. Likewise [`workload.yaml`](../../../src/phase0/workload.yaml) says correctness can use cosine or expected substrings, while [`benchmark_cuda.py`](../../../src/phase0/benchmark_cuda.py) actually performs case-insensitive substring checks and records unavailable timing fields as zero or wall-derived approximations. [`correctness_prompts.jsonl`](../../../src/phase0/correctness_prompts.jsonl) is the frozen twelve-case substring fixture; it is not a proof API or an oracle for bank arithmetic.

There is retained-evidence drift. Both checked-in baseline JSON files report `all_pass: false` with eight passing prompt records. `freeze.yaml` records a graph baseline of 242.63 tok/s and `11/12`, while [`baseline_cuda_graph.json`](../../../src/phase0/baseline_cuda_graph.json) records 247.36 tok/s and a different prompt result. The eager JSON records 24.14 tok/s. These are separate recorded runs/claims and must not be merged into one supposedly authenticated fact. [`guidellm_config.yaml`](../../../src/phase0/guidellm_config.yaml) is another graph throughput recipe using synthetic 128/128 traffic for 300 seconds; it is not the same workload as `benchmark_cuda.py`. [`env.sh`](../../../src/phase0/env.sh) supplies build/JIT concurrency and model environment values only.

Phase 3 turns exact values into executable provenance checks:

1. [`generate_reference.py`](../../../src/phase3/generate_reference.py) validates source and NVFP4 config contracts, exact checkpoint index totals, full bank size/SHA-256 and the canonical sorted shard manifest.
2. It audits the SBEXP001 header and every selected record byte for layers `{0,31}` and experts `{0,1,7,63,127,191,254,255}`, including the raw float32 reciprocal bits.
3. It constructs eight deterministic, distinct, finite fp16 input rows.
4. It decodes BF16 source weights and NVFP4 packed weights independently, evaluates stable SiLU expert forwards in float64 and stores both result families.
5. It requires each selected expert and the aggregate to satisfy relative RMSE `<= 0.18` and cosine `>= 0.98` unless explicitly running calibration mode.
6. It writes a temporary fixture, validates the packed header/64-byte layout, renames, revalidates, and requires the final whole-file digest.

[`ReferenceFixture::open`](../../../src/phase3/reference_fixture.cpp) independently requires the exact 4,227,456-byte file and frozen SHA-256 before mapping. It then validates magic/version/endian tag, H/I/top-k/counts, both snapshot IDs, bank and manifest digests, reserved zeros, exact aligned offsets, padding zeros, fixed unique ID sets and finiteness of every input/output. Bounded accessors reject out-of-range input, layer and expert requests.

## 7. Oracle mathematics and error budgets

[`math_cases.cpp`](../../../src/phase3/math_cases.cpp) keeps canonical global routes separate from provider-private compact IDs. `stage_remote_routes` validates normalized, finite, nonnegative top-k rows; copies only tokens with at least one remote route; and preserves canonical IDs, weights, masks, original row map and route positions. A paired mixed case changes only local IDs/weights and must produce byte-identical remote materialization. A zero-remote case produces no staged rows and must not publish a ring slot.

The oracle accumulates each remote route's `weight * fixture NVFP4 expert output` in double precision. For each output element, `output_budget` is:

`1e-6 * sum_abs_weights + (1e-2 + gamma) * weighted_magnitude`,

where `gamma` is the standard fp32 accumulation term for twice the remote-route count. A dedicated sensitivity check removes the single `2^-12` route and requires at least one resulting delta to exceed its comparison budget; this prevents a permissive tolerance from making that route unobservable.

These values are **source contracts backed by concrete fixture bytes and executable comparisons**. Reyu represents the outcomes as observations (`relativeRmseOk`, `cosineOk`, `withinOutputBudget`, `smallRouteObserved`). It does not symbolically encode E2M1, E4M3, BF16, SiLU, matrix multiplication, rounding or CUDA/SYCL arithmetic, and therefore does not prove the float thresholds.

[`provider_math_test.cpp`](../../../src/phase3/provider_math_test.cpp) applies the oracle through the actual Phase-2 process ring and Phase-1 provider: authenticated fixture and runtime bank, stale placement/weight bootstrap negatives, zero-remote bypass, a one-route M=1..128 sweep, duplicate/unsorted/boundary/small-weight all-remote routes, local-route invariance, exact wire identity/status/extents, and split/fused sequence-bound injected failures. Failure must expose no payload and must preserve both wire and client poison. Shutdown checks dispatch counts and unchanged allocation baselines. Direct provider calls reject M=0 and M=129 without dispatch/allocation changes.

The Phase-3 [`Makefile`](../../../src/phase3/Makefile) wires generator, reference/math objects, Phase-1 provider and Phase-2 ring/provider objects. It builds normal and test-fault providers separately. It is a build/harness definition, not behavioral evidence by itself.

## 8. Executable scenarios and properties

`banks.ryu` contains reachable witnesses and scenarios for:

* successful SBEXP001 extraction with inferred contiguous identity;
* successful SBINT401 extraction with an explicit resident map;
* Marlin's valid but non-atomic direct destination publication edge;
* B12x ratio/alpha factorization;
* geometry rejection, foreign int4 payload rejection and wrong source identity.

Its invariant requires a published artifact to have accepted geometry, source/header/payload, packing, resident map, scale contract and comparison evidence. It deliberately does not require atomic replacement for Marlin.

`reference.ryu` contains reachable witnesses and scenarios for:

* a fully authenticated, record-audited oracle;
* successful wire comparison within the source budget;
* zero-remote no-publication;
* fixture identity rejection, bank-record mismatch, quality failure and output-budget failure;
* post-kernel dispatch failure with poison preserved.

Its invariant makes Phase-0 recording always true but never sufficient for later states. Authentication, byte audit and measured quality are ordered prerequisites.

## 9. Complete assigned source inventory

| Source | Classification and role |
|---|---|
| [`src/phase0/baseline_cuda_eager.json`](../../../src/phase0/baseline_cuda_eager.json) | Recorded eager CUDA evidence; eight substring cases pass, throughput/timing fields are historical measurements, not executable validation. |
| [`src/phase0/baseline_cuda_graph.json`](../../../src/phase0/baseline_cuda_graph.json) | Recorded graph CUDA evidence; differs from the frozen summary values and remains a separate run artifact. |
| [`src/phase0/benchmark_cuda.py`](../../../src/phase0/benchmark_cuda.py) | CUDA-only benchmark harness for prompt substring checks, single/batched decode and prefill; writes mode-specific JSON. |
| [`src/phase0/capability_manifest.yaml`](../../../src/phase0/capability_manifest.yaml) | Frozen model/provider geometry, format, hash and intended startup gates. |
| [`src/phase0/env.sh`](../../../src/phase0/env.sh) | Environment preset for compile concurrency and model choice. |
| [`src/phase0/freeze.yaml`](../../../src/phase0/freeze.yaml) | Frozen versions, snapshots, bank identity, geometry and measured baseline narrative; documentary, not an authenticator. |
| [`src/phase0/guidellm_config.yaml`](../../../src/phase0/guidellm_config.yaml) | Alternative synthetic graph-throughput recipe. |
| [`src/phase0/provider_protocol.yaml`](../../../src/phase0/provider_protocol.yaml) | Historical logical protocol v1 schema; superseded by the executable Phase-2 wire. |
| [`src/phase0/workload.yaml`](../../../src/phase0/workload.yaml) | Fixed decode/batched/prefill workload and intended correctness acceptance. |
| [`src/phase0/correctness_prompts.jsonl`](../../../src/phase0/correctness_prompts.jsonl) | Twelve frozen prompts and expected substrings consumed by the benchmark; not a formal or numerical oracle. |
| [`src/phase1/extract_experts.py`](../../../src/phase1/extract_experts.py) | SBEXP001 discovery, serialization, bounded layer residency, size check and atomic rename. |
| [`src/phase1/extract_experts_int4.py`](../../../src/phase1/extract_experts_int4.py) | SBINT401 v2 builder/validator, explicit residents, qzero/g_idx gates, dry run, sparsity and sampled comparisons. |
| [`src/phase1/build_marlin_bank.py`](../../../src/phase1/build_marlin_bank.py) | SBINT401-to-SBMARL01 CUDA repack, bounded reusable storage/RSS and fresh-repack byte checks. |
| [`src/phase1/build_b12x_bank.py`](../../../src/phase1/build_b12x_bank.py) | Specialized native-FP4 B12x builder with ratio/alpha scale factorization and optional byte comparison. |
| [`src/phase1/validate_expert_bank.py`](../../../src/phase1/validate_expert_bank.py) | Independent SBEXP001 exact-byte and layer-shift diagnostic. |
| [`src/phase1/generate_golden.py`](../../../src/phase1/generate_golden.py) | Legacy single-expert deterministic NVFP4 golden generator. |
| [`src/phase1/validate_reference.py`](../../../src/phase1/validate_reference.py) | Expert-0 bank byte/global multiplier and compressed-tensors golden cross-check. |
| [`src/phase3/Makefile`](../../../src/phase3/Makefile) | Reference generation and provider mathematics build/orchestration, including separately compiled fault path. |
| [`src/phase3/generate_reference.py`](../../../src/phase3/generate_reference.py) | Authenticated BF16/NVFP4 float64 reference generator, bank auditor and fixture publisher. |
| [`src/phase3/reference_fixture.hpp`](../../../src/phase3/reference_fixture.hpp) | Packed 256-byte fixture ABI, fixed domain constants and bounded read-only API. |
| [`src/phase3/reference_fixture.cpp`](../../../src/phase3/reference_fixture.cpp) | Whole-file digest, schema/layout/identity/finite validation, mmap lifetime and accessors. |
| [`src/phase3/math_cases.hpp`](../../../src/phase3/math_cases.hpp) | Canonical route/staging/oracle/metric types and public case functions. |
| [`src/phase3/math_cases.cpp`](../../../src/phase3/math_cases.cpp) | Case construction, remote staging, double oracle, invariance/sensitivity, budget and artifact metrics. |
| [`src/phase3/provider_math_test.cpp`](../../../src/phase3/provider_math_test.cpp) | End-to-end authenticated provider/ring harness and failure/identity/allocation checks. |
| [`src/phase4/int4_bank_roundtrip_test.py`](../../../src/phase4/int4_bank_roundtrip_test.py) | Real extractor-to-canonical-reader/admission/plane-view round-trip harness. |
| [`src/phase4/src/shooting_brake_vllm/int4_bank_format.py`](../../../src/phase4/src/shooting_brake_vllm/int4_bank_format.py) | Canonical version-2 SBINT401 header, geometry, validation and header-only reader. |
| [`src/phase4/src/shooting_brake_vllm/marlin_bank_format.py`](../../../src/phase4/src/shooting_brake_vllm/marlin_bank_format.py) | Canonical SBMARL01 transformed sidecar header, strict geometry/size reader and path resolution. |
| [`src/phase4/src/shooting_brake_vllm/b12x_bank_format.py`](../../../src/phase4/src/shooting_brake_vllm/b12x_bank_format.py) | SBB12X01 header/geometry/path helpers; six live planes despite the stale four-plane prose. |
| [`src/phase4/src/shooting_brake_vllm/bank_source.py`](../../../src/phase4/src/shooting_brake_vllm/bank_source.py) | Pageable or CUDA-registered read-only bank data view, with fatal registration failure. |

## 10. Limits, dependencies and CPU implementation comparisons

External boundaries include Hugging Face snapshot/index layout, safetensors, PyTorch/NumPy, compressed-tensors, vLLM Marlin ops, FlashInfer scale conversion/B12x execution, CUDA runtime registration and the native provider/ring/backend. The bank model preserves their observable contracts but does not implement them.

Useful CPU-executable/source-inspection comparisons for the parent verification layer are:

1. Construct tiny `Int4BankHeader` instances, serialize/parse them, and mutate magic/version/padding/ID order/plane offset to confirm fail-closed behavior without a GPU.
2. Run the existing tiny real int4 builder/reader round trip when its checkpoint fixture is available; compare canonical and admission header fields and all six plane views.
3. Create synthetic Marlin headers and compare `plane_geometry`, `pack`, `parse_marlin_bank_header` and exact-size rejection; no repack kernel is needed for header checks.
4. Create B12x headers with `make_header`, parse them and independently recompute offsets/stride/file extent. This explicitly exposes that the current reader itself does not enforce all derived/file-size checks.
5. Compare `validate_expert_bank.py::expected_record` with extractor ordering on a tiny synthetic safetensors snapshot, including a nonzero first source layer, gate/up mismatch and shifted-row diagnostic.
6. Compare `generate_golden.py`'s manual NVFP4 decode with compressed-tensors on one CPU tensor, then separately compare Phase-3's NumPy decode; agreement is evidence across independently written decoders.
7. Mutate a copied Phase-3 fixture's digest, reserved bytes, offsets, fixed IDs or non-finite values and inspect that `ReferenceFixture::open` rejects each before exposing spans. This is CPU-only C++ behavior once the fixture exists.
8. Evaluate `output_budget` and `small_route_is_sensitivity_checked` on the fixture without a provider to verify route-count, weight and weighted-magnitude dependence. That checks the comparator contract, not device-kernel accuracy.

No gate, test, build, formatter, linter, GPU workload or server workload was run while authoring this chapter.
