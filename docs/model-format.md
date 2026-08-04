# Model Format and Expert Artifact Contract

## Purpose, authority, and status

This document defines the source-checkpoint, provider-artifact, compact-ownership, and capability-manifest contract for the architecture in [`../plan.md`](../plan.md).

The production deployment derives separate CUDA and B70 artifacts from one identical higher-precision BF16/FP16 source checkpoint. Provider formats are not interchangeable. This is a normative design and qualification contract, not a claim that the planned QuixiCore-XPU production provider or its model artifacts have passed.

The existing Colibri signed-S4 GS64 native worker is proven reference evidence and remains a comparator. Its reference artifact is distinct from the production NVFP4 contract, and it does not prove a batched QuixiCore-XPU provider or the secondary llm-scaler INT4 path.

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## One source, two provider artifacts

```text
identical BF16/FP16 higher-precision source checkpoint
    ├── CUDA artifact
    │     CUDA-owned routed experts in a qualified vLLM format
    │     all dense/router/shared-expert/attention/GDN/LM-head weights
    └── B70 artifact
          B70-owned routed experts in a qualified NVFP4 (E2M1 weights + E4M3 block scales) layout
```

The source checkpoint is the rebuild and semantic authority. Its model identity and exact tensor content MUST be fixed by provenance and cryptographic fingerprints. Both provider artifacts MUST name that same source identity. One artifact MUST NOT be derived by reinterpreting or requantizing the other provider's already-quantized bytes.

At runtime:

- the RTX 5090 loads only CUDA-owned routed experts for the hybrid placement;
- the B70 loads only B70-owned routed experts into compact slots;
- dense, router, shared-expert, attention, recurrent-state, LM-head, and sampling weights remain CUDA-owned;
- each routed expert has exactly one normal-path owner in the active placement;
- no expert weight moves or is quantized during decode or prefill; and
- CPU holds no normal-path expert execution role.

An explicitly validated all-CUDA placement may use a complete CUDA expert artifact. The hybrid placement MUST NOT assume that a B70-owned expert is also resident on CUDA; exact recovery availability is a separate declared capability.

## Source checkpoint contract

The source record MUST bind:

- source repository/model provenance and immutable revision;
- model architecture and configuration;
- tokenizer/config identities needed to bind execution semantics;
- exact source tensor names, shapes, dtypes, and checksums;
- layer, global expert, gate, up, and down tensor identities;
- any source sharding and reconstruction rules;
- source BF16/FP16 values or immutable byte ranges; and
- one model-source fingerprint used by the CUDA artifact, B70 artifact, placement, and runtime manifests.

A loader or converter MUST NOT infer missing model or quantization facts from a filename, marketing format name, matching dimensions, or provider default. Unknown source metadata fails the build.

## Manifest-driven logical shapes

Logical expert shapes come from the qualified model manifest, not from Colibri's older GLM dimensions and not from a kernel filename.

For model hidden size `H` and routed intermediate size `I`, each routed expert computes:

```text
input           [M, H]
gate projection H -> I
up projection   H -> I
SiLU(gate) * up [M, I]
down projection I -> H
expert output   [M, H]
```

For the inspected Qwen production target:

```text
H = 2048
I = 512
num_experts = 256
top_k = 8
```

Those values are a model-manifest example, not permission to accept another Qwen-family model by name. Every architecture/configuration MUST declare and qualify its actual `H`, `I`, expert count, top-k, router normalization, activation, tensor orientation, padding, and shared-expert semantics.

Logical dimensions do not specify physical row/column order, packing, tiling, interleaving, or padding. Every converter validates those independently.

## Provider artifacts are private derived records

A provider artifact is a reproducible, versioned conversion from the common source into one kernel ABI. It MAY reorder, transpose, interleave gate/up data, tile, pad, change scale placement, or quantize, but only through its named converter and declared precision policy.

Every artifact MUST:

- identify source model fingerprint and converter revision;
- identify provider (`cuda-vllm` or `b70-xpu`) and kernel ABI/bundle;
- identify architecture, semantic shapes, logical tensor identities, and ownership subset;
- declare weight code mapping, group or block size, scale dtype/orientation, zero-point policy, physical layout, padding/alignment, and checksums;
- declare activation, route-weight, accumulator, and output dtypes;
- identify the exact compact-slot map or CUDA-local map;
- be reproducible without using another provider's quantized artifact as source;
- remain immutable while its weight generation is active; and
- fail validation if any required field is absent, unknown, mismatched, out of bounds, misaligned, or unsupported.

Provider-private bytes MUST NOT become the source authority and MUST NOT be passed to another provider merely because format names resemble one another.

## CUDA artifact

The CUDA artifact contains the active CUDA placement's routed experts in a format explicitly supported by the pinned upstream-vLLM CUDA backend, such as a qualified NVFP4, FP8, or Marlin representation. Its manifest MUST record the exact backend, quantization recipe, physical layout, scales, expected kernel behavior, and CUDA-local slot mapping.

The artifact also binds the CUDA-owned router and shared-expert semantics, even though the B70 never consumes those tensors. `HybridMoERunner` MUST receive the canonical vLLM top-k IDs and weights produced by this model execution; provider artifacts cannot override them.

A CUDA artifact is not compatible with the B70 because its weights have the same semantic shapes. CUDA-addressable pointer tables and CUDA quantization layouts MUST NOT cross the provider boundary.

## B70 artifact and format choices

The B70 artifact contains only B70-owned routed experts in compact `(layer, global expert) -> B70 slot` order. It MUST select exactly one qualified physical weight contract for a loaded bank.

### Proven Colibri GS64 reference

The current native Colibri comparator has proven this contract for its supported shapes:

```text
weight codes:   signed S4
group size:     64
scale storage:  FP16
layout:         K-major / marlin-derived native-worker layout (Colibri reference only)
execution:      native fused gate/up/SiLU/down (Colibri reference only)
input staging:  FP16
result:         routing-weighted hidden-size partial
```

The existing conversion is exact for Colibri's captured signed-S4 GS64 semantics and agrees numerically with its CPU reference. These are reference-path facts only: GS64 is not the production B70 format, and this evidence does not qualify QuixiCore-XPU NVFP4 batching, every Qwen shape, an llm-scaler GS128 fallback, or the production provider process.

### Production B70 alternatives

The production B70 formats are:

1. **Primary — QuixiCore-XPU NVFP4:** packed E2M1 weight nibbles, E4M3 block scales with block size 16, and one FP32 global scale per expert. The selected `nvfp4_moe` fused or split operation MUST accept the canonical preselected routes; `multiply_router_weight` and asynchronous event chaining are provider capability facts, not claims that the production process is implemented.
2. **Secondary — llm-scaler / vllm-xpu-kernels INT4 W4A16:** a separately converted signed-S4 GS128 artifact MAY be qualified only if primary NVFP4 quality is insufficient. It remains a fallback and MUST NOT become the CUDA host.

The primary QuixiCore-XPU physical records are:

```text
w13:                [E, 2I, K/2]  uint8 packed E2M1 nibbles
w13_scales:         [E, 2I, K/16] uint8 E4M3 block scales
w13_global_scales:  [E]            float32
w2:                 [E, K, I/2]   uint8 packed E2M1 nibbles
w2_scales:          [E, K, I/16]  uint8 E4M3 block scales
w2_global_scales:   [E]            float32
```

NVFP4, GS64, and GS128 are different numerical and storage contracts. Their codes, scale counts, and boundaries MUST NOT be reinterpreted across formats. Every production or fallback B70 artifact MUST be quantized independently from the same BF16/FP16 higher-precision source used for the CUDA artifact, never from another provider's quantized artifact. Conversion or requantization MUST NOT occur on the token path.

Kernel family, group or block size, scale dtype, layout, supported shapes, and precision tolerance are capability facts. The provider MUST NOT choose them by an undocumented model-name conditional.

## Quantized-format questions

For every CUDA or B70 quantized artifact, the converter and kernel ABI record and validate:

1. code type and signedness, including the exact 4-bit mapping where applicable;
2. zero-point or no-zero-point convention;
3. low/high nibble placement and traversal order;
4. gate/up identities and split, concatenated, tiled, or interleaved order;
5. scale dtype, stored bit pattern, granularity, orientation, count, and group boundaries;
6. physical major order, transpose, tiling, and offsets;
7. logical-to-physical gate, up, and down shapes;
8. row, column, group, and byte padding plus legal kernel read bounds;
9. alignment and arena offset/length requirements;
10. source-specific corrections and the stage at which they apply;
11. activation, accumulation, route-weight multiplication, and output dtype/rounding; and
12. supported sequential shapes and persistent-scratch lifetime.

An unanswered item is incompatibility, not permission to use a default. Names such as `INT4`, `W4`, `S4`, `NVFP4`, or matching dimensions do not establish compatibility.

## Canonical dual-artifact qualification fixture

Before bulk conversion, the project MUST freeze one reviewable source fixture containing:

- source model fingerprint and full provenance;
- one layer/global-expert identity;
- higher-precision gate, up, and down source tensors;
- semantic shapes and declared activation;
- deterministic edge inputs and token-row cases;
- comparison points after source projection, gate/up, activation, down projection, routing-weight multiplication, and complete expert output; and
- separately labeled oracle outputs with revision, dtype, and tolerance.

The fixture produces independent CUDA and B70 provider artifacts. Their prepacked bytes, metadata, and outputs remain separately labeled. Regenerating one MUST NOT modify the source fixture or overwrite the other provider's record.

Validation proceeds in order:

1. source identity, tensor names, shapes, and checksums;
2. provider quantization codes, scales, orientation, and padding;
3. representative dequantized rows;
4. gate/up projection outputs;
5. SiLU/multiply output;
6. down-projection output;
7. complete unweighted expert output;
8. route-weighted expert output; and
9. CUDA-local plus B70-remote summed routed output.

A later match does not waive an earlier mismatch. Representation fields and identities compare exactly. Floating-point points use predeclared absolute/relative tolerance and NaN/Inf rules. Both provider artifacts MUST pass independently against the common source before the heterogeneous sum is accepted.

Required cases include zero, small, saturating, seeded random, repeated, and changing inputs; `M=1`, `M=2..32`, and representative prefill `M`; changing/non-sorted/duplicate IDs under canonical semantics; multiple rows choosing one expert; unequal and near-zero weights; boundary compact slots; zero-row experts; and different supported shapes sequentially in one process.

Passing the QuixiCore-XPU kernel correctness smoke gate does not qualify a converted production model artifact. Bulk conversion is prohibited until the one-expert fixture passes for the chosen provider contract. No production QuixiCore-XPU NVFP4 artifact or secondary llm-scaler INT4 artifact is recorded here as having passed that artifact gate.

## Model, placement, and provider manifests

Minimum model semantics:

```yaml
source_model_fingerprint: <immutable BF16/FP16 source>
model_architecture: Qwen3_5MoeForCausalLM
hidden_size: 2048
routed_intermediate_size: 512
num_layers: 40
num_experts: 256
top_k: 8
shared_intermediate_size: 512
router_normalization: canonical-vllm
```

Minimum per-artifact facts:

```yaml
artifact_provider: cuda-vllm | b70-xpu
artifact_fingerprint: <payload-and-manifest fingerprint>
source_model_fingerprint: <same source for both artifacts>
converter_revision: <pinned>
kernel_bundle: <pinned and qualified>
weight_format: <explicit>
group_or_block_size: <explicit>
scale_dtype: <explicit>
scale_orientation: <explicit>
layout: <explicit>
activation_dtypes: [...]
route_weight_dtypes: [...]
accumulator_dtype: <explicit>
output_dtypes: [...]
ownership_fingerprint: <placement subset and slot mapping>
weight_generation: <immutable loaded generation>
```

Minimum B70 provider/protocol facts additionally include:

```yaml
provider_protocol: 1
provider_backend: quixicore-xpu-nvfp4
device_identity: <qualified B70>
supported_hidden_sizes: [...]
supported_intermediate_sizes: [...]
supported_top_k: [...]
supported_block_sizes: [16]
secondary_int4_group_sizes: [] | [128]
decode_kernels: [...]
prefill_kernels: [...]
max_tokens: <qualified>
max_routes_per_token: <qualified>
max_inflight_slots: <qualified>
provider_generation: <current>
placement_fingerprint: <current>
```

The placement manifest maps every qualified routed expert exactly once:

```text
(layer, global expert) -> CUDA local slot
(layer, global expert) -> B70 compact slot
```

It MUST bind the model source, both artifact fingerprints, active weight generation, and provider capabilities. B70 is not represented as a fake EP rank.

## Load, validation, and fail-closed behavior

Before hybrid startup, validation occurs in this order:

1. match model architecture/configuration and source fingerprint;
2. match CUDA and B70 artifacts to that same source;
3. validate artifact checksums, converter/kernel ABI, quantization, group size, scales, layout, offsets, lengths, and alignment;
4. validate that placement ownership is complete, disjoint, and matches each artifact's slots;
5. compare adapter requirements with provider protocol/capabilities and fixed capacities;
6. load each compact B70 expert once into a fixed arena;
7. validate loaded weight generation and placement fingerprint; and
8. admit hybrid requests only after all checks succeed.

Any absent, unknown, unsupported, mismatched, corrupted, out-of-bounds, or misaligned fact fails closed. A cache record MAY be rebuilt from the immutable higher-precision source with the exact selected converter, then fully revalidated. It MUST NOT be repaired by relabeling bytes, borrowing another provider's prepack, ignoring a checksum, changing group size metadata, or quantizing during inference.

An unsupported model continues on stock upstream vLLM CUDA when a valid all-CUDA artifact is available. It MUST NOT be silently forced through a generic B70 path. A requested hybrid placement with a missing required artifact or owner MUST fail startup; CPU execution is not a normal format fallback.

## Prohibited shortcuts

The following violate this contract:

- deriving CUDA and B70 artifacts from different base checkpoints;
- deriving GS128 from an already-quantized GS64 artifact;
- treating GS64 and GS128 scale surfaces as interchangeable;
- passing CUDA tensors or CUDA provider bytes directly to XPU operators;
- using one provider's prepack under another provider's ABI;
- loading router or shared-expert work onto B70;
- assuming older GLM dimensions for the Qwen production target;
- selecting a kernel from model name rather than manifest capability;
- accepting final-output similarity while source, representation, ownership, or intermediate checks fail;
- moving, packing, allocating, or quantizing expert weights on the token path;
- bulk conversion before the one-expert dual-artifact gate; or
- describing the planned production artifact/provider as proven because the Colibri native GS64 comparator passed.
