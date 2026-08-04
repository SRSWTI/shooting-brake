# Model Format and Expert Conversion Contract

## Purpose and status

This document defines the format boundary between Colibri model weights and provider-specific resident expert banks. It elaborates the [quantization compatibility gate in `architecture.md`](architecture.md#quantization-compatibility-gate); it does not replace the repository overview.

**Status:** design contract and validation plan. The Colibri-to-XPU conversion, provider compatibility, kernel selection, and resident-bank cache described here are **not established as passing**. A format name, matching matrix dimensions, or matching group size is not evidence of compatibility.

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## Evidence boundaries

The design keeps four kinds of statement separate:

- **Source fact:** a property captured directly from the selected Colibri model tensors and quantization metadata.
- **Provider claim:** a property documented or exposed by a kernel provider, but not thereby proven against Colibri.
- **Design target:** behavior Shooting Brake requires before that provider can be used.
- **Unverified assumption:** any proposed equivalence that has not passed the protocol in this document.

In particular:

- Colibri integer W4 is the source-format contract to be captured exactly.
- The patched llm-scaler XPU INT4 path and the current standalone vLLM-XPU INT4 path are distinct provider contracts.
- Triton MXFP4 is an E2M1, block-size-32 representation and is not Colibri integer W4.
- Xe-Fuse BF16 is a BF16 execution path, not a W4 packed representation.
- Whether any provider reproduces the canonical Colibri expert remains unverified until the one-expert gate passes.

## Source format and provider prepack are different layers

### Source-format boundary

The source side consists of the original expert tensors plus their exact Colibri quantization metadata. It is the authority for:

- integer nibble interpretation;
- zero-point or no-zero-point convention;
- group scale values, dtype, shape, and orientation;
- logical gate, up, and down identities;
- tensor dimensions and any source padding;
- source packing order;
- Colibri-specific quantization corrections;
- model, layer, and expert identity.

Source bytes and metadata MUST be retained unchanged as the rebuild authority. A loader MUST NOT infer omitted quantization properties from a filename, a generic label such as `W4` or `INT4`, provider defaults, matching dimensions, or matching group size.

### Provider-prepack boundary

A provider prepack is a derived, disposable cache record optimized for one kernel ABI. It may reorder rows, interleave tensors, transpose packing axes, add padding, change scale placement, or dequantize to BF16, but only through an explicit, versioned conversion from the validated source artifact.

A provider prepack:

- MUST identify its kernel ABI and prepack version in the resident-bank manifest;
- MUST be reproducible from the unchanged source artifact;
- MUST NOT become the authority for source quantization metadata;
- MUST NOT be passed to another provider merely because both providers use the name `INT4`, `W4`, or `MXFP4`;
- MUST NOT be silently reinterpreted after a kernel, prepack, scale, or quantization-policy change;
- MUST be rejected when its manifest or checksum does not validate.

Conversion is therefore explicit:

```text
validated Colibri integer-W4 source artifact
    -> provider-specific converter
    -> versioned provider prepack
    -> provider kernel ABI
```

Triton MXFP4 and Xe-Fuse BF16 require explicit numerical conversion from the same canonical source. Renaming Colibri bytes or changing only metadata is prohibited.

## Exact W4 compatibility questions

Every integer-W4 converter and kernel ABI review MUST answer all of the following from bytes, metadata, and numerical probes. An unanswered item is incompatibility, not permission to use a default.

1. **Nibble signedness:** are 4-bit codes signed two's-complement, biased, unsigned, or interpreted by another mapping?
2. **Zero-point convention:** is there an explicit zero point; if so, what is its dtype, granularity, orientation, and position in dequantization?
3. **Nibble placement:** which logical values occupy the low and high nibbles, and in what traversal order?
4. **Gate/up row ordering:** which logical rows are gate and which are up?
5. **Gate/up storage:** are gate and up split, concatenated, tiled, or interleaved?
6. **Scale dtype:** what stored scalar type and bit pattern represent each scale?
7. **Scale granularity and orientation:** which weight elements share a scale, and along which logical axis are scale groups traversed?
8. **Packing major order:** is the packed representation N-major, K-major, transposed, or tile-major?
9. **Group boundaries:** how do group size and partial groups map to the logical K dimension?
10. **Padding and alignment:** which rows, columns, groups, and byte ranges are padded, and may a kernel legally read that padding?
11. **Logical-to-physical shape:** how are the gate, up, and down matrices represented in provider storage?
12. **Colibri corrections:** are there source-specific quantization corrections, and at which point are they applied?

Compatibility requires identical declared semantics and passing comparisons. It MUST NOT be inferred from the shared names `W4` or `INT4`.

## Kernel ABI compatibility matrix

The matrix below records boundaries and required evidence; it does not select a winner.

| Path | Numeric/storage contract | Permitted role | Required compatibility work | Current conclusion |
|---|---|---|---|---|
| Colibri integer W4 | Integer 4-bit source codes with captured Colibri group-scale semantics and packing | Canonical source and reference dequantization contract | Capture every W4 question above; preserve original bytes and metadata | Source contract to freeze; provider compatibility unverified |
| llm-scaler patched XPU INT4 | Provider-specific INT4 prepack and the pinned llm-scaler kernel ABI | Controlled B70 reference and candidate provider | Reconcile explicit format arguments, grouped-GEMM API, activation behavior, scale layout, and output semantics; retain the patch's scale-prefetch safety behavior | Distinct provider ABI; no equivalence to Colibri or current XPU implied |
| Current standalone vLLM-XPU INT4 | Current provider-specific INT4 prepack and simplified dispatch | Candidate long-term B70 provider | Compare with the patched path on identical canonical inputs; establish scale-prefetch safety, layout conversion, `gelu_tanh` behavior, supported shapes, and sequential-shape safety | Must not replace the patched path merely because it is newer |
| Triton MXFP4 | MXFP4 E2M1 with block size 32 | Alternative converted provider for later comparison | Perform an explicit numerical conversion and validate its own block scales, packing, ABI, and error against the canonical references | Not a format-compatible Colibri W4 drop-in |
| Xe-Fuse BF16 | BF16 weights/operations rather than packed W4 | Dequantized BF16 comparison or upper-bound path | Dequantize explicitly from the canonical source, define BF16 layout and epilogue semantics, then compare at declared floating-point tolerances | Not an INT4/MXFP4 prepack and not byte-compatible with any W4 path |

For each provider revision, the compatibility record MUST freeze and test:

| ABI surface | What must be recorded and compared |
|---|---|
| Revision and patches | Exact repository revision; the llm-scaler pinned kernel revision and patch are retained as distinct provenance |
| Format selection | Explicit INT4/MXFP4 arguments in the older path versus current simplified dispatch; no default may stand in for a declared format |
| Weight arguments | Accepted packed layout, alignment, offsets, lengths, and lifetime |
| Scale arguments | Scale dtype, orientation, surface dimensions, alignment, and ownership |
| Expert grouping | Expert offsets, remapping, active/zero-row behavior, and grouped-GEMM API shape |
| Activation/epilogue | First-GEMM output convention, gate/up split, SwiGLU composition, `gelu_tanh` behavior where exposed, weighting, and final output dtype |
| Shapes | All supported logical shapes, physical padding, and exact output shapes |
| State lifetime | Sequential execution of different shapes in one process, with no stale persistent-buffer reuse or memory growth |
| Safety | No scale overread; the scale-prefetch guard must cover small or insufficiently aligned scale surfaces, including the documented `K=1408`, `group_num=11` hazard class |
| Numerical result | Declared comparison point, oracle, tolerance, NaN/Inf rule, and result for every supported shape |

An upstream kernel MUST NOT be selected until scale-prefetch safety is present, API differences are reconciled, compatible results are demonstrated from controlled inputs, and sequential shapes are safe.

## Canonical one-expert artifact

Before modifying or selecting kernels, the project MUST define exactly one canonical expert input/output artifact. The artifact is a reviewable test fixture, not evidence that conversion has passed.

It MUST bind:

- the source model hash;
- exact source repository/model provenance used to read the expert;
- layer and expert identity;
- original gate, up, and down packed bytes;
- the complete Colibri quantization policy and metadata needed to answer every W4 compatibility question;
- independent checksums for the retained source content;
- logical tensor identities and semantic shapes;
- the fixed edge-case inputs, route/expert identities, and token-row cases used by the gate;
- comparison points for dequantized rows, GEMM1, SwiGLU, down projection, and complete expert output;
- outputs from each oracle, labeled by oracle, revision, dtype, and declared tolerance rather than merged into a purported universal answer.

Provider-prepacked bytes MUST live outside the canonical source portion and MUST be labeled with their provider, kernel ABI version, and prepack version. Regenerating a provider prepack MUST NOT modify the source portion or overwrite another provider's result.

## Logical expert shapes and packed-layout checks

The semantic operation has these exact GLM dimensions:

```text
gate:    6144 -> 2048
up:      6144 -> 2048
gate/up: 6144 -> 4096  (the two 2048 projections before SwiGLU)
SwiGLU:  4096 -> 2048
down:    2048 -> 6144
```

These are logical dimensions, not permission to assume a physical matrix orientation. For both gate/up and down, conversion MUST check:

1. source byte length against the captured packing and padding rules;
2. expected logical input and output dimensions;
3. the mapping from logical row and column coordinates to packed byte and nibble positions;
4. low/high-nibble order using known codes, including negative, zero, positive, and saturating values where the source convention admits them;
5. scale count, dtype, exact stored values, group boundaries, and logical orientation;
6. zero-point data and application, if the source policy defines it;
7. gate versus up identity for the first and last rows and at packing/tile boundaries;
8. split, concatenated, or interleaved gate/up ordering across the entire tensor, not only its first tile;
9. down-projection orientation independently of gate/up orientation;
10. row/column padding contents, alignment, provider read bounds, and the exclusion of padding from logical results;
11. provider-prepack length and offsets before upload;
12. dequantized sentinel rows against the Colibri CPU dequantization reference before any GEMM is accepted.

A shape match without these checks fails the format gate.

## One-expert conversion protocol

### 1. Freeze provenance

Record the relevant repository revisions, preserve the llm-scaler pinned revision and patch as a distinct reference, capture exact Colibri quantization metadata, and bind the canonical artifact to one model hash, layer, and expert.

### 2. Decode without reinterpretation

Decode the selected expert only according to captured Colibri metadata. Inspect nibble values, scales, zero-point convention, group boundaries, gate/up ordering, packing major order, and padding. Provider defaults MUST NOT fill missing metadata.

### 3. Convert exactly one expert

Convert exactly one gate/up pair and one down projection. Generate a separate prepack for each provider under evaluation. Colibri integer W4, patched XPU INT4, current XPU INT4, Triton MXFP4, and Xe-Fuse BF16 remain separately named formats throughout conversion and reporting.

### 4. Validate representation and intermediates in order

The following validation steps are mandatory and MUST be reported separately:

1. nibble values;
2. scales;
3. dequantized rows;
4. gate/up ordering;
5. first GEMM output;
6. SwiGLU output;
7. down output;
8. complete expert output.

A later match MUST NOT waive an earlier mismatch. In particular, a complete-output match does not excuse wrong nibbles, scales, or row ordering.

### 5. Compare against every exact oracle

Use the same canonical source expert and the same labeled input case to compare against:

- Colibri CPU dequantization/reference execution;
- 5090 execution;
- llm-scaler patched B70 execution;
- standalone vLLM-XPU B70 execution.

Representation comparisons—identities, shapes, integer codes, source metadata, and checksums—are exact. Floating-point comparisons at dequantized rows and execution intermediates use a tolerance declared before the run for that dtype and comparison point. Results MUST include both absolute and relative differences (including the handling of zero reference values), NaN/Inf detection, output shape, and oracle provenance. Tolerances MUST NOT be widened after observing a failure.

MXFP4 and BF16 are compared as explicit conversions at their declared floating-point tolerances. Their numerical proximity MUST NOT be reported as integer-W4 byte or layout compatibility.

### 6. Run the required edge-case corpus

The single expert MUST pass all of these cases before promotion:

- all-zero input;
- small values;
- saturating quantized values;
- random input with its seed and generation rule recorded;
- repeated identical input;
- changing token-row counts;
- changing expert IDs;
- zero-row experts.

Different supported shapes MUST also execute sequentially in one process to expose stale persistent buffers. Each case records the representation checks, intermediate comparisons, final comparison, exact output shape, NaN/Inf status, stale-buffer status, memory-growth status, and scale-read safety. A zero-row case MUST perform no invalid weight or scale access.

### 7. Hard promotion gate

**Bulk conversion is prohibited until this one gate/up/down expert passes every validation step, every oracle comparison applicable to the provider, and every edge case.** A partial pass, a performance result, or a match from only one backend does not authorize bulk conversion. No provider is currently recorded by this document as having passed.

## Versioned resident-bank manifest

Every cached prepack uploaded into a fixed B70 arena MUST have one manifest entry containing every field below:

| Required field | Meaning and validation |
|---|---|
| `model hash` | Exact source model identity; MUST match the canonical source used for rebuild |
| `quantization policy` | Complete policy identity for the source-to-provider conversion; MUST match captured Colibri semantics and converter selection |
| `kernel ABI version` | Exact consumer ABI; MUST match the loaded kernel before the entry is used |
| `prepack version` | Exact physical-layout/converter version; MUST match the decoder/uploader for the cached bytes |
| `layer` | Source layer identity; MUST match placement and request identity |
| `expert` | Source expert identity; MUST match placement and request identity |
| `device` | Intended resident device identity; MUST match the arena being populated |
| `offset` | Byte offset in the fixed resident arena/cache record; MUST satisfy the provider ABI's bounds and alignment |
| `length` | Exact byte length; MUST match the prepack version and remain within the arena |
| `scale format` | Provider scale encoding, dtype, grouping, and orientation identity; MUST match the kernel ABI |
| `checksum` | Integrity checksum for the exact versioned cache payload; its algorithm and byte scope are fixed by the prepack version and MUST validate before upload/use |

The manifest is invalid if any required field is absent, unknown, mismatched, out of bounds, misaligned, duplicated inconsistently, or unsupported by the active provider. String similarity and numeric coercion are not matches.

### Load, checksum, rebuild, and fallback behavior

For each entry, the loader MUST perform these steps in order:

1. match `model hash`, `quantization policy`, `layer`, and `expert` to the validated source request;
2. match `kernel ABI version`, `prepack version`, and `scale format` to the selected provider;
3. match `device` to the target resident bank;
4. bounds-check and ABI-alignment-check `offset` and `length` before reading or uploading bytes;
5. validate `checksum` over the exact byte scope defined by the prepack version;
6. accept the entry only after every check succeeds.

On any mismatch or checksum failure, the loader MUST NOT upload or execute the entry. It SHOULD rebuild that one prepack from the validated source artifact using the exact selected converter, write a fresh versioned entry, and validate the full entry again. It MUST NOT repair a failure by relabeling bytes, borrowing another provider's prepack, ignoring the checksum, or changing quantization metadata.

If a validated rebuild cannot be completed, execution MUST use the safe Colibri CPU reference/fallback path rather than reinterpret incompatible bytes. If that fallback is unavailable, the operation MUST fail closed; it MUST NOT continue with an unvalidated resident expert. Cache failure and provider incompatibility do not alter the canonical source artifact.

## Prohibited shortcuts

The following are format-contract violations:

- treating `W4`, `INT4`, or matching group sizes as format equivalence;
- passing Colibri integer-W4 bytes directly to patched or current XPU INT4 without a proven converter;
- passing integer-W4 bytes to Triton MXFP4 or Xe-Fuse BF16;
- using one provider's prepack under another provider's ABI;
- silently choosing signedness, zero point, scale orientation, gate/up order, packing major order, padding, or correction rules;
- accepting only final-output similarity while representation or intermediate checks fail;
- choosing current upstream code solely because it is newer;
- bulk-converting experts before the canonical one-expert gate passes;
- describing an unrun or failed comparison as compatible.
