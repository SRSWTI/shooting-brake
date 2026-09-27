# Placement and route partition

The placement branch answers two different questions in order:

1. **Static ownership:** for every absolute model layer and global routed-expert id, which concrete target owns the normal-path weights, and what is that target's dense local slot?
2. **Per-invocation partition:** for each router-selected top-k entry, which tier must compute it, and how is a global id safely translated for a compact CUDA tensor?

Read this branch breadth first through the static [placement model](../models/placement.ryu), then through the runtime [partition model](../models/partition.ryu). The models deliberately keep ownership separate from execution: they do not model B70 transport, kernel arithmetic, request scheduling, or result joining.

## 1. Responsibility, actors, and interfaces

The authoritative static module is [`placement.py`](../../../src/phase4/src/shooting_brake_vllm/placement.py). Its actors are:

- a qualified model, supplying `num_layers`, `num_experts`, absolute `b70_capable_layers`, and bank topology;
- a `PlacementPolicy`, used only while building an immutable owner table;
- one `DeviceTarget` per concrete CUDA, B70, or CPU destination;
- an `ExpertOwner` for every `(absolute layer, global expert)` pair;
- optional `DeviceCapacity` records, which bound resident bytes on named concrete targets;
- consumers such as weight surgery and runtime route partitioning, which read the published placement but do not invoke the policy on the hot path.

The authoritative runtime module is [`partition.py`](../../../src/phase4/src/shooting_brake_vllm/partition.py). It consumes the immutable placement and router tensors, producing masks for CUDA, B70, and CPU. It also owns the global-to-compact CUDA mapping and the zero-weight dummy-slot convention needed by the fused CUDA kernel.

The boundary is intentionally not vLLM expert parallelism. A B70 target is an external accelerator with an explicit index, not a synthetic vLLM rank. Placement is stable during an inference step and carries a `generation` for coarse-boundary replacement; neither module implements the future policy that decides when to replace a generation.

## 2. State and operation map

| Source operation | Inputs | Observable result | Failure contract | Reyu action or observation |
|---|---|---|---|---|
| `PlacementPolicy.assign` / `build_placement` | model dimensions, capable layers, generation, optional capacities | complete immutable owner matrix | construction or validation rejects before publication | policy-specific `build...` actions in `placement.ryu` |
| `validate_placement` | complete placement | accepted shape, dense slots per concrete target, capable-layer discipline, listed capacity budgets | `PlacementError` for invalid dimensions/shape/type, negative/duplicate/gapped slots, illegal offload, duplicate capacity, or overcommit | `inv`, rejection scenarios |
| `b70_bank_covers` | placement plus bank layer/source ids | Boolean coverage of every B70 **and CPU** owner | returns `false`, leaving `build_for_qualified` to raise | `bankCovers`, sparse-bank scenarios |
| `policy_from_name` | environment-facing policy string | a built-in policy object | malformed, unknown, or unarmed CPU-tier request raises | policy success/error scenarios |
| `build_cuda_expert_maps` | placement and layer | ascending local-to-global CUDA ids and `-1`-sentinel global-to-local map | invalid layer, slot mismatch, or failed round trip raises | compact-map abstraction in `partition.ryu` |
| `compact_cuda_routes` | original ids/weights and CUDA map | local ids, masked CUDA weights, CUDA mask | id/weight shape mismatch or non-1D map raises | `compactCuda`, `shapeMismatchTest` |
| `partition_routes` | original ids/weights and one device-map row | CUDA/B70/CPU/valid masks | classification itself is a gather; later validation enforces semantics | `classifyThreeWay`, `classifyAllCuda` |
| `validate_partition` | a `RoutePartition` and capable-layer set | covering, disjoint, capable, finite route partition | uncovered, double-owned, illegal offload, or non-finite valid weight raises | named validation error scenarios |
| `validate_cuda_dummy_slot_placement` | placement | assurance that every offloaded row retains a real CUDA expert | all-remote row rejects because no no-CUDA fused path exists | `zeroCudaDummySlotErrorTest` |
| `validate_dispatch_buffer_shapes` | `DispatchBufferGeometry` plus four staging tensors | exact hidden and route shapes | one diagnostic reports all mismatched buffers | documented here; shape rejection abstracted in `partition.ryu` |

The placement model's ten-expert row is a finite behavioral witness, not a production dimension. Its sets preserve complete/disjoint ownership and exact small policy results. The partition model's route positions represent a fixed row with global ids `[0, 3, 7, 9, -1]`: two CUDA routes, one B70 route, one CPU route, and padding.

## 3. Static ownership mechanics

### 3.1 Owner identity and compact slots

`owners[layer][global_expert]` is total: every table cell contains exactly one `ExpertOwner`. An owner has a device kind, a non-negative compact slot, and a device index. Only B70 currently permits a nonzero index; CUDA and CPU indices other than zero are rejected by `DeviceTarget`.

Slot density is checked **per layer and per concrete `(device, index)` target**. Thus B70 card 0 and card 1 may each own slot 0. For each target present in a layer, the observed slots must equal `[0, count)` with no duplicates or gaps. CUDA global ids need not be contiguous, but `cuda_expert_ids(layer)` enumerates them in ascending global-id order and that order defines compact CUDA ids. This is why post-load slicing and preemptive compact allocation can share one remap contract.

The manifest serializer uses schema `shooting-brake.placement.v1` when no nonzero device index or capacities are present and v2 otherwise. v1 omits a zero `device_index` to retain legacy bytes. Deserialization always calls `validate_placement`; JSON is not trusted simply because it names a supported schema.

### 3.2 Capable and sparse layers

`b70_capable_layers` is an absolute model-layer set, not a count and not necessarily a prefix. Both B70 and CPU ownership are forbidden outside this set because both tiers source routed weights from the offline bank. A non-capable routed layer—such as the Qwen FP8 tail—must therefore remain CUDA. Laguna makes the distinction sharper: model layer 0 is dense and unbanked, while routed/banked layers are absolute ids 1 through 47 and compact bank rows are 0 through 46.

`b70_bank_covers` accepts explicit `bank_layer_ids` and `bank_source_expert_ids`. When absent, it retains the legacy prefix assumptions `range(bank_layers)` and `range(bank_experts_per_layer)`. It checks original global expert ids, never compact slots. This supports sparse source sets such as `{0,4,9}` and avoids serving Laguna layer $L+1$ weights for model layer $L$.

A source-visible representational limitation is preserved: `Placement` has a row for every model layer and cannot express “no routed module exists.” The topology harness therefore fills Laguna's dense layer 0 with CUDA owners solely as a placeholder and verifies that no remote owner appears there. The chapter does not reinterpret those placeholder owners as real dense-layer routed computation.

### 3.3 Built-in policies

All built-ins first produce a complete row and then rely on common validation:

- **`AllCudaPolicy`:** every expert remains CUDA, and each global id is its local slot.
- **`ExpertGroupPolicy`:** explicit half-open ranges assign concrete targets. `validate_expert_groups` rejects out-of-bounds groups, overlap, and gaps. This is the direct multi-card ownership API but is not exposed by `policy_from_name`.
- **`FractionalRemotePolicy`:** computes `cuda_n = round(num_experts * cuda_fraction)`, assigns the first `remote_n = num_experts - cuda_n` ids to B70 cards in positional order, then assigns the tail to CUDA. Unweighted counts use `divmod`, giving earlier positions one extra expert. Weighted counts use largest-remainder apportionment: floor each proportional share, sort positions by descending fractional remainder, and add one to the first `leftover` positions. Python's `round` semantics, including ties-to-even, determine `cuda_n`; this matters in differential tests.
- **`SplitPolicy`:** clamps `cuda_per_layer` into `[0,num_experts]`; low ids are CUDA and the remaining tail is B70 card 0.
- **`InterleavedPolicy`:** requires `period >= 2`; ids satisfying `e % period == period - 1` are B70 and all others are CUDA. Slots for each tier advance independently.
- **`LayerSubsetPolicy`:** chooses the last `active_layers` from sorted capable layer ids. Those rows use a split, while all other rows are CUDA. Slicing means requesting more active layers than exist simply selects all capable layers. Active rows share one resident expert set.
- **`AllOutPolicy`:** chooses the same last-layer subset, then partitions each active row into low-id CUDA hot experts, a contiguous B70 middle, and high-id CPU cold experts. It is the only built-in CPU owner. `policy_from_name` rejects `cpu_per_layer > 0` unless `SHOOTING_BRAKE_ALL_OUT=1`; a policy object constructed directly does not itself read the environment.

For an exact CPU comparison, the model's weighted example uses 10 experts, `cuda_fraction=0.5`, two device indices, and weights `(3,2)`. It must produce B70 card 0 ids `{0,1,2}`, B70 card 1 ids `{3,4}`, and CUDA ids `{5,6,7,8,9}`. A second useful boundary is 181 experts, no CUDA, and two unweighted cards: `divmod(181,2)` produces counts 91 and 90.

### 3.4 Capacity and generation

Capacity validation is opt-in and concrete-target-specific. For every listed `DeviceCapacity`, required bytes are

$$
\text{count_target(target)} \times \text{bytes_per_expert}.
$$

A strict `required > capacity_bytes` comparison rejects overcommit. Missing capacity records do not impose an implicit limit; duplicate capacity records for one target reject. Counts span all layers because each layer has distinct weights.

A generation is serialized and may be incremented at a coarse swap boundary. The hot route partition has no generation-transition protocol: a caller must ensure it does not mix owner maps within one inference step. The Reyu generation action is explicitly a boundary abstraction, not a claim of an implemented online predictor.

## 4. Runtime partition and remapping

### 4.1 Device-map gather

`build_device_map` encodes every owner as int8: CUDA 0, B70 1, CPU 2. The whole map is built on CPU; a caller moves and caches the relevant layer row. `partition_routes` treats `topk_ids >= 0` as valid, clamps negative padding to zero before the gather, and intersects each equality mask with validity. Padding therefore belongs to no compute tier even though it temporarily indexes element zero.

The returned `RoutePartition` retains the original ids and weights. It does not materialize three rewritten route lists. `validate_partition` sums the Boolean tier masks and demands exactly one owner for every valid entry. It separately rejects B70 or CPU routes outside capable layers and rejects non-finite weights at valid positions. The partition identity

$$
\sum_k w_k E_k(x)=
\sum_{k\in CUDA}w_kE_k(x)+
\sum_{k\in B70}w_kE_k(x)+
\sum_{k\in CPU}w_kE_k(x)
$$

is justified only by disjoint, covering masks; the module does not prove kernel numerical equality.

### 4.2 CUDA compaction and the dummy slot

`build_cuda_expert_maps` enumerates ascending CUDA global ids. `local_to_global[local]` is the selected global id; `global_to_local[global]` is its local position, while every offloaded global id remains `-1`. It checks that the placement's stored CUDA slot agrees with this enumeration and checks the round trip.

`compact_cuda_routes` performs these operations in a critical order:

1. validate matching id/weight shapes and a one-dimensional map;
2. clamp padding before indexing;
3. gather local ids, preserving `-1` for non-CUDA globals;
4. derive the CUDA mask from `local_id >= 0` and valid input id;
5. multiply original weights by that mask;
6. only then clamp negative local ids to dummy local slot 0;
7. convert local ids back to the router's original integer dtype.

The final dtype conversion is behavior, not decoration: silently widening router int32 ids to int64 caused the downstream CUTLASS kernel to reject them. A dummy id alone is not safe; its corresponding weight must be zero.

The current fused CUDA call always executes. Therefore a row that offloads experts but retains zero real CUDA experts has no real local slot 0. `validate_cuda_dummy_slot_placement` rejects such a placement until a no-CUDA fast path exists. This is an intentional cross-module distinction: `FractionalRemotePolicy(cuda_fraction=0.0)` can build and pass `validate_placement`, but downstream partition setup rejects it. The two Reyu models expose both reachable observations rather than hiding the contradiction.

### 4.3 Dispatch geometry

`DispatchBufferGeometry` requires positive `max_batch`, `hidden_size`, and `top_k`. Hidden input/output buffers must be exactly `(max_batch, hidden_size)`; id and weight buffers must be exactly `(max_batch, top_k)`. Validation happens before native poller registration and reports every mismatched name with observed and expected shapes. The owned module checks shape, not dtype, contiguity, pinning, or native lifetime; those belong to transport/provider branches.

## 5. Named executable scenarios and properties

### Placement scenarios

- `allCudaTest`, `splitTest`, and `interleavedTest` cover the baseline policies.
- `explicitGroupTest` covers indexed multi-card ranges.
- `weightedLargestRemainderTest` is the exact 10-expert `(3,2)` differential fixture.
- `subsetActiveTest` and `subsetInactiveTest` distinguish capable from actually active layers.
- `allOutRequiresArmTest` and `allOutSuccessTest` preserve the CPU opt-in gate.
- `sparseCapableLayerTest` forces a non-capable row to CUDA.
- `allRemoteAcceptedHereTest` makes the placement/partition boundary visible.
- capacity, group-gap, sparse-bank, and generation scenarios expose rejection and replacement observations.

The placement invariant requires complete, pairwise-disjoint ownership; dense slots; bank coverage; capacity acceptance; no offload from a non-capable row; and CPU ownership only in armed all-out mode.

### Partition scenarios

- `threeWayPartitionTest` classifies a CUDA/B70/CPU row, excludes padding, and observes remote/padding entries mapped to dummy slot 0 with zero CUDA weight.
- `allCudaPassThroughTest` and `sparseLayerForcedCudaTest` cover rows with no remote routes.
- `zeroCudaDummySlotErrorTest` and `cannotCompactWithoutCudaTest` expose the missing no-CUDA fast path.
- shape, uncovered, double-owned, non-capable offload, and non-finite-weight scenarios are named failures.

The partition invariant separates classification from validation and records preservation of original weights and route-id dtype. It does not assert output-token parity or floating-point tolerances.

## 6. Source-backed harness evidence

The following files are harnesses and are not runtime implementation:

- [`placement_unit_test.py`](../../../src/phase4/placement_unit_test.py) exercises exact int4 resident sets, CUDA compaction, the dummy-slot rejection, dispatch geometry, sparse source ids, explicit multi-card groups, balanced fractional placement, and capacity overcommit. It also covers unrelated admission, allocation, telemetry, and runtime seams that these models do not claim to own.
- [`placement_test.py`](../../../src/phase5/placement_test.py) is the original CPU-only qualified-35B manifest gate. It checks all-CUDA/split/interleaved/subset placement, Qwen FP8 tail forcing, legacy bank coverage, generation, manifest round trip, non-dense slots, and unknown policy rejection.
- [`layer_topology_unit_test.py`](../../../src/phase6/layer_topology_unit_test.py) pins Qwen prefix topology and Laguna's sparse absolute-layer mapping. It explicitly documents the CUDA-placeholder row for Laguna's dense layer 0.
- [`partition_unit_test.py`](../../../src/phase6/partition_unit_test.py) classifies synthetic routes, padding, all-CUDA/interleaved rows, invalid ownership, non-capable offload, and a valid three-tier partition.
- [`partition_integration_test.py`](../../../src/phase6/partition_integration_test.py) launches an all-CUDA baseline and `split:128`, requiring exact token parity and at least one remote-route marker. With no B70 device it deliberately recomputes the masked remote partial on CUDA, so it checks partition/merge behavior, not physical B70 execution.
- [`shadow_validation_test.py`](../../../src/phase6/shadow_validation_test.py) launches eager `split:128` shadow mode and checks `Y_cuda + Y_b70 ≈ Y_full` with maximum absolute error below 0.1, cosine above 0.999, and no NaN. Its “B70-only” partial is a masked reference computation in this gate; the file says the physical device kernel was validated separately.

No harness was run while authoring this branch. Harness statements are source observations, not new verification claims.

## 7. Contradictions, limits, and external dependencies

- **All-remote policy versus executable CUDA path:** placement validation accepts zero CUDA owners; partition setup rejects offloaded rows without a real CUDA slot 0. This is the most important source-visible constraint.
- **Dense layers versus rectangular placement:** the manifest has one expert row per model layer, even when a model layer has no routed module. The Laguna harness uses CUDA placeholders rather than extending the representation.
- **“Weight-preserving” scope:** `partition_routes` retains the original tensor object; `compact_cuda_routes` intentionally creates masked CUDA weights. The models distinguish original router weights from CUDA-call weights.
- **Capabilities versus activity:** a bank-covered layer may remain all-CUDA. Dispatch cost follows active remote layers, while memory follows total remote experts.
- **Capacity scope:** only explicitly supplied capacities are checked. The module does not discover hardware memory or account for non-expert allocations.
- **Policy exposure:** `ExpertGroupPolicy` is programmatic; `policy_from_name` exposes all-cuda, split, interleaved, subset, allout, and fractional families, not arbitrary explicit ranges.
- **Numerics and execution:** ownership masks make split summation structurally valid but do not establish kernel parity, transport success, or token parity. vLLM router/kernel behavior, PyTorch tensor semantics, physical bank files, and native providers are external dependencies at this branch.
- **Current production defaults:** this chapter does not infer a current launcher placement from historical comments. The static API supports weighted fractional strings such as `fractional:2:0.22:95,75`; the chosen launcher's executable environment remains the authority for a particular deployment.

Useful CPU implementation comparisons are pure-Python policy construction/manifest round trips, exact owner/slot enumeration, bank-coverage predicates, capacity arithmetic, and torch-only synthetic partition/compaction. They can establish correspondence for finite inputs without a B70, but they do not substitute for integration or hardware evidence.
