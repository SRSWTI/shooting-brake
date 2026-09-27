# Admission and launch

This branch explains how a process becomes a Shooting Brake vLLM process and how the checked-in shell recipes start one. Read it after the repository HLD and before placement or execution. The executable models are [admission](../models/admission.ryu) and [launch](../models/launch.ryu).

## 1. Responsibility, actors, and boundary

Admission has three actors. The shell recipe selects a checkpoint, plugin, bank paths, execution modes, and vLLM CLI. vLLM discovers the `shooting_brake_vllm` general-plugin entry point and owns its API server, request scheduler, attention, model construction, and CLI parsing. The plugin gates itself, patches a few vLLM seams, validates the effective model and banks, and installs the `RoutedExperts` and `MoERunner` replacements. It does **not** install a request scheduler.

The production wrapper adds process replacement, a bounded `/health` wait, live `/v1/models` inspection, a chat completion, and a banner. Direct benchmark recipes stop at `exec vllm serve`; they do not implement the wrapper's readiness workflow. `scripts/b70_tune.sh` changes card frequency or power state before/alongside serving and is not model admission.

```text
shell environment + CLI
        |
        v
vLLM plugin discovery -- phase4_enabled? -- no --> ordinary vLLM process
        | yes
        v
file overrides -> ordered global patches -> OOT conflict check
        |
        v
model construction -> require_qualified_config -> reject or QualifiedModel
        |
        v
vLLM-owned scheduling/API -> routed-layer integration (other chapters)
```

## 2. Source-backed state and operations

| Model action / state | Source operation | Observable contract |
|---|---|---|
| `passPluginGate`, `skipPlugin` | [`config.py::phase4_enabled`](../../../src/phase4/src/shooting_brake_vllm/config.py) and [`__init__.py::register`](../../../src/phase4/src/shooting_brake_vllm/__init__.py) | Registration runs only when `SHOOTING_BRAKE_PHASE4=all-cuda` **and** `SHOOTING_BRAKE_MODEL` is already a registry key. Otherwise it is a no-op. |
| `loadOverrides` | `__init__.py::_apply_file_env_overrides` | `/tmp/sb_env_overrides.json` is read after the gate. Valid key/value pairs become strings in `os.environ`; missing file is a no-op; malformed input is warned and ignored. It cannot rescue a process that failed the earlier gate. |
| `qualify` / rejection actions | `config.py::require_qualified_config` | Effective architecture, text model type, exact geometry, text-only requirement, bank identity, rank modes, and int4 opt-in/ownership must all match. Failure raises `QualificationError`. |
| `installPatches` | `__init__.py::_patch_*` and `install_preemptive_alloc_hook` | Patches are installed before model/layer construction. Most are independently gated. Preemptive allocation must precede `RoutedExperts.__init__`. |
| `registerReplacements` | `__init__.py::register` | Empty OOT entries are filled; the same implementation is idempotently accepted; any foreign `RoutedExperts` or `MoERunner` replacement is fatal. |
| `prepareProduction` | [`serve_production.sh`](../../../serve_production.sh) plus [`serve_jota_r15_dual.sh`](../../../benchmarks/serve_jota_r15_dual.sh) | Wrapper KV/prefix flags are prepended to inherited `SB_EXTRA_ARGS`; recipe defaults and named `SB_*` overrides are resolved; trailing extra CLI arguments remain last. |
| `pollUnhealthy`, `observeHealthy`, `healthTimeout` | `serve_production.sh` health loop | `/health` is retried until success or `SB_BOOT_TIMEOUT_S` (default 900 seconds); timeout prints the log tail and exits 1. |
| `inspectServer`, `requestSmoke`, `printBanner` | production wrapper after health | `/v1/models`, log-derived facts, and one chat completion populate the banner. Because the script enables `pipefail` but not `errexit`, failed command substitutions do not form a fail-closed smoke gate; READY can still be printed with empty/invalid fields. |
| `execDirect` | benchmark `serve_*.sh` files | Direct recipes replace the shell with vLLM. Their success and health are external to these scripts. |

The Reyu admission model uses finite profile, architecture, and bank categories while carrying the five compared geometry fields as concrete integers; scenarios use the checked-in 35B, split-88B, and Laguna-r15 values. `BankStatus` abstracts header parsing plus, for int4, per-device resident ownership. This preserves the accept/reject boundary without modeling tensor bytes. The launch model bounds the health retry counter at two attempts instead of 900 seconds and distinguishes catalog/completion success, curl failure, and JSON failure.

## 3. Model registry and fail-closed qualification

[`config.py`](../../../src/phase4/src/shooting_brake_vllm/config.py) registers six checkpoint contracts:

- Qwen 35B NVFP4: 40 layers, 256 experts, top-k 8, hidden 2048, routed intermediate 512.
- Qwen 122B NVFP4: 48/256/top-k 8, hidden 3072, intermediate 1024.
- Split 88B: dense checkpoint `axe-superveloce-88b-nvfp4a16`, routed checkpoint `axe-superveloce-88b-int4`, 48 layers, 180 experts, GPTQ int4/group128, and text-only.
- 99B NVFP4: 48 layers, 205 experts, CausalLM architecture, and text-only.
- Laguna r20/r15 NVFP4: 48 transformer layers but routed modules and bank rows only for absolute layers 1 through 47. r20 has 205 experts; r15 has 218; both use top-k 10. Model layer 1 maps to compact bank row 0.

A Hugging Face cache snapshot path may be converted back to its `org/repo` identity. Other names fail closed. Qualification requires the registered architecture and text model type, each exact geometry field, format alignment (16 for NVFP4, 128 for routed int4), tensor parallel 1, pipeline parallel 1, and EPLB disabled. Text-only registry entries require the effective multimodal state to be absent or explicitly language-only.

Bank absence is accepted only when hybrid offload is not selected. In hybrid mode a missing file is fatal. NVFP4 `SBEXP001` and int4 `SBINT401` are the only admitted headers. Every populated bank must match routed format, hidden/intermediate width, logical expert count, source-layer identity when present, and may not have more rows than the model. All per-card banks must agree on row coverage. A Laguna bank with 48 rows is rejected rather than shifted; its registered topology expects 47 rows.

The split 88B hybrid path additionally requires `SHOOTING_BRAKE_B70_INT4=1`, SBINT401 version/format, group size 128, 4 bits, zero point 8, exact source geometry, a shared resident set across layers, one bank per remote device, no CPU-owned experts, and exact disjoint equality between each card's bank IDs and placement IDs on every layer. Gaps, duplicates, overlaps, extra IDs, and wrong card counts are errors.

The qualified multi-card view deliberately differs by format. Explicit-ID int4 card banks are unioned. Repeated monolithic NVFP4 banks represent the same full file while provider resident lists select disjoint card subsets. This chapter models the identity gate; byte formats and placement construction are deeper branches.

## 4. Registration order and mode checks

`register()` first evaluates `phase4_enabled`, then applies file overrides, then installs patches in this order: force PIECEWISE for breakable hybrid graphs; normalize nested 99B checkpoint names; normalize nested quantization ignore names; arm command-streamer doorbells after warmup; add optional seam/profiler step hooks; install optional Laguna shared-expert fusion; install optional hidden-state capture. It imports the replacements only after those steps. If preemptive surgery is enabled, its allocation hook lands before any routed layer calls `create_weights`.

The nested-name and ignore patches are installed for every selected registry model but are no-ops when prefixes already match. PIECEWISE forcing requires both breakable graphs and hybrid. Command-streamer arming requires its own switch plus hybrid and happens only after `Worker.compile_or_warm_up_model`; `SHOOTING_BRAKE_B70_GRAPH=1` alone is not the command-streamer switch. Seam profiling and hidden capture are explicit opt-ins. None of these hooks changes vLLM's scheduler.

Registration checks both OOT slots. An empty slot is replaced. A slot already holding the identical Shooting Brake class is accepted. A different implementation raises rather than composing two replacements or silently winning by import order.

## 5. Recipe matrix and executable precedence

| Source | Effective purpose and notable live defaults |
|---|---|
| [`serve_production.sh`](../../../serve_production.sh) | Stops prior vLLM/EngineCore processes, prepends prefix caching and `--kv-cache-memory` to `SB_EXTRA_ARGS`, launches the tracked r15 recipe detached on port 8017, waits for health, performs inspection/smoke, then prints a banner. `HF_HUB_OFFLINE` defaults to 1 only for the spawned recipe. |
| [`serve_jota_r15_dual.sh`](../../../benchmarks/serve_jota_r15_dual.sh) | Current tracked production recipe: Laguna r15, two B70s, default placement `fractional:2:0.22018348623853212:95,75` (48 CUDA, 170 remote), graph doorbell arm, grouped NVFP4, fp16 wire, pipeline 2, MML 131072, **executable MNBT 2048**, MNS 6, GPU util 0.85, Poolside reasoning/tools. `SB_SPEC` is optional, never default. |
| [`serve_jota_r20_dual.sh`](../../../benchmarks/serve_jota_r20_dual.sh) | Laguna r20, 54 local / 151 remote default, 47-row repeated monolithic bank, MML 32768, MNBT 512 (explicitly inferred rather than measured on r20), no reasoning parser, no Marlin. |
| [`serve_99b_dual.sh`](../../../benchmarks/serve_99b_dual.sh) | 99B NVFP4, default 54 local / 151 remote despite its historical header describing the original one-local split, dual BDF selectors, MML 32768, MNBT 2048, MNS 4. |
| [`serve_88b.sh`](../../../benchmarks/serve_88b.sh) | 88B int4, split 54/126 on one selected B70, graph + preemptive surgery, MML 32768, MNBT 256, MNS 4. |
| [`serve_88b_128k.sh`](../../../benchmarks/serve_88b_128k.sh) | 88B long-context variant, MML 131072, MNBT 8192 and MNS 16 by default. `SHOOTING_BRAKE_BANK_REGISTER=1` changes the default B70 max batch from 2048 to 256; explicit caller values win. Marlin defaults on. |
| [`serve_88b_128k_profile.sh`](../../../benchmarks/serve_88b_128k_profile.sh) | Attribution-only profile variant: MNBT 8192, MNS 64, B70 profile markers and torch profiler output. Its comments explicitly reject these runs as performance measurements. |
| [`serve_hybrid.sh`](../../../benchmarks/serve_hybrid.sh) | Older 35B Track A recipe, default subset placement and 131072 context. Source-visible bank/library paths are `$REPO_ROOT/phase1` and `$REPO_ROOT/phase7`, while this checkout contains `src/phase1` and `src/phase7`; absent external symlinks, those defaults do not name the checked-in artifacts. |
| [`results/.../serve_vllm_rtx_pro_6000.sh`](../../../benchmarks/results/rtx_pro_6000_r15_slo/serve_vllm_rtx_pro_6000.sh) | Separate RTX Pro 6000 baseline under a `guidellm` project. It verifies its own vLLM and CUDA compiler, creates CUDA library aliases if missing, serves r15 without selecting the Shooting Brake plugin, disables prefix caching, and uses MML 160000/MNS 6. |
| [`scripts/b70_tune.sh`](../../../scripts/b70_tune.sh) | Root-only sysfs helper. PCI aliases choose test or serving B70; operations pin/floor/reset frequency, cap power, or report state. It does not launch vLLM and its measured comments are not runtime controls. |

Shell precedence matters. `${VAR:-default}` admits a nonempty inherited environment value; unconditional `export` or `unset` overwrites inheritance. In the production path, the wrapper constructs `SB_EXTRA_ARGS` as KV flag, then prefix flag, then pre-existing extras. The r15 recipe appends the completed string after its fixed CLI. Therefore a duplicate option in inherited extras is last and is the effective CLI candidate, subject to vLLM's parser behavior. Comments and echo/banner text never override executable assignments.

## 6. Source-visible contradictions and limits

The r15 prose still says MNBT 512 is the measured knee, but the executed default is `${SB_MNBT:-2048}`. The production banner reports its scheduler from the same 2048 default, so the executable and banner agree while the historical recipe discussion does not.

The current r15 placement executes 95 experts on device 0 and 75 on device 1. `serve_production.sh` prints “85/card,” an older equal-split description. Both total 170, so the error is invisible if only the total is inspected. The launch model exposes `sourceContradictionWitness` rather than normalizing the banner to source truth.

The production comment says one real completion “proves the pipeline end to end.” It is a useful live smoke, not proof, and the shell does not enforce it: only `set -o pipefail` is set. Failure in `MODELS_JSON`, parsing, log grep, or `SMOKE` assignment can continue to the banner. The health timeout is fail-closed; post-health verification is not. `failedSmokeStillPrintsBannerTest` preserves this exact distinction.

The scripts' scheduler flags configure vLLM's external scheduler. The plugin only makes per-layer execution decisions after vLLM supplies a batch. No model here invents a Shooting Brake request queue, admission order, preemption rule, or fairness guarantee.

## 7. Named scenarios and properties

`admission.ryu` includes successful Laguna hybrid registration; all-CUDA admission with an absent bank; an override file that cannot rescue the earlier plugin gate; wrong bank geometry; missing int4 opt-in; and a conflicting OOT replacement. Its invariant ties `Registered` to Shooting Brake ownership and keeps every rejected path associated with an explicit error.

`launch.ryu` includes healthy production startup, trailing CLI precedence, bounded health timeout, the non-gating failed-smoke edge, and a direct profiling recipe with no readiness loop. `readyWitness`, `timeoutWitness`, `precedenceWitness`, `smokeNotAGateWitness`, and `sourceContradictionWitness` are reachable observations, not claims about real hardware timing.

## 8. Build/packaging and external dependencies

The root [`pyproject.toml`](../../../pyproject.toml) packages the phase4 source tree into `shooting-brake` 0.2.0, registers the plugin entry point, and declares harness dependencies while deliberately leaving CUDA-specific torch/vLLM installation to the operator; its optional engine group pins torch 2.11.0 and vLLM 0.26.0. Phase4's [`pyproject.toml`](../../../src/phase4/pyproject.toml) is a narrower `shooting-brake-vllm` 0.1.0 package with the same entry point and vLLM 0.26.0 dependency. These are alternative installation surfaces for the same Python module, not two runtime plugins.

External behavior includes vLLM plugin discovery/CLI duplicate handling, HF config objects, filesystem bank headers, oneAPI setup, CUDA/Level Zero libraries, HTTP availability, process signals, PCI/sysfs permissions, and model responses. Numerical correctness, hardware selection, provider health, placement, and result joining are owned by deeper chapters.

## 9. Potential CPU implementation comparisons

Without accelerator work, a parent verification pass can compare the Reyu branch table against existing CPU-only admission fixtures: supported versus snapshot-path model identity; absent bank in all-CUDA versus hybrid; wrong format/geometry; text-only, TP/PP and EPLB rejection; and exact int4 ownership gaps/overlaps. A source-inspection comparison can tokenize each recipe's final `exec` arguments under controlled environments and confirm that r15 defaults to MNBT 2048, the 95/75 split, and trailing `SB_EXTRA_ARGS`. A shell subprocess with fake `curl`, `setsid`, `nohup`, `pgrep`, and `pkill` executables could also demonstrate the post-health smoke non-gate without a GPU, but such a comparison must be reported as shell-control-flow evidence, not server validation. No gate or workload was run while authoring this chapter.
