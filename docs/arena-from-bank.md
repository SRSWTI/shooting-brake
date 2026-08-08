# Sourcing the DRAM arena from the expert bank

## Why this has to change

`_load_host_experts` fills the host arena by `index_select`-ing the layer's
weight tensors with global expert ids. Post-hoc surgery makes that correct:
the tensors still hold all 256 experts when it runs. Pre-emptive allocation
never creates the offloaded ones, so the same call either indexes out of
bounds or — when an offloaded global id happens to fall below `num_cuda` —
quietly loads a different expert than the one requested.

`_reject_unsupported_preemptive_tiers` currently refuses that combination
outright. It has to, because a wrong arena produces plausible tokens rather
than an exception. But the 122B needs both at once: its bank is 59.48 GiB
against 31.89 GiB of B70, so ~12 GiB of experts must live in host DRAM, and
its 66 GiB load peak means pre-emptive allocation is not optional.

## What the bank holds versus what VRAM holds

The bank is written from the raw checkpoint (`phase1/extract_experts.py`).
The VRAM tensors have been through
`prepare_nvfp4_moe_layer_for_fi_or_cutlass`. They differ in three ways, and
every one is silent if got wrong — a wrong arena computes a different
function without raising.

| quantity | VRAM source (today) | bank source |
|---|---|---|
| `w13` quant | `[up, gate]` on FlashInfer backends, so the loader swaps halves via `_w13_is_reordered()` | `[gate, up]` as the checkpoint stores it — **no swap** |
| `w13` / `w2` block scales | swizzled by `swizzle_blockscale`, undone per load by `linear_sf()` | already linear — **no conversion** |
| `w2` quant | direct | direct |
| `w13` global scale | `qconfig._w1.alpha_or_gscale` | `bank_w13_inv / a1_gscale` |
| `w2` global scale | `qconfig._w2.alpha_or_gscale` | `bank_w2_inv / a2_gscale` |

`linear_sf()` exists to undo a swizzle vLLM applied, not one the checkpoint
had. Reading the bank therefore skips it rather than applying it twice.

## The global scale is folded, and the fold must be reproduced

The bank stores `1 / weight_global_scale`. The quant config's
`alpha_or_gscale` is that value **divided by the activation global scale**.
Measured on the 35B, layer 16, experts 0-7:

| | bank `1/w_global` | `alpha_or_gscale` | ratio | activation gscale |
|---|---|---|---|---|
| w13 | 2.1230e-05 | 3.878e-07 | **54.744** | a1 = 54.75 |
| w2 | 2.2195e-05 | 1.441e-07 | **154.02** | a2 = 154.0 |

So a bank-sourced arena must divide by the same activation scalar. Using
the raw bank value would make the host tiers' weights 54.75x (w13) and 154x
(w2) too large.

That the current, folded value is the correct one is established
independently: the prefill streamer reads this same arena, and at 94.7% of
routes it measured **-0.0248 nats/token**. A scale error of that magnitude
would have produced roughly -0.49, the signature of dropping every route.

### Which `a2` — and why the gate cannot check it

`qconfig._a2.alpha_or_gscale` is a reduction over the experts *resident on
CUDA*. Under post-hoc surgery that is all 256, so it equals the global max
(154 at layer 16) and the table above is unambiguous. Under pre-emptive it
is the max over the CUDA-owned set alone — 173 at `subset:16:8` — while the
arena holds the **CPU-owned** experts, which are not in that set at all.

So "divide by the activation gscale from qconfig" reproduces today's
behaviour only under post-hoc. Under pre-emptive the same expression bakes
in a constant chosen by which experts happened to land on CUDA, applied to
weights belonging to experts that did not. The byte-for-byte gate below
will not catch it: its reference is a post-hoc run, where the two
expressions agree exactly.

This is a decision, not a detail, and it is the one thing in this document
that following the rest of it correctly will still get wrong. The bank's
raw `1 / weight_global_scale` is placement-independent, which is a point in
its favour. Decide what the CPU kernel's convention actually requires by
reading what it does with the `PackedPlane` scalar — not by making the two
paths agree on the single configuration where agreement is guaranteed.

## Gate before switching the source

Do not compare arena internals. Hash the arrays handed to
`host.load_expert` / `PackedPlane` — w13 quant bytes, linearised block
scales, and the two global scalars — per `(layer, expert)`, under both
sources, and require byte-for-byte equality. The tensors are already in
hand at the call site, so this needs no new accessor to read packed memory
back out, and equality there implies equality in the arena.

Reference configuration: 35B, `allout` placement, post-hoc surgery — the
cold tier's currently-working setup, and the one the pre-emptive guard
still permits.

This is the same discipline that validated the other two migrations in this
area: the bank extractor was gated on rebuilding the 35B bank byte-identical
to the working one, and pre-emptive surgery on a weight digest showing every
expert weight, block scale and remap identical to post-hoc. Neither was
resolvable by looking at tokens — the correctness harness has a measured
~0.11 nats/token noise floor, established by running identical code twice
and getting 4/8 sequences.
