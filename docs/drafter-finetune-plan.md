# DFlash Drafter Finetune Plan (r15)

The single largest decode lever left: speculative decoding with a drafter
finetuned on r15 itself. Target: 60-80% acceptance -> ~4-6 ms effective
decode vs today's 12.5 ms, at every context length (unlike ngram, which
inverts at long context).

## Why finetune at all
`poolside/Laguna-S-2.1-DFlash` (2.1 GB, k=15) is architecturally correct for
this family and already wired: vLLM ships `DFlashLagunaForCausalLM`, our serve
script exposes it via `SB_SPEC`. Measured on r15: **0 of 7,665 drafted tokens
accepted** -- the drafter was trained against unpruned Laguna-S-2.1, and r15
is 15%-REAP-pruned. The drafter is not broken; it is mistrained for this
target. Finetuning on r15's own hidden states is the repair, warm-starting
from the official checkpoint.

## Pipeline (three stages)

| stage | needs | runs where |
|---|---|---|
| 1. Data | real workload text: agentic/code prompts + r15 outputs | HERE (server live; corpus tooling exists) |
| 2. Hidden-state capture | full 118B forwards, dumping per-token hidden states | decision below |
| 3. Drafter training | 2.1 GB drafter + captured states (SpecForge) | HERE (5090; grad-accum for batch) |

## Stage 2: the only real decision
DFlash conditions on the TARGET model's hidden states; capture = running the
118B over ~100M+ tokens of representative text.

- **Rent one 96 GB GPU (recommended).** r15 NVFP4 is 58.2 GiB -- fits a
  single RTX PRO 6000 class card, and that exact serving is proven (the
  reference benchmark bundle came from one). SpecForge's
  `prepare_hidden_states` runs stock, zero engineering. Capture ~3-6 h,
  finetune ~6-24 h: a 1-2 day rental total.
- **On-rig alternative.** Build a hidden-state dump hook into our vLLM plugin
  (recon-scoped at ~120-180 production LOC; the Laguna DFlash training class
  port is 3 seams + a registered subclass) and capture at our prefill speeds
  (~10-20 h box time, not serving meanwhile). Zero cash, real engineering +
  downtime.

## Non-negotiables either way
1. **Warm start** from the official DFlash checkpoint; do not train from
   scratch.
2. **Train on the serving distribution**: agentic coding loops, long system
   prompts, repo context -- the workload this box actually serves.
3. **The acceptance gate runs on THIS rig**: greedy-equivalence vs baseline
   (temperature 0, identical prompts), acceptance rate from /metrics, and
   effective ITL across the 1K->127K ladder -- same instruments as every
   other lever (`benchmarks/decode_ladder_probe.py`).
4. Ship only behind `SB_SPEC` with the 120-prompt quality sweep green, like
   every other arm.
