"""Probe: what does a >max_batch prefill actually compute after VRAM surgery?

``forward_modular`` sends any batch larger than ``SHOOTING_BRAKE_B70_MAX_BATCH``
to the parent CUDA kernel with *raw global* expert ids.  After surgery the
CUDA weight tensor holds only CUDA-owned experts, so those ids are out of
range for every offloaded route.  The comment on that branch claims the two
paths were measured byte-identical but calls the mechanism unexplained.

Phase 6 replaces exactly this branch, so the current behaviour has to be
pinned down first.  This runs one greedy completion over a prompt long enough
to force the prefill branch and prints the token ids, so the caller can diff
configurations that differ only in which prefill path they take.

Driven entirely by environment variables (see ``prefill_probe.sh``); it holds
no policy of its own so the same file serves every configuration.
"""

from __future__ import annotations

import json
import os
import sys


#: Long enough that prefill exceeds the default 128-token dispatch threshold,
#: and ordinary enough that a degraded expert path shows up as visibly broken
#: text rather than a plausible alternate phrasing.
PROMPT = (
    "You are reading a technical description of a computer system. "
    "A mixture-of-experts language model routes each token to a small "
    "number of expert networks chosen from a much larger pool. Most "
    "experts are idle for any given token, so the memory holding them is "
    "mostly cold. A system can exploit that by keeping frequently used "
    "experts in fast memory close to the processor and moving rarely used "
    "experts to slower, cheaper memory further away. The cost of this "
    "trade is paid only when a cold expert is actually selected, and the "
    "benefit is that a much larger model fits in the same budget. "
    "Considering only the information above, explain in one sentence why "
    "moving rarely used experts to slower memory is usually a good trade."
)

MAX_TOKENS = 32


def main() -> int:
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=os.environ["SHOOTING_BRAKE_MODEL"],
        gpu_memory_utilization=float(os.environ.get("SB_GPU_UTIL", "0.90")),
        max_model_len=8192,
        # GDN allocates one Mamba cache block per decode sequence, and the
        # all-CUDA case has far less VRAM left over for them than the
        # offloaded ones do. Pinning this well under the all-CUDA ceiling
        # (~83) keeps every case on identical batching so the token diff
        # reflects the expert path and nothing else.
        max_num_seqs=int(os.environ.get("SB_MAX_SEQS", "64")),
        enforce_eager=os.environ.get("SB_EAGER") == "1",
    )

    tok = llm.get_tokenizer()
    n_prompt = len(tok.encode(PROMPT))

    # prompt_logprobs is the sensitive measurement here. Token ids only
    # reveal a broken prefill when the damage flips an argmax, and the first
    # generated token of a prompt like this one is predictable enough to
    # survive substantial damage. Prompt logprobs are computed *during*
    # prefill, one per prompt position, so they read that pass directly: a
    # path that drops routed-expert output shows up as a clearly worse
    # cumulative logprob even when every sampled token matches.
    out = llm.generate(
        [PROMPT],
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            logprobs=5,
            prompt_logprobs=0,
        ),
    )[0]
    gen = out.outputs[0]

    # Sum the prompt logprobs, skipping position 0 (no prediction exists).
    plp = [
        next(iter(d.values())).logprob
        for d in out.prompt_logprobs[1:]
        if d
    ]

    first = gen.logprobs[0] if gen.logprobs else {}
    result = {
        "label": os.environ.get("SB_LABEL", "?"),
        "placement": os.environ.get("SHOOTING_BRAKE_PLACEMENT", "all-cuda"),
        "max_batch": os.environ.get("SHOOTING_BRAKE_B70_MAX_BATCH", "128"),
        "prompt_tokens": n_prompt,
        "prompt_logprob_sum": sum(plp),
        "prompt_logprob_mean": sum(plp) / max(len(plp), 1),
        "first_token_top5": {
            int(k): round(v.logprob, 5) for k, v in first.items()
        },
        "token_ids": list(gen.token_ids),
        "text": gen.text,
    }

    dest = os.environ.get("SB_OUT")
    if dest:
        with open(dest, "w") as fh:
            json.dump(result, fh, indent=2)

    print("PROBE_RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
