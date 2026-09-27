"""Opt-in mixed-traffic experiment; not registered in production.

Retains vLLM's async scheduling and allocation policy. Limits the total step
budget while a running request has entered decode, restoring the original
budget on every exit. Cold prefill keeps the configured budget.
"""

from __future__ import annotations

import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


def interactive_budget(requests, baseline: int, cap: int) -> int:
    for request in requests:
        if (
            request.num_output_tokens > 0
            and request.num_computed_tokens >= request.num_prompt_tokens
        ):
            return min(baseline, cap)
    return baseline


class InteractiveScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interactive_token_cap = int(os.environ.get("SB_INTERACTIVE_TOKEN_CAP", "512"))
        if self._interactive_token_cap < self.max_num_running_reqs:
            raise ValueError("interactive token cap must accommodate every running request")
        if self.vllm_config.speculative_config is not None:
            raise ValueError("interactive scheduler experiment excludes speculative decoding")

    def schedule(self, throttle_prefills: bool = False):
        baseline = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = interactive_budget(
            self.running, baseline, self._interactive_token_cap
        )
        try:
            return super().schedule(throttle_prefills=throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = baseline
