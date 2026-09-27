from vllm import SamplingParams
from vllm.v1.request import Request

from experiments.interactive_scheduler import interactive_budget


def request():
    return Request("budget-test", [1] * 100, SamplingParams(max_tokens=32), None)


def test_cold_prefill_and_empty_queue_keep_original_budget():
    pending = request()
    assert interactive_budget([], 2048, 512) == 2048
    assert interactive_budget([pending], 2048, 512) == 2048
    pending.num_computed_tokens = pending.num_prompt_tokens
    assert interactive_budget([pending], 2048, 512) == 2048


def test_decode_caps_mixed_steps_without_increasing_smaller_budget():
    decoding = request()
    decoding.num_computed_tokens = decoding.num_prompt_tokens
    decoding.append_output_token_ids(2)
    assert interactive_budget([request(), decoding], 2048, 512) == 512
    assert interactive_budget([decoding], 128, 512) == 128


def test_preempted_decode_does_not_throttle_prompt_recomputation():
    resumed = request()
    resumed.append_output_token_ids(2)
    resumed.num_computed_tokens = resumed.num_prompt_tokens - 1
    assert interactive_budget([resumed], 2048, 512) == 2048
    resumed.num_computed_tokens += 1
    assert interactive_budget([resumed], 2048, 512) == 512
