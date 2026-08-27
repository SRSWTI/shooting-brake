from __future__ import annotations

import os

import torch

from shooting_brake_vllm.hidden_capture import (
    FORMAT,
    TARGET_LAYER_IDS,
    CaptureAccumulator,
    capture_filename,
    make_capture_request_id,
    parse_capture_request_id,
)


def test_chunk_accumulation_and_atomic_specforge_serialization(tmp_path) -> None:
    corpus_id = "pilot/15:00042"
    request_id = make_capture_request_id(corpus_id, response_start=2, total_tokens=5)
    assert parse_capture_request_id(f"cmpl-{request_id}-0") == (corpus_id, 2, 5)

    layers = len(TARGET_LAYER_IDS)
    hidden = 4
    all_states = torch.arange(
        5 * layers * hidden, dtype=torch.float32
    ).view(5, layers, hidden)
    accumulator = CaptureAccumulator(tmp_path)

    first = accumulator.add_cpu_chunk(
        f"cmpl-{request_id}-0",
        torch.tensor([11, 12], dtype=torch.int64),
        all_states[:2],
    )
    assert first is None
    assert not list(tmp_path.glob("*.part"))

    output = accumulator.add_cpu_chunk(
        f"cmpl-{request_id}-0",
        torch.tensor([13, 14, 15, 999], dtype=torch.int64),
        all_states[2:].new_tensor(all_states[2:].tolist() + [all_states[0].tolist()]),
    )
    assert output == tmp_path / capture_filename(corpus_id)
    assert output.is_file()
    assert not list(tmp_path.glob("*.part"))

    record = torch.load(output, map_location="cpu", weights_only=False)
    assert record["format"] == FORMAT
    assert record["id"] == corpus_id
    assert record["response_start"] == 2
    assert record["target_layer_ids"].tolist() == list(TARGET_LAYER_IDS)
    assert record["input_ids"].dtype == torch.int32
    assert record["input_ids"].tolist() == [11, 12, 13, 14, 15]
    assert record["loss_mask"].tolist() == [False, False, True, True, True]
    assert record["hidden_states_by_layer"].shape == (5, layers, hidden)
    assert record["hidden_states_by_layer"].dtype == torch.bfloat16
    assert record["hidden_states"].shape == (5, layers * hidden)
    torch.testing.assert_close(
        record["hidden_states_by_layer"], all_states.to(torch.bfloat16)
    )
    torch.testing.assert_close(
        record["hidden_states"], all_states.to(torch.bfloat16).view(5, -1)
    )
    # The explicit and flattened feature tensors are views of one serialized
    # storage, rather than doubling the NVMe footprint.
    assert (
        record["hidden_states"].untyped_storage().data_ptr()
        == record["hidden_states_by_layer"].untyped_storage().data_ptr()
    )

    before = os.stat(output).st_mtime_ns
    duplicate = accumulator.add_cpu_chunk(
        f"cmpl-{request_id}-0",
        torch.tensor([1]),
        torch.zeros(1, layers, hidden),
    )
    assert duplicate == output
    assert os.stat(output).st_mtime_ns == before


def test_non_capture_request_is_ignored(tmp_path) -> None:
    accumulator = CaptureAccumulator(tmp_path)
    result = accumulator.add_cpu_chunk(
        "cmpl-normal-request-0",
        torch.tensor([1]),
        torch.zeros(1, len(TARGET_LAYER_IDS), 2),
    )
    assert result is None
    assert list(tmp_path.iterdir()) == []
