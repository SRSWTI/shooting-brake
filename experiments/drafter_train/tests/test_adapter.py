from __future__ import annotations

import json
import sys
import weakref
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parents[1]
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(REPO_ROOT / "vendor" / "SpecForge"))

import adapter  # noqa: E402
from adapter import (  # noqa: E402
    FORMAT,
    HIDDEN_SIZE,
    SPECFORGE_FEATURE_KEYS,
    SPECFORGE_FORMAT,
    TARGET_LAYER_IDS,
    stage_capture,
)
from specforge.algorithms.common.hidden_states_data import (  # noqa: E402
    normalize_offline_sample,
)
from specforge.runtime.data_plane.offline_reader import (  # noqa: E402
    OfflineManifestReader,
)


def make_record(corpus_id: str, length: int = 5) -> dict[str, Any]:
    by_layer = torch.zeros(
        length,
        len(TARGET_LAYER_IDS),
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )
    response_start = 2
    return {
        "id": corpus_id,
        "format": FORMAT,
        "input_ids": torch.arange(length, dtype=torch.int32),
        "loss_mask": torch.arange(length) >= response_start,
        "response_start": response_start,
        "hidden_states": by_layer.view(length, -1),
        "hidden_states_by_layer": by_layer,
        "target_layer_ids": torch.tensor(TARGET_LAYER_IDS, dtype=torch.int64),
    }


def write_record(path: Path, corpus_id: str, length: int = 5) -> None:
    torch.save(make_record(corpus_id, length), path)


def test_stage_capture_links_files_in_specforge_offline_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "capture"
    output = tmp_path / "staged"
    source.mkdir()
    write_record(source / "b.pt", "b")
    write_record(source / "a.pt", "a")

    manifest = stage_capture(source, output)

    assert manifest["records"] == 2
    assert manifest["tokens"] == 10
    assert manifest["supervised_tokens"] == 6
    assert manifest["specforge_format"] == SPECFORGE_FORMAT
    assert manifest["feature_keys"] == list(SPECFORGE_FEATURE_KEYS)
    assert "samples" not in manifest
    assert json.loads((output / "manifest.json").read_text()) == manifest

    staged = sorted(output.glob("*.ckpt"))
    assert [path.name for path in staged] == ["a.ckpt", "b.ckpt"]
    assert staged[0].samefile(source / "a.pt")
    assert staged[1].samefile(source / "b.pt")

    reader = OfflineManifestReader(
        str(output),
        strategy="dflash",
        feature_keys=SPECFORGE_FEATURE_KEYS,
        target_repr=None,
    )
    refs = list(reader)
    assert len(refs) == 2
    assert all(not ref.feature_specs for ref in refs)
    raw_path = Path(refs[0].feature_store_uri.removeprefix("file://"))
    raw = torch.load(raw_path, weights_only=False, mmap=True)
    assert all(torch.is_tensor(raw[key]) for key in SPECFORGE_FEATURE_KEYS)
    normalized = normalize_offline_sample(raw, max_len=5)
    assert normalized["input_ids"].shape == (1, 5)
    assert normalized["loss_mask"].shape == (1, 5)
    assert normalized["hidden_states"].shape == (
        1,
        5,
        len(TARGET_LAYER_IDS) * HIDDEN_SIZE,
    )
    assert (
        normalized["hidden_states"].untyped_storage().data_ptr()
        == raw["hidden_states"].untyped_storage().data_ptr()
    )


def test_stage_capture_releases_each_mmap_record_before_loading_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "capture"
    output = tmp_path / "staged"
    source.mkdir()
    for index in range(3):
        (source / f"{index}.pt").touch()

    class TrackedRecord(dict[str, Any]):
        pass

    loaded: list[weakref.ReferenceType[TrackedRecord]] = []

    def tracked_load(path: Path) -> dict[str, Any]:
        if loaded:
            assert loaded[-1]() is None, "the prior tensor record was retained"
        record = TrackedRecord(make_record(path.stem))
        loaded.append(weakref.ref(record))
        return record

    monkeypatch.setattr(adapter, "load_record", tracked_load)

    assert stage_capture(source, output)["records"] == 3
    assert len(loaded) == 3
    assert all(reference() is None for reference in loaded)


def detach_flat_hidden(record: dict[str, Any]) -> None:
    hidden_states = record["hidden_states"]
    assert torch.is_tensor(hidden_states)
    record["hidden_states"] = hidden_states.clone()


def corrupt_loss_mask(record: dict[str, Any]) -> None:
    loss_mask = record["loss_mask"]
    assert torch.is_tensor(loss_mask)
    loss_mask[0] = True


def corrupt_layer_ids(record: dict[str, Any]) -> None:
    layer_ids = record["target_layer_ids"]
    assert torch.is_tensor(layer_ids)
    layer_ids[0] = 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (detach_flat_hidden, "without a tensor copy"),
        (corrupt_loss_mask, "loss_mask does not match"),
        (corrupt_layer_ids, "target_layer_ids"),
    ],
)
def test_stage_capture_rejects_invalid_capture_contract(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    source = tmp_path / "capture"
    source.mkdir()
    path = source / "bad.pt"
    record = make_record("bad")
    mutate(record)
    torch.save(record, path)

    with pytest.raises(ValueError, match=message):
        stage_capture(source, tmp_path / "staged")


def test_adapter_window_matches_training_config() -> None:
    """The staging filter must use the same window SpecForge truncates to.

    If these drift, records whose supervised region falls outside the
    training window reach SpecForge and abort the run with "require two
    consecutive supervised tokens" (cost: one failed pilot, 2026-08-26).
    """
    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "pilot.yaml").read_text()
    )
    assert adapter.MAX_LENGTH_DEFAULT == config["data"]["max_length"]
