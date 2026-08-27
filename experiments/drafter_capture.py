#!/usr/bin/env python3
"""Replay the on-policy drafter corpus as fixed-token prefill captures.

The server must be started with ``SB_HIDDEN_CAPTURE_DIR`` set (normally via
``/tmp/sb_env_overrides.json``).  This driver obtains the exact chat-template
token sequence from ``/tokenize``, submits those token ids to
``/v1/completions`` with one generated token, and waits for the EngineCore hook
to atomically persist the corresponding feature record.  ``max_tokens=1`` is
the smallest supported completion and exercises the full fixed prompt in
prefill; the generated token is not part of the capture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PHASE4_SRC = REPO / "src" / "phase4" / "src"
if str(PHASE4_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE4_SRC))

from shooting_brake_vllm.hidden_capture import (  # noqa: E402
    CAPTURE_ENV,
    capture_filename,
    make_capture_request_id,
)

# DISTINCT from datagen's pause file on purpose: datagen is paused
# precisely so the capture server can have the box to itself, and sharing
# one file made capture sleep through its own window (0 records in 1200 s,
# 2026-08-26). Pause capture with /tmp/sb_capture.pause.
PAUSE_FILE = "/tmp/sb_capture.pause"
DEFAULT_CORPUS = "benchmarks/results/drafter_corpus/pilot.jsonl"


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected response from {url}: {type(result).__name__}")
    return result


def _tokenize_record(
    base_url: str,
    model: str,
    prompt: str,
    raw_completion: str,
    timeout: float,
) -> tuple[list[int], int]:
    """Prompt render + verbatim raw completion -> one token sequence.

    The corpus stores the RAW completion text (thinking markup included)
    generated against the rendered chat prompt, so the replay is simply
    prompt tokens + completion tokens. No chat-template reassembly, no
    prefix ambiguity. The text->token round trip of the completion can
    differ from the original sampled ids at rare merge boundaries, which
    is acceptable for drafter training data.
    """
    prompt_result = _post_json(
        base_url + "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "continue_final_message": False,
        },
        timeout,
    )
    completion_result = _post_json(
        base_url + "/tokenize",
        {
            "model": model,
            "prompt": raw_completion,
            "add_special_tokens": False,
        },
        timeout,
    )
    prompt_tokens = [int(token) for token in prompt_result["tokens"]]
    completion_tokens = [int(token) for token in completion_result["tokens"]]
    if not completion_tokens:
        raise RuntimeError("corpus record has an empty completion")
    return prompt_tokens + completion_tokens, len(prompt_tokens)


def _load_corpus(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                corpus_id = str(raw["id"])
                prompt = raw["prompt"]
                completion = raw["raw_completion"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid corpus record at {path}:{line_number}") from exc
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(f"non-text prompt/completion at {path}:{line_number}")
            if corpus_id in seen:
                # Duplicates are a collection artifact, not a capture fault
                # (two datagen instances overlapped on 2026-08-26 and wrote
                # 86 shared ids). Capturing one twice would double-weight it
                # in training, so keep the first and account for the rest --
                # but never stall a multi-hour stage over upstream noise.
                duplicates.append(corpus_id)
                continue
            seen.add(corpus_id)
            records.append({"id": corpus_id, "prompt": prompt,
                            "raw_completion": completion})
    if duplicates:
        print(
            f"[drafter-capture] skipped {len(duplicates)} duplicate corpus "
            f"id(s), keeping first occurrence (e.g. {duplicates[0]})",
            flush=True,
        )
    return records


def capture(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    capture_dir.mkdir(parents=True, exist_ok=True)
    records = _load_corpus(Path(args.corpus))
    pending = [
        record
        for record in records
        if not (capture_dir / capture_filename(record["id"])).exists()
    ]
    print(
        f"[drafter-capture] {len(records) - len(pending)} done, "
        f"{len(pending)} remaining -> {capture_dir}",
        flush=True,
    )
    if not pending:
        return 0

    lock = threading.Lock()
    queue = list(reversed(pending))
    stats = {"ok": 0, "failed": 0, "tokens": 0, "started": time.time()}

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                record = queue.pop()
            output = capture_dir / capture_filename(record["id"])
            while Path(PAUSE_FILE).exists():
                time.sleep(args.pause_poll)
            for attempt in range(1, args.retries + 1):
                while Path(PAUSE_FILE).exists():
                    time.sleep(args.pause_poll)
                if output.exists():
                    break
                try:
                    tokens, response_start = _tokenize_record(
                        args.base_url,
                        args.model,
                        record["prompt"],
                        record["raw_completion"],
                        args.timeout,
                    )
                    request_id = make_capture_request_id(
                        record["id"], response_start, len(tokens)
                    )
                    _post_json(
                        args.base_url + "/v1/completions",
                        {
                            "model": args.model,
                            "prompt": tokens,
                            "add_special_tokens": False,
                            "max_tokens": 1,
                            "temperature": 0.0,
                            "echo": False,
                            "request_id": request_id,
                            # Prefix-cache hits are fatal for capture: cached
                            # tokens never run through the model, so their
                            # hidden states are never produced and the record
                            # never completes (observed rows=17 of 49,
                            # 2026-08-25). A unique salt forces a full run.
                            "cache_salt": request_id,
                        },
                        args.timeout,
                    )
                    deadline = time.monotonic() + args.write_timeout
                    while not output.exists() and time.monotonic() < deadline:
                        time.sleep(0.1)
                    if not output.exists():
                        raise TimeoutError(f"capture hook did not write {output}")
                    with lock:
                        stats["ok"] += 1
                        stats["tokens"] += len(tokens)
                        if stats["ok"] % args.report_every == 0:
                            elapsed = max(time.time() - stats["started"], 1e-6)
                            print(
                                f"[drafter-capture] {stats['ok']} records, "
                                f"{stats['tokens']} tokens, "
                                f"{stats['tokens'] / elapsed:.1f} tok/s, "
                                f"{stats['failed']} failed attempts",
                                flush=True,
                            )
                    break
                except Exception as exc:  # server reboot and capture retries are expected
                    with lock:
                        stats["failed"] += 1
                    if attempt == args.retries:
                        print(
                            f"[drafter-capture] giving up id={record['id']!r}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    time.sleep(min(args.retry_max_wait, args.retry_wait * attempt))

    threads = [
        threading.Thread(target=worker, name=f"capture-{index}", daemon=True)
        for index in range(args.concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    missing = sum(
        not (capture_dir / capture_filename(record["id"])).exists()
        for record in records
    )
    print(
        f"[drafter-capture] complete: {stats['ok']} captured, "
        f"{missing} missing, {stats['tokens']} tokens",
        flush=True,
    )
    return 1 if missing else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument(
        "--capture-dir",
        default=os.environ.get(CAPTURE_ENV),
        help=f"output directory (default: ${CAPTURE_ENV})",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8017")
    parser.add_argument("--model", default="shooting-brake-jota-r15")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--write-timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=1000)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--retry-max-wait", type=float, default=60.0)
    parser.add_argument("--pause-poll", type=float, default=10.0)
    parser.add_argument("--report-every", type=int, default=25)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not args.capture_dir:
        parser.error(f"--capture-dir or {CAPTURE_ENV} is required")
    if args.concurrency < 1 or args.retries < 1 or args.report_every < 1:
        parser.error("concurrency, retries, and report-every must be positive")
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
