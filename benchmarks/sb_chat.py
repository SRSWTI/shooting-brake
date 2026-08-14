#!/usr/bin/env python3
"""Terminal chat client for the Shooting Brake 88B server.

Multi-turn, streaming. Two things it is deliberately careful about:

- Token counts come from the server's ``usage.completion_tokens`` via
  ``stream_options.include_usage``, never from counting SSE chunks. A chunk
  is not a token, so counting frames gives a plausible-looking but wrong
  tok/s.
- ``reasoning_content`` is displayed but NOT written into history. Only the
  final ``content`` goes back to the model. Feeding hidden chain-of-thought
  back as ordinary assistant text changes the conversation the model sees.

Commands:  /reset  clear history      /stats  cumulative totals
           /sys <text>  set system prompt and clear history
           /think  toggle showing reasoning
           /quit or Ctrl-D  exit
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8016/v1/chat/completions"
MODEL = "shooting-brake-88b"
DEFAULT_SYSTEM = "you are jesco-- a coding agent"
# Thinking off by default: this model reasons at length and rarely emits final
# content inside a sane max_tokens, so thinking-on looks like an empty reply.
THINK_DEFAULT = False

DIM, BOLD, CYAN, GREEN, YELLOW, MAGENTA, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[35m", "\033[0m",
)


class Turn:
    __slots__ = ("content", "reasoning", "tokens", "ttft_s", "total_s")

    def __init__(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.tokens: int | None = None   # None => server did not report usage
        self.ttft_s = 0.0
        self.total_s = 0.0


def wait_for_server(timeout_s: float = 1200.0) -> bool:
    start = time.monotonic()
    spin = "|/-\\"
    i = 0
    while time.monotonic() - start < timeout_s:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8016/health", timeout=2
            ) as r:
                if r.status == 200:
                    sys.stdout.write(
                        f"\r{GREEN}server up{RESET} after "
                        f"{time.monotonic() - start:.0f}s"
                        + " " * 30 + "\n"
                    )
                    return True
        except Exception:
            pass
        sys.stdout.write(
            f"\r{DIM}waiting for server {spin[i % 4]} "
            f"{time.monotonic() - start:5.0f}s  "
            f"(weights, JIT, autotune, graph capture){RESET}"
        )
        sys.stdout.flush()
        i += 1
        time.sleep(1.0)
    sys.stdout.write("\n")
    return False


def stream_turn(messages: list[dict], show_think: bool, thinking: bool) -> Turn:
    payload = {
            "model": MODEL,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 4096,
            "temperature": 0.6,
            "top_p": 0.95,
    }
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )

    turn = Turn()
    content: list[str] = []
    reasoning: list[str] = []
    in_think = False
    first = True
    t0 = time.monotonic()

    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                turn.tokens = int(usage["completion_tokens"])

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            think = delta.get("reasoning") or delta.get("reasoning_content") or ""
            text = delta.get("content") or ""
            if not think and not text:
                continue
            if first:
                turn.ttft_s = time.monotonic() - t0
                first = False

            if think:
                reasoning.append(think)
                if show_think:
                    if not in_think:
                        sys.stdout.write(f"{MAGENTA}{DIM}")
                        in_think = True
                    sys.stdout.write(think)
                    sys.stdout.flush()
            if text:
                if in_think:
                    sys.stdout.write(f"{RESET}\n")
                    in_think = False
                content.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()

    if in_think:
        sys.stdout.write(RESET)
    turn.content = "".join(content)
    turn.reasoning = "".join(reasoning)
    turn.total_s = time.monotonic() - t0
    return turn


def main() -> int:
    print(
        f"{BOLD}Shooting Brake 88B{RESET} "
        f"{DIM}— 5090 dense NVFP4 + Arc Pro B70 int4 experts (126/layer){RESET}"
    )
    print(f"{DIM}system: {DEFAULT_SYSTEM}{RESET}")
    print(f"{DIM}/reset  /stats  /sys <text>  /reason  /nothink  /quit{RESET}\n")

    if not wait_for_server():
        print(f"{YELLOW}server did not come up; check /tmp/sb_serve.log{RESET}")
        return 1

    messages: list[dict] = [{"role": "system", "content": DEFAULT_SYSTEM}]
    show_think = False
    thinking = THINK_DEFAULT
    tot_tok = 0           # all completion tokens, for reporting volume
    tot_decode_tok = 0    # tokens attributable to decode: excludes each turn's first
    tot_decode_s = 0.0
    turns = 0
    unknown_usage = 0

    while True:
        try:
            user = input(f"\n{CYAN}you ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
            print(f"{DIM}history cleared, default system prompt restored{RESET}")
            continue
        if user == "/nothink":
            thinking = False
            print(f"{DIM}thinking disabled{RESET}")
            continue
        if user == "/reason":
            thinking = True
            show_think = True
            print(f"{DIM}thinking enabled and shown (needs a high token cap){RESET}")
            continue
        if user == "/think":
            show_think = not show_think
            print(f"{DIM}reasoning display {'on' if show_think else 'off'}{RESET}")
            continue
        if user == "/stats":
            if turns == 0:
                print(f"{DIM}no turns yet{RESET}")
            else:
                avg = tot_decode_tok / tot_decode_s if tot_decode_s else 0.0
                note = (
                    f" ({unknown_usage} turn(s) without server usage, excluded)"
                    if unknown_usage
                    else ""
                )
                print(
                    f"{DIM}{turns} turns · {tot_tok} completion tokens · "
                    f"{tot_decode_s:.1f}s decode · {avg:.2f} tok/s{note}{RESET}"
                )
            continue
        if user.startswith("/sys "):
            messages = [{"role": "system", "content": user[5:]}]
            print(f"{DIM}system prompt set, history cleared{RESET}")
            continue

        messages.append({"role": "user", "content": user})
        print(f"\n{GREEN}88b ▸ {RESET}", end="", flush=True)
        try:
            turn = stream_turn(messages, show_think, thinking)
        except (urllib.error.URLError, ConnectionError) as exc:
            print(f"\n{YELLOW}request failed: {exc}{RESET}")
            messages.pop()
            continue
        except KeyboardInterrupt:
            print(f"\n{YELLOW}interrupted{RESET}")
            messages.pop()
            continue

        # Only the final content re-enters the conversation. Reasoning is
        # shown on request but never fed back as assistant text.
        messages.append({"role": "assistant", "content": turn.content})
        turns += 1

        decode_s = max(turn.total_s - turn.ttft_s, 1e-9)
        if turn.tokens is None:
            unknown_usage += 1
            rate = "unavailable (server reported no usage)"
        else:
            tot_tok += turn.tokens
            tot_decode_tok += max(turn.tokens - 1, 0)
            tot_decode_s += decode_s
            rate = (
                f"{turn.tokens} tok · "
                f"{turn.tokens / turn.total_s:.2f} tok/s wall · "
                f"{max(turn.tokens - 1, 0) / decode_s:.2f} tok/s decode"
            )
        think_note = (
            f" · {len(turn.reasoning)} reasoning chars hidden"
            if turn.reasoning and not show_think
            else ""
        )
        print(f"\n{DIM}ttft {turn.ttft_s * 1000:.0f} ms · {rate}{think_note}{RESET}")

    if turns:
        avg = tot_decode_tok / tot_decode_s if tot_decode_s else 0.0
        print(
            f"\n{DIM}session: {turns} turns, {tot_tok} tokens, "
            f"{avg:.2f} tok/s decode{RESET}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
