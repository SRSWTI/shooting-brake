#!/usr/bin/env python3
"""Terminal agent for the Shooting Brake 88B server: chat + tools + thinking.

Talks to the running vLLM server directly (no API key, no external deps --
stdlib only, matches sb_chat.py's approach). Adds local tool execution
(read/write/edit/glob/grep/bash) on top of sb_chat.py's proven streaming
primitives (reasoning-field separation, usage-based token accounting).

Tool-call protocol: this checkpoint's chat template (checked directly --
tokenizer_config's chat_template.jinja) does NOT use Hermes-style JSON
<tool_call>{"name":...} tags. It uses a custom XML dialect:

    <tool_call>
    <function=read>
    <parameter=path>
    src/foo.py
    </parameter>
    </function>
    </tool_call>

Measured directly against the live server: sending `tools=[...]` in the
request body 400s outright ("'auto' tool choice requires
--enable-auto-tool-choice and --tool-call-parser to be set") -- vLLM gates
the `tools` param on that flag at the REQUEST layer, not just structured
response parsing, and the server was not launched with it. Rather than
restart the server, this client renders the exact same tool-list block the
chat template would have produced (copied verbatim from
chat_template.jinja) directly into the system message -- ordinary content,
no request-layer gate. Tool RESULTS still go back as role="tool" messages,
which the template natively wraps in <tool_response> regardless of how the
tool list got into the prompt.

Reasoning is shown live (gray) by default -- this model reasons at length,
so max_tokens defaults higher than a non-agentic chat session needs.
Reasoning is NEVER written back into history (sb_chat.py's rule): only
final content, and the model's own tool_call XML when it makes one.

Commands:  /reset  clear history        /stats  cumulative totals
           /sys <text>  set system prompt and clear history
           /think  toggle showing reasoning (default: shown)
           /health  curl-equivalent server status
           /quit or Ctrl-D  exit
"""
from __future__ import annotations

import glob as globlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8016/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8016/health"
MODELS_URL = "http://127.0.0.1:8016/v1/models"
METRICS_URL = "http://127.0.0.1:8016/metrics"
MODEL = "shooting-brake-88b"
DEFAULT_SYSTEM = (
    "You are jesco, a terse, precise coding agent operating directly on the "
    "user's filesystem via the provided tools. Read before you edit. Prefer "
    "small, verified changes. State assumptions; don't invent file contents."
)
MAX_TOOL_ITERS = 20
BASH_TIMEOUT_S = 60

DIM, BOLD, CYAN, GREEN, YELLOW, MAGENTA, RED, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m",
    "\033[35m", "\033[31m", "\033[0m",
)

# -- local tools --------------------------------------------------------------


MAX_READ_LINES = 2000
MAX_READ_CHARS = 40_000  # catches single huge/minified lines a line cap misses


def _read(args: dict) -> str:
    """Read with hard caps regardless of what the model requests.

    A prior session hit a real incident from this: the model read a 15 MB
    JSON report with no `limit`, the unbounded result went into history as
    a tool message, and every subsequent turn -- including plain "hi" --
    400'd with a context-length error, because the giant message never
    left history. These caps make that structurally impossible: `limit` can
    only shrink the read, never grow past MAX_READ_LINES, and the returned
    string is additionally capped in bytes for the pathological case of a
    few enormous (e.g. minified) lines.
    """
    try:
        offset = max(0, int(args.get("offset", 0) or 0))
        requested = args.get("limit")
        limit = min(int(requested), MAX_READ_LINES) if requested else MAX_READ_LINES
        selected = []
        with open(args["path"]) as f:
            for _ in range(offset):
                if f.readline() == "":
                    break
            for i, ln in enumerate(f):
                if i >= limit:
                    selected.append(None)  # marks "more remains"
                    break
                selected.append(ln)
        truncated_by_lines = selected and selected[-1] is None
        if truncated_by_lines:
            selected = selected[:-1]
        out = "".join(f"{offset + i + 1:5}| {ln}" for i, ln in enumerate(selected))
        truncated_by_chars = len(out) > MAX_READ_CHARS
        if truncated_by_chars:
            out = out[:MAX_READ_CHARS]
        if truncated_by_lines or truncated_by_chars:
            why = "line cap" if truncated_by_lines else "char cap"
            out += (
                f"\n... [truncated at {why}: use offset={offset + limit} to "
                f"continue, or a narrower limit]"
            )
        return out or "(empty selection)"
    except Exception as e:
        return f"error: {e}"


def _write(args: dict) -> str:
    try:
        path = args["path"]
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            f.write(args["content"])
        return f"ok: wrote {len(args['content'])} bytes to {path}"
    except Exception as e:
        return f"error: {e}"


def _edit(args: dict) -> str:
    try:
        path = args["path"]
        with open(path) as f:
            text = f.read()
        old, new = args["old"], args["new"]
        if old not in text:
            return "error: old string not found verbatim in file"
        count = text.count(old)
        do_all = str(args.get("all", "")).lower() in ("true", "1", "yes")
        if not do_all and count > 1:
            return f"error: old string appears {count} times; pass all=true or make it unique"
        replacement = text.replace(old, new) if do_all else text.replace(old, new, 1)
        with open(path, "w") as f:
            f.write(replacement)
        return f"ok: {count if do_all else 1} replacement(s) in {path}"
    except Exception as e:
        return f"error: {e}"


def _glob(args: dict) -> str:
    try:
        base = args.get("path", ".")
        pattern = (base.rstrip("/") + "/" + args["pat"]).replace("//", "/")
        files = globlib.glob(pattern, recursive=True)
        files.sort(key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0, reverse=True)
        return "\n".join(files[:200]) or "(no matches)"
    except Exception as e:
        return f"error: {e}"


def _grep(args: dict) -> str:
    try:
        pattern = re.compile(args["pat"])
        base = args.get("path", ".")
        hits = []
        for fp in globlib.glob(base.rstrip("/") + "/**", recursive=True):
            if not os.path.isfile(fp) or ".git/" in fp:
                continue
            try:
                with open(fp, errors="ignore") as f:
                    for n, line in enumerate(f, 1):
                        if pattern.search(line):
                            line_text = line.rstrip()
                            if len(line_text) > 300:
                                line_text = line_text[:300] + "...[line truncated]"
                            hits.append(f"{fp}:{n}:{line_text}")
                            if len(hits) >= 200:
                                raise StopIteration
            except StopIteration:
                break
            except Exception:
                continue
        out = "\n".join(hits[:200]) or "(no matches)"
        return out[:MAX_READ_CHARS]
    except Exception as e:
        return f"error: {e}"


def _bash(args: dict) -> str:
    try:
        timeout = float(args.get("timeout", BASH_TIMEOUT_S) or BASH_TIMEOUT_S)
        result = subprocess.run(
            args["cmd"], shell=True, capture_output=True, text=True, timeout=timeout,
        )
        out = (result.stdout + result.stderr).strip()
        if len(out) > 8000:
            out = out[:8000] + f"\n... [truncated, {len(out)} bytes total]"
        return out or "(empty output)"
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {args.get('timeout', BASH_TIMEOUT_S)}s"
    except Exception as e:
        return f"error: {e}"


# name -> (description, {param: (json_type, required, description)}, fn)
TOOLS: dict[str, tuple[str, dict[str, tuple[str, bool, str]]]] = {
    "read": (
        "Read a file with line numbers. Use offset/limit for large files.",
        {
            "path": ("string", True, "file path to read"),
            "offset": ("integer", False, "0-indexed line to start at"),
            "limit": ("integer", False, "max lines to return"),
        },
    ),
    "write": (
        "Create a file or overwrite it entirely with new content.",
        {
            "path": ("string", True, "file path to write"),
            "content": ("string", True, "full file content"),
        },
    ),
    "edit": (
        "Replace an exact substring in a file. `old` must match verbatim and "
        "be unique unless all=true.",
        {
            "path": ("string", True, "file path to edit"),
            "old": ("string", True, "exact text to replace"),
            "new": ("string", True, "replacement text"),
            "all": ("boolean", False, "replace every occurrence, not just the first"),
        },
    ),
    "glob": (
        "Find files matching a glob pattern (e.g. '**/*.py'), newest first.",
        {"pat": ("string", True, "glob pattern"), "path": ("string", False, "base dir, default .")},
    ),
    "grep": (
        "Search files under a directory for a regex pattern.",
        {"pat": ("string", True, "regex pattern"), "path": ("string", False, "base dir, default .")},
    ),
    "bash": (
        "Run a shell command and return combined stdout+stderr.",
        {"cmd": ("string", True, "shell command"), "timeout": ("number", False, "seconds, default 60")},
    ),
}
TOOL_FNS = {"read": _read, "write": _write, "edit": _edit, "glob": _glob, "grep": _grep, "bash": _bash}


def make_tool_schema() -> list[dict]:
    out = []
    for name, (desc, params) in TOOLS.items():
        properties = {p: {"type": t, "description": d} for p, (t, req, d) in params.items()}
        required = [p for p, (t, req, d) in params.items() if req]
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        })
    return out


# Verbatim from chat_template.jinja (the "if you choose to call a function"
# instruction block) -- the server 400s on the `tools` API param without
# --enable-auto-tool-choice, so the tool list is rendered by hand into the
# system message instead. Kept byte-identical to what the template would
# have produced so the model sees exactly what it was trained on.
_TOOL_CALL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
    "</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow "
    "the specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags\n- Required parameters "
    "MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT "
    "after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>"
)


def render_system_prompt(user_system: str) -> str:
    lines = ["# Tools\n\nYou have access to the following functions:\n\n<tools>"]
    for schema in make_tool_schema():
        lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("</tools>")
    block = "\n".join(lines) + _TOOL_CALL_INSTRUCTIONS
    content = user_system.strip()
    return block + ("\n\n" + content if content else "")


def run_tool(name: str, args: dict) -> str:
    fn = TOOL_FNS.get(name)
    if fn is None:
        return f"error: unknown tool '{name}'"
    try:
        return fn(args)
    except Exception as e:
        return f"error: {e}"


def _coerce(name: str, param: str, raw: str) -> object:
    raw = raw.strip("\n")
    spec = TOOLS.get(name, ("", {}))[1].get(param)
    jtype = spec[0] if spec else "string"
    if jtype == "integer":
        try:
            return int(raw.strip())
        except ValueError:
            return raw
    if jtype == "number":
        try:
            return float(raw.strip())
        except ValueError:
            return raw
    if jtype == "boolean":
        return raw.strip().lower() in ("true", "1", "yes")
    return raw


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([\w.\-]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_tool_calls(content: str) -> list[tuple[str, dict]]:
    """Parse this checkpoint's XML tool-call dialect out of assistant content."""
    calls = []
    for m in TOOL_CALL_RE.finditer(content):
        name, body = m.group(1), m.group(2)
        args = {}
        for pm in PARAM_RE.finditer(body):
            pname, pval = pm.group(1), pm.group(2)
            args[pname] = _coerce(name, pname, pval)
        calls.append((name, args))
    return calls


def strip_tool_calls(content: str) -> str:
    """Prose portion of a turn, with tool_call XML removed (for history/display)."""
    return TOOL_CALL_RE.sub("", content).strip()


# -- server I/O -----------------------------------------------------------


def wait_for_server(timeout_s: float = 1200.0) -> bool:
    start = time.monotonic()
    spin = "|/-\\"
    i = 0
    while time.monotonic() - start < timeout_s:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    sys.stdout.write(
                        f"\r{GREEN}server up{RESET} after {time.monotonic() - start:.0f}s"
                        + " " * 30 + "\n"
                    )
                    return True
        except Exception:
            pass
        sys.stdout.write(
            f"\r{DIM}waiting for server {spin[i % 4]} {time.monotonic() - start:5.0f}s{RESET}"
        )
        sys.stdout.flush()
        i += 1
        time.sleep(1.0)
    sys.stdout.write("\n")
    return False


def health_check() -> str:
    lines = []
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as r:
            lines.append(f"health: HTTP {r.status}")
    except Exception as e:
        return f"{RED}server unreachable: {e}{RESET}"
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=3) as r:
            data = json.loads(r.read())
            m = data["data"][0]
            lines.append(f"model: {m['id']}  max_model_len: {m.get('max_model_len')}")
    except Exception as e:
        lines.append(f"models endpoint: error {e}")
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=3) as r:
            text = r.read().decode("utf-8", "replace")
        for key in ("num_requests_running", "num_requests_waiting", "gpu_cache_usage_perc"):
            m = re.search(rf'vllm:{key}\{{[^}}]*\}}\s+([\d.eE+-]+)', text)
            if m:
                lines.append(f"{key}: {m.group(1)}")
    except Exception:
        pass
    return "\n".join(lines)


class TagFilter:
    """Hides <tool_call>...</tool_call> spans from a live text stream.

    Chunks arrive as arbitrary substrings, so the open tag can straddle two
    deltas; holds back a suffix that could be a partial tag match rather than
    printing it prematurely.
    """
    OPEN, CLOSE = "<tool_call>", "</tool_call>"

    def __init__(self) -> None:
        self._pending = ""
        self._inside = False
        self._announced = False

    def feed(self, chunk: str) -> str:
        self._pending += chunk
        out = []
        while True:
            if not self._inside:
                idx = self._pending.find(self.OPEN)
                if idx == -1:
                    holdback = min(len(self.OPEN) - 1, len(self._pending))
                    out.append(self._pending[:len(self._pending) - holdback] if holdback else self._pending)
                    self._pending = self._pending[len(self._pending) - holdback:] if holdback else ""
                    break
                out.append(self._pending[:idx])
                self._pending = self._pending[idx + len(self.OPEN):]
                self._inside = True
                self._announced = False
            else:
                idx = self._pending.find(self.CLOSE)
                if not self._announced:
                    out.append(f"{DIM}{YELLOW}[tool call]{RESET}")
                    self._announced = True
                if idx == -1:
                    break
                self._pending = self._pending[idx + len(self.CLOSE):]
                self._inside = False
        return "".join(out)


class ChatRequestError(Exception):
    """A 400/4xx/5xx from the server, with vLLM's actual error message.

    Bare urllib.error.HTTPError prints as "HTTP Error 400: Bad Request" --
    the useful text is in the response body, which HTTPError does not
    surface by default. This reads it once at the failure site.
    """

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        try:
            self.message = json.loads(body).get("error", {}).get("message", body)
        except Exception:
            self.message = body
        super().__init__(f"HTTP {status}: {self.message}")


class Turn:
    __slots__ = ("content", "reasoning", "tokens", "ttft_s", "total_s")

    def __init__(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.tokens: int | None = None
        self.ttft_s = 0.0
        self.total_s = 0.0


def stream_turn(messages: list[dict], show_think: bool, max_tokens: int) -> Turn:
    payload = {
        "model": MODEL,
        "messages": messages,
        # NOT "tools"/"tool_choice": the server 400s on those without
        # --enable-auto-tool-choice. The tool list is already baked into
        # messages[0] by render_system_prompt() -- see module docstring.
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})

    turn = Turn()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    in_think = False
    first = True
    t0 = time.monotonic()
    tag_filter = TagFilter()

    try:
        resp_cm = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        raise ChatRequestError(e.code, e.read().decode("utf-8", "replace")) from None

    with resp_cm as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
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
                reasoning_parts.append(think)
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
                content_parts.append(text)
                visible = tag_filter.feed(text)
                if visible:
                    sys.stdout.write(visible)
                    sys.stdout.flush()

    if in_think:
        sys.stdout.write(RESET)
    turn.content = "".join(content_parts)
    turn.reasoning = "".join(reasoning_parts)
    turn.total_s = time.monotonic() - t0
    return turn

CONTEXT_ERROR_MARKERS = ("maximum context length", "context length", "context_length_exceeded")
HISTORY_CHAR_BUDGET = 300_000  # conservative proxy, well under 131,072 tokens


def trim_oversized_history(messages: list[dict]) -> int:
    """Stub out the largest messages in history until under budget.

    Protects messages[0] (system) and the very last message (the turn that
    just failed) so the retry is otherwise identical. Stubs content in
    place rather than deleting the message: role/position sequencing is
    what the chat template's tool_call/tool_response pairing depends on,
    and removing a message could desync that pairing in ways a missing
    tool response would not.
    """
    protect = {0, len(messages) - 1}
    sized = sorted(
        (i for i in range(len(messages)) if i not in protect),
        key=lambda i: len(str(messages[i].get("content", ""))),
        reverse=True,
    )
    total = sum(len(str(m.get("content", ""))) for m in messages)
    dropped = 0
    for i in sized:
        if total <= HISTORY_CHAR_BUDGET:
            break
        sz = len(str(messages[i].get("content", "")))
        if sz < 2000:
            break  # remaining messages are small; stop, not worth stubbing
        messages[i]["content"] = (
            f"[pruned: this {sz}-char result was too large and was dropped "
            f"from history to recover from a context-length error]"
        )
        total -= sz
        dropped += 1
    return dropped

def preview(text: str, n: int = 100) -> str:
    lines = text.split("\n")
    p = lines[0][:n]
    if len(lines) > 1:
        p += f" ... +{len(lines) - 1} lines"
    elif len(lines[0]) > n:
        p += "..."
    return p


def main() -> int:
    if len(sys.argv) > 1:
        try:
            os.chdir(sys.argv[1])
        except FileNotFoundError:
            print(f"{RED}directory not found: {sys.argv[1]}{RESET}")
            return 1

    print(f"{BOLD}Shooting Brake 88B agent{RESET} {DIM}— {os.getcwd()}{RESET}")
    print(f"{DIM}tools: {', '.join(TOOLS)}{RESET}")
    print(f"{DIM}/reset  /stats  /sys <text>  /think  /health  /quit{RESET}\n")

    if not wait_for_server():
        print(f"{YELLOW}server did not come up{RESET}")
        return 1
    print(health_check())

    messages: list[dict] = [{"role": "system", "content": render_system_prompt(DEFAULT_SYSTEM)}]
    show_think = True
    max_tokens = int(os.environ.get("SB_AGENT_MAX_TOKENS", "8192"))
    tot_tok = 0
    tot_decode_tok = 0
    tot_decode_s = 0.0
    turns = 0
    unknown_usage = 0

    while True:
        try:
            user = input(f"\n{CYAN}you \u25b8 {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            messages = [{"role": "system", "content": render_system_prompt(DEFAULT_SYSTEM)}]
            print(f"{DIM}history cleared{RESET}")
            continue
        if user == "/think":
            show_think = not show_think
            print(f"{DIM}reasoning display {'on' if show_think else 'off'}{RESET}")
            continue
        if user == "/health":
            print(health_check())
            continue
        if user == "/stats":
            if turns == 0:
                print(f"{DIM}no turns yet{RESET}")
            else:
                avg = tot_decode_tok / tot_decode_s if tot_decode_s else 0.0
                note = f" ({unknown_usage} without usage){RESET}" if unknown_usage else ""
                print(f"{DIM}{turns} turns \u00b7 {tot_tok} completion tokens \u00b7 "
                      f"{tot_decode_s:.1f}s decode \u00b7 {avg:.2f} tok/s{note}")
            continue
        if user.startswith("/sys "):
            messages = [{"role": "system", "content": render_system_prompt(user[5:])}]
            print(f"{DIM}system prompt set, history cleared{RESET}")
            continue

        messages.append({"role": "user", "content": user})
        print(f"\n{GREEN}88b \u25b8 {RESET}", end="", flush=True)

        for _ in range(MAX_TOOL_ITERS):
            try:
                turn = stream_turn(messages, show_think, max_tokens)
            except ChatRequestError as exc:
                is_context_err = any(m in exc.message.lower() for m in CONTEXT_ERROR_MARKERS)
                if is_context_err:
                    dropped = trim_oversized_history(messages)
                    if dropped:
                        print(f"\n{YELLOW}context length exceeded -- pruned "
                              f"{dropped} oversized message(s) from history, "
                              f"retrying{RESET}")
                        continue
                print(f"\n{YELLOW}server error: {exc.message}{RESET}")
                messages.pop()
                break
            except (urllib.error.URLError, ConnectionError) as exc:
                print(f"\n{YELLOW}request failed: {exc}{RESET}")
                messages.pop()
                break
            except KeyboardInterrupt:
                print(f"\n{YELLOW}interrupted{RESET}")
                break

            turns += 1
            decode_s = max(turn.total_s - turn.ttft_s, 1e-9)
            if turn.tokens is None:
                unknown_usage += 1
            else:
                tot_tok += turn.tokens
                tot_decode_tok += max(turn.tokens - 1, 0)
                tot_decode_s += decode_s
            print(f"\n{DIM}ttft {turn.ttft_s * 1000:.0f} ms \u00b7 "
                  f"{turn.tokens or '?'} tok{RESET}")

            calls = parse_tool_calls(turn.content)
            messages.append({"role": "assistant", "content": turn.content})

            if not calls:
                break

            for name, args in calls:
                arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                print(f"{GREEN}\u23fa {name}{RESET}({DIM}{arg_str[:80]}{RESET})")
                result = run_tool(name, args)
                print(f"  {DIM}\u23bf {preview(result)}{RESET}")
                messages.append({"role": "tool", "content": result})

            print(f"\n{GREEN}88b \u25b8 {RESET}", end="", flush=True)
        else:
            print(f"{YELLOW}hit {MAX_TOOL_ITERS} tool iterations, stopping{RESET}")

        print()

    if turns:
        avg = tot_decode_tok / tot_decode_s if tot_decode_s else 0.0
        print(f"\n{DIM}session: {turns} turns, {tot_tok} tokens, {avg:.2f} tok/s decode{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
