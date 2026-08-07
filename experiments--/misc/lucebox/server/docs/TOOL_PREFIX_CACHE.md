# Tool-heavy agent prefix caching

Tool schemas do not need a separate server or a second snapshot protocol.
`dflash_server` renders them into the stable system prefix and the native
turn-boundary cache snapshots that prefix after the first request. Later turns
restore the complete backend state and prefill only the new conversation
suffix.

For Qwen chat templates, the first request is effectively:

```text
[system + tool schemas] [user request] [assistant start]
                       ^ native snapshot boundary
```

The snapshot is a normal, backend-owned prefix snapshot. This is important for
hybrid architectures such as Qwen3.5/3.6: it contains attention KV, recurrent
state, convolution state, and the last-token seed together. Composing a
tool-only attention snapshot with an unrelated recurrent-state snapshot is not
a valid restore.

## Runtime behavior

- Turn 1 pays the tool-schema prefill once and commits a native inline snapshot.
- Turn 2 restores the snapshot keyed by the system/tool boundary.
- Later turns can restore progressively deeper conversation boundaries.
- The cache key is the exact token prefix. Changing a tool name, description,
  parameter, system prompt, or template produces a miss; incompatible KV state
  is never reused.
- With FlowKV enabled, the system/tool prefix and recent turns stay verbatim;
  only aged message content is eligible for compression.
- Client disconnects use the same C++ request lifecycle as every other request,
  so an unfinished snapshot reservation is aborted instead of being published.

Some backends round the physical snapshot down to a deterministic prefill-chunk
boundary. The cache key remains the longer safe chat boundary, while
`usage.timings.cached_prefix_tokens` reports the physical token count actually
restored. This is conservative: the backend recomputes the small remainder and
never restores past the stable prefix.

No tool-specific flag is required. The native server default enables the
in-memory prefix cache with 32 slots. Direct container launches inherit that
default; set `DFLASH_PREFIX_CACHE_SLOTS=N` to override it or explicitly set it
to `0` to disable prefix reuse.

## Reproducible benchmark

Start the production server with at least four prefix-cache slots, then run:

```bash
python3 server/scripts/benchmark_tool_prefix_cache.py \
  --url http://127.0.0.1:8080 \
  --json-out /tmp/tool-prefix-cache.json
```

The benchmark creates a unique, large tool schema and checks:

1. deterministic outputs stay identical and non-empty;
2. the first request is a cache miss;
3. every identical-tool follow-up is a cache hit;
4. restored and prefilled token counts exactly cover each effective backend
   prompt, including when FlowKV/PFlash rewrites the raw prompt;
5. changing one tool forces a cache miss;
6. cold prefill time is at least 3× the median warm prefill time by default.

It reads backend-only timing and cache metadata from `usage.timings`, rather
than inferring cache behavior from noisy end-to-end latency. Use
`--min-speedup` to change only the performance gate. In normal mode, all
correctness gates are enforced.

For an A/B against an unmodified `main` server, use the same command with
`--timing-only`. That mode accepts the older three-field `usage.timings` shape
and reports prefill speedup, but deliberately does not claim cache correctness
because `main` cannot expose the restored-token counts.
