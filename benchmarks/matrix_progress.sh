#!/usr/bin/env bash
# Append one progress line for a running benchmark matrix.
#
# Built for cron rather than a sleep loop: an 8.5-hour matrix outlives any
# shell, and both server deaths on 2026-08-21 were logind reaping the session
# (graceful SIGTERM, 1h46m idle, no CUDA error, no OOM). `loginctl
# enable-linger` plus cron is the combination that survives a logout; setsid
# nohup alone is not.
#
# Records the three things that actually kill a long run here:
#   1. cells completed          -- is it progressing, or wedged on one cell?
#   2. server health            -- an EngineCore death strands the rest
#   3. host MemAvailable/SwapFree -- device USM on the B70s costs host RAM 1:1
#      (~48 GiB for two cards, measured), so this box serves with single-digit
#      GiB free and swap is the first thing to go.
#
# Usage: matrix_progress.sh <output-root> [log-file]
set -euo pipefail

ROOT="${1:?usage: matrix_progress.sh <output-root> [log-file]}"
LOG="${2:-/tmp/sb_matrix_progress.log}"

ts=$(date '+%Y-%m-%d %H:%M:%S')

# One report.json per completed cell. find, not ls: the tree is nested
# <model>/ctx_<n>/<profile>/ and depth varies by profile.
cells=$(find "$ROOT" -name report.json 2>/dev/null | wc -l | tr -d ' ')
newest=$(find "$ROOT" -name report.json -printf '%T@ %p\n' 2>/dev/null \
         | sort -rn | head -1 | cut -d' ' -f2- || true)
age="-"
if [ -n "${newest:-}" ]; then
  age="$(( ($(date +%s) - $(stat -c %Y "$newest")) / 60 ))m"
fi

if curl -fsS --max-time 10 http://127.0.0.1:8017/health >/dev/null 2>&1; then
  health=UP
else
  health=DOWN
fi

runner=$(pgrep -cf 'matrix_runner' || true)
gllm=$(pgrep -cf 'guidellm' || true)

mem=$(awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo)
swap=$(awk '/^SwapFree:/ {printf "%.1f", $2/1048576}' /proc/meminfo)
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo '?')

printf '%s cells=%-4s last=%-6s health=%-4s runner=%s guidellm=%s mem=%sG swap=%sG gpu=%sMiB\n' \
  "$ts" "$cells" "$age" "$health" "$runner" "$gllm" "$mem" "$swap" "$gpu" >> "$LOG"

# Surface a stall loudly in the log itself: a live runner whose newest result is
# hours old is the failure mode that looks like success.
if [ "$runner" -gt 0 ] && [ "$health" = DOWN ]; then
  printf '%s   WARNING runner alive but server DOWN -- remaining cells will error\n' \
    "$ts" >> "$LOG"
fi
if [ "$runner" -eq 0 ] && [ "$cells" -gt 0 ]; then
  printf '%s   runner gone -- matrix finished or was killed (%s cells banked)\n' \
    "$ts" "$cells" >> "$LOG"
fi
