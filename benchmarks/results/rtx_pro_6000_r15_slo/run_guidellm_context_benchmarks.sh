#!/usr/bin/env bash
set -Eeuo pipefail

# GuideLLM context-length matrix for a local OpenAI-compatible vLLM server.
# Usage:
#   ./run_guidellm_context_benchmarks.sh [--skip-existing] [--results-dir PATH]
# Resume safely by reusing the same directory:
#   ./run_guidellm_context_benchmarks.sh \
#     --results-dir benchmark-results/axe-context-matrix --skip-existing

TARGET="${TARGET:-http://127.0.0.1:8016}"
MODEL="${MODEL:-srswti/axe-superveloce-jota-118b-r15-nvfp4}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
MAX_DURATION="${MAX_DURATION:-60}"
MAX_REQUESTS="${MAX_REQUESTS:-100}"
MAX_ERRORS="${MAX_ERRORS:-3}"
SWEEP_SIZE="${SWEEP_SIZE:-8}"
STREAMS_CSV="${STREAMS_CSV:-1,2,3,4,5,6,8,12}"
RESULTS_DIR="${RESULTS_DIR:-benchmark-results/$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

while (($#)); do
  case "$1" in
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --results-dir)
      if (($# < 2)); then
        printf 'Error: --results-dir requires a path\n' >&2
        exit 2
      fi
      RESULTS_DIR="$2"
      shift 2
      ;;
    --help)
      printf 'Usage: %s [--skip-existing] [--results-dir PATH]\n' "$0"
      exit 0
      ;;
    *)
      printf 'Error: unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

# Labels use the requested names; values are binary-K token counts.
CONTEXTS=(
  "1k:1024"
  "4k:4096"
  "8k:8192"
  "16k:16384"
  "32k:32768"
  "64k:65536"
  "96k:98304"
  "127k:130048"
)

GUIDELLM=(uv run --no-sync guidellm)
COMMON_ARGS=(
  --backend "kind=openai_http,target=${TARGET},model=${MODEL},request_format=/v1/chat/completions"
  --constraint "kind=max_duration,seconds=${MAX_DURATION}"
  --constraint "kind=max_requests,count=${MAX_REQUESTS}"
  --constraint "kind=max_errors,count=${MAX_ERRORS}"
  --label "prefix_cache=disabled"
  --disable-console-interactive
)

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'COMMAND:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi

  "$@"
}

run_profile() {
  local label="$1"
  local prompt_tokens="$2"
  local profile="$3"
  local profile_config="$4"
  shift 4

  local output_prefix="${RESULTS_DIR}/${label}/${profile}"
  local json_path="${output_prefix}/results.json"
  local csv_path="${output_prefix}/results.csv"
  mkdir -p "${output_prefix}"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${json_path}" && -s "${csv_path}" ]]; then
    printf 'SKIP: %s %s already has JSON and CSV results\n' "${label}" "${profile}"
    return
  fi

  run_command "${GUIDELLM[@]}" run \
    "${COMMON_ARGS[@]}" \
    --profile "${profile_config}" \
    "$@" \
    --data "kind=synthetic_text,prompt_tokens=${prompt_tokens},output_tokens=${OUTPUT_TOKENS}" \
    --output "kind=json,path=${json_path}" \
    --output "kind=csv,path=${csv_path}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  curl --fail --silent --show-error "${TARGET}/health" >/dev/null
fi

mkdir -p "${RESULTS_DIR}"
printf 'GuideLLM benchmark results: %s\n' "${RESULTS_DIR}"
printf 'Target: %s\nModel: %s\nOutput tokens: %s\n' \
  "${TARGET}" "${MODEL}" "${OUTPUT_TOKENS}"
printf 'Limits per strategy: %ss or %s requests; max errors: %s\n' \
  "${MAX_DURATION}" "${MAX_REQUESTS}" "${MAX_ERRORS}"
printf 'Concurrent streams: %s\n\n' "${STREAMS_CSV}"
printf 'Skip completed profiles: %s\n\n' "${SKIP_EXISTING}"

for context in "${CONTEXTS[@]}"; do
  label="${context%%:*}"
  prompt_tokens="${context##*:}"

  printf '=== %s context (%s prompt tokens) ===\n' "${label}" "${prompt_tokens}"

  # Sequential baseline: exactly one in-flight request.
  run_profile "${label}" "${prompt_tokens}" synchronous "kind=synchronous"

  # Adaptive request-rate sweep: baseline, throughput, and interpolated rates.
  run_profile "${label}" "${prompt_tokens}" sweep \
    "kind=sweep,sweep_size=${SWEEP_SIZE}"

  # Fixed concurrent-stream strategies: 1,2,3,4,5,6,8,12 by default.
  run_profile "${label}" "${prompt_tokens}" concurrent "kind=concurrent" \
    --override profile.streams "${STREAMS_CSV}"
done

printf '\nAll GuideLLM benchmarks completed. Results: %s\n' "${RESULTS_DIR}"
