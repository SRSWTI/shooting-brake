# RTX PRO 6000 GuideLLM SLO benchmark bundle

This folder contains the GuideLLM source snapshot, the context-length benchmark runner, and a vLLM launcher for `srswti/axe-superveloce-jota-118b-r15-nvfp4` on one NVIDIA RTX PRO 6000 Blackwell GPU (SM120).

The benchmark is an **uncached-prefill** test. The vLLM server is launched with prefix caching disabled, and every GuideLLM result is labeled `prefix_cache=disabled`.

## Contents

- `guidellm/`: runnable GuideLLM source snapshot.
- `guidellm/run_guidellm_context_benchmarks.sh`: resumable benchmark matrix.
- `serve_vllm_rtx_pro_6000.sh`: vLLM server launcher using the virtual environment's CUDA 13.2 compiler.

Generated benchmark results and the local `.venv` are deliberately not included.

## 1. Create the environment

Install `uv`, then run from this `benchmark_slo` folder:

```bash
cd guidellm
uv sync
uv pip install --python .venv/bin/python 'vllm==0.27.1'
uv pip install --python .venv/bin/python \
  'nvidia-cuda-nvcc==13.2.86' \
  'nvidia-cuda-crt==13.2.86' \
  'nvidia-nvvm==13.2.86' \
  'nvidia-nvjitlink==13.2.86'
cd ..
chmod +x serve_vllm_rtx_pro_6000.sh guidellm/run_guidellm_context_benchmarks.sh
```

If the model is gated or private, export your own Hugging Face token in the shell. Never store it in either script:

```bash
export HF_TOKEN='your_token_here'
```

The launcher expects Python 3.13 at `guidellm/.venv/lib/python3.13/site-packages`. For a different Python minor version, set `PYTHON_SITE` to that environment's `site-packages` directory.

## 2. Start vLLM on the RTX PRO 6000

From the `benchmark_slo` folder:

```bash
./serve_vllm_rtx_pro_6000.sh 2>&1 | tee vllm-8016.log
```

In another terminal, wait for readiness:

```bash
curl --fail http://127.0.0.1:8016/health
curl --fail http://127.0.0.1:8016/v1/models
```

### Equivalent vLLM command

Run this from `benchmark_slo/guidellm`:

```bash
CUDA_HOME="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13" \
PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH" \
.venv/bin/vllm serve srswti/axe-superveloce-jota-118b-r15-nvfp4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 160000 \
  --max-num-seqs 6 \
  --port 8016 \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --enable-mfu-metrics \
  --no-enable-prefix-caching
```

### vLLM arguments

| Argument | Purpose |
| --- | --- |
| `--trust-remote-code` | Allows the model repository's custom Laguna architecture code. |
| `--tensor-parallel-size 1` | Uses one GPU. |
| `--gpu-memory-utilization 0.90` | Lets vLLM use up to 90% of GPU memory for weights, compilation, and KV cache. |
| `--max-model-len 160000` | Sets the maximum prompt-plus-generation sequence length. |
| `--max-num-seqs 6` | Allows at most six active sequences in a scheduler iteration; excess benchmark streams queue. |
| `--port 8016` | Exposes the OpenAI-compatible API on port 8016. |
| `--enable-auto-tool-choice` | Enables automatic tool selection for requests containing tools. |
| `--tool-call-parser poolside_v1` | Parses the model's Poolside-format tool calls. |
| `--reasoning-parser poolside_v1` | Parses the model's Poolside-format reasoning output. |
| `--enable-mfu-metrics` | Enables vLLM model-flops-utilization metrics. |
| `--no-enable-prefix-caching` | Disables cross-request prefix-KV reuse so every prompt performs full prefill. |

### Why the CUDA environment is explicit

The RTX PRO 6000 Blackwell is SM120. FlashInfer JIT-compiles the model's NVFP4 dense and Mixture-of-Experts kernels. `nvidia-smi` reports the driver's supported CUDA level, not the compiler selected for that JIT. `CUDA_HOME` and `PATH` therefore point to the matching CUDA 13.2.86 compiler and headers installed inside `.venv`; using a system CUDA 12.8 `nvcc` fails SM120 compilation.

The launcher also creates two aliases sometimes absent from the pip CUDA package layout:

```text
nvidia/cu13/lib64 -> lib
nvidia/cu13/lib/libcudart.so -> libcudart.so.13
```

The first startup can pause for several minutes while FlashInfer compiles NVFP4 kernels. Later launches reuse the JIT cache.

## 3. Run the uncached GuideLLM matrix

Keep the vLLM terminal running. From another terminal, enter the downloaded GuideLLM directory and start the resumable run:

```bash
cd benchmark_slo/guidellm
./run_guidellm_context_benchmarks.sh \
  --results-dir benchmark-results/axe-context-matrix-no-prefix-cache \
  --skip-existing \
  2>&1 | tee benchmark-results/axe-context-matrix-no-prefix-cache.log
```

Re-run the exact same command after an interruption. With `--skip-existing`, a context/profile is skipped only when both its JSON and CSV files exist and are nonempty.

### Benchmark matrix

- Prompt contexts: 1,024; 4,096; 8,192; 16,384; 32,768; 65,536; 98,304; and 130,048 tokens.
- Requested generation: 512 tokens per request.
- Profiles per context:
  - `synchronous`: exactly one request in flight, establishing the sequential baseline.
  - `sweep`: eight adaptive request-rate strategies (`sweep_size=8`).
  - `concurrent`: fixed stream counts `1,2,3,4,5,6,8,12`.
- Per-strategy stopping constraints: 60 seconds, 100 requests, or 3 errors, whichever applies first.
- Outputs: `results.json` and `results.csv` under `<results-dir>/<context>/<profile>/`.
- Result metadata label: `prefix_cache=disabled`.

GuideLLM's synthetic prompts begin with a per-sample unique value and do not request shared synthetic prefixes. The authoritative cache control is the server-side `--no-enable-prefix-caching` argument. The GuideLLM label documents the server mode; it does not control vLLM caching.

### Benchmark arguments and environment overrides

| Variable | Default | Meaning |
| --- | --- | --- |
| `TARGET` | `http://127.0.0.1:8016` | vLLM base URL. |
| `MODEL` | `srswti/axe-superveloce-jota-118b-r15-nvfp4` | Served model identifier. |
| `OUTPUT_TOKENS` | `512` | Requested output length. |
| `MAX_DURATION` | `60` | Maximum seconds per strategy. |
| `MAX_REQUESTS` | `100` | Maximum requests per strategy. |
| `MAX_ERRORS` | `3` | Error cutoff per strategy. |
| `SWEEP_SIZE` | `8` | Number of sweep strategies. |
| `STREAMS_CSV` | `1,2,3,4,5,6,8,12` | Concurrent stream strategies. |
| `RESULTS_DIR` | UTC timestamped directory | Output root when `--results-dir` is omitted. |
| `DRY_RUN` | `0` | Set to `1` to print commands without sending requests. |
| `SKIP_EXISTING` | `0` | Set to `1`, or pass `--skip-existing`, to resume. |

`--results-dir PATH` chooses a stable output directory. `--skip-existing` provides safe resume behavior. The runner performs a health check before sending benchmark traffic.

Example shorter smoke run:

```bash
MAX_DURATION=30 MAX_REQUESTS=2 OUTPUT_TOKENS=32 \
  ./run_guidellm_context_benchmarks.sh \
  --results-dir benchmark-results/smoke \
  --skip-existing
```

Preview all generated commands without sending requests:

```bash
DRY_RUN=1 ./run_guidellm_context_benchmarks.sh \
  --results-dir benchmark-results/preview
```

## Result interpretation

Because prefix caching is disabled, prompt-processing throughput and time-to-first-token include the entire prefill cost for each request. This is useful for raw long-context and SLO characterization. It is intentionally more demanding than production workloads that reuse large system or document prefixes.
