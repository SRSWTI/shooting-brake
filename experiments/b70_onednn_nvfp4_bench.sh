#!/usr/bin/env bash
# oneDNN experimental grouped matmul with NVFP4 weights, on the B70 -- gate 1
# of the "use oneDNN for the prefill GEMM core" decision (see chat 2026-08-24).
#
# Measures f16 x f4_e2m1 (+ f8_e4m3 group-16 scales, NVFP4 exactly) grouped
# GEMM at r15 geometry across the fill ladder, both production GEMM shapes:
#   gate/up: [M_total x 3072] x [85 x 3072 x 2048]
#   w2     : [M_total x 1024] x [85 x 1024 x 3072]
# Bars to clear (docs/campaign-scorecard.md + fill-ladder session):
#   ours (NVFP4+dequant): 20.5 TFLOP/s @ ~30 rows/expert, ~23.6 @ ~120
#   Intel sycl-tla bf16 : 9.9-12.9   @ 30,            38-48 @ 120
# DISQUALIFIER: impl name containing "ref" (reference fallback = not usable).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../vendor/oneDNN" || exit 1
BD=build/tests/benchdnn/benchdnn
[ -x "$BD" ] || { echo "benchdnn not built"; exit 1; }

EXPERTS=85
sizes() { local r=$1 out=""; for _ in $(seq 1 $EXPERTS); do out+="${r}+"; done; printf '%s' "${out%+}"; }

echo "=== correctness + impl name (rows=30, gate/up shape) ==="
ONEDNN_VERBOSE=0 $BD --matmul --engine=gpu -v1 --mode=C \
  --dt=f16:f4_e2m1:f16 --stag=abx --wtag=abx --dtag=abx \
  --attr-scales=wei:per_tensor:f8_e4m3:16x1 \
  --grouped=0:${EXPERTS}:$(sizes 30) \
  $((30*EXPERTS))x3072:${EXPERTS}x3072x2048 2>&1 | grep -E "impl|PASSED|FAILED|UNIMPL|SKIPPED" | head -5

for rows in 30 120 244; do
  M=$((rows*EXPERTS))
  echo "=== perf rows/expert=${rows} (M_total=${M}) ==="
  for shape in "${M}x3072:${EXPERTS}x3072x2048" "${M}x1024:${EXPERTS}x1024x3072"; do
    $BD --matmul --engine=gpu --mode=P --max-ms-per-prb=2000 \
      --dt=f16:f4_e2m1:f16 --stag=abx --wtag=abx --dtag=abx \
      --attr-scales=wei:per_tensor:f8_e4m3:16x1 \
      --grouped=0:${EXPERTS}:$(sizes $rows) \
      "$shape" 2>&1 | grep -E "^0:|perf" | head -2
  done
done
