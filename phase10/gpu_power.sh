#!/usr/bin/env bash
# Shooting Brake — GPU power management for repeatable benchmarks.
#
# Two ways to keep the RTX 5090 from sitting at sustained 600 W for the
# whole matrix run. Use either or both.
#
#   cap     nvidia-smi -pl <watts>  — driver-level TDP cap. The card
#                                    throttles its own clocks to stay
#                                    under the limit under load and
#                                    drops to idle (~18 W) when idle.
#                                    Range on this board: 400..600 W.
#                                    This is the clean answer: no code
#                                    changes, no inter-request delays.
#
#   clocks  nvidia-smi --lock-gpu-clocks — pin clocks instead of power.
#                                    Tighter run-to-run variance, but
#                                    the card still pulls what the
#                                    workload demands at those clocks.
#
# Both require root (the driver power/clock controls are privileged).
# This script re-invokes itself under sudo when not already root, so you
# will be prompted for your password exactly once per invocation.
#
# Usage:
#   bash phase10/gpu_power.sh status          # current draw + limits
#   bash phase10/gpu_power.sh cap 570         # cap TDP at 570 W
#   bash phase10/gpu_power.sh cap 550
#   bash phase10/gpu_power.sh reset           # back to 600 W default
#   bash phase10/gpu_power.sh clocks 2400     # lock graphics clocks (MHz)
#   bash phase10/gpu_power.sh unlock          # remove clock lock
#
set -euo pipefail

# Re-exec as root if needed — nvidia-smi -pl / --lock-gpu-clocks are
# privileged, regardless of who owns the device node.
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

GPU="${GPU:-0}"

status () {
  nvidia-smi --query-gpu=power.draw,power.limit,clocks.gr,clocks.sm,clocks.mem \
    --format=csv -i "$GPU"
  echo "--- supported power range ---"
  nvidia-smi -q -d POWER -i "$GPU" \
    | grep -iE "Current Power Limit|Min Power Limit|Max Power Limit" || true
}

case "${1:-status}" in
  status)
    status
    ;;
  cap)
    watts="${2:?usage: gpu_power.sh cap <watts>}"
    nvidia-smi -i "$GPU" --power-limit="$watts"
    echo "[gpu_power] power limit set to ${watts} W on GPU $GPU"
    status
    ;;
  reset)
    nvidia-smi -i "$GPU" --power-limit=600
    echo "[gpu_power] reset to 600 W default"
    status
    ;;
  clocks)
    mhz="${2:?usage: gpu_power.sh clocks <mhz>}"
    # Lock both graphics and SM to the same target; memory left to default.
    nvidia-smi -i "$GPU" --lock-gpu-clocks="$mhz,$mhz"
    echo "[gpu_power] graphics clocks locked to ${mhz} MHz"
    status
    ;;
  unlock)
    nvidia-smi -i "$GPU" --reset-gpu-clocks || true
    echo "[gpu_power] clock lock removed"
    status
    ;;
  *)
    echo "usage: $0 {status|cap <watts>|reset|clocks <mhz>|unlock}" >&2
    exit 2
    ;;
esac
