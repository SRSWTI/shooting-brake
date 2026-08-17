#!/usr/bin/env bash
# B70 frequency/power tuning — the one script that needs root.
#
# Whitelist once in sudoers (visudo):
#   shooting-brake007 ALL=(root) NOPASSWD: /home/shooting-brake007/srswti/shooting-brake/scripts/b70_tune.sh
#
# Cards on this box (stable by PCI address, resolved at runtime):
#   0000:11:00.0  idle/test B70   (ZE_AFFINITY_MASK=0)
#   0000:15:00.0  serving B70     (ZE_AFFINITY_MASK=1, Gen4 x4)
#
# Usage:
#   b70_tune.sh <test|serve> pin <MHz>     # min_freq = max_freq = MHz (fixed request)
#   b70_tune.sh <test|serve> floor <MHz>   # min_freq only
#   b70_tune.sh <test|serve> reset         # min 400 / max 2800 (hardware defaults)
#   b70_tune.sh <test|serve> cap <W>       # hwmon power1_cap in watts
#   b70_tune.sh <test|serve> status        # read-only: freq + power state
#
# Findings that shaped this (benchmarks/results/b70_gemv_audit/):
#   - the min==max pin HOLDS on this kernel (act_freq 2633-2800 under load);
#     no slpc_ignore_eff_freq needed
#   - pin is mandatory for standalone microbenches (2.6x cold variance),
#     buys NOTHING for serving ITL at C=1 (decode self-warms the card)
#   - act_freq is PCODE truth; cur_freq only shows GuC's request
set -euo pipefail

card_for() {
  local pci
  case "$1" in
    test)  pci="0000:11:00.0" ;;
    serve) pci="0000:15:00.0" ;;
    *) echo "unknown card alias: $1 (want test|serve)" >&2; exit 2 ;;
  esac
  for c in /sys/class/drm/card*/device; do
    [ "$(basename "$(readlink -f "$c")")" = "$pci" ] && { dirname "$c"; return; }
  done
  echo "no drm card for $pci" >&2; exit 2
}

CARD="$(card_for "${1:?card alias}")"
FREQ="$CARD/device/tile0/gt0/freq0"
OP="${2:?operation}"

hwmon_cap() {
  local h
  for h in "$CARD"/device/hwmon/hwmon*; do
    [ -e "$h/power1_cap" ] && { echo "$h/power1_cap"; return; }
  done
  echo "no power1_cap under $CARD/device/hwmon" >&2; exit 2
}

case "$OP" in
  pin)
    MHZ="${3:?MHz}"
    echo "$MHZ" > "$FREQ/max_freq"
    echo "$MHZ" > "$FREQ/min_freq"
    ;;
  floor)
    echo "${3:?MHz}" > "$FREQ/min_freq"
    ;;
  reset)
    echo 400  > "$FREQ/min_freq"
    echo 2800 > "$FREQ/max_freq"
    ;;
  cap)
    W="${3:?watts}"
    echo $((W * 1000000)) > "$(hwmon_cap)"
    ;;
  status)
    for f in min_freq max_freq cur_freq act_freq rpe_freq power_profile; do
      printf '%-14s %s\n' "$f" "$(cat "$FREQ/$f")"
    done
    if cap=$(hwmon_cap 2>/dev/null); then
      printf '%-14s %s uW\n' power1_cap "$(cat "$cap")"
    fi
    ;;
  *)
    echo "unknown op: $OP (want pin|floor|reset|cap|status)" >&2; exit 2 ;;
esac
echo "ok: $1 ($CARD) $OP ${3:-}"
