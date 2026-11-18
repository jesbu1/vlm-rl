#!/usr/bin/env bash
# Run a command under a hard memory cap so it can't take a shared machine down with it.
#   scripts/memcap.sh 48G python scripts/train.py ...      # killed (exit 137) if it exceeds 48 GB of RAM
#   MEMCAP_MARGIN=0.2 scripts/memcap.sh 32G make test-slow
# Uses a transient systemd user scope (cgroup v2 MemoryMax, no swap): the kernel OOM-kills only this command.
# Refuses to start (exit 75) unless MemAvailable >= cap * (1 + MEMCAP_MARGIN) (default 10%).
# Falls back to running uncapped (with a warning) where user systemd scopes are unavailable (macOS, containers).
set -euo pipefail
CAP=${1:?usage: memcap.sh <cap, e.g. 48G> <command...>}; shift
[ $# -gt 0 ] || { echo "memcap: no command" >&2; exit 2; }
to_kb() { local v=${1^^}; case $v in *G) echo $(( ${v%G} * 1024 * 1024 ));; *M) echo $(( ${v%M} * 1024 ));; *) echo $(( v / 1024 ));; esac; }
need=$(to_kb "$CAP")
if [ -r /proc/meminfo ]; then
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  want=$(awk -v n="$need" -v m="${MEMCAP_MARGIN:-0.1}" 'BEGIN {printf "%d", n * (1 + m)}')
  if [ "$avail" -lt "$want" ]; then
    echo "memcap: only $((avail / 1048576)) GB available, need $((want / 1048576)) GB for cap $CAP; not starting" >&2
    exit 75
  fi
fi
if ! command -v systemd-run >/dev/null || ! systemd-run --user --scope -q true 2>/dev/null; then
  echo "memcap: no user systemd scopes here; running WITHOUT a memory cap" >&2
  exec "$@"
fi
unit="memcap-$$-$(date +%s)"
set +e
systemd-run --user --scope -q -p MemoryMax="$CAP" -p MemorySwapMax=0 --unit="$unit" -- "$@"
rc=$?
set -e
[ $rc -eq 137 ] && echo "memcap: '$*' exceeded $CAP and was killed" >&2
exit $rc
