#!/bin/bash
# ------------------------------------------------------------------
# Run the monitor + the demo sentinel on this machine (macOS/Linux),
# detached from the terminal, with logs and PID files.
#
#   bash deploy/run_local.sh start     # start both (sentinel honours .env: SENTINEL_DRY_RUN)
#   bash deploy/run_local.sh stop      # stop both
#   bash deploy/run_local.sh status    # are they running? + last log lines
#   bash deploy/run_local.sh restart
#
# Not a substitute for systemd (docs/DEPLOY.md): processes die on reboot.
# ------------------------------------------------------------------
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/.run"; mkdir -p "$RUN"

start_one() {  # name, workdir, command...
    local name="$1" dir="$2"; shift 2
    if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
        echo "  $name: already running (pid $(cat "$RUN/$name.pid"))"; return
    fi
    (cd "$dir" && nohup "$@" >> "$RUN/$name.out" 2>&1 < /dev/null & echo $! > "$RUN/$name.pid")
    sleep 1
    if kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
        echo "  $name: started (pid $(cat "$RUN/$name.pid")) -> $RUN/$name.out"
    else
        echo "  $name: FAILED to start - see $RUN/$name.out"; tail -5 "$RUN/$name.out"
    fi
}

stop_one() {
    local name="$1"
    if [ -f "$RUN/$name.pid" ]; then
        local pid; pid="$(cat "$RUN/$name.pid")"
        if kill -0 "$pid" 2>/dev/null; then kill "$pid" && echo "  $name: stopped (pid $pid)"; else echo "  $name: not running"; fi
        rm -f "$RUN/$name.pid"
    else
        echo "  $name: not running"
    fi
}

status_one() {
    local name="$1"
    if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
        echo "  $name: RUNNING (pid $(cat "$RUN/$name.pid"))"
    else
        echo "  $name: stopped"
    fi
    [ -f "$RUN/$name.out" ] && tail -3 "$RUN/$name.out" | sed 's/^/      /'
}

case "${1:-}" in
    start)
        [ -x "$PY" ] || { echo "venv missing: $PY"; exit 1; }
        start_one monitor  "$ROOT/gold_monitor" "$PY" main.py --monitor
        start_one sentinel "$ROOT"              "$PY" -m sentinel.main --run
        ;;
    stop)    stop_one sentinel; stop_one monitor ;;
    restart) "$0" stop; sleep 1; "$0" start ;;
    status)  status_one monitor; status_one sentinel ;;
    *) echo "usage: bash deploy/run_local.sh {start|stop|restart|status}"; exit 1 ;;
esac
