#!/bin/bash
# Headless batch driver: replays every TRACE Benchmark scenario in real
# CARLA physics (ns_launch_scene_bm_logging.py), one CARLA server boot per
# scenario (matches run_benchmark.sh's own per-scenario restart pattern,
# since each crash uses a different OpenDRIVE map). No X server available
# on this host, so the server is launched with -RenderOffScreen.
set -u
CURRENT_DIR=$(pwd)
CARLA_ROOT=$(realpath "$CURRENT_DIR/../../")
SERVER_PATH="$CARLA_ROOT/CarlaUE4.sh"
MAP_DIR="Benchmark/maps"
SIM_DIR="Benchmark/simulations"
OUT_DIR="Benchmark/replay"
mkdir -p "$OUT_DIR"

crash_ids=$(ls "$SIM_DIR" | sed -n 's/simulation_\(.*\)\.json/\1/p' | sort)
total=$(echo "$crash_ids" | wc -l)
i=0

for CRASH_ID in $crash_ids; do
    i=$((i+1))
    OUT_FILE="$OUT_DIR/replay_${CRASH_ID}.json"
    if [ -f "$OUT_FILE" ]; then
        echo ">> [$i/$total] $CRASH_ID already replayed, skipping"
        continue
    fi

    MAP_PATH="${MAP_DIR}/map_${CRASH_ID}.xodr"
    if [ ! -f "$MAP_PATH" ]; then
        echo ">> [$i/$total] $CRASH_ID: map not found at $MAP_PATH, skipping"
        continue
    fi

    echo ">> [$i/$total] Replaying crash $CRASH_ID ..."
    pkill -9 -f CarlaUE4 >/dev/null 2>&1
    sleep 2

    nohup "$SERVER_PATH" -nullrhi > "/tmp/carla_server_${CRASH_ID}.log" 2>&1 &
    SERVER_PID=$!
    sleep 12

    python3.8 config.py -x "$MAP_PATH" > "/tmp/config_${CRASH_ID}.log" 2>&1
    sleep 2

    timeout 200 python3.8 ns_launch_scene_bm_logging.py "$CRASH_ID"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo ">> [$i/$total] $CRASH_ID: ns_launch_scene_bm_logging.py exited with code $RC"
    fi

    kill "$SERVER_PID" >/dev/null 2>&1
    pkill -9 -f CarlaUE4 >/dev/null 2>&1
    sleep 2
done

echo "Done. Replayed $(ls "$OUT_DIR" | wc -l) / $total scenarios."
