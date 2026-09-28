#!/bin/bash

# --- 0. Check for Arguments ---
if [ -z "$1" ]; then
    echo "Error: No Crash ID provided."
    echo "Usage: ./launch_custom.sh <Crash ID>"
    exit 1
fi

CRASH_ID=$1

CURRENT_DIR=$(pwd)

CARLA_ROOT=$(realpath "$CURRENT_DIR/../../")

SERVER_PATH="$CARLA_ROOT/CarlaUE4.sh"

MAP_DIR="Benchmark/maps"

MAP_PATH="${MAP_DIR}/map_${CRASH_ID}.xodr"

if [ ! -f "$MAP_PATH" ]; then
    echo "Error: Map file not found at: $MAP_PATH"
    exit 1
fi

pkill -f CarlaUE4

echo ">> Launching CARLA Server from: $CARLA_ROOT"

nohup "$SERVER_PATH" > /dev/null 2>&1 &
SERVER_PID=$!

echo ">> Server started (PID: $SERVER_PID). Waiting for initialization..."

sleep 10 

echo ">> Loading OpenDRIVE map..."

python3.8 config.py -x "$MAP_PATH"

echo ">> Moving Spectator..."
python3.8 ns_change_spectator_bm.py "$CRASH_ID"

sleep 5

python3.8 ns_launch_scene_bm.py "$CRASH_ID"

wait $SERVER_PID