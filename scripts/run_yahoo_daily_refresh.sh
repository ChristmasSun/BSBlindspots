#!/bin/zsh

set -euo pipefail

REPO_DIR="/Users/andyque/BSBlindspots"
LOG_DIR="/Users/andyque/Library/Logs/BSBlindspots"
LOCK_DIR="/tmp/bsblindspots-yahoo-refresh.lock"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

export PATH="/Users/andyque/Library/Python/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Refresh already running, exiting."
    exit 0
fi

trap 'rmdir "$LOCK_DIR"' EXIT

uv run python -m src.data.refresh_all --force --large-universe --top-stock-count 200 --top-etf-count 20 --sleep-seconds 0.25
uv run python -m src.data.export_excel
