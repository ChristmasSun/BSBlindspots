#!/usr/bin/env bash
# Daily options data collector for BS Blindspots.
#
# Add to crontab: 0 17 * * 1-5 /Users/vedan/Documents/programming/BSBlindspots/scripts/collect_daily.sh
# (Runs at 5pm ET on weekdays, after market close)
#
# To edit crontab: crontab -e
# To verify: crontab -l

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/raw/options/daily"
LOG_FILE="${LOG_DIR}/collection.log"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---" >> "${LOG_FILE}"
uv run python scripts/collect_daily.py 2>&1 | tee -a "${LOG_FILE}"
echo "" >> "${LOG_FILE}"
