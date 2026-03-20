#!/bin/bash
set -euo pipefail
export SEHBMASTER_API="http://localhost:8000/api"
export SEHBMASTER_API_KEY="dev-secret"
export SCRAPER_ID="server-lobbywatch"

LOG_DIR="/root/sehbmaster/scraper/logs"
mkdir -p ""
LOG_FILE="/lobbywatch-2026-03-20.log"
find "" -name 'lobbywatch-*.log' -mtime +30 -delete 2>/dev/null || true

cd /root/sehbmaster/scraper
source .venv/bin/activate
python lobbywatch.py >> "" 2>&1
