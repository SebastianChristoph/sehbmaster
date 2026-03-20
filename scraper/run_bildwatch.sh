#!/bin/bash
set -euo pipefail

export SEHBMASTER_API="http://localhost:8000/api"
export SEHBMASTER_API_KEY="dev-secret"
export SCRAPER_ID="server-bildwatch"

LOG_DIR="/root/sehbmaster/scraper/logs"
LOG_FILE="$LOG_DIR/bildwatch-$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

echo "--- $(date -Iseconds) START ---" >> "$LOG_FILE"
/root/sehbmaster/scraper/.venv/bin/python /root/sehbmaster/scraper/bildwatch.py >> "$LOG_FILE" 2>&1
EXIT=$?
echo "--- $(date -Iseconds) END (exit=$EXIT) ---" >> "$LOG_FILE"

# Logs älter als 30 Tage löschen
find "$LOG_DIR" -name "bildwatch-*.log" -mtime +30 -delete

exit $EXIT
