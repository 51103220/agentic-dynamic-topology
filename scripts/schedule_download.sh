#!/bin/bash
HOST="1.34.56.15"
PORT="63833"
KEY="$HOME/.ssh/id_ed25519"
REMOTE="root@${HOST}:/workspace/results_v2/"
LOCAL="$HOME/Downloads/results_v2/"

download() {
    local label="$1"
    echo "[$(date '+%H:%M:%S')] $label — starting download..."
    mkdir -p "$LOCAL"
    scp -P "$PORT" -i "$KEY" \
        -o StrictHostKeyChecking=no \
        -o BatchMode=yes \
        -r "$REMOTE" "$LOCAL"
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] $label — done. Files:"
        ls -lh "$LOCAL"
    else
        echo "[$(date '+%H:%M:%S')] $label — FAILED"
    fi
}

echo "Scheduled downloads at +2h, +3h, +4h from now ($(date '+%H:%M:%S'))"
echo "Saving to: $LOCAL"
echo "Press Ctrl+C to cancel."
echo ""

sleep 7200 && download "2h"
sleep 3600 && download "3h"
sleep 3600 && download "4h"

echo "All done."
