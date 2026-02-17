#!/bin/bash
# ============================================================
# Upload Trading Monitor files to VPS
# Usage: bash deploy/upload_to_vps.sh user@ip_address
# Example: bash deploy/upload_to_vps.sh root@143.47.100.50
# ============================================================

if [ -z "$1" ]; then
    echo "Usage: bash deploy/upload_to_vps.sh user@ip_address"
    echo "Example: bash deploy/upload_to_vps.sh root@143.47.100.50"
    exit 1
fi

VPS="$1"
REMOTE_DIR="/opt/trading-monitor"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uploading Trading Monitor to $VPS..."
echo "Local: $LOCAL_DIR"
echo "Remote: $REMOTE_DIR"
echo ""

# Create remote directory
ssh "$VPS" "mkdir -p $REMOTE_DIR"

# Upload all Python files and configs (exclude venv, __pycache__, logs)
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='venv' \
    --exclude='*.pyc' \
    --exclude='logs/' \
    --exclude='output/' \
    --exclude='.env' \
    "$LOCAL_DIR/" "$VPS:$REMOTE_DIR/"

echo ""
echo "Files uploaded! Now SSH into your VPS and run:"
echo "  ssh $VPS"
echo "  cd $REMOTE_DIR"
echo "  bash deploy/setup_vps.sh"
