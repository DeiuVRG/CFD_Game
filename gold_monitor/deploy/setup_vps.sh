#!/bin/bash
# ============================================================
# Trading Monitor - VPS Setup Script
# Run this on a fresh Ubuntu VPS (22.04 or 24.04)
# Usage: bash setup_vps.sh
# ============================================================

set -e

echo "============================================"
echo "  Trading Monitor - VPS Setup"
echo "============================================"

# Update system
echo "[1/6] Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "[2/6] Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git

# Create app directory
echo "[3/6] Setting up application..."
APP_DIR="/opt/trading-monitor"
sudo mkdir -p "$APP_DIR"
sudo chown "$USER:$USER" "$APP_DIR"

# Copy application files (run this after scp-ing the files)
if [ -d "./gold_monitor" ]; then
    cp -r ./gold_monitor/* "$APP_DIR/"
    echo "  Files copied from local directory"
else
    echo "  NOTE: Copy your gold_monitor files to $APP_DIR manually"
fi

# Create virtual environment
echo "[4/6] Creating Python virtual environment..."
cd "$APP_DIR"
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install yfinance xgboost scikit-learn joblib python-dotenv requests pandas numpy

# Create necessary directories
mkdir -p models output logs

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "  IMPORTANT: Create .env file with your Discord webhook:"
    echo "  echo 'DISCORD_WEBHOOK_URL=your_webhook_url_here' > $APP_DIR/.env"
    echo ""
fi

# Install systemd service
echo "[5/6] Installing systemd service..."
sudo tee /etc/systemd/system/trading-monitor.service > /dev/null << 'EOF'
[Unit]
Description=Trading Monitor - AI Trading Signals
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/trading-monitor
ExecStart=/opt/trading-monitor/venv/bin/python3 main.py --monitor
Restart=always
RestartSec=30
StandardOutput=append:/opt/trading-monitor/logs/service.log
StandardError=append:/opt/trading-monitor/logs/service_error.log

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Install cron job for daily training (at 00:00 UTC, Sunday)
echo "[6/6] Setting up automatic weekly training..."
CRON_CMD="0 0 * * 0 cd /opt/trading-monitor && /opt/trading-monitor/venv/bin/python3 main.py --train >> /opt/trading-monitor/logs/train_cron.log 2>&1"

# Add cron job (avoid duplicates)
(crontab -l 2>/dev/null | grep -v "trading-monitor.*--train"; echo "$CRON_CMD") | crontab -

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Copy your .env file:"
echo "     echo 'DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...' > $APP_DIR/.env"
echo ""
echo "  2. Train AI models (first time):"
echo "     cd $APP_DIR && source venv/bin/activate && python3 main.py --train"
echo ""
echo "  3. Start the service:"
echo "     sudo systemctl daemon-reload"
echo "     sudo systemctl enable trading-monitor"
echo "     sudo systemctl start trading-monitor"
echo ""
echo "  4. Check status:"
echo "     sudo systemctl status trading-monitor"
echo "     tail -f $APP_DIR/logs/service.log"
echo ""
echo "  Auto-training: Every Sunday at 00:00 UTC"
echo "  The service auto-restarts if it crashes."
echo ""
