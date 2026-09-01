# DEPLOY — hosting 24/7 pentru gold_monitor

> **Când merită pornit 24/7**: abia după ce un instrument trece **Poarta 1**
> (criteriile OOS din `RESULTS.md`: ≥30 trades, PF ≥ 1.15, expectanță > 0,
> maxDD < 15%, trade-Sharpe > 0.5) și primește `ENABLED=True`. Până atunci
> monitorul pornește, explică de ce nu are ce monitoriza și iese — un
> serviciu 24/7 ar face restart în buclă degeaba. Acest document pregătește
> totul dinainte; nu deploy-a acum.

## Cerințe (toate scenariile)

- **Hardware**: 1 CPU / 2GB RAM suficient (XGBoost pe ~11k candele 1h;
  reantrenarea la 6h e cel mai greu moment). ~1GB disc.
- **OS**: Linux 64-bit obligatoriu — **xgboost nu are wheel pe 32-bit**
  (relevant la Raspberry Pi: folosește Raspberry Pi OS **64-bit**).
- **Python ≥ 3.10** cu `venv`.
- **Rețea**: aplicația face **doar conexiuni outbound** (TradingView,
  TwelveData, Yahoo Finance, webhook Discord). **Nu** trebuie IP public,
  port forwarding sau firewall inbound — merge și în spatele unui NAT/CGNAT
  (deci și de acasă, și pe un VPS ieftin).
- **Secrete**: `.env` se creează **pe mașină**, nu vine din git și nu se
  copiază în repo.

## Ce fac scripturile existente (`gold_monitor/deploy/`)

- `upload_to_vps.sh <user@ip>` — rsync-uiește `gold_monitor/` pe server în
  `/opt/trading-monitor` (exclude venv, `__pycache__`, logs, output, `.env`).
- `setup_vps.sh` — pe server: `apt` update + instalează Python/git, creează
  `/opt/trading-monitor`, venv + pip install, instalează un serviciu
  **systemd** `trading-monitor` cu `Restart=always` (auto-restart la crash,
  la 30s) care rulează `main.py --monitor`, și un **cron săptămânal**
  (duminică 00:00 UTC) pentru reantrenare `--train`.

⚠️ **Limitare cunoscută a scripturilor**: ele copiază DOAR `gold_monitor/`,
dar din v3 codul importă `common/indicators.py` de la **rădăcina
repo-ului** (`gold_monitor/data/indicators.py:15`) — pe un server pe care ai
copiat doar `gold_monitor/`, monitorul crapă cu `ModuleNotFoundError:
common`. Procedurile de mai jos ocolesc problema clonând **întregul repo**
prin git (oricum mai bun pentru update-uri). Alte observații: serviciul
rulează ca `root` (mai jos folosim un user dedicat) și pip-ul instalează
pachete enumerate manual în loc de `requirements.txt`.

---

## A. VPS Ubuntu 22.04 / 24.04

### 1. Instalare

```bash
# ca user cu sudo
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git

# user dedicat, fără shell de login (serviciul nu rulează ca root)
sudo useradd -r -m -d /opt/cfd -s /usr/sbin/nologin cfd

# clona repo-ului (public, nu cere autentificare)
sudo -u cfd git clone https://github.com/DeiuVRG/CFD_Game.git /opt/cfd/CFD_Game

# venv + dependențe
sudo -u cfd python3 -m venv /opt/cfd/venv
sudo -u cfd /opt/cfd/venv/bin/pip install --upgrade pip
sudo -u cfd /opt/cfd/venv/bin/pip install -r /opt/cfd/CFD_Game/gold_monitor/requirements.txt
```

### 2. Configurare `.env`

```bash
sudo -u cfd cp /opt/cfd/CFD_Game/gold_monitor/.env.example /opt/cfd/CFD_Game/gold_monitor/.env
sudo -u cfd nano /opt/cfd/CFD_Game/gold_monitor/.env
# completează DISCORD_WEBHOOK_URL și, opțional, DISCORD_MENTION / TWELVEDATA_API_KEY
sudo chmod 600 /opt/cfd/CFD_Game/gold_monitor/.env

# test webhook
cd /opt/cfd/CFD_Game/gold_monitor
sudo -u cfd /opt/cfd/venv/bin/python main.py --test-discord
```

### 3. Modele (gitignored — se antrenează pe mașină)

```bash
cd /opt/cfd/CFD_Game/gold_monitor
sudo -u cfd /opt/cfd/venv/bin/python main.py --train
```

### 4. Serviciu systemd

```bash
sudo tee /etc/systemd/system/cfd-monitor.service > /dev/null << 'EOF'
[Unit]
Description=CFD_Game Trading Monitor (semnale, fara executie)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=cfd
WorkingDirectory=/opt/cfd/CFD_Game/gold_monitor
ExecStart=/opt/cfd/venv/bin/python main.py --monitor
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable cfd-monitor    # pornește la boot
sudo systemctl start cfd-monitor
```

Comenzi utile:

```bash
sudo systemctl status cfd-monitor          # stare
sudo journalctl -u cfd-monitor -f          # log live (stdout-ul serviciului)
sudo systemctl restart cfd-monitor         # restart
sudo systemctl stop cfd-monitor            # oprire
```

⚠️ **Bug cunoscut, de reparat ÎNAINTE de primul deploy sub systemd**:
listener-ul de tastatură (`engine/monitor_engine.py:514`, `_keyboard_listener`)
face `select()` pe stdin; sub systemd stdin e `/dev/null`, care e mereu
„readable”, iar `read(1)` întoarce `''` instant → **buclă infinită fără
sleep, un core la 100%** (verificat: ~1.7M iterații/2s). Fix minim (cod):
porni thread-ul doar dacă `sys.stdin.isatty()`, sau ieși din buclă la
`read` gol (EOF). Semnalele și Discord-ul merg oricum, dar CPU-ul arde
degeaba — pe un Pi e inacceptabil. Dashboard-ul de terminal fără TTY e
inofensiv (ignoră afișajul din journal).

### 4b. Santinela (demo) ca al doilea serviciu

Rulează în paralel cu monitorul (citește doar `signals.db`). Cere în `.env`
cheia Anthropic și credențialele Capital.com **demo** (vezi
`gold_monitor/.env.example`, secțiunea Sentinel) și
`pip install -r sentinel/requirements.txt` în același venv.

```bash
sudo tee /etc/systemd/system/cfd-sentinel.service > /dev/null << 'UNIT'
[Unit]
Description=CFD_Game Sentinel (DEMO account supervisor)
After=network-online.target cfd-monitor.service
Wants=network-online.target

[Service]
Type=simple
User=cfd
WorkingDirectory=/opt/cfd/CFD_Game
ExecStart=/opt/cfd/venv/bin/python -m sentinel.main --run
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload && sudo systemctl enable --now cfd-sentinel
sudo journalctl -u cfd-sentinel -f
```

Verifică întâi epic-urile pe demo: `python -m sentinel.main --markets gold`
și `--markets bitcoin`; pune valorile în `SENTINEL_EPIC_GOLD` /
`SENTINEL_EPIC_BTC`. Prima săptămână rulează cu `--dry-run` (sau
`SENTINEL_DRY_RUN=true`) ca să vezi deciziile fără ordine. Backup:
`sentinel/data/decisions.db` merge în același cron ca `signals.db`.

### 5. Reantrenare săptămânală (cron)

```bash
sudo -u cfd crontab -e
# duminică 00:00 UTC
0 0 * * 0 cd /opt/cfd/CFD_Game/gold_monitor && /opt/cfd/venv/bin/python main.py --train >> logs/train_cron.log 2>&1
```

(Monitorul se reantrenează oricum singur la 6h în fundal —
`engine/monitor_engine.py`, `RETRAIN_INTERVAL_SEC`; cron-ul e plasă de
siguranță pentru perioadele în care serviciul e oprit.)

### 6. Unde sunt datele

| Ce | Unde |
|---|---|
| Log aplicație | `/opt/cfd/CFD_Game/gold_monitor/logs/monitor.log` |
| Log serviciu | `journalctl -u cfd-monitor` |
| Baza de semnale | `/opt/cfd/CFD_Game/gold_monitor/data/signals.db` |
| CSV semnale (legacy) | `/opt/cfd/CFD_Game/gold_monitor/output/signals.csv` |
| Modele | `/opt/cfd/CFD_Game/gold_monitor/models/*.pkl` |
| Deciziile santinelei | `/opt/cfd/CFD_Game/sentinel/data/decisions.db` + `sentinel/logs/sentinel.log` |

### 7. Update

```bash
sudo systemctl stop cfd-monitor
cd /opt/cfd/CFD_Game && sudo -u cfd git pull origin main
sudo -u cfd /opt/cfd/venv/bin/pip install -r gold_monitor/requirements.txt   # dacă s-au schimbat dependențele
cd gold_monitor && sudo -u cfd /opt/cfd/venv/bin/python -m pytest ../tests/ -q   # opțional, dar recomandat
sudo systemctl start cfd-monitor
```

`signals.db`, modelele și `.env` sunt gitignored, deci `git pull` nu le
atinge.

### 8. Backup `signals.db`

`signals.db` e **dovada** proiectului — singurul fișier cu adevărat de
neînlocuit. Backup zilnic cu snapshot SQLite consistent (sigur în timp ce
monitorul scrie):

```bash
sudo -u cfd crontab -e
# zilnic la 23:50 UTC, păstrează 30 de zile
50 23 * * * sqlite3 /opt/cfd/CFD_Game/gold_monitor/data/signals.db ".backup /opt/cfd/backups/signals-$(date +\%F).db" && find /opt/cfd/backups -name 'signals-*.db' -mtime +30 -delete
```

(`sudo apt install sqlite3`; creează întâi `/opt/cfd/backups`.) Ideal,
copiază periodic și off-site: `scp cfd@vps:/opt/cfd/backups/... .`

---

## B. Raspberry Pi (Raspberry Pi OS 64-bit)

**Obligatoriu OS 64-bit** (Bookworm arm64): `uname -m` trebuie să arate
`aarch64`. Pe 32-bit (`armv7l`) `pip install xgboost` eșuează — nu există
wheel.

Procedura e identică cu VPS-ul (Raspberry Pi OS e Debian, aceleași comenzi
`apt`/`systemd`/`cron`), cu diferențele:

```bash
sudo apt install -y libgomp1   # runtime OpenMP cerut de xgboost (de regulă lipsește pe Lite)
```

- Un Pi 4/5 cu 2GB+ RAM e suficient; pe 2GB adaugă swap dacă antrenarea dă
  OOM: `sudo dphys-swapfile swapoff && sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile && sudo dphys-swapfile setup && sudo dphys-swapfile swapon`.
- Antrenarea durează mai mult (minute, nu zeci de secunde) — normal.
- Cardurile SD mor de la scris continuu: pune log-urile și backup-urile pe
  un SSD USB dacă rulezi luni de zile, sau măcar fă backup-ul `signals.db`
  către altă mașină.
- Pi-ul stă în spatele routerului tău — nicio configurare de porturi
  (doar outbound).

Restul: urmează pașii A.1 → A.8 întocmai.

---

## C. Un PC Linux oarecare (24/7 acasă)

Orice distribuție cu systemd merge. Diferă doar managerul de pachete:

- **Debian/Ubuntu/Mint**: `sudo apt install python3 python3-venv git`
- **Fedora**: `sudo dnf install python3 git`
- **Arch**: `sudo pacman -S python git`

Apoi pașii A.1 → A.8 identic (venv-ul izolează totul de Python-ul
sistemului). Dacă distribuția nu are systemd, alternativa minimă e un
`@reboot` în cron + un loop de restart, dar recomandarea rămâne systemd.

De verificat pe un PC de casă:
- dezactivează suspend/hibernate (`sudo systemctl mask sleep.target
  suspend.target hibernate.target hybrid-sleep.target`);
- în BIOS: „Restore on AC Power Loss" → On, ca să revină după o pană de
  curent (systemd `enable` face restul).

---

## Corecții propuse pentru scripturile existente (minime)

Dacă vrem ca `deploy/setup_vps.sh` să rămână calea oficială, ar avea nevoie
de trei schimbări mici (nu le-am aplicat — cer acordul întâi):

1. **Deploy din repo întreg, nu doar `gold_monitor/`** — altfel lipsește
   `common/` și monitorul crapă la import. Cel mai simplu: `git clone` în
   loc de rsync, cu `WorkingDirectory` setat la `.../CFD_Game/gold_monitor`.
2. **`pip install -r requirements.txt`** în loc de lista hardcodată de
   pachete (care poate diverge de `requirements.txt`).
3. **Serviciul să nu ruleze ca `root`** (`User=cfd` ca mai sus) și
   comentariul „daily training" corectat în „weekly" (cron-ul e deja
   săptămânal).
4. **(cod, nu script) listener-ul de tastatură să pornească doar cu TTY** —
   vezi avertismentul de la pasul A.4; fără acest fix serviciul consumă un
   core întreg permanent.
