# CFD_Game — semnale de trading cu porți de dovezi

> **Filosofia proiectului: colectăm dovezi, nu bani.** Un instrument devine
> tranzacționabil doar după ce își demonstrează edge-ul pe date
> out-of-sample, după costuri, cu execuție pesimistă. Până atunci: semnale
> dezactivate, execuție în carantină, date colectate.

## Starea curentă (v3, 2026-08-25)

| Componentă | Stare |
|---|---|
| `gold_monitor/` | **Inima proiectului.** Monitor multi-instrument (XGBoost + 2 strategii clasice → semnale Discord). Zero execuție. |
| Instrumente | **Toate `ENABLED=False`.** Gold și BTC au picat poarta OOS v3 (cifrele exacte: [RESULTS.md](RESULTS.md)). EUR/USD și GBP/USD erau deja dezactivate. |
| `execution_capital/` | Bot Capital.com **în carantină** — inactiv până la Poarta 2, cu bug-urile de risc reparate dormant. **Nu rula pe live.** |
| `trading_game/` | Joc competitiv de descoperire a ponderilor optime de indicatori (spec: `docs/trading_game_prompt.md`), cu validare statistică pe test set. |
| `common/` | Modulul unic de indicatori tehnici, folosit de ambele aplicații. |

## Porțile (gates)

1. **Poarta 1 — semnale**: instrumentul trece criteriile OOS din Faza 4
   (≥30 trades, PF ≥ 1.15, expectanță > 0, maxDD < 15%, trade-Sharpe > 0.5,
   toate după costuri, pe fereastra OOS rulată **o singură dată** per
   configurație) → `ENABLED=True`, semnalele curg pe Discord și în
   `signals.db`.
2. **Poarta 2 — execuție demo**: după o perioadă de colectare live cu
   statistici pozitive (`python main.py --report`), botul din
   `execution_capital/` poate porni pe **demo**.
3. **Poarta 3 — live**: doar după dovezi pe demo.

Regulă permanentă: **nu se optimizează pe fereastra OOS**. Dacă schimbi
strategia după ce ai văzut OOS-ul, următorul test cere date noi.

## Modelul de execuție v3 (backtest cinstit)

Implementat o singură dată în `gold_monitor/ai/backtester.py` și folosit de
backtester, walk-forward și optimizer deopotrivă:

- semnal pe **close-ul** candelei → intrare pe **open-ul candelei următoare**;
- SL/TP verificate pe **HIGH/LOW** (fitilurile contează), nu pe close;
- SL + TP în aceeași candelă → **SL primul** (conservator);
- gap prin SL → execuție la **open** (preț mai prost); TP exact la nivelul
  TP (niciodată mai bine);
- închiderea forțată de final de perioadă **actualizează equity-ul**;
- walk-forward folosește **același motor**, cu poziția, semnalul în
  așteptare și equity-ul transportate între ferestre.

Cât umflau vechile bug-uri: pe aur, execuția legacy arăta **+2.3% OOS** acolo
unde execuția cinstită arată **−16.2%** ([RESULTS.md](RESULTS.md)).

## Setup

```bash
pip install -r gold_monitor/requirements.txt
cd gold_monitor
cp .env.example .env    # DISCORD_WEBHOOK_URL, DISCORD_MENTION (opțional)
```

## Comenzi — gold_monitor

```bash
cd gold_monitor
python main.py --train [gold|btc|...]   # antrenare model (filtrul explicit atinge și instrumente dezactivate)
python main.py --optimize [instrument]  # grid search parametri + prag etichetare (fereastra de optimizare)
python main.py --backtest [instrument]  # backtest v3: fereastra de optimizare + OOS, raportate separat
python main.py --backtest-wf [...]      # walk-forward cu reantrenare periodică (același motor v3)
python main.py --monitor                # monitorizare live (doar instrumente ENABLED)
python main.py --report                 # raport colectare semnale (win rate, expectanță, export CSV)
python main.py --test-discord           # test webhook
python tools/run_evaluation.py all      # protocolul complet Faza 4 (optimizer → OOS one-shot → verdict)
```

## Colectarea de date (signals.db)

Fiecare semnal emis de monitor se persistă append-only în
`gold_monitor/data/signals.db`: timestamp UTC, instrument, direcție,
confidence, probabilități, ADX/regim, entry/SL/TP, **versiunea modelului**
(hash .pkl + data antrenării). `position_tracker` completează ulterior
outcome-ul ipotetic (TP/SL/EOD, preț de ieșire, P&L brut și net de costuri),
o singură dată per semnal. `--report` agregă totul per instrument și exportă
CSV pentru audit extern.

## Instrumente & costuri

- **Gold (GC=F / XAU-USD)** — model pips: spread 3 pips × $0.10.
- **BTC (BTC-USD)** — model procentual (`SPREAD_PCT`): 0.30% round-trip
  (spread tipic CFD retail 0.20–0.35%), prioritar față de pips. Sesiune
  24/7, exceptat de la închiderea EOD; pragul de etichetare pornește la 0.01
  cu grid 0.008–0.02 în optimizer.
- Sesiunile sunt per instrument: aurul păstrează fereastra London+NY,
  weekendul fără candele nu crapă nimic (retry cu backoff + degradare la
  frame gol).

## Trading game (`trading_game/`)

Simulare competitivă: mai mulți jucători autonomi (RF, XGBoost, algoritm
genetic, corelație IC, euristici momentum/mean-reversion, baseline) își
calculează propriile ponderi de indicatori pe 2015-2019, concurează 24 de
luni pe 2020-2021 (eliminări lunare, recalibrări trimestriale ≤30% shift,
costuri realiste prin Crupier) și sunt judecați pe 2022-2023 cu ponderi
înghețate: bootstrap CI pe Sharpe, permutation test vs S&P 500, stress pe
ferestre de criză. Detalii și rezultate: `trading_game/README.md` +
`trading_game/results/REPORT.md`.

```bash
python -m trading_game.main              # date reale (yfinance, cache)
python -m trading_game.main --synthetic  # complet offline
```

## Teste

```bash
python -m pytest tests/          # tot: execuție v3, instrumente, signal store,
                                 # trading game, suita execution_capital (subproces)
```

## Structura repo-ului

```
CFD_Game/
├── README.md / RESULTS.md       # acest fișier + dovezile Faza 4
├── common/indicators.py         # UNICA implementare de indicatori
├── gold_monitor/                # aplicația activă (semnale, fără execuție)
│   ├── ai/                      # feature engineering, XGBoost, backtester v3, optimizer
│   ├── engine/                  # monitor loop, position tracker (outcome-uri)
│   ├── data/                    # fetch (retry/backoff), signal_store (SQLite)
│   ├── strategies/              # AI (1h) + scalping/momentum (5m)
│   └── tools/run_evaluation.py  # protocolul Faza 4
├── execution_capital/           # bot Capital.com ÎN CARANTINĂ (Poarta 2)
├── trading_game/                # jocul competitiv de ponderi
├── docs/                        # spec joc + JSON-urile evaluărilor
└── tests/                       # pytest
```

## Reguli absolute

- **Zero execuție live.** Modul live al botului rămâne blocat de proces.
- **Zero secrete în repo.** `.env` e gitignored; nicio valoare personală
  hardcodată (mention-ul Discord vine din `DISCORD_MENTION`).
- **Zero LLM în bucla de decizie.** Semnalele = XGBoost + reguli
  deterministe.
- **ENABLED doar prin poarta OOS.** Restul rămâne dezactivat, cu cifrele în
  comentariu.
