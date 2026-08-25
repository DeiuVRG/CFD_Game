# Gold Monitor - Documentatie Completa

## Ce este?
Aplicatie de monitorizare a pretului aurului (XAU/USD) care trimite semnale de tranzactionare pe Discord.
**NU tranzactioneaza automat** - doar trimite notificari, tu executi manual pe XTB.

---

## Structura Proiect

```
gold_monitor/
├── main.py                  # Entry point (--train / --monitor / --test-discord)
├── .env                     # Discord webhook URL
├── models/
│   └── gold_xgb.pkl         # Modelul AI antrenat (XGBoost)
├── ai/
│   ├── feature_engineer.py  # Calculeaza 20 features din candle-uri
│   ├── model.py             # GoldPredictor - incarca/ruleaza modelul
│   └── trainer.py           # Antrenare pe date istorice Yahoo Finance
├── strategies/
│   ├── ai_strategy.py       # Strategie AI (50% din vot)
│   ├── scalping_strategy.py # RSI + Bollinger Bands (25% din vot)
│   └── momentum_strategy.py # MACD + EMA crossover (25% din vot)
├── engine/
│   ├── monitor_engine.py    # Loop-ul principal de monitorizare
│   └── signal.py            # Clasa Signal (BUY/SELL + SL/TP)
├── data/
│   ├── gold_fetcher.py      # Descarca preturi live de pe Yahoo Finance (GC=F)
│   └── indicators.py        # RSI, MACD, Bollinger Bands, ATR, EMA
├── notifications/
│   └── discord_notify.py    # Trimite semnale pe Discord via webhook
├── ui/
│   └── terminal_ui.py       # Dashboard live in terminal
├── config/
│   └── settings.py          # Configurari (intervale, thresholds, ponderi)
├── output/
│   └── signals.csv          # Log cu toate semnalele generate
└── logs/
    └── gold_monitor.log     # Log-uri aplicatie
```

---

## Comenzi

| Comanda | Ce face |
|---|---|
| `cd gold_monitor && python main.py --train` | Antreneaza/re-antreneaza modelul AI pe 2 ani de date istorice |
| `cd gold_monitor && python main.py --monitor` | Porneste monitorizarea LIVE (loop infinit) |
| `cd gold_monitor && python main.py --test-discord` | Trimite mesaj test pe Discord |
| `cd gold_monitor && python main.py --backtest gold` | Backtest v3 (fereastra optimizare + OOS separat) |
| `cd gold_monitor && python main.py --optimize gold` | Grid search parametri + prag etichetare |
| `cd gold_monitor && python main.py --report` | Raport colectare semnale + export CSV |
| `cd gold_monitor && python tools/run_evaluation.py all` | Protocolul complet de evaluare (Faza 4) |

---

## Cum functioneaza monitorizarea (--monitor)

### La fiecare 5 secunde:
- Ia pretul LIVE al aurului de pe Yahoo Finance (GC=F)
- Afiseaza dashboard-ul in terminal

### La fiecare 60 secunde (analiza):
1. Descarca ultimele candle-uri de 5 minute (display + strategii clasice)
2. Separat, pastreaza un cache de candele de 1h (30 zile) pentru calea AI
3. Calculeaza indicatorii tehnici (RSI, MACD, BB, ATR, EMA, ADX)
4. Ruleaza **3 strategii in paralel**:

#### a) AI (XGBoost) - 50% din votul final [v3: pe candele de 1h]
- Ruleaza pe TRAIN_INTERVAL (1h) - acelasi timeframe pe care modelul a fost
  antrenat si backtestat; gate-ul de regim ADX se calculeaza tot pe 1h
- Calculeaza features din candle-urile de 1h
- Modelul prezice BUY / SELL / HOLD
- Doar daca confidenta trece pragul instrumentului

#### b) Scalping (RSI + Bollinger Bands) - 25% din vot
- RSI < 30 + pret sub BB lower → BUY
- RSI > 70 + pret peste BB upper → SELL

#### c) Momentum (MACD + EMA crossover) - 25% din vot
- EMA(9) trece peste EMA(21) + MACD pozitiv → BUY
- EMA(9) trece sub EMA(21) + MACD negativ → SELL

4. **VOTARE**: combina cele 3 strategii ponderat
   - Daca scorul combinat > 0.50 → SEMNAL VALID
5. Daca e semnal NOU (diferit de ultimul):
   - Calculeaza Stop Loss + Take Profit (bazat pe ATR)
   - Trimite pe Discord (cu ping daca DISCORD_MENTION e setat)
   - Persista in `output/signals.csv` SI in `data/signals.db` (SQLite,
     append-only, cu versiunea modelului); outcome-ul ipotetic (TP/SL/EOD,
     P&L brut/net) se completeaza ulterior de position tracker
   - Raport agregat oricand cu `python main.py --report`

---

## Modelul AI (XGBoost)

- **Date antrenare**: ~11.500 candle-uri de 1h, 2 ani de Gold Futures (Yahoo Finance)
- **Features (20)**: RSI, MACD, Bollinger Bands, ATR, EMA, shadow-uri, pattern-uri temporale
- **Labels**: daca pretul creste >0.5% in urmatoarele 6h → BUY, scade >0.5% → SELL, altfel → HOLD
- **Split**: 80% train / 20% test
- **Accuracy**: 53.4%
- **Fisier model**: `models/gold_xgb.pkl`
- **Re-antrenare**: optional, o data pe saptamana/luna cu `--train`

---

## Discord

- Webhook URL configurat in `.env`
- Tag user: configurat prin `DISCORD_MENTION` in `.env` (optional)
- Notificare exemplu:
  ```
  🟢 CUMPARA AUR (XAU/USD)
  💰 Pret: $2,045.30
  🎯 Target: $2,052.00
  🛑 Stop Loss: $2,040.00
  📊 Strategie: AI + Momentum
  💪 Confidenta: 73%
  ⚖️ R:R 1:1.7
  ```

---

## Cerinte tehnice

- **Python** cu bibliotecile din `requirements.txt` (xgboost, pandas, yfinance, etc.)
- **Internet** activ (pentru Yahoo Finance)
- **Laptopul deschis** cu terminalul ruland `--monitor`
- **NU** trebuie XTB/chart/browser deschis - doar terminalul

---

## Workflow utilizator

1. Pornesti: `cd gold_monitor && python main.py --monitor`
2. Astepti notificare pe Discord (pe telefon)
3. Cand primesti semnal: deschizi XTB, executi tranzactia manual
4. Pui SL si TP conform notificarii
5. Ctrl+C in terminal cand vrei sa opresti

---

## Note importante

- Semnalele vin DOAR cand strategiile sunt de acord - poate dura ore
- Piata aurului: luni-vineri, ~non-stop (weekend inchis)
- 53.4% accuracy = instrument ajutator, NU garantie de profit
- Semnalele se logheaza in `output/signals.csv` pentru analiza ulterioara
