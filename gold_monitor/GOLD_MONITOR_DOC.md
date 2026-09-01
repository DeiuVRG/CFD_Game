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

Din v3.1 monitorul are doua moduri (`MonitorConfig.SIGNAL_MODE` in
`config/settings.py`). **Implicit este `ai_only`**: calea live reproduce
exact modelul validat de backtester, ca dovezile din `signals.db` sa fie
despre strategia care a fost efectiv testata.

### La fiecare 10 secunde:
- Ia pretul LIVE (TradingView → TwelveData → Yahoo)
- Afiseaza dashboard-ul in terminal

### La fiecare 60 secunde (`ai_only`):
1. Descarca candele de 5 minute doar pentru indicatorii din dashboard
2. Ia candelele de 1h (TRAIN_INTERVAL) si **arunca candela in formare** -
   modelul a fost antrenat si validat pe candele complete
3. Daca a aparut o candela 1h **noua**:
   - **Outcome**: pozitia ipotetica deschisa e verificata pe candela noua
     cu regulile v3 (`engine/execution_rules.py`): fitilurile conteaza,
     SL are prioritate daca SL+TP sunt atinse in aceeasi candela, gap prin
     SL = iesire la open. La inchidere: Discord + outcome in `signals.db`
   - Daca nu e pozitie deschisa: gate ADX(1h) ≥ `adx_min`, predictie
     XGBoost pe candelele complete, filtru cost + R:R minim identic cu
     `Backtester._simulate_trades`
   - Semnal: intrare = pretul live din momentul semnalului (≈ open-ul
     candelei urmatoare, ca in backtest), SL/TP la distantele ATR din
     candela de semnal → Discord + `output/signals.csv` + `data/signals.db`
     (cu versiunea modelului)
4. **Nu exista** in acest mod: filtru de sesiune, inchidere EOD, trailing
   SL, vot intre strategii - niciuna nu face parte din modelul validat
5. La pornire, semnalele ramase fara outcome (dupa un crash/restart) sunt
   **rejucate** din candele (`PositionTracker.restore_from_store`)

### Modul `vote` (legacy, nevalidat de backtester)
AI (50%) + Scalping RSI+BB (25%) + Momentum MACD+EMA (25%) pe candele 5m,
ponderi ajustate pe regim, prag de vot 0.35, SL/TP verificate pe tick cu
trailing, filtru de sesiune London+NY si inchidere fortata la 21:00 UTC.
Pastrat pentru experimente; nu-l folosi pentru colectarea de dovezi.

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
