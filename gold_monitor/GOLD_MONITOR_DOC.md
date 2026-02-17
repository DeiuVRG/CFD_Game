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

---

## Cum functioneaza monitorizarea (--monitor)

### La fiecare 5 secunde:
- Ia pretul LIVE al aurului de pe Yahoo Finance (GC=F)
- Afiseaza dashboard-ul in terminal

### La fiecare 60 secunde (candle noua):
1. Descarca ultimele 100 candle-uri de 5 minute
2. Calculeaza indicatorii tehnici (RSI, MACD, BB, ATR, EMA)
3. Ruleaza **3 strategii in paralel**:

#### a) AI (XGBoost) - 50% din votul final
- Calculeaza 20 features din candle-uri
- Modelul prezice BUY / SELL / HOLD
- Doar daca confidenta > 60%

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
   - Trimite pe Discord cu ping
   - Salveaza in `output/signals.csv`

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
- Tag user: `<@266234847532548096>`
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
