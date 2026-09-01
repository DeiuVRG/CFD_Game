# RESULTS — Evaluarea v3 (Gold + Bitcoin), 2026-08-25

> Filosofia proiectului: **colectăm dovezi, nu bani.** Un instrument devine
> `ENABLED=True` doar dacă își demonstrează edge-ul **out-of-sample, după
> costuri, cu execuție pesimistă**. Acest document este dovada — inclusiv
> dovada că vechea execuție (pre-v3) umfla artificial rezultatele.

## Verdict

| Instrument | Fereastra de optimizare | Fereastra OOS (decizie) | Verdict |
|---|---|---|---|
| **XAU/USD (Gold)** | **+30.51%** (PF 1.61) | **−16.17%** (PF 0.82) | **ENABLED=False** — overfit, zero edge OOS |
| **BTC/USD (Bitcoin)** | **−16.40%** (PF 0.91) | **−24.32%** (PF 0.82) | **ENABLED=False** — negativ chiar și in-sample |

Niciun instrument nu trece poarta de activare. Monitorul rămâne fără
instrumente active pe semnale până când o configurație viitoare trece
criteriile pe date noi.

## Protocol (anti-overfitting)

1. **Date**: 2 ani de candele 1h prin yfinance, descărcate o singură dată și
   cache-uite (`gold_monitor/data_cache/`), astfel încât optimizer-ul și
   backtester-ul văd exact aceleași date.
   - Gold `GC=F`: 11.459 candele, 2024-08-25 → 2026-08-25
   - BTC `BTC-USD`: 17.488 candele, 2024-08-25 → 2026-08-25
2. **Split 50/25/25**: model antrenat pe primii 50%; grid search pe mijlocul
   de 25% („fereastra de optimizare”); ultimii 25% = **OOS, rulat o singură
   dată per configurație**, doar pentru decizie.
3. **Regula de selecție** (declarată înainte de orice rulare OOS): cel mai
   bun scor `sharpe × profit_factor` dintre combinațiile cu ≥30 trades în
   fereastra de optimizare (fallback ≥15, apoi best overall).
4. **Execuție v3 peste tot** (optimizer + backtester, același motor):
   intrare la open-ul candelei următoare semnalului; SL/TP pe HIGH/LOW
   (fitilurile contează); SL prioritar dacă SL+TP sunt atinse în aceeași
   candelă; gap prin SL executat la open (preț mai prost); TP niciodată mai
   bine decât nivelul TP; închiderea forțată de final de perioadă
   actualizează equity-ul.
5. **Costuri**: gold — spread 3 pips × $0.10 (model pips); BTC — **0.30%
   round-trip procentual** (`SPREAD_PCT`, spread tipic CFD retail 0.20–0.35%:
   XTB ~0.22% țintă, Capital.com ~0.35% în ore volatile) + slippage 0.5×
   spread ⇒ ~0.45% efectiv per trade.

## Criterii de activare (toate, pe OOS, după costuri)

minimum 30 trades · profit factor ≥ 1.15 · expectanță medie/trade > 0 ·
max drawdown < 15% · trade-Sharpe > 0.5

---

## XAU/USD (Gold) — GC=F, 1h

**Configurația evaluată** (aleasă de optimizer pe fereastra de optimizare):
prag etichetare 0.005, SL = 2.5×ATR, TP = 4.0×ATR, confidence ≥ 0.50,
ADX ≥ 15, R:R ≥ 1.0. Model: XGBoost, CV walk-forward 56.4% (train 5.710
candele; optim 2.855; OOS 2.856).

### Execuție v3 (cinstită)

| Fereastră | Return | Trades | WinRate | PF | Avg/trade | Trade-Sharpe | MaxDD | Costuri cum. |
|---|---|---|---|---|---|---|---|---|
| Optimizare (in-sample) | **+30.51%** | 62 | 53.2% | 1.61 | +0.460% | 0.44 | −9.34% | 1.28% |
| **OOS (decizie)** | **−16.17%** | 92 | 38.0% | 0.82 | −0.173% | −0.25 | **−33.71%** | 1.87% |

Criterii OOS: trades ✔ (92) · PF ✘ (0.82 < 1.15) · expectanță ✘ (−0.173%) ·
maxDD ✘ (−33.7%) · trade-Sharpe ✘ (−0.25) → **ENABLED=False**.

Interpretare: +30.5% pe fereastra pe care s-au ales parametrii și −16.2%
imediat după este semnătura clasică a overfitting-ului de parametri. Modelul
reantrenat pe toate datele confirmă degradarea: accuracy pe ultimii 20% din
date = 29.4% (CV 40.4% ± 11.7%) — regimul recent al aurului nu seamănă cu
cel pe care s-a învățat.

### Execuția veche (legacy, pre-v3) vs v3 — cât umflau bug-urile

Același model, aceleași date, aceiași parametri; singura diferență e motorul
de execuție. Legacy = intrare pe close-ul candelei de semnal, SL/TP
verificate doar pe close (fitilurile ignorate), gap-urile umplute exact la
nivelul SL, închiderea finală fără actualizare de equity.

| Fereastră | v3 (cinstit) | Legacy (buggy) | Umflare |
|---|---|---|---|
| Optimizare — Return | +30.51% | **+51.91%** | **+21.4 pp** |
| Optimizare — PF / WR / trades | 1.61 / 53.2% / 62 | 2.68 / 60.4% / 48 | — |
| **OOS — Return** | **−16.17%** | **+2.26%** | **+18.4 pp** |
| OOS — PF / WR / trades | 0.82 / 38.0% / 92 | 1.06 / 42.3% / 71 | — |
| OOS — MaxDD | −33.71% | −13.99% | de 2.4× mai adânc în realitate |

Negru pe alb: **execuția veche transforma o strategie perdantă OOS (−16%)
într-una aparent pe break-even (+2%)** — „edge-ul” era în bug-uri, nu în
semnal. Mecanismele: (1) SL-urile atinse de fitiluri intrabar nu se
declanșau → pierderile reale dispăreau sau deveneau exit-uri mai târzii mai
bune; (2) intrarea pe close-ul candelei de semnal folosea un preț
necunoscut în timp real; (3) gap-urile prin SL se umpleau la nivelul SL, nu
la open-ul real; (4) candelele cu SL+TP simultan alegeau implicit varianta
optimistă.

---

## BTC/USD (Bitcoin) — BTC-USD, 1h

**Configurația evaluată**: grid de prag 0.008/0.01/0.015/0.02 (CV model:
52.8% / 57.4% / 69.0% / 79.8% — creșterea vine din dominanța clasei HOLD,
nu din putere predictivă), ales prag 0.015, SL = 2.5×ATR, TP = 4.0×ATR,
confidence ≥ 0.45, ADX ≥ 25, R:R ≥ 1.0. Train 8.725 / optim 4.362 /
OOS 4.363 candele. 2.048 combinații testate prin motorul v3.

### Execuție v3

| Fereastră | Return | Trades | WinRate | PF | Avg/trade | Trade-Sharpe | MaxDD | Costuri cum. |
|---|---|---|---|---|---|---|---|---|
| Optimizare (in-sample) | **−16.40%** | 124 | 45.2% | 0.91 | −0.110% | −0.11 | −25.18% | 55.80% |
| **OOS (o singură rulare)** | **−24.32%** | 119 | 38.7% | 0.82 | −0.208% | −0.24 | −38.11% | 53.55% |

Criterii OOS: trades ✔ (119) · PF ✘ · expectanță ✘ · maxDD ✘ ·
trade-Sharpe ✘ → **ENABLED=False**.

Interpretare: **nicio combinație din grid nu a fost profitabilă nici măcar
in-sample** (top-10 integral negativ). Cauza dominantă e structurală:
~0.45% cost efectiv per trade × ~120 trades ⇒ **~54% din capital tocat pe
costuri** pe fereastră. La spread-ul CFD retail, un model de 1h cu ținte de
2.5–4×ATR pe BTC nu are unde să găsească margine. Pentru legacy (doar
context): optim −19.49% / OOS −33.25% — aici și execuția veche pierdea.

---

## Consecințe & pași următori

1. `gold_monitor/config/settings.py`: gold și BTC au `ENABLED=False` cu
   metricile exacte în comentarii (standardul EURUSD: „no edge” = disabled).
2. Bot-ul de execuție Capital.com rămâne în carantină
   (`execution_capital/`, Poarta 2 neatinsă — nicio dovadă de edge).
3. Infrastructura de colectare (signals.db, outcome tracking, `--report`)
   este gata pentru orice configurație viitoare care trece poarta.
4. Orice re-testare a acestor instrumente cere **o configurație nouă și/sau
   date noi** (regula: OOS o singură dată per configurație — fereastra OOS
   folosită aici e consumată).
5. Rezervele de îmbunătățire onestă (fără a re-folosi OOS-ul curent):
   features de regim mai bune pe 1h, orizonturi mai lungi (4h/1d) unde
   costurile pesează proporțional mai puțin, praguri de etichetare legate de
   ATR, execuție pe spread-uri instituționale mai mici.

**Artefacte**: JSON-urile complete ale evaluării (parametri, top-10, toate
metricile, verdicte) sunt versionate în `docs/evaluations/`. Datele brute și
modelele antrenate rămân locale (`gold_monitor/data_cache/`,
`gold_monitor/models/` — gitignored).

---

## Notă v3.1 (2026-09-02) — recalibrarea trade-Sharpe

Metricile de mai sus au fost produse cu anualizare la **252 perioade/an**
(corectă pentru candele zilnice), deși evaluarea rulează pe candele **1h**
(~5.980/an pentru aur, 8.760/an pentru BTC). Efect: Sharpe-ul pe candele și
**trade-Sharpe-ul folosit de poartă erau subestimate de ~4,9× (aur) / ~5,9×
(BTC)** — criteriul „trade-Sharpe > 0,5" era, în realitate, „> ~2,5".

Ce **nu** se schimbă: semnul metricilor, clasamentul din optimizer (factorul
e constant per instrument) și verdictele — ambele instrumente aveau
trade-Sharpe negativ, deci rămân `ENABLED=False`. Ce se schimbă din v3.1:
`BacktestMetrics` anualizează cu `InstrumentConfig.candles_per_year()`.
Tabelele de mai sus **nu au fost re-rulate** (fereastra OOS e consumată);
pragul rămâne declarat la 0,5 și se re-declară explicit, înainte de
următoarea rulare OOS, dacă vrem să-l schimbăm.
