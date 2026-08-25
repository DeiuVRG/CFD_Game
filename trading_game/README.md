# Trading Game — descoperirea competitivă a ponderilor de indicatori

Implementarea specificației din `docs/trading_game_prompt.md`: o simulare
competitivă între jucători autonomi care își construiesc fiecare propria
metodă de ponderare a indicatorilor tehnici, cu scopul de a găsi ponderi
care rezistă **out-of-sample**, nu doar pe istoric.

## Cum rulezi

```bash
# din rădăcina repo-ului
pip install -r gold_monitor/requirements.txt   # numpy/pandas/sklearn/xgboost/yfinance

python -m trading_game.main                # date reale (yfinance, cache CSV)
python -m trading_game.main --synthetic    # complet offline (date sintetice)
python -m trading_game.main --players 4    # mai puțini jucători
```

Rapoartele se scriu în `trading_game/results/`:
`REPORT.md`, `castigator_report.json`, `comparative_analysis.json`,
`recommendations.json`, `audit_trade_log.json`.

## Arhitectură (fidelă spec-ului)

| Componentă | Rol |
|---|---|
| `config.py` | segmente 60/20/20 (2015-19 / 2020-21 / 2022-23), capital $100k, costuri, praguri |
| `data_loader.py` | 20 large-caps S&P 500 + SPY + ^VIX zilnic (yfinance, cache) sau generator sintetic cu regimuri |
| `indicators.py` | catalogul de indicatori (trend/momentum/volatilitate/volum) → semnale standardizate în [−1,+1] |
| `crupier.py` | **toate** tranzacțiile trec pe aici: validare, costuri realiste, lichiditate (max 1% ADV), portofolii, audit, integritate temporală, aprobat/registru date externe, eliminări, validare recalibrări |
| `costs.py` | comision 0.1% (min $1) + spread 0.05% + slippage 0.03% + impact de mărime |
| `metrics.py` | Sharpe/Sortino/Calmar/WinRate/PF → scor compozit; penalizare de complexitate; evaluare pe regimuri (VIX + trend S&P) |
| `players/` | 7 metode diferite (vezi mai jos) |
| `game.py` | pregătire → 24 luni competiție (eliminări lunare, recalibrare trimestrială ≤30% shift) → test set cu ponderi înghețate |
| `validation.py` | bootstrap CI 95% pe Sharpe, permutation test vs piață, stress pe ferestre de criză |
| `reporting.py` | raport câștigător (cu ablation study), analiză comparativă, recomandări |

## Jucătorii (metode implementate)

1. `equal_weight` — baseline: 5 indicatori clasici, ponderi egale.
2. `correlation_ic` — statistic: pondere ∝ |corelația semnal↔randament forward 21d|, orientarea = semnul corelației.
3. `random_forest` — ML: importanțele unui RandomForest care prezice randamente forward.
4. `xgboost` — ML: importanțele XGBoost pe orizont mai scurt (10d), alt set de indicatori.
5. `genetic` — optimizare: algoritm genetic (numpy) care maximizează Sharpe-ul unui portofoliu lunar pe training.
6. `momentum` — euristic: trend-following, oscilatoare inversate.
7. `mean_reversion` — euristic: contrarian pe oscilatoare + poziția în benzi.

Fiecare jucător e liber la metodă (spec 1.1); ponderile finale respectă
regulile: sumă 1, fiecare ∈ [0,1], 3–20 indicatori. „Orientarea" (±1 per
indicator) este interpretarea proprie a semnalului standardizat — partea de
metodă a jucătorului.

## Reguli aplicate de Crupier

- min 1 / max 50 tranzacții pe lună (încălcare → eliminare);
- eliminare lunară dacă ultimul e la ≥20% de penultimul (scor compozit cu
  penalizare de complexitate + scor multi-regim după luna 6);
- drawdown sub 50% din capital → eliminare imediată;
- recalibrare doar trimestrial, respinsă dacă vreun indicator sare cu >30%;
- test set: ponderi înghețate, o singură trecere;
- câștigătorul REAL: 0.40·scor test + 0.25·semnificație statistică +
  0.20·bate piața semnificativ + 0.15·stress test.

## Rezultatele rulării pe date reale (2015-2023)

Raport complet: [`results/REPORT.md`](results/REPORT.md) (+ JSON-uri de audit).
Pe scurt, rularea de referință (7 jucători, seed 42):

- Crash-ul COVID (feb-mar 2020) a prăbușit scorurile tuturor în lunile 2-3;
  cinci jucători au fost eliminați pe regula gap-ului de 20% între mai și
  octombrie 2020 (xgboost, genetic, mean_reversion, random_forest,
  correlation_ic — în această ordine).
- Supraviețuitori: `equal_weight` și `momentum`. Pe validare, momentum a
  făcut +27.0% (Sharpe 0.93, maxDD −6.7%).
- **Pe test (2022-2023, ponderi înghețate) câștigătorul de validare a făcut
  −0.44%**, bootstrap CI [−1.47, 1.40] → nesemnificativ statistic, nu bate
  piața (p=0.854); trece doar stress-testul. Ablation: un singur indicator
  cu impact pozitiv pe test (EMA_CROSS).

Interpretarea onestă (aceeași lecție ca în `RESULTS.md` al repo-ului):
clasamentul de validare nu garantează nimic out-of-sample; „ponderile
optime" descoperite au valoare doar ca prior de pornire, iar criteriul care
separă edge-ul de noroc este semnificația statistică pe date nevăzute — pe
care nimeni nu a atins-o în această rulare.

## Abateri documentate față de spec

- **Univers**: 20 de acțiuni lichide S&P 500 în loc de toate cele ~500
  (configurabil în `config.py`) — păstrează mecanica identică la un cost de
  rulare rezonabil.
- **Stress scenarios**: 2008/dot-com preced datele (2015+), deci reziliența
  se măsoară pe crizele existente în date (COVID 2020, bear 2022) + cea mai
  urâtă fereastră de piață detectată automat; același criteriu de
  supraviețuire (maxDD > −50%).
- **Long-only**: SELL înseamnă închiderea poziției (fără short) — simplifică
  contabilitatea fără să schimbe întrebarea studiată (care ponderi separă
  semnalul de zgomot).
- **GPU**: metodele implementate sunt ieftine pe CPU; punctele naturale de
  accelerare (GA population eval, RF/XGB) folosesc deja toate core-urile și
  pot fi mutate pe GPU fără schimbarea interfeței.
- **Date externe**: API-ul de aprobare + registrul cu hash există în Crupier
  (`approve_external_source` / `register_external_data`, cu verificare de
  look-ahead); jucătorii impliciți folosesc doar preț/volum, ca jocul să fie
  reproductibil offline.
