# Trading Game — Raport final

- **Univers**: 20 simboluri S&P 500, date zilnice (reale (yfinance))
- **Segmente**: train 2015-01-01→2019-12-31 · validation 2020-01-01→2021-12-31 · test 2022-01-01→2023-12-31
- **Jucători**: 7 · capital inițial $100,000 · costuri: 0.10% comision (min $1) + 0.050% spread + slippage 0.030%+impact · lichiditate max 1% ADV

## 🏆 Câștigător: `P6_momentum`

Metoda: Experience-based heuristic: heavy trend/momentum weights; oscillators flipped from contrarian to trend-following (overbought = strength, not a fade).

### Ponderi finale (descoperite)

| Indicator | Pondere | Orientare | Impact ablation (test) |
|---|---|---|---|
| SMA_RATIO | 0.160 | + | -0.0071
| EMA_CROSS | 0.160 | + | +0.0114
| MACD_HIST | 0.140 | + | -0.0052
| ADX_TREND | 0.120 | + | -0.0066
| ROC | 0.120 | + | -0.0024
| AROON | 0.100 | + | +0.0038
| OBV_TREND | 0.080 | + | -0.0339
| RSI | 0.060 | − | -0.0169
| VOLUME_SURGE | 0.060 | + | -0.0304

### Performanță

| | Validation (competiție) | Test (necunoscut, ponderi fixe) |
|---|---|---|
| Return | +27.02% | -0.44% |
| Sharpe | 0.93 | -0.16 |
| Sortino | 1.07 | -0.22 |
| Max DD | -6.66% | -11.88% |
| Bootstrap CI 95% (Sharpe) | — | [-1.47, 1.40] |
| Semnificativ statistic | — | NU |
| Bate piața (semnificativ) | — | NU (p=0.854) |
| Stress test | — | PASS |

### Scenarii de stres

- **covid_crash_2020**: maxDD -6.66% → supraviețuiește
- **bear_market_2022**: maxDD -9.27% → supraviețuiește
- **worst_auto_window**: maxDD -6.66% → supraviețuiește

## Clasament final (scor compozit pe test + validare statistică)

| # | Jucător | Indicatori | Scor final | Eliminat |
|---|---|---|---|---|
| 1 | P6_momentum | 9 | 0.176 | — |
| 2 | P1_equal_weight | 5 | 0.150 | — |
| 3 | P2_correlation_ic | 14 | — | score gap >= 20% |
| 4 | P3_random_forest | 12 | — | score gap >= 20% |
| 5 | P7_mean_reversion | 8 | — | score gap >= 20% |
| 6 | P5_genetic | 11 | — | score gap >= 20% |
| 7 | P4_xgboost | 11 | — | score gap >= 20% |

## Pattern-uri comune la câștigători (top 3)

- Indicatori comuni: ADX_TREND, MACD_HIST, RSI
- Recalibrări medii: 7.0
- Corelație complexitate↔performanță: n/a

## Recomandări (ponderi baseline descoperite)

| Indicator | Pondere recomandată |
|---|---|
| MACD_HIST | 0.170 |
| ADX_TREND | 0.160 |
| RSI | 0.130 |
| BB_POSITION | 0.100 |
| CMF | 0.100 |
| EMA_CROSS | 0.080 |
| SMA_RATIO | 0.080 |
| ROC | 0.060 |
| AROON | 0.050 |
| OBV_TREND | 0.040 |
| VOLUME_SURGE | 0.030 |

- Indicatori esențiali (ablation > 0): EMA_CROSS
- Indicatori redundanți (impact ≤ 0): ROC, MACD_HIST, ADX_TREND, SMA_RATIO, RSI, VOLUME_SURGE, OBV_TREND

### Best practices
- 5-10 indicatori e sweet spot-ul (penalizarea de complexitate + ablation)
- Recalibreaza trimestrial, dar gradual (max 30% shift per indicator)
- Scorul final se decide pe TEST, nu pe validation - nu te atasa de clasamentul intermediar
- Semnificatia statistica (bootstrap CI > 0) separa edge-ul de noroc
- Stress-test pe ferestrele de criza inainte de orice deployment

### Red flags
- Drawdown > 50% = eliminare directa
- Sharpe < 1.0 pe validation rareori supravietuieste pe test
- Performanta concentrata intr-un singur regim de piata
- Castig pe validation + esec bootstrap pe test = overfitting

## Eliminări pe parcurs

- 2020-05: **P4_xgboost** — performance gap
- 2020-07: **P5_genetic** — performance gap
- 2020-08: **P7_mean_reversion** — performance gap
- 2020-09: **P3_random_forest** — performance gap
- 2020-10: **P2_correlation_ic** — performance gap

---
*Toate tranzacțiile au trecut prin Crupier (validare, costuri, lichiditate, audit). Registrul complet: `audit_trade_log.json`.*