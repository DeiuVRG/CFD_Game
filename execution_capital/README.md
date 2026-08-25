# execution_capital — Modul de execuție Capital.com (CARANTINĂ)

**Modul de execuție. Inactiv până la Poarta 2. Nu rula pe live.**

Acest director conține botul de execuție automată pe Capital.com (demo/live).
Este păstrat ca infrastructură viitoare, dar **nu se pornește și nu se
dezvoltă** până când un instrument nu trece poarta de activare din
`gold_monitor` (criteriile OOS din `RESULTS.md`).

## De ce e în carantină

Filosofia proiectului: *colectăm dovezi, nu bani*. Un instrument devine
tranzacționabil doar după ce își demonstrează edge-ul out-of-sample, după
costuri, cu execuție pesimistă. Până atunci, singura componentă activă este
`gold_monitor/` (semnale pe Discord, fără execuție).

## Porți (gates)

1. **Poarta 1 — Semnale**: instrumentul trece criteriile OOS (≥30 trades,
   PF ≥ 1.15, expectanță > 0, maxDD < 15%, trade-Sharpe > 0.5) și primește
   `ENABLED=True` în `gold_monitor/config/settings.py`. Semnalele se
   colectează în `signals.db` și se urmăresc outcome-urile ipotetice.
2. **Poarta 2 — Execuție demo**: după o perioadă de colectare live a
   semnalelor cu statistici pozitive (`python main.py --report`), botul de
   aici poate fi pornit pe **demo**.
3. **Poarta 3 — Live**: doar după dovezi pe demo. Nu înainte.

## Fix-uri aplicate în carantină (bug-uri cunoscute, reparate dormant)

- `PositionSizer` cere explicit rata de conversie a monedei de cotare;
  simbolurile necotate în moneda contului sunt respinse dacă rata lipsește
  (înainte: dimensionare greșită pe perechi gen USDJPY).
- `MAX_TRADES_PER_DAY` este acum aplicat (înainte: `_trades_today` era
  incrementat dar nefolosit).
- Regulă anti-corelare: maximum o poziție FX simultan.
- Risc redus: 1% per trade (era 3%), limită zilnică 3% (era 10%).

## Rulare (DOAR când se ajunge la Poarta 2)

```bash
cd execution_capital
cp .env.example .env   # completează credențialele Capital.com (mode=demo!)
pip install -r requirements.txt
python main.py --mode demo
```

`CAPITAL_MODE=live` rămâne blocat de proces (poarta 3), nu de cod — nu-l
folosi.
