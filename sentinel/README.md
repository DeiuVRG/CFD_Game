# Sentinel — santinela pe cont DEMO

**Ce este**: un supraveghetor care ia semnalele deterministe emise de
`gold_monitor` (XGBoost pe candele 1h + regulile v3), le dă lui Claude
(Fable 5.1) cu tot contextul — snapshot tehnic, track record-ul semnalelor,
starea contului și o cercetare de piață cu web search — și, **în limitele
hard din cod**, execută pe contul **DEMO** Capital.com și gestionează
pozițiile deschise.

**Ce NU este**: nu e un trader autonom. Modelul nu poate deschide poziții
fără semnal determinist, nu poate mări riscul, nu poate lărgi stop-ul, nu
poate depăși limitele zilnice și nu poate atinge contul live
(`CAPITAL_MODE=live` e refuzat de cod).

Decizie de proiect (2026-09-02): utilizatorul a ales explicit această
abatere de la regula „zero LLM în bucla de decizie”, **pe demo**, cu
fiecare decizie logată ca să putem măsura dacă modelul adaugă valoare față
de calea deterministă (`--no-llm` = grupul de control).

## Cum funcționează

```
gold_monitor --monitor  ──►  signals.db (tier=demo)
                                   │  cursor (fetch_since)
                                   ▼
                     sentinel: research (web search, cache 1h)
                               → decizie APPROVE/VETO + size_fraction (JSON strict)
                               → reguli hard (rules.py)
                               → sizing 1% risc × size_fraction
                               → ordin DEMO (Capital.com) → decisions.db → Discord
                     la 15 min: poziții deschise → HOLD / CLOSE / TIGHTEN_SL
                               → reguli hard → ordin DEMO → decisions.db
                     la fiecare poll: poziții dispărute de la broker → outcome
```

Reguli hard (`sentinel/rules.py`, testate): semnal mai vechi de 15 min =
ignorat; risc/trade max 1% (modelul poate doar reduce); pierdere zilnică ≥3%
= niciun trade nou; max 5 trade-uri/zi; max 2 poziții simultan; o singură
poziție per instrument; R:R ≥ 1; SL-ul se mută doar în favoarea poziției și
niciodată peste prețul curent. Dacă modelul refuză, dă eroare sau răspunde
neparsabil → **fail closed** (nu deschide, nu schimbă nimic).

Modelul primește context strict factual (`sentinel/context.py`) și răspunde
JSON validat de schemă (`sentinel/schema.py`); prompturile sunt în
`sentinel/brain.py`. Apelurile folosesc `claude-fable-5-1` cu
`fallbacks: "default"` (dacă clasificatorii de siguranță refuză cererea,
API-ul o re-rulează automat pe modelul recomandat de Anthropic).

## Configurare

Toate secretele în `gold_monitor/.env` (vezi `gold_monitor/.env.example`):

```
ANTHROPIC_API_KEY=sk-ant-...
CAPITAL_API_KEY=...          CAPITAL_IDENTIFIER=...   CAPITAL_PASSWORD=...
CAPITAL_MODE=demo            # singura valoare acceptată de sentinel
DISCORD_WEBHOOK_URL=...      # opțional
SENTINEL_EPIC_GOLD=GOLD      SENTINEL_EPIC_BTC=BITCOIN   # verifică cu --markets
SENTINEL_DRY_RUN=false       SENTINEL_WEB_SEARCH=true
```

```bash
pip install -r sentinel/requirements.txt
python -m sentinel.main --markets gold        # confirmă epic-urile pe demo
python -m sentinel.main --run --dry-run       # decizii logate, zero ordine
python -m sentinel.main --run                 # DEMO
python -m sentinel.main --run --no-llm        # control determinist
python -m sentinel.main --report              # rezumat decisions.db
```

Rulează în paralel cu `cd gold_monitor && python main.py --monitor`
(sentinel-ul doar citește `signals.db`). La prima pornire cursorul se
setează la ultimul semnal existent — istoricul nu e rejucat.

## Unde sunt datele

- `sentinel/data/decisions.db` — fiecare decizie (semnal, răspunsul
  modelului, motivarea, riscurile, acțiunea finală + motivul regulii,
  tokens, deal_id, outcome/P&L). Append-only.
- tabela `research` — brief-urile de piață (cache 1h per instrument).
- `sentinel/logs/sentinel.log`.

## Cum măsurăm valoarea modelului

Fiecare semnal are un outcome ipotetic în `signals.db` (calea
deterministă) și, dacă a fost executat, un P&L real pe demo în
`decisions.db`. Semnalele **respinse** de model au totuși outcome ipotetic —
deci se poate calcula exact cât a câștigat sau pierdut veto-ul. Rulează o
perioadă cu `--no-llm` ca grup de control.

## Limitări cunoscute

- P&L-ul pozițiilor închise de broker (TP/SL) e ultimul P&L văzut la poll
  (la 60s), nu valoarea exactă de închidere.
- Sesiunea Capital.com e demo-only prin cod; contul live este Poarta 3 și
  rămâne o decizie separată, explicită.
- Web search face rezultatele nereproductibile — brief-urile sunt salvate
  integral tocmai ca să poată fi auditate ulterior.
