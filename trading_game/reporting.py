"""Reporting (spec Part VII): winner report with indicator-importance
ablation, comparative analysis of all players, and recommendations for the
future. Writes JSON artifacts + a human-readable REPORT.md."""
import json
import math
import os
from typing import Dict

import numpy as np
import pandas as pd

from trading_game.metrics import AdvancedMetrics, ComplexityPenalty

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Series):
        return None  # equity curves are summarized, not dumped
    return obj


def indicator_importance_ablation(game, player) -> Dict[str, dict]:
    """Ablation study on the TEST period: re-run the winner's strategy with
    each indicator removed (weights renormalized); impact = drop in the
    composite test score without it."""
    metrics = game.metrics
    base = game.test_results[player.id]["test_metrics"]["composite_score"]

    importance = {}
    for indicator in list(player.weights.keys()):
        if player.weights.get(indicator, 0) == 0:
            importance[indicator] = {"impact": 0.0,
                                     "weight": 0.0, "efficiency": 0.0}
            continue
        reduced = {k: v for k, v in player.weights.items() if k != indicator}
        total = sum(reduced.values())
        if total <= 0:
            continue
        reduced = {k: v / total for k, v in reduced.items()}
        equity, trades = game.simulate_period(player, game.config.test_start,
                                              game.config.test_end,
                                              weights_override=reduced)
        returns = equity.pct_change().dropna().to_numpy()
        score_without = metrics.composite_score(returns, equity.to_numpy(), trades)
        impact = base - score_without
        weight = player.weights[indicator]
        importance[indicator] = {
            "impact": impact,
            "weight": weight,
            "efficiency": impact / weight if weight > 0 else 0.0,
        }
    return dict(sorted(importance.items(),
                       key=lambda kv: kv[1]["impact"], reverse=True))


def winner_report(game, winner_id: str) -> dict:
    player = next(p for p in game.players if p.id == winner_id)
    result = game.test_results[winner_id]

    val_equity = game.crupier.equity_series(winner_id)
    val_returns = val_equity.pct_change().dropna()
    val_metrics = game.metrics.full_metrics(
        val_returns.to_numpy(), val_equity.to_numpy(),
        game.crupier.closed_trades[winner_id])

    regimes = game.regime_eval.multi_regime_score(val_returns, game.metrics)

    trades = game.crupier.closed_trades[winner_id]
    notable = sorted(trades, key=lambda t: t["profit"])
    notable = {"worst": notable[:3], "best": notable[-3:][::-1]}

    importance = indicator_importance_ablation(game, player)

    return {
        "identity": {
            "player_id": player.id,
            "method": player.method_description,
            "recalibrations_used": player.recalibrations,
        },
        "final_weights": dict(sorted(player.weights.items(),
                                     key=lambda kv: -kv[1])),
        "orientations": player.orientation,
        "performance_validation": val_metrics,
        "performance_test": {
            **result["test_metrics"],
            "confidence_interval": result["bootstrap"]["confidence_interval"],
            "statistically_significant": result["bootstrap"]["is_significant"],
            "beats_market": result["vs_market"]["beats_market"],
            "beats_market_significant": result["vs_market"]["is_significant"],
            "p_value_vs_market": result["vs_market"]["p_value"],
        },
        "regime_breakdown": regimes["per_regime"],
        "stress_test": result["stress"],
        "indicator_importance_ablation": importance,
        "notable_trades": notable,
        "final_score_breakdown": {
            "final_score": result["final_score"],
            "test_score_adjusted": result["test_score_adjusted"],
            "is_significant": result["bootstrap"]["is_significant"],
            "beats_market": result["vs_market"]["beats_market"],
            "passes_stress": result["stress"]["pass_stress_test"],
        },
    }


def comparative_analysis(game) -> dict:
    ranking = []
    for player in game.players:
        entry = {
            "player_id": player.id,
            "method": player.method_description,
            "n_indicators": len(player.indicators),
            "eliminated": player.id in game.crupier.eliminated,
            "elimination_reason": game.crupier.eliminated.get(player.id),
            "last_monthly_score": (game.monthly_scores[player.id][-1]
                                   if game.monthly_scores[player.id] else None),
            "final_score": (game.test_results.get(player.id, {})
                            .get("final_score")),
            "weights": dict(sorted(player.weights.items(), key=lambda kv: -kv[1])),
        }
        ranking.append(entry)
    ranking.sort(key=lambda e: (e["final_score"] is None,
                                -(e["final_score"] or 0),
                                -(e["last_monthly_score"] or 0)))

    top3 = [e for e in ranking if not e["eliminated"]][:3]
    top_ids = [e["player_id"] for e in top3]
    top_players = [p for p in game.players if p.id in top_ids]

    common_indicators = (set.intersection(*[set(p.weights) for p in top_players])
                         if top_players else set())
    weight_ranges = {}
    for indicator in sorted(set().union(*[set(p.weights) for p in top_players])
                            if top_players else []):
        vals = [p.weights.get(indicator, 0.0) for p in top_players]
        weight_ranges[indicator] = {
            "min": min(vals), "max": max(vals),
            "avg": float(np.mean(vals)),
        }

    popularity = {}
    for player in game.players:
        for indicator, weight in player.weights.items():
            entry = popularity.setdefault(indicator, {"players": 0, "avg_weight": 0.0})
            entry["players"] += 1
            entry["avg_weight"] += weight
    for entry in popularity.values():
        entry["avg_weight"] /= max(entry["players"], 1)

    complexity_perf = [
        (e["n_indicators"], e["final_score"])
        for e in ranking if e["final_score"] is not None
    ]
    corr = None
    if len(complexity_perf) >= 3:
        xs, ys = zip(*complexity_perf)
        if np.std(xs) > 0 and np.std(ys) > 0:
            corr = float(np.corrcoef(xs, ys)[0, 1])

    return {
        "final_ranking": ranking,
        "winner_patterns": {
            "common_indicators_top3": sorted(common_indicators),
            "weight_ranges_top3": weight_ranges,
            "avg_recalibrations_top3": float(np.mean(
                [p.recalibrations for p in top_players])) if top_players else 0,
        },
        "eliminated_mistakes": [
            {"player": e["player_id"], "reason": e["elimination_reason"]}
            for e in ranking if e["eliminated"]
        ],
        "indicator_popularity": dict(sorted(
            popularity.items(), key=lambda kv: -kv[1]["avg_weight"])),
        "complexity_vs_performance_corr": corr,
        "eliminations": game.elimination_log,
    }


def recommendations(game, winner: dict, comparison: dict) -> dict:
    ablation = winner.get("indicator_importance_ablation", {})
    essential = [k for k, v in ablation.items() if v["impact"] > 0.005]
    redundant = [k for k, v in ablation.items() if v["impact"] <= 0.0]

    top3_ranges = comparison["winner_patterns"]["weight_ranges_top3"]
    baseline = {k: round(v["avg"], 4) for k, v in sorted(
        top3_ranges.items(), key=lambda kv: -kv[1]["avg"]) if v["avg"] > 0.02}

    return {
        "baseline_recommended_weights": baseline,
        "essential_indicators": essential,
        "redundant_indicators": redundant,
        "best_practices": [
            "5-10 indicatori e sweet spot-ul (penalizarea de complexitate + ablation)",
            "Recalibreaza trimestrial, dar gradual (max 30% shift per indicator)",
            "Scorul final se decide pe TEST, nu pe validation - nu te atasa de clasamentul intermediar",
            "Semnificatia statistica (bootstrap CI > 0) separa edge-ul de noroc",
            "Stress-test pe ferestrele de criza inainte de orice deployment",
        ],
        "red_flags": [
            "Drawdown > 50% = eliminare directa",
            "Sharpe < 1.0 pe validation rareori supravietuieste pe test",
            "Performanta concentrata intr-un singur regim de piata",
            "Castig pe validation + esec bootstrap pe test = overfitting",
        ],
    }


def _fmt_pct(x):
    return "n/a" if x is None else f"{x:+.2f}%"


def write_markdown(game, winner_id, winner, comparison, recs) -> str:
    lines = []
    add = lines.append
    add("# Trading Game — Raport final")
    add("")
    cfg = game.config
    add(f"- **Univers**: {len(cfg.universe)} simboluri S&P 500, date zilnice"
        f" ({'sintetice' if cfg.synthetic else 'reale (yfinance)'})")
    add(f"- **Segmente**: train {cfg.train_start}→{cfg.train_end} · "
        f"validation {cfg.validation_start}→{cfg.validation_end} · "
        f"test {cfg.test_start}→{cfg.test_end}")
    add(f"- **Jucători**: {len(game.players)} · capital inițial "
        f"${cfg.initial_capital:,.0f} · costuri: {cfg.commission_percent:.2%} "
        f"comision (min ${cfg.commission_fixed:.0f}) + {cfg.spread_percent:.3%} "
        f"spread + slippage {cfg.slippage_base:.3%}+impact · lichiditate max "
        f"{cfg.max_percent_daily_volume:.0%} ADV")
    add("")
    add(f"## 🏆 Câștigător: `{winner_id}`")
    add("")
    add(f"Metoda: {winner['identity']['method']}")
    add("")
    add("### Ponderi finale (descoperite)")
    add("")
    add("| Indicator | Pondere | Orientare | Impact ablation (test) |")
    add("|---|---|---|---|")
    ablation = winner["indicator_importance_ablation"]
    for ind, w in winner["final_weights"].items():
        orient = winner["orientations"].get(ind, 1.0)
        imp = ablation.get(ind, {}).get("impact")
        add(f"| {ind} | {w:.3f} | {'+' if orient >= 0 else '−'} | "
            f"{imp:+.4f}" if imp is not None else
            f"| {ind} | {w:.3f} | {'+' if orient >= 0 else '−'} | n/a")
    add("")
    pv = winner["performance_validation"]
    pt = winner["performance_test"]
    add("### Performanță")
    add("")
    add("| | Validation (competiție) | Test (necunoscut, ponderi fixe) |")
    add("|---|---|---|")
    add(f"| Return | {_fmt_pct(pv['total_return_pct'])} | {_fmt_pct(pt['total_return_pct'])} |")
    add(f"| Sharpe | {pv['sharpe']:.2f} | {pt['sharpe']:.2f} |")
    add(f"| Sortino | {pv['sortino']:.2f} | {pt['sortino']:.2f} |")
    add(f"| Max DD | {pv['max_drawdown']:.2%} | {pt['max_drawdown']:.2%} |")
    ci = pt["confidence_interval"]
    add(f"| Bootstrap CI 95% (Sharpe) | — | [{ci[0]:.2f}, {ci[1]:.2f}] |")
    add(f"| Semnificativ statistic | — | {'DA' if pt['statistically_significant'] else 'NU'} |")
    add(f"| Bate piața (semnificativ) | — | "
        f"{'DA' if pt['beats_market_significant'] and pt['beats_market'] else 'NU'} "
        f"(p={pt['p_value_vs_market']:.3f}) |")
    stress = winner["stress_test"]
    add(f"| Stress test | — | {'PASS' if stress['pass_stress_test'] else 'FAIL'} |")
    add("")
    add("### Scenarii de stres")
    add("")
    for name, res in stress["scenarios"].items():
        add(f"- **{name}**: maxDD {res['max_drawdown']:.2%} → "
            f"{'supraviețuiește' if res['survives'] else 'NU supraviețuiește'}")
    add("")
    add("## Clasament final (scor compozit pe test + validare statistică)")
    add("")
    add("| # | Jucător | Indicatori | Scor final | Eliminat |")
    add("|---|---|---|---|---|")
    for i, entry in enumerate(comparison["final_ranking"], 1):
        fs = entry["final_score"]
        add(f"| {i} | {entry['player_id']} | {entry['n_indicators']} | "
            f"{fs:.3f}" if fs is not None else
            f"| {i} | {entry['player_id']} | {entry['n_indicators']} | —"
            )
        if lines[-1].count("|") < 6:
            lines[-1] += (f" | {entry['elimination_reason'] or '—'} |")
    add("")
    add("## Pattern-uri comune la câștigători (top 3)")
    add("")
    add(f"- Indicatori comuni: "
        f"{', '.join(comparison['winner_patterns']['common_indicators_top3']) or '—'}")
    add(f"- Recalibrări medii: "
        f"{comparison['winner_patterns']['avg_recalibrations_top3']:.1f}")
    corr = comparison["complexity_vs_performance_corr"]
    add(f"- Corelație complexitate↔performanță: "
        f"{corr:+.2f}" if corr is not None else
        "- Corelație complexitate↔performanță: n/a")
    add("")
    add("## Recomandări (ponderi baseline descoperite)")
    add("")
    add("| Indicator | Pondere recomandată |")
    add("|---|---|")
    for ind, w in recs["baseline_recommended_weights"].items():
        add(f"| {ind} | {w:.3f} |")
    add("")
    add(f"- Indicatori esențiali (ablation > 0): "
        f"{', '.join(recs['essential_indicators']) or '—'}")
    add(f"- Indicatori redundanți (impact ≤ 0): "
        f"{', '.join(recs['redundant_indicators']) or '—'}")
    add("")
    add("### Best practices")
    for bp in recs["best_practices"]:
        add(f"- {bp}")
    add("")
    add("### Red flags")
    for rf in recs["red_flags"]:
        add(f"- {rf}")
    add("")
    add("## Eliminări pe parcurs")
    add("")
    if game.elimination_log:
        for e in game.elimination_log:
            add(f"- {e['month']}: **{e['player']}** — {e['reason']}")
    else:
        add("- nicio eliminare")
    add("")
    add("---")
    add("*Toate tranzacțiile au trecut prin Crupier (validare, costuri, "
        "lichiditate, audit). Registrul complet: `audit_trade_log.json`.*")
    return "\n".join(lines)


def generate_all_reports(game) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    winner_id = game.determine_winner()
    if winner_id is None:
        return {}

    winner = winner_report(game, winner_id)
    comparison = comparative_analysis(game)
    recs = recommendations(game, winner, comparison)

    artifacts = {
        "castigator_report.json": winner,
        "comparative_analysis.json": comparison,
        "recommendations.json": recs,
        "audit_trade_log.json": {
            "trades": game.crupier.trade_log,
            "external_sources": game.crupier.external_sources,
            "external_data_registry": game.crupier.external_data_registry,
        },
    }
    for name, payload in artifacts.items():
        with open(os.path.join(RESULTS_DIR, name), "w") as f:
            json.dump(_json_safe(payload), f, indent=2, default=str)

    markdown = write_markdown(game, winner_id, winner, comparison, recs)
    with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w") as f:
        f.write(markdown)

    print(f"\n  Rapoarte generate in {RESULTS_DIR}:")
    for name in list(artifacts) + ["REPORT.md"]:
        print(f"    - {name}")

    return {"winner": winner, "comparison": comparison,
            "recommendations": recs, "results_dir": RESULTS_DIR}
