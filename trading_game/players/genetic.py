"""Genetic-algorithm player: evolves the weight vector by maximizing the
Sharpe ratio of a fast vectorized monthly-rebalance portfolio simulation on
the TRAINING set. Pure numpy (no external GA dependency); population-based
search parallelizes trivially and would map to GPU, but CPU is plenty here."""
import numpy as np
import pandas as pd

from trading_game.players.base import BasePlayer


class GeneticPlayer(BasePlayer):
    name = "genetic"
    method_description = (
        "Genetic algorithm (pop=48, gens=60, tournament selection, blend "
        "crossover, gaussian mutation) maximizing training-set Sharpe of a "
        "monthly top-K portfolio built from the weighted signal score."
    )

    POP = 48
    GENS = 60
    TOP_K = 6
    REBALANCE = 21

    def __init__(self, player_id, config, seed=0):
        super().__init__(player_id, config, seed)
        self.indicators = [
            "SMA_RATIO", "EMA_CROSS", "MACD_HIST", "ADX_TREND", "AROON",
            "RSI", "ROC", "CCI", "BB_POSITION", "OBV_TREND", "CMF",
        ]

    # ------------------------------------------------------------------

    def _prepare_panels(self, train_data):
        """3-D signal tensor (days x symbols x indicators) + daily returns
        matrix, aligned on a common calendar."""
        symbols = sorted(train_data.keys())
        calendars = [train_data[s]["signals"].index for s in symbols]
        common = calendars[0]
        for cal in calendars[1:]:
            common = common.intersection(cal)
        common = common.sort_values()

        sig_cube = np.zeros((len(common), len(symbols), len(self.indicators)))
        rets = np.zeros((len(common), len(symbols)))
        for j, symbol in enumerate(symbols):
            bundle = train_data[symbol]
            sig = bundle["signals"].reindex(common)[self.indicators]
            sig_cube[:, j, :] = np.nan_to_num(sig.to_numpy(), nan=0.0)
            close = bundle["close"].reindex(common)
            rets[:, j] = close.pct_change().fillna(0.0).to_numpy()
        return sig_cube, rets

    def _fitness(self, weights: np.ndarray, orient: np.ndarray,
                 sig_cube: np.ndarray, rets: np.ndarray) -> float:
        scores = sig_cube @ (weights * orient)          # days x symbols
        n_days = scores.shape[0]
        port_rets = []
        holdings = None
        for start in range(60, n_days - 1, self.REBALANCE):
            row = scores[start]
            candidates = np.argsort(row)[::-1][: self.TOP_K]
            candidates = candidates[row[candidates] > self.config.signal_buy_threshold]
            holdings = candidates
            end = min(start + self.REBALANCE, n_days)
            if holdings is None or len(holdings) == 0:
                port_rets.extend([0.0] * (end - start - 1))
                continue
            window = rets[start + 1:end, holdings].mean(axis=1)
            port_rets.extend(window.tolist())
        arr = np.array(port_rets)
        if len(arr) < 30 or np.std(arr) == 0:
            return -5.0
        return float(np.mean(arr) / np.std(arr) * np.sqrt(252))

    def fit(self, train_data):
        rng = np.random.default_rng(self.seed)
        sig_cube, rets = self._prepare_panels(train_data)

        # Orientation learned once from pooled correlation (kept fixed while
        # the GA searches the magnitude space). Row-major flattening aligns
        # each (day t, symbol j) signal with the day t+1 return.
        n_ind = len(self.indicators)
        orient = np.ones(n_ind)
        fwd_aligned = np.vstack([rets[1:], np.zeros((1, rets.shape[1]))]).reshape(-1)
        for i in range(n_ind):
            col = sig_cube[:, :, i].reshape(-1)
            if np.std(col) > 0 and np.std(fwd_aligned) > 0:
                c = np.corrcoef(col, fwd_aligned)[0, 1]
                orient[i] = 1.0 if (np.isnan(c) or c >= 0) else -1.0
        self.orientation = dict(zip(self.indicators, orient.tolist()))

        pop = rng.random((self.POP, n_ind))
        pop /= pop.sum(axis=1, keepdims=True)

        def evaluate(population):
            return np.array([
                self._fitness(ind, orient, sig_cube, rets) for ind in population
            ])

        fitness = evaluate(pop)
        for _ in range(self.GENS):
            new_pop = [pop[int(np.argmax(fitness))].copy()]  # elitism
            while len(new_pop) < self.POP:
                # Tournament selection
                a, b = rng.choice(self.POP, 2, replace=False)
                p1 = pop[a] if fitness[a] >= fitness[b] else pop[b]
                a, b = rng.choice(self.POP, 2, replace=False)
                p2 = pop[a] if fitness[a] >= fitness[b] else pop[b]
                # Blend crossover + gaussian mutation
                alpha = rng.random()
                child = alpha * p1 + (1 - alpha) * p2
                child += rng.normal(0, 0.08, n_ind)
                child = np.clip(child, 0.0, None)
                if child.sum() <= 0:
                    child = rng.random(n_ind)
                child /= child.sum()
                new_pop.append(child)
            pop = np.array(new_pop)
            fitness = evaluate(pop)

        best = pop[int(np.argmax(fitness))]
        self.weights = self._normalize(dict(zip(self.indicators, best.tolist())))
