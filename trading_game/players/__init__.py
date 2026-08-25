from trading_game.players.base import BasePlayer
from trading_game.players.heuristic import EqualWeightPlayer, MomentumPlayer, MeanReversionPlayer
from trading_game.players.correlation import CorrelationPlayer
from trading_game.players.ml import RandomForestPlayer, XGBoostPlayer
from trading_game.players.genetic import GeneticPlayer

DEFAULT_PLAYER_CLASSES = [
    EqualWeightPlayer,
    CorrelationPlayer,
    RandomForestPlayer,
    XGBoostPlayer,
    GeneticPlayer,
    MomentumPlayer,
    MeanReversionPlayer,
]

__all__ = [
    "BasePlayer", "EqualWeightPlayer", "MomentumPlayer", "MeanReversionPlayer",
    "CorrelationPlayer", "RandomForestPlayer", "XGBoostPlayer", "GeneticPlayer",
    "DEFAULT_PLAYER_CLASSES",
]
