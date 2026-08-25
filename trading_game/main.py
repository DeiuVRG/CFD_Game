#!/usr/bin/env python3
"""CLI for the competitive trading game.

Usage (from the repo root):
    python -m trading_game.main                 # real data (yfinance, cached)
    python -m trading_game.main --synthetic     # fully offline
    python -m trading_game.main --players 4     # fewer players
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_game.config import GameConfig
from trading_game.game import TradingGameSystem
from trading_game.players import DEFAULT_PLAYER_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Competitive trading game")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (fully offline)")
    parser.add_argument("--players", type=int, default=len(DEFAULT_PLAYER_CLASSES),
                        help=f"Number of players (2-{len(DEFAULT_PLAYER_CLASSES)})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    n = max(2, min(args.players, len(DEFAULT_PLAYER_CLASSES)))
    config = GameConfig(synthetic=args.synthetic, seed=args.seed)
    game = TradingGameSystem(config, player_classes=DEFAULT_PLAYER_CLASSES[:n])
    outcome = game.run_complete_game()
    print(f"\nWinner: {outcome['winner']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
