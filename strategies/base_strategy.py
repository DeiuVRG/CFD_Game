from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from engine.signal import Signal


class BaseStrategy(ABC):
    def __init__(self, name: str, timeframe: str):
        self.name = name
        self.timeframe = timeframe

    @abstractmethod
    def analyze(self, epic: str, df: pd.DataFrame) -> Optional[Signal]:
        pass

    @abstractmethod
    def get_required_candles(self) -> int:
        pass
