## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[2]
sys.path.append(str(PJT_PATH))

from quant.backtest.metric import Metric
import numpy as np
import pandas as pd

class Volatility:
    def __init__(self, rebal_price: pd.DataFrame,
                 freq: str, n_sel: int,
                 long_only: bool=True) -> None:
        pass