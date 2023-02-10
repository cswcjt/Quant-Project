import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs

from scipy.optimize import minimize
from scipy.stats import norm

from typing import *

import yfinance as yf

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[2]
sys.path.append(str(PJT_PATH))

from price.price_processing import get_price, add_cash, rebal_dates, price_on_rebal, calculate_portvals, port_cum_rets
from backtest.metric import Metric
from strategy.factors.momentum import MomentumFactor
from strategy.optimize.cross_sectional import Equalizer, Optimization
from strategy.optimize.time_series import TimeSeries

class BackTest:
    # 초기화 함수
    def __init__(self, start_date: str, end_date: str, universe: List[str], 
                 rebal_freq: str, rebal_method: str, 
                 factor: str, factor_params: Dict[str, Any], 
                 opt_method: str, opt_params: Dict[str, Any]):
        self.start_date = start_date
        self.end_date = end_date
        self.universe = universe
        self.rebal_freq = rebal_freq
        self.rebal_method = rebal_method
        self.factor = factor
        self.factor_params = factor_params
        self.opt_method = opt_method
        self.opt_params = opt_params

        self.factor_obj = self.get_factor()
        self.opt_obj = self.get_opt()
    
    def run(self):
        pass