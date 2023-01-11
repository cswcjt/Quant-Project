import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# 종적 배분 모델 클래스
class TimeSeries:
    def __init__(self, port_rets, param):
        self.port_rets = port_rets
        self.param = param

    # VT   
    def vt(self, vol_target=0.1):
        vol = self.port_rets.rolling(self.param).std().fillna(0) * np.sqrt(self.param)
        weights = (vol_target / vol).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
        weights[weights > 1] = 1
        return weights
    
    # CVT
    def cvt(self, delta=0.01, cvar_target=0.05):
        def calculate_CVaR(rets, delta=0.01):
            VaR = rets.quantile(delta)    
            return rets[rets <= VaR].mean()
        
        rolling_CVaR = -self.port_rets.rolling(self.param).apply(calculate_CVaR, args=(delta,)).fillna(0)
        weights = (cvar_target / rolling_CVaR).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
        weights[weights > 1] = 1
        return weights
    
    # KL
    def kl(self):
        sharpe_ratio = (self.port_rets.rolling(self.param).mean() * np.sqrt(self.param) / self.port_rets.rolling(self.param).std())
        weights = pd.Series(2 * norm.cdf(sharpe_ratio) - 1, index=port_rets.index).fillna(0)
        weights[weights < 0] = 0
        weights = weights.shift(1).fillna(0)
        return weights
    
    # CPPI
    def cppi(self, m=3, floor=0.7, init_val=1):
        n_steps = len(self.port_rets)
        port_value = init_val
        floor_value = init_val * floor
        peak = init_val

        port_history = pd.Series(dtype=np.float64).reindex_like(self.port_rets)
        weight_history = pd.Series(dtype=np.float64).reindex_like(self.port_rets)
        floor_history = pd.Series(dtype=np.float64).reindex_like(self.port_rets)

        for step in range(n_steps):
            peak = np.maximum(peak, port_value)
            floor_value = peak * floor

            cushion = (port_value - floor_value) / port_value
            weight = m * cushion

            risky_alloc = port_value * weight
            safe_alloc = port_value * (1 - weight)
            port_value = risky_alloc * (1 + self.port_rets.iloc[step]) + safe_alloc

            port_history.iloc[step] = port_value
            weight_history.iloc[step] = weight
            floor_history.iloc[step] = floor_value

        return weight_history.shift(1).fillna(0)