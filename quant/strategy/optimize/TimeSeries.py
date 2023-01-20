# 패키지 임포트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from scipy.optimize import minimize
from scipy.stats import norm


class TimeSeries:
    # 종적 배분 클래스

    def __init__(self, 
                port_rets,
                param: int,
                cs_port_rets: pd.Series,
                cs_weights: pd.DataFrame,
                rebal_price: pd.DataFrame,
                ):

        self.port_rets = port_rets
        self.param = param
        self.cs_port_rets = cs_port_rets
        self.cs_weights = cs_weights
        self.rets = rebal_price.pct_change().dropna()

        # 거래비용
        self.cost = 0.0005

    # VT
    def vt(self, vol_target=0.1):
        vol = self.port_rets.rolling(self.param).std().fillna(0) * np.sqrt(self.param)
        weights = (
            vol_target / vol).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
        weights[weights > 1] = 1
        return weights

    # CVT
    def cvt(self, port_rets, param, delta=0.01, cvar_target=0.05):
        def calculate_CVaR(rets, delta=0.01):
            VaR = rets.quantile(delta)
            return rets[rets <= VaR].mean()

        rolling_CVaR = - \
            port_rets.rolling(param).apply(
                calculate_CVaR, args=(delta,)).fillna(0)
        weights = (
            cvar_target / rolling_CVaR).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
        weights[weights > 1] = 1
        return weights

    # KL
    def kl(self, port_rets, param):
        sharpe_ratio = (port_rets.rolling(param).mean() *
                        np.sqrt(param) / port_rets.rolling(param).std())
        weights = pd.Series(2 * norm.cdf(sharpe_ratio) - 1,
                            index=port_rets.index).fillna(0)
        weights[weights < 0] = 0
        weights = weights.shift(1).fillna(0)
        return weights

    # CPPI

    def cppi(self, port_rets, m=3, floor=0.7, init_val=1):
        n_steps = len(port_rets)
        port_value = init_val
        floor_value = init_val * floor
        peak = init_val
        """
        init_val = 스타트 시점
        m = 래버리지 승수
        floor = 하한선 비율 (e.g. init_val 이 1이고 floor이 0.7이면 하한선은 30%)
        """
        port_history = pd.Series(dtype=np.float64).reindex_like(port_rets)
        weight_history = pd.Series(dtype=np.float64).reindex_like(port_rets)
        floor_history = pd.Series(dtype=np.float64).reindex_like(port_rets)

        for step in range(n_steps):
            peak = np.maximum(peak, port_value)
            floor_value = peak * floor

            cushion = (port_value - floor_value) / port_value
            weight = m * cushion

            risky_alloc = port_value * weight
            safe_alloc = port_value * (1 - weight)
            port_value = risky_alloc * (1 + port_rets.iloc[step]) + safe_alloc

            port_history.iloc[step] = port_value
            weight_history.iloc[step] = weight
            floor_history.iloc[step] = floor_value

        return weight_history.shift(1).fillna(0)

    def transaction_cost(self, weights_df, rets_df, cost=0.0005):
        # 이전 기의 투자 가중치
        prev_weights_df = (weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:, :])) \
            .div((weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:, :])).sum(axis=1), axis=0)

        # 거래비용 데이터프레임
        cost_df = abs(weights_df - prev_weights_df) * cost
        cost_df.fillna(0, inplace=True)
        return cost_df

    # 백테스팅 실행 함수

    def run(self, ts_model, cost):
        # 빈 딕셔너리
        backtest_dict = {}

        # 일별 수익률 데이터프레임
        rets = self.rets

        # 종적 배분 모델 선택 및 실행
        if ts_model == 'VT':
            ts_weights = self.vt(self.cs_port_rets, self.param)
        elif ts_model == 'CVT':
            ts_weights = (self.cvt(self.cs_port_rets, self.param))
        elif ts_model == 'KL':
            ts_weights = (self.kl(self.cs_port_rets, self.param))
        elif ts_model == 'CPPI':
            ts_weights = (self.cppi(self.cs_port_rets))
        elif ts_model == None:
            ts_weights = 1

        # 최종 포트폴리오 투자 가중치
        port_weights = self.cs_weights.multiply(ts_weights, axis=0)

        # 거래비용 데이터프레임
        cost = self.transaction_cost(port_weights, rets)

        # 최종 포트폴리오 자산별 수익률
        port_asset_rets = port_weights.shift() * rets - cost

        # # 최종 포트폴리오 수익률
        port_rets = port_asset_rets.sum(axis=1)
        port_rets.index = pd.to_datetime(port_rets.index).strftime("%Y-%m-%d")

        return port_weights, port_asset_rets, port_rets