# 패키지 임포트
import numpy as np
import pandas as pd

from scipy.stats import norm

class TimeSeries:
    """
    종적 자산배분: 
    횡적 자산배분으로 투자자산 간 투자비중을 산출은 위험자산에 투자할 비중을 구하는 것이다.
    이후 최종적으로 위험자산과 안전자산(현금)의 비중으로 구하는 것이 종적 자산배분이다. 
    """
    def __init__(self, port_rets: pd.Series, param: int, vol_target: int=0.1, delta: int=0.01, cvar_target: int=0.05, m: int=3, floor: int=0.7):
        """_summary_

        Args:
            port_rets (pd.Series): 리밸런싱 날의 포트폴리오 수익률 정보를 담고 있는 pd series 
            param (int): 연율화를 위한 상수값 설정
            vol_target (int, optional): 
                vt를 위한 변수. 포트폴리오의 타켓 변동성 설정값. Defaults to 0.1.
            delta (int, optional): 
                cvt를 위한 변수. 변동성의 quantile이라고 보면 된다. Defaults to 0.01.
            cvar_target (int, optional): 
                cvt를 위한 변수. 포트폴리오의 타겟 cvar 설정값. Defaults to 0.05.
            m (int, optional): 
                cppi를 위한 변수. 레버리지 승수 설정값.Defaults to 3.
            floor (int, optional): 
                cppi를 위한 변수. floor = 하한선 비율(e.g. init_val 이 1이고 floor이 0.7이면 하한선은 30%). Defaults to 0.7.
        """
        self.port_rets = port_rets
        self.param = param
        
        self.vol_target = vol_target
        
        self.delta = delta
        self.cvar_target = cvar_target
        
        self.m = m
        self.floor = floor

    def vt(self):
        vol = self.port_rets.rolling(self.param).std().fillna(0) * np.sqrt(self.param)
        weights = (self.vol_target / vol).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
        weights[weights > 1] = 1
        return weights

    def cvt(self):
        def calculate_CVaR(rets: int, delta: int):
            """_summary_

            Args:
                rets (int):  pd.Series인 self.port_rets를 row별로 슬라이싱한 int 값
                delta (int): apply 함수 사용할 때 들어가는 변수로 rets.quantile(delta) 계산시 사용 

            Returns:
                _type_: 
            """
            VaR = rets.quantile(delta)
            return rets[rets <= VaR].mean()

        rolling_CVaR = - \
            self.port_rets.rolling(self.param).apply(
                calculate_CVaR, args=(self.delta,)).fillna(0)
        weights = (
            self.cvar_target / rolling_CVaR).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
        weights[weights > 1] = 1
        return weights

    # KL
    def kl(self):
        sharpe_ratio = (self.port_rets.rolling(self.param).mean() *
                        np.sqrt(self.param) / self.port_rets.rolling(self.param).std())
        weights = pd.Series(2 * norm.cdf(sharpe_ratio) - 1,
                            index=self.port_rets.index).fillna(0)
        weights[weights < 0] = 0
        weights = weights.shift(1).fillna(0)
        return weights

    # CPPI
    def cppi(self, init_val=1):
        n_steps = len(self.port_rets)
        port_value = init_val
        floor_value = init_val * self.floor
        peak = init_val
        """
        init_val = 스타트 시점
        m = 래버리지 승수
        floor = 하한선 비율(e.g. init_val 이 1이고 floor이 0.7이면 하한선은 30%)
        """
        port_history = pd.Series(dtype=np.float64).reindex_like(self.port_rets)
        weight_history = pd.Series(dtype=np.float64).reindex_like(self.port_rets)
        floor_history = pd.Series(dtype=np.float64).reindex_like(self.port_rets)

        for step in range(n_steps):
            peak = np.maximum(peak, port_value)
            floor_value = peak * self.floor

            cushion = (port_value - floor_value) / port_value
            weight = self.m * cushion

            risky_alloc = port_value * weight
            safe_alloc = port_value * (1 - weight)
            port_value = risky_alloc * (1 + self.port_rets.iloc[step]) + safe_alloc

            port_history.iloc[step] = port_value
            weight_history.iloc[step] = weight
            floor_history.iloc[step] = floor_value

        return weight_history.shift(1).fillna(0)