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
    def __init__(self, port_rets: pd.Series, cs_weight: pd.DataFrame, param: int, call_method: int):
        """_summary_

        Args:
            port_rets (pd.Series): 리밸런싱 날의 포트폴리오 수익률 정보를 담고 있는 pd series 
            cs_weight (pd.DataFrame): 횡적 자산분배 이후 투자비중 df
            param (int): 연율화를 위한 상수값 설정
            call_method (str): 종적 자산분배 모델 이름
        
        """
        self.port_rets = port_rets
        self.cs_weight = cs_weight
        self.param = param
        self.call_method = call_method
        
    def ew(self, weight_target: int=0.7) -> pd.DataFrame:
        """_summary_

        Args:
            weight_target (int, optional): 위험자산에 투자할 비중. Defaults to 0.7.

        Returns:
            pd.DataFrame: 투자비중 df
        """
        
        weights = pd.DataFrame({'PORTFOLIO': weight_target}, index=self.port_rets.index)
        
        return weights

    def vt(self, vol_target: int=0.1) -> pd.DataFrame:
        """_summary_

        Args:
            vol_target (int, optional): 
                vt를 위한 변수. 포트폴리오의 타켓 변동성 설정값. Defaults to 0.1.

        Returns:
            pd.DataFrame: 투자비중 df
        """

        vol = self.port_rets.rolling(self.param).std().fillna(0) * np.sqrt(self.param)
        weights = (vol_target / vol).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
        weights[weights > 1] = 1
        return weights

    def cvt(self, delta: int=0.01, cvar_target: int=0.05) -> pd.DataFrame:
        """_summary_

        Args:
            delta (int, optional): 
                cvt를 위한 변수. 변동성의 quantile이라고 보면 된다. Defaults to 0.01.
            cvar_target (int, optional): 
                cvt를 위한 변수. 포트폴리오의 타겟 cvar 설정값. Defaults to 0.05.

        Returns:
            pd.DataFrame: 투자비중 df
        """
        
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

        rolling_CVaR = -self.port_rets.rolling(self.param).apply(calculate_CVaR, args=(delta, )).fillna(0)
        weights = (cvar_target / rolling_CVaR).replace([np.inf, -np.inf], 0).shift(1).fillna(0)
        
        # 레버리지 사용 불가
        weights[weights > 1] = 1
        
        return weights

    # KL
    def kl(self):
        sharpe_ratio = (self.port_rets.rolling(self.param).mean() *
                        np.sqrt(self.param) / self.port_rets.rolling(self.param, ).std())
        weights = pd.Series(2 * norm.cdf(sharpe_ratio) - 1,
                            index=self.port_rets.index).fillna(0)
        weights[weights < 0] = 0
        weights = weights.shift(1).fillna(0)
        return weights

    # CPPI
    def cppi(self, m: int=3, floor: int=0.7, init_val=1) -> pd.DataFrame:
        """_summary_

        Args:
            m (int, optional): 래버리지 승수. Defaults to 3.
            floor (int, optional): 하한선 비율(e.g. init_val 이 1이고 floor이 0.7이면 하한선은 30%). Defaults to 0.7.
            init_val (int, optional): 포트폴리오의 시작 가치를 1로 설정. Defaults to 1.

        Returns:
            pd.DataFrame: 투자비중 df
        """

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
    
    def run(self):
        method = self.call_method
        cs_weight = self.cs_weight
        
        if method == 'ew':
            ts_weight = self.ew()
            
        elif method == 'vt':
            ts_weight = self.vt()
            
        elif method == 'cvt': 
            ts_weight = self.cvt()
            
        elif method == 'kl':
            ts_weight = self.kl()
            
        elif method == 'cppi': 
            ts_weight = self.cppi()
            
        # ts weight만 확인
        ts_weight = pd.concat([ts_weight, 1-ts_weight], axis=1, join='inner')
        ts_weight.columns = ['PORTFOLIO', 'CASH']

        # cs weight에 ts weight 적용한 최종 투자비중 산출
        cs_ts_port_weight = cs_weight.multiply(ts_weight['PORTFOLIO'], axis=0)
        cs_ts_port_weight['CASH'] = ts_weight["CASH"]
        cs_ts_port_weight.dropna(inplace=True)
        
        return ts_weight, cs_ts_port_weight