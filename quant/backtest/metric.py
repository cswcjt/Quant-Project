import numpy as np
import pandas as pd
from typing import *
from itertools import groupby, chain

class Metric:
    def __init__(self, portfolio: Union[pd.DataFrame, pd.Series],
                 freq: str='month'):
        """Metric class

        Args:
            portfolio (Union[pd.DataFrame, pd.Series]): 포트폴리오 자산별 가격 혹은 총자산 가격표.
            freq (str, optional): 데이터 수집 주기. Defaults to 'day'.
        """
        if isinstance(portfolio, pd.DataFrame):
            self.portfolio = portfolio.sum(axis=1)
        elif isinstance(portfolio, pd.Series):
            self.portfolio = portfolio
        else:
            raise TypeError()
        
        self.freq2day = self.convert_to_day(freq)
        self.rets = self.portfolio.pct_change().fillna(0)
        self.cum_rets = (1 + self.rets).cumprod()
        
        if isinstance(freq, str):
            self.param = self.annualize_scaler(freq)
        else:
            raise TypeError()
    
    def annualize_scaler(param: str) -> int:
        annualize_scale_dict = {
            'day': 252,
            'week': 52,
            'month': 12,
            'quarter': 4,
            'half-year': 2,
            'year': 1
        }
        try:
            scale: int = annualize_scale_dict[param]
        except:
            raise Exception("freq is only ['day', 'week', 'month', \
                'quarter', 'half-year', 'year']")
        
        return scale
    
    def convert_to_day(self, freq: str) -> int:
        convert_to_days = {
            'day': 1,
            'week': 7,
            'month': 30,
            'quarter': 90,
            'half-year': 180,
            'year': 365
        }
        return convert_to_days[freq]

    def calc_lookback(lookback, scale) -> int:
        if isinstance(lookback, int):
            return lookback * scale
        elif isinstance(lookback, float):
            return int(lookback * scale)

    def sharp_ratio(self, returns: pd.Series=None, 
                    yearly_rfr: float=0.03, rolling: bool=False,
                    lookback: int=1) -> Union[pd.Series, float]:
        '''Sharp ratio method

        Args:
            price (Union[pd.DataFrame, pd.Series]):
                - DataFrame -> 포트폴리오 테이블
                - Series -> 개별종목 시계열 데이터
            freq (str, optional):
                포트폴리오 시간 간격 -> ['day', 'week', 'month', 'quarter', 'half-year', 'year'] 중 택1. 
                Defaults to 'day'.
            yearly_rfr (float, optional): 무위험자산 수익률(예금 이자). Defaults to 0.04.
            rolling (bool, optional):
                False - 전체 연율화 샤프지수
                Ture - (lookback)년 롤링 연율화 샤프지수
                Defaults to False.
            lookback (int, optional): 수익률 롤링 윈도우(단위: 년). Defaults to 1.
        
        Returns:
            Union[pd.Series, float]:
                - Series -> (lookback)년 롤링 연율화 샤프지수
                - float -> 연율화 샤프지수
        '''
        if rets is None:
            rets = self.rets.copy()
        else:
            rets = returns.copy()
        
        lookback = self.calc_lookback(lookback, self.param)
        
        if rolling:
            rets = rets.rolling(lookback)
            
        sharp = (rets.mean() * self.param - yearly_rfr)  / (rets.std() * np.sqrt(self.param))
        sharp = sharp.fillna(0) if rolling else sharp
        
        return sharp

    def sortino_ratio(self, returns: pd.Series=None,
                      yearly_rfr: float=0.03, rolling: bool=False,
                      lookback: Union[float, int]=1) -> Union[pd.Series, float]:
        """Sortino ratio calculation method

        Args:
            price (Union[pd.DataFrame, pd.Series]):
                - DataFrame -> 포트폴리오 테이블
                - Series -> 개별종목 시계열 데이터
            freq (str, optional):
                포트폴리오 시간 간격 -> ['day', 'week', 'month', 'quarter', 'half-year', 'year'] 중 택1. 
                Defaults to 'day'.
            yearly_rfr (float, optional): 무위험자산 수익률(예금 이자). Defaults to 0.04.
            rolling (bool, optional):
                False - 전체 연율화 소르티노 지수
                Ture - (lookback)년 롤링 연율화 소르티노 지수
                Defaults to False.
            lookback (int, optional): 수익률 롤링 윈도우(단위: 년). Defaults to 1.

        Returns:
            Union[pd.Series, float]:
                - Series -> (lookback)년 롤링 연율화 소르티노 지수
                - float -> 연율화 소르티노 지수
        """
        def downside_deviation(rets):
            rets_copy = rets.copy()
            rets_copy[rets_copy >= 0] = 0
            return rets_copy.std()
        
        if rets is None:
            rets = self.rets.copy()
        else:
            rets = returns.copy()
        
        lookback: int = self.calc_lookback(lookback, self.param)
        
        if rolling:
            rets = rets.rolling(lookback)
        
        dev = rets.apply(downside_deviation) * np.sqrt(self.param)
        
        sortino = (rets.mean() * self.param - yearly_rfr) / dev
        sortino = sortino.fillna(0) if rolling else sortino
        return sortino

    def calmar_ratio(self, returns: pd.Series=None, rolling: bool=False, 
                     lookback: Union[float, int]=1, MDD_lookback: Union[float, int]=3) -> Union[pd.Series, float]:
        '''Calmar ratio calculation method
        
        Args:
            price (Union[pd.DataFrame, pd.Series]):
                - DataFrame -> 포트폴리오 테이블
                - Series -> 개별종목 시계열 데이터
            freq (str, optional):
                포트폴리오 시간 간격 -> ['day', 'week', 'month', 'quarter', 'half-year', 'year'] 중 택1. 
                Defaults to 'day'.
            rolling (bool, optional):
                False - 전체 연율화 칼머 지수
                Ture - (lookback)년 롤링 연율화 칼머 지수
                Defaults to False.
            lookback (Union[float, int], optional): 수익률 롤링 윈도우(단위: 년). Defaults to 1.
            MDD_lookback (Union[float, int], optional): MDD 롤링 윈도우(단위: 년). Defaults to 3.

        Returns:
            Union[pd.Series, float]:
                - Series -> (lookback)년 롤링 연율화 칼머 지수
                - float -> 연율화 칼머 지수
        '''
        dd = self.price / self.price.cummax() - 1
        if rets is None:
            rets = self.rets.copy()
        else:
            rets = returns.copy()
        
        lookback: int = self.calc_lookback(lookback, self.param)
        MDD_lookback = self.calc_lookback(MDD_lookback, self.param)
        
        if rolling:
            rets = rets.rolling(lookback)
            dd = dd.rolling(MDD_lookback)
        
        calmar = - rets.mean() * self.param / dd.min()
        calmar = calmar.fillna(0) if rolling else calmar
        return calmar

    def VaR(self, returns: pd.Series=None, delta: float=0.01):
        if returns is None:
            rets = self.rets.copy()
        else:
            rets = returns.copy()
        return rets.quantile(delta)
    
    def VaR_ratio(self, returns: pd.Series=None, rolling: bool=False,
                  lookback: int=1, delta: float=0.01) -> Union[pd.Series, float]:
        """VaR ratio calculation method

        Args:
            price (Union[pd.DataFrame, pd.Series]):
                - DataFrame -> 포트폴리오 테이블
                - Series -> 개별종목 시계열 데이터
            freq (str, optional):
                포트폴리오 시간 간격 -> ['day', 'week', 'month', 'quarter', 'half-year', 'year'] 중 택1. 
                Defaults to 'day'.
            rolling (bool, optional):
                False - 전체 연율화 VaR 지수
                Ture - (lookback)년 롤링 연율화 VaR 지수
                Defaults to False.
            lookback (Union[float, int], optional): 수익률 롤링 윈도우(단위: 년). Defaults to 1.
            delta (float, optional): 위험구간(z-value corresponding to %VaR). Defaults to 0.01.

        Returns:
            Union[pd.Series, float]:
                - Series -> (lookback)년 롤링 연율화 VaR 지수
                - float -> 연율화 VaR 지수
        """
        if returns is None:
            rets = self.rets.copy()
        else:
            rets = returns.copy()
        lookback: int = self.calc_lookback(lookback, self.param)
        
        if rolling:
            rets = self.rets.rolling(lookback)
        
        ratio = -rets.mean() / self.VaR(rets, delta)
        ratio = ratio.fillna(0) if rolling else ratio
        return ratio

    def CVaR(self, returns: pd.Series=None, delta=0.01):
        if returns is None:
            rets = self.rets
        else:
            rets = returns.copy()
        return rets[rets <= self.VaR(rets, delta)].mean()

    def CVar_Ratio(self, returns: pd.Series=None, 
                   rolling: bool=False, lookback: int=1, 
                   delta=0.01) -> Union[pd.Series, float]:
        """CVaR ratio calculation method

        Args:
            price (Union[pd.DataFrame, pd.Series]):
                - DataFrame -> 포트폴리오 테이블
                - Series -> 개별종목 시계열 데이터
            freq (str, optional):
                포트폴리오 시간 간격 -> ['day', 'week', 'month', 'quarter', 'half-year', 'year'] 중 택1. 
                Defaults to 'day'.
            rolling (bool, optional):
                False - 전체 연율화 CVaR 지수
                Ture - (lookback)년 롤링 연율화 CVaR 지수
                Defaults to False.
            lookback (Union[float, int], optional): 수익률 롤링 윈도우(단위: 년). Defaults to 1.
            delta (float, optional): 위험구간(z-value corresponding to %VaR). Defaults to 0.01.

        Returns:
            Union[pd.Series, float]:
                - Series -> (lookback)년 롤링 연율화 CVaR 지수
                - float -> 연율화 CVaR 지수
        """
        if returns is None:
            rets = self.rets
        else:
            rets = returns.copy()
        lookback: int = self.calc_lookback(lookback, self.param)
        
        if rolling:
            rets = rets.rolling(lookback)
            ratio = -rets.mean() / rets.apply(lambda x: self.CVaR(x, delta))
            ratio = ratio.fillna(0)
        else:
            ratio = -rets.mean() / self.CVaR(rets, delta)
            
        return ratio

    def hit_ratio(self, returns: pd.Series=None,
                  rolling: bool=False, lookback: int=1,
                  delta=0.01) -> Union[pd.Series, float]:
        """Hit ratio calculation method

        Args:
            price (Union[pd.DataFrame, pd.Series]):
                - DataFrame -> 포트폴리오 테이블
                - Series -> 개별종목 시계열 데이터
            freq (str, optional):
                포트폴리오 시간 간격 -> ['day', 'week', 'month', 'quarter', 'half-year', 'year'] 중 택1. 
                Defaults to 'day'.
            rolling (bool, optional):
                False - 전체 연율화 HR
                Ture - (lookback)년 롤링 연율화 HR
                Defaults to False.
            lookback (Union[float, int], optional): 수익률 롤링 윈도우(단위: 년). Defaults to 1.
            delta (float, optional): 위험구간(z-value corresponding to %VaR). Defaults to 0.01.

        Returns:
            Union[pd.Series, float]:
                - Series -> (lookback)년 롤링 연율화 HR
                - float -> 연율화 HR
        """
        hit = lambda rets: len(rets[rets > 0.0]) / len(rets[rets != 0.0])
        
        if returns is None:
            rets = self.rets
        else:
            rets = returns.copy()
        lookback: int = self.calc_lookback(lookback, self.param)
        
        if rolling:
            rets = rets.rolling(lookback)
            ratio = rets.apply(hit)
        else:
            ratio = hit(rets)
            
        return ratio

    def GtP_ratio(self, returns: pd.Series=None,
                  rolling: bool=False, lookback: int=1,
                  delta=0.01) -> Union[pd.Series, float]:
        """Gain-to-Pain ratio(GPR) calculation method

        Args:
            price (Union[pd.DataFrame, pd.Series]):
                - DataFrame -> 포트폴리오 테이블
                - Series -> 개별종목 시계열 데이터
            freq (str, optional):
                포트폴리오 시간 간격 -> ['day', 'week', 'month', 'quarter', 'half-year', 'year'] 중 택1. 
                Defaults to 'day'.
            rolling (bool, optional):
                False - 전체 연율화 GPR
                Ture - (lookback)년 롤링 연율화 GPR
                Defaults to False.
            lookback (Union[float, int], optional): 수익률 롤링 윈도우(단위: 년). Defaults to 1.
            delta (float, optional): 위험구간(z-value corresponding to %VaR). Defaults to 0.01.

        Returns:
            Union[pd.Series, float]:
                - Series -> (lookback)년 롤링 연율화 GPR
                - float -> 연율화 GPR
        """
        GPR = lambda rets: rets[rets > 0.0].mean() / -rets[rets < 0.0].mean()
        
        if returns is None:
            rets = self.rets
        else:
            rets = returns.copy()
        lookback: int = self.calc_lookback(lookback, self.param)
        
        if rolling:
            rets = rets.rolling(lookback)
            ratio = rets.apply(GPR)
        else:
            ratio = GPR(rets)
        return ratio
    
    def skewness(self):
        return self.rets.skew()
    
    def kurtosis(self):
        return self.rets.kurtosis()
    
    def drawdown(self, returns: pd.Series=None) -> pd.Series:
        if returns is None:
            cum_rets = self.cum_rets
        else:
            cum_rets = (1 + returns).cumprod()
            
        return cum_rets.div(cum_rets.cummax()).sub(1)
    
    def drawdown_duration(self, returns: pd.Series=None) -> pd.Series:
        if returns is None:
            rets = self.rets
        else:
            rets = returns.copy()
        
        dd = self.drawdown(rets)
        
        ddur_count = list(chain.from_iterable((np.arange(len(list(j))) + 1).tolist() if i==1 else [0] * len(list(j)) for i, j in groupby(dd != 0)))
        ddur_count = pd.Series(ddur_count, index=dd.index)
        temp_df= ddur_count.reset_index()
        temp_df.columns = ['date', 'counts']
        
        count_0 = temp_df.counts.apply(lambda x: 0 if x > 0 else 1)
        cumdays = temp_df.date.diff().dt.days.fillna(0).astype(int).cumsum()
        ddur = cumdays - (count_0 * cumdays).replace(0, np.nan).ffill().fillna(0).astype(int)
        ddur.index = dd.index
        return ddur
        
    def MDD(self, returns: pd.Series=None) -> float:
        if returns is None:
            cum_rets = self.cum_rets
        else:
            cum_rets = (1 + returns).cumprod()
        return cum_rets.div(cum_rets.cummax()).sub(1).min()
        
    def MDD_duration(self, returns: pd.Series=None) -> float:
        """MDD 지속기간 계산 메서드

        Args:
            timeseries (bool, optional): _description_. Defaults to False.

        Returns:
            - pd.Series: 모든 시간에 따른 MDD 지속기간 값
            - float: 총 MDD 지속기간 값
        """
        return self.drawdown_duration().max()
        
    def get_rets(self):
        return self.rets
    
    def get_cumulat_rets(self):
        return self.cum_rets