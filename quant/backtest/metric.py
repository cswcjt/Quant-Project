import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from itertools import groupby, chain
from typing import Union

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[2]
sys.path.append(str(PJT_PATH))

from scaling import convert_freq, annualize_scaler

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
        
        self.freq = convert_freq(freq)
        self.param = annualize_scaler(self.freq)
        self.freq2day = int(252 / self.param)
        
        self.rets = self.portfolio.pct_change().fillna(0)
        self.cum_rets = (1 + self.rets).cumprod()
    
    def calc_lookback(self, lookback, scale) -> int:
        # lookback을 주기에 맞게 변환해주는 함수
        if isinstance(lookback, int):
            return lookback * scale
        elif isinstance(lookback, float):
            return int(lookback * scale)
    
    def rolling(func):
        # 옵션 활용을 위한 데코레이터
        def wrapper(self, returns=None, rolling=False, lookback=1, *args, **kwargs):
            if returns is None:
                rets = self.rets.copy()
            else:
                try:
                    rets = returns.copy()
                except AttributeError:
                    rets = returns
            
            lookback = self.calc_lookback(lookback, self.param)
            
            if rolling:
                rets = rets.rolling(lookback)
            
            result = func(self, returns=rets, *args, **kwargs)
            return result
        return wrapper
    
    def external(func):
        # 옵션 활용을 위한 데코레이터
        def wrapper(self, returns=None, *args, **kwargs):
            if returns is None:
                rets = self.rets.copy()
            else:
                try:
                    rets = returns.copy()
                except AttributeError:
                    rets = returns
            
            result = func(self, returns=rets, *args, **kwargs)
            return result
        return wrapper
    
    @external
    def CAGR(self, returns: pd.Series=None) -> float:
        try:
            return returns.add(1).prod() ** (self.param / len(returns)) - 1
        except Exception:
            return returns.apply(lambda x: self.CAGR(x))
    
    @external
    def annualized_volatility(self, returns: pd.Series=None) -> float:
        return returns.std() * np.sqrt(self.param)
    
    @rolling
    def sharp_ratio(self, returns: pd.Series,
                    rolling: bool=False,
                    lookback: Union[float, int]=1,
                    yearly_rfr: float=0.04) -> Union[pd.Series, float]:
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
        return (self.CAGR(returns) - yearly_rfr) / self.annualized_volatility(returns)
    
    @rolling
    def sortino_ratio(self, returns: pd.Series=None,
                      rolling: bool=False,
                      lookback: Union[float, int]=1,
                      yearly_rfr: float=0.03) -> Union[pd.Series, float]:
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
        def downside_std(returns):
            try:
                returns[returns >= 0] = 0
                return returns.std() * np.sqrt(self.param)
            except TypeError:
                return returns.apply(lambda x: downside_std(x))
        
        return (self.CAGR(returns) - yearly_rfr) / downside_std(returns)

    @external
    def calmar_ratio(self, returns: pd.Series=None,
                     rolling: bool=False,
                     lookback: Union[float, int]=1,
                     MDD_lookback: Union[float, int]=3
                     ) -> Union[pd.Series, float]:
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
        dd = self.drawdown(returns)
        lookback = self.calc_lookback(lookback, self.param)
        MDD_lookback = self.calc_lookback(MDD_lookback, self.param)
        
        if rolling:
            returns = returns.rolling(lookback)
            dd = dd.rolling(MDD_lookback)
        
        calmar = - self.CAGR(returns) / dd.min()
        return calmar
    
    @external
    def VaR(self, returns: pd.Series=None, delta: float=0.01):
        return returns.quantile(delta)
    
    @rolling
    def VaR_ratio(self, returns: pd.Series=None, 
                  rolling: bool=False, lookback: int=1,
                  delta: float=0.01) -> Union[pd.Series, float]:
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
        ratio = -returns.mean() / self.VaR(returns, delta=delta)
        return ratio

    @external
    def CVaR(self, returns: pd.Series=None, delta=0.01):
        try:
            return returns[returns <= self.VaR(returns, delta=delta)].mean()
        except TypeError:
            return returns.apply(lambda x: self.CVaR(x))

    @rolling
    def CVaR_ratio(self, returns: pd.Series=None, 
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
        if rolling:
            ratio = -returns.mean() / returns.apply(lambda x: self.CVaR(x, delta=delta))
        else:
            ratio = -returns.mean() / self.CVaR(returns, delta=delta)
            
        return ratio

    @external
    def hit_ratio(self, returns: pd.Series=None,
                  rolling: bool=False, lookback: int=1
                  ) -> Union[pd.Series, float]:
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
        lookback = self.calc_lookback(lookback, self.param)
        return returns.rolling(lookback).apply(hit) if rolling else hit(returns)

    @external
    def GtP_ratio(self, returns: pd.Series=None,
                  rolling: bool=False, lookback: int=1
                  ) -> Union[pd.Series, float]:
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
        lookback = self.calc_lookback(lookback, self.param)
        return returns.rolling(lookback).apply(GPR) if rolling else GPR(returns)
    
    @external
    def skewness(self, returns: pd.Series=None) -> float:
        # skewness 계산 메서드
        return self.rets.skew()
    
    @external
    def kurtosis(self, returns: pd.Series=None) -> float:
        # kurtosis 계산 메서드
        return self.rets.kurtosis()
    
    @external
    def drawdown(self, returns: pd.Series=None) -> pd.Series:
        """기간내 최고점 대비 수익하락율(drawdown) 계산 메서드

        Args:
            returns (pd.Series, optional): 시간에 따른 수익률 리스트(Series). Defaults to None.

        Returns:
            - pd.Series: 주기에 따른 drawdown 리스트(Series)
        """
        try:
            cum_rets = (1 + returns).cumprod()
            return cum_rets.div(cum_rets.cummax()).sub(1)
        except TypeError:
            return returns.apply(self.drawdown)
    
    @external
    def drawdown_duration(self, returns: pd.Series=None) -> pd.Series:
        
        """drawdown 지속기간 계산 메서드(일 단위)

        Args:
            returns (pd.Series, optional): 시간에 따른 수익률 리스트(Series). Defaults to None.

        Returns:
            - pd.Series: 주기에 따른 drawdown 지속시간 리스트(Series, 일 단위)
        """
        dd = self.drawdown(returns=returns)
        
        ddur_count = list(chain.from_iterable((np.arange(len(list(j))) + 1).tolist() if i==1 else [0] * len(list(j)) for i, j in groupby(dd != 0)))
        ddur_count = pd.Series(ddur_count, index=dd.index)
        temp_df= ddur_count.reset_index()
        temp_df.columns = ['date', 'counts']
        
        count_0 = temp_df.counts.apply(lambda x: 0 if x > 0 else 1)
        cumdays = temp_df.date.diff().dt.days.fillna(0).astype(int).cumsum()
        ddur = cumdays - (count_0 * cumdays).replace(0, np.nan).ffill().fillna(0).astype(int)
        ddur.index = dd.index
        return ddur
    
    @external
    def MDD(self, returns: pd.Series=None) -> float:
        # MDD 계산 메서드
        return self.drawdown(returns).min()
    
    @external
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
    
    def total_returns(self, returns: pd.Series=None) -> float:
        return (1 + returns).prod()
    
    @external
    def print_report(self, returns: pd.Series=None, delta: float=0.01):
        print(f'Total Returns: {100 * self.total_returns(returns):.2f}%')
        print(f'CAGR: {100 * self.CAGR(returns):.2f}%')
        print(f'Annualized Volatility: {100 * self.annualized_volatility(returns):.2f}%')
        print(f'Skewness: {self.skewness(returns):.2f}')
        print(f'Kurtosis: {self.kurtosis(returns):.2f}')
        print(f'Max Drawdown: {self.MDD(returns):.2%}')
        print(f'Max Drawdown Duration: {self.MDD_duration(returns):.2f} days')
        print(f'Annualized Sharp Ratio: {self.sharp_ratio(returns):.2f}')
        print(f'Annualized Sortino Ratio: {self.sortino_ratio(returns):.2f}')
        print(f'Annualized Calmar Ratio: {self.calmar_ratio(returns):.2f}')
        print(f'Annualized VaR: {self.VaR(returns, delta=delta):.2f}')
        print(f'Annualized VaR Ratio: {self.VaR_ratio(returns, delta=delta):.2f}')
        print(f'Annualized CVaR: {self.CVaR(returns, delta=delta):.2f}')
        print(f'Annualized CVaR Ratio: {self.CVaR_ratio(returns, delta=delta):.2f}')
        print(f'Annualized hit Ratio: {self.hit_ratio(returns):.2f}')
        print(f'Annualized GtP Ratio: {self.GtP_ratio(returns):.2f}')
    
    @external        
    def numeric_metric(self, returns: pd.Series=None,
                       delta: float=0.01, dict: bool=True) -> Union[dict, pd.Series]:
        result = {
            'returns': f'{100 * self.total_returns(returns):.2f}',
            'CAGR': f'{100 * self.CAGR(returns):.2f}',
            'volatility': f'{self.annualized_volatility(returns):.2f}',
            'skewness': f'{self.skewness(returns):.2f}',
            'kurtosis': f'{self.kurtosis(returns):.2f}',
            'MDD': f'{self.MDD(returns):.2f}',
            'MDD_duration': f'{self.MDD_duration(returns):.2f}',
            'sharp': f'{self.sharp_ratio(returns):.2f}',
            'sortino': f'{self.sortino_ratio(returns):.2f}',
            'calmar': f'{self.calmar_ratio(returns):.2f}',
            'VaR': f'{self.VaR(returns, delta=delta):.2f}',
            'VaR_ratio': f'{self.VaR_ratio(returns, delta=delta):.2f}',
            'CVaR': f'{self.CVaR(returns, delta=delta):.2f}',
            'CVaR_ratio': f'{self.CVaR_ratio(returns, delta=delta):.2f}',
            'hit': f'{self.hit_ratio(returns):.2f}',
            'GtP': f'{self.GtP_ratio(returns):.2f}'
        }
        return result if dict else pd.Series(result)
    
    def rolling_metric(self, returns: pd.Series=None,
                       lookback: Union[float, int]=1,
                       MDD_lookback: Union[float, int]=3,
                       delta: float=0.01) -> pd.DataFrame:
        rolling = True
        
        dd = self.drawdown(returns)
        ddur = self.drawdown_duration(returns)
        sharp = self.sharp_ratio(returns, rolling=rolling, lookback=lookback)
        sortino = self.sortino_ratio(returns, rolling=rolling, lookback=lookback)
        calmar = self.calmar_ratio(returns, rolling=rolling, 
                                   lookback=lookback, MDD_lookback=MDD_lookback)
        VaR = self.VaR(returns, delta=delta)
        VaR_ratio = self.VaR_ratio(returns, rolling=rolling,
                                   lookback=lookback, delta=delta)
        CVaR = self.CVaR(returns, delta=delta)
        CVaR_ratio = self.CVaR_ratio(returns, rolling=rolling,
                                     lookback=lookback, delta=delta)
        hit = self.hit_ratio(returns, rolling=rolling, lookback=lookback)
        GtP = self.GtP_ratio(returns, rolling=rolling, lookback=lookback)
        
        result = pd.concat([dd, ddur, sharp, sortino, calmar,
                            VaR_ratio, CVaR_ratio, hit, GtP], axis=1)
        result.columns = ['dd', 'ddur', 'sharp', 'sortino', 'calmar',
                          'VaR_ratio', 'CVaR_ratio', 'hit', 'GtP']
        return result
    
    def plot_report(self, returns: pd.Series=None,
                    lookback: Union[float, int]=1,
                    MDD_lookback: Union[float, int]=3,
                    delta: float=0.01) -> None:
        
        report = self.rolling_metric(returns=returns, lookback=lookback,
                                    MDD_lookback=MDD_lookback, delta=delta)
        report.reset_index(inplace=True, names=['Date'])
        
        _, ax = plt.subplots(4, 2, figsize=(8, 16))
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        
        sns.lineplot(data=report, x='Date', y='dd', ax=ax[0][0])
        ax[0][0].set_title('Drawdown')
        sns.lineplot(data=report, x='Date', y='ddur', ax=ax[0][1])
        ax[0][1].set_title('Drawdown Duration')
        sns.lineplot(data=report, x='Date', y='sharp', ax=ax[1][0])
        ax[1][0].set_title(f'{lookback}-year Sharp Ratio')
        sns.lineplot(data=report, x='Date', y='calmar', ax=ax[1][1])
        ax[1][1].set_title(f'{lookback}-year Calmar Ratio')
        sns.lineplot(data=report, x='Date', y='VaR_ratio', ax=ax[2][0])
        ax[2][0].set_title(f'{lookback}-year VaR Ratio')
        sns.lineplot(data=report, x='Date', y='CVaR_ratio', ax=ax[2][1])
        ax[2][1].set_title(f'{lookback}-year CVaR Ratio')
        sns.lineplot(data=report, x='Date', y='hit', ax=ax[3][0])
        ax[3][0].set_title(f'{lookback}-year hit Ratio')
        sns.lineplot(data=report, x='Date', y='GtP', ax=ax[3][1])
        ax[3][1].set_title(f'{lookback}-year GtP Ratio')
        plt.show()

import yfinance as yf

if __name__ == '__main__':
    data = yf.download('SPY TLT', start='2002-07-30')['Adj Close']
    test = Metric(data)
    test.print_report()
    test.rolling_metric()
