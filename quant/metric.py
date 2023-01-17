import numpy as np
import pandas as pd
from typing import *

def annualize_scaler(freq: str) -> int:
    """_summary_

    Args:
        freq (str): _description_

    Raises:
        Exception: _description_

    Returns:
        int: _description_
    """
    annualize_scale_dict = {
        'day': 252,
        'week': 52,
        'month': 12,
        'quarter': 4,
        'half-year': 2,
        'year': 1
    }
    try:
        scale: int = annualize_scale_dict[freq]
    except:
        raise Exception("freq is only ['day', 'week', 'month', 'quarter']")
    
    return scale

def calc_lookback(lookback, scale) -> int:
    if isinstance(lookback, int):
        return lookback * scale
    elif isinstance(lookback, float):
        return int(lookback * scale)

def sharp_ratio(price: Union[pd.DataFrame, pd.Series], freq: str='day',
                yearly_rfr: float=0.04, rolling: bool=False,
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
    scale = annualize_scaler(freq)
    lookback = calc_lookback(lookback, scale)
    
    if isinstance(price, pd.Series):
        rets = price.pct_change().fillna(0)
        if rolling:
            rets = rets.rolling(lookback)
            
        sharp = rets.mean() * np.sqrt(scale) / rets.std()
        
    elif isinstance(price, pd.DataFrame):
        rets = price.sum(axis=1).pct_change().fillna(0)
        if rolling:
            rets = rets.rolling(lookback)
            
        sharp = (rets.mean() * scale - yearly_rfr)  / (rets.std() * np.sqrt(scale))
    
    sharp = sharp.fillna(0) if rolling else sharp
        
    return sharp

def sortino_ratio(price: Union[pd.DataFrame, pd.Series], freq: str='day',
                  yearly_rfr: float=0.04, rolling: bool=False,
                  lookback: Union[float, int]=1) -> Union[pd.Series, float]:
    '''Sortino ratio calculation method

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
    '''
    def downside_deviation(rets):
        rets_copy = rets.copy()
        rets_copy[rets_copy >= 0] = 0
        return rets_copy.std()
    
    scale: int = annualize_scaler(freq)
    lookback = calc_lookback(lookback, scale)

        
    if isinstance(price, pd.Series):
        # 개별종목 리스트(시리즈)인 경우
        rets = price.pct_change().fillna(0)
        if rolling:
            rets = rets.rolling(lookback)
        
        dev = rets.apply(downside_deviation) * np.sqrt(scale)
        sortino = rets.mean() * scale / dev
    
    elif isinstance(price, pd.DataFrame):
        # 전체종목 포트폴리오 테이블(DataFrame)인 경우
        rets = price.sum(axis=1).pct_change().fillna(0)
        if rolling:
            rets = rets.rolling(lookback)
            
        dev = rets.apply(downside_deviation) * np.sqrt(scale)
        sortino = (rets.mean() * scale - yearly_rfr) / dev
        
    sortino = sortino.fillna(0) if rolling else sortino

    return sortino

def calmar_ratio(price: Union[pd.DataFrame, pd.Series], freq: str='day',
                 rolling: bool=False, lookback: Union[float, int]=1,
                 MDD_lookback: Union[float, int]=3) -> Union[pd.Series, float]:
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
    dd = price / price.cummax() - 1
    rets = price.pct_change().fillna(0)
    
    scale: int = annualize_scaler(freq)
    lookback = calc_lookback(lookback, scale)
    MDD_lookback = calc_lookback(MDD_lookback, scale)
        
    if rolling:
        rets = rets.rolling(lookback)
        dd = dd.rolling(MDD_lookback)
    
    calmar = - rets.mean() * scale / dd.min()
    calmar = calmar.fillna(0) if rolling else calmar
    return calmar

def VaR_ratio(price: Union[pd.DataFrame, pd.Series], freq: str='day',
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
    rets = price.pct_change().fillna(0)
    
    scale: int = annualize_scaler(freq)
    lookback = calc_lookback(lookback, scale)
    
    if rolling:
        rets = rets.rolling(lookback)
    
    VaR = rets.quantile(delta)
    ratio = -rets.mean() / VaR
    ratio = ratio.fillna(0) if rolling else ratio
    return ratio

def calculate_CVaR(rets, delta=0.01):
    VaR = rets.quantile(delta)
    return rets[rets <= VaR].mean()

def CVar_Ratio(price: Union[pd.DataFrame, pd.Series], freq: str='day',
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
    rets = price.pct_change().fillna(0)
    scale: int = annualize_scaler(freq)
    lookback = calc_lookback(lookback, scale)
    
    if rolling:
        rets = rets.rolling(lookback)
        ratio = -rets.mean() / rets.apply(lambda x: calculate_CVaR(x, delta))
        ratio = ratio.fillna(0)
    else:
        ratio = -rets.mean() / calculate_CVaR(rets)
        
    return ratio

def hit_ratio(price: Union[pd.DataFrame, pd.Series], freq: str='day',
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
    rets = price.pct_change().fillna(0)
    hit = lambda rets: len(rets[rets > 0.0]) / len(rets[rets != 0.0])
    scale: int = annualize_scaler(freq)
    lookback = calc_lookback(lookback, scale)
    
    if rolling:
        rets = rets.rolling(lookback)
        ratio = rets.apply(hit)
    else:
        ratio = hit(rets)
        
    return ratio

def GtP_ratio(price: Union[pd.DataFrame, pd.Series], freq: str='day',
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
    rets = price.pct_change().fillna(0)
    GPR = lambda rets: rets[rets > 0.0].mean() / -rets[rets < 0.0].mean()
    scale: int = annualize_scaler(freq)
    lookback = calc_lookback(lookback, scale)
    
    if rolling:
        rets = rets.rolling(lookback)
        ratio = rets.apply(GPR)
    else:
        ratio = GPR(rets)
    return ratio