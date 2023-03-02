import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *

from django.conf import settings

from backtest.services import Metric
from price.services import *

import yfinance as yf
from fredapi import Fred

def business_cycle(starting_date: str='1992-12-01', recession: bool=False) -> pd.DataFrame:        
    fred = Fred(api_key=settings.FRED_API_KEY)
    regime_list = []
    
    us_cli = fred.get_series('USALOLITONOSTSAM', observation_start=starting_date).dropna()
    us_cli = pd.concat([us_cli.iloc[1:,], us_cli.pct_change().dropna()], axis=1)
    
    core_pce = fred.get_series('PCEPILFE', observation_start=starting_date)
    core_pce = core_pce.pct_change(12).dropna()
    
    business_cycle = pd.concat([us_cli, core_pce], axis=1)
    business_cycle.columns = ['cli', 'cli_change', 'pce']
    
    business_cycle = business_cycle.shift()
    business_cycle = business_cycle.dropna()
    
    deflation = ((business_cycle['cli'] < 100) & (business_cycle['cli_change'] < 0) & (business_cycle['pce'] < 0.025)) * 1
    inflation = ((business_cycle['cli'] < 100) & (business_cycle['cli_change'] < 0) & (business_cycle['pce'] >= 0.025)) * 1
    recovery = ((business_cycle['cli'] < 100) & (business_cycle['cli_change'] >= 0)) * 1
    expansion = ((business_cycle['cli'] >= 100) & (business_cycle['cli_change'] >= 0)) * 1
    
    regime_df = pd.concat([deflation, inflation, recovery, expansion], axis=1).dropna()
    regime_df.columns = ['deflation', 'inflation', 'recovery', 'expansion']
    
    if recession:
        sahm_rule = fred.get_series('SAHMREALTIME')
        sahm_rule = pd.DataFrame(sahm_rule)
        sahm_rule = sahm_rule.rename(columns={0:'sahm'})

        sahm_rule['3m'] = sahm_rule.sahm.rolling(window=3).mean()
        sahm_rule['min'] = sahm_rule.sahm.rolling(window=12).min()
        sahm_rule['recession'] = (sahm_rule['3m'] - sahm_rule['min'])
        sahm_rule['recession'] = (sahm_rule['recession'] >= 0.5) * 1
        
        regime_df = pd.concat([regime_df, sahm_rule['recession']], axis=1)
        
    return regime_df
        
def asset_indicators(asset_tickers: list=['SPY','TLT', 'GSG', 'VNQ', 'UUP']):
    
    asset_list = []
    for ticker in asset_tickers:
        asset_price = yf.download(ticker)['Adj Close']
        asset_price.index = pd.to_datetime(pd.to_datetime(asset_price.index).date)
        asset_list.append(asset_price)
        asset_df = pd.concat(asset_list, axis=1)
    
    asset_df.columns = asset_tickers
    
    return asset_df

########################################################################################
def factor_with_regime(regime_df: pd.DataFrame, factor_monthly_rets: pd.Series, factor_rets: str) -> pd.DataFrame:
    """_summary_

    Args:
        regime_df (pd.DataFrame): 
            - 각 월별로 1개의 regime만 1로 표시한 데이터프레임
        factor_monthly_rets (pd.Series):
            - 각 월별로 factor return을 가진 시리즈
        factor_name (str): 
            - factor의 이름 + rets: ex) 'mom_rets'

    Returns:
        pd.DataFrame: 
    """
    
    fr_with_rg = pd.concat([factor_monthly_rets, regime_df], axis=1)
    fr_with_rg.columns = regime_df.columns.insert(0, factor_rets)
    
    return fr_with_rg

def check_factor_with_regime(factor_with_regime: pd.DataFrame, factor_rets: str, freq: str, plot: bool=False):
    """_summary_

    Args:
        factor_with_regime (pd.DataFrame): 
            - factor_with_regime()의 아웃풋 df
        factor_rets (str): 
            - factor_with_regime()의 변수 중 factor_rets와 동일한 스트링
        freq (str):     
            - Metric 클래스의 변수로 사용되는 freq: 'day', 'month', 'year', etc.
        plot (bool, optional): 
            - plt 아니면 Metric 중에 하나만 선택. Defaults to False.
    """
    
    each_regime = factor_with_regime.columns[factor_with_regime.columns != f'{factor_rets}']
    
    for regime in each_regime:

        if plot:
            print((1 + factor_with_regime[factor_rets].mul(factor_with_regime[regime], axis=0)).cumprod().plot(title=f'{factor_rets} with {regime}', lw=1))
            plt.show()
        
        else:
            print(f'{factor_rets} with {regime}')
            Metric(portfolio=(1 + factor_with_regime[factor_rets]).cumprod(), freq=freq).print_report()
            print()
            
            
def multi_asset_df(asset_df: pd.DataFrame, factor_with_regime_df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        asset_df (pd.DataFrame): 
            - 벤치마크와 대체자산을 포함한 데이터프레임
        factor_with_regime_df (pd.DataFrame): 
            - factor_with_regime()의 아웃풋 데이터프레임
            
    Returns:
        pd.DataFrame: 
    """
    
    start_date = factor_with_regime_df.iloc[0].name - DateOffset(months = 2)
    asset_df = asset_df.loc[start_date:,:].dropna().resample('M').last()

    multi_asset_df = pd.merge(asset_df.pct_change(), factor_with_regime_df, left_index=True, right_index=True, how='left').dropna()
    
    return multi_asset_df

def check_best_regime(multi_asset_df, freq: str, _plot: bool=False):
    each_regime = multi_asset_df.columns[-4:]
    tickers = multi_asset_df.columns[:-4]
    
    for regime in each_regime:
        if _plot:
            print((1+multi_asset_df[tickers].mul(multi_asset_df[regime], axis=0)).cumprod().plot(lw=1))

        else:
            for ticker in tickers:
                print(f'{ticker} with {regime}')
                Metric(portfolio=(1 + multi_asset_df[ticker] * multi_asset_df[regime]).cumprod(), freq=freq).print_report()
                print()
                
def invest_asset_df(asset_df: pd.DataFrame, factor_cum_ret: pd.Series, alternate_asset_list: list) -> pd.DataFrame:
    """_summary_
    
    Args:
        asset_df (pd.DataFrame): 
            - 벤치마크와 대체자산을 포함한 데이터프레임
        factor_cum_ret (pd.Series): 
            - '누적수익률.pct_change = 일별 수익률'이기 때문에 누적수익률을 사용
        alternate_asset_list (list):
            - 대체자산 리스트
            
    Returns:
        pd.DataFrame: 팩터와 현금을 포함한 투자자산 데이터프레임
    """
    start_date = factor_cum_ret.index[0]
    end_date = factor_cum_ret.index[-1]
    
    indexes_price = asset_df[alternate_asset_list].loc[start_date:end_date,]
    indexes_price.index = pd.to_datetime(indexes_price.index)
    indexes_price.index.name = 'date_time'
    indexes_price['factor_rets'] = factor_cum_ret
    invest_price = add_cash(indexes_price, 252, 0.04)
    
    return invest_price

def regime_signal(ma_regime_df: pd.DataFrame, regime_asset_dict: dict) -> pd.DataFrame:
    """_summary_
    ma_regime_df (pd.DataFrame): 
        - invest_asset_df()의 아웃풋 데이터프레임과 시황을 결합한 데이터프레임
    regime_asset_dict (dict):
        - 각 시황을 key, 투자할 자산을 value로 하는 딕셔너리
        
    Returns:
        pd.DataFrame: 시황을 반영해 각 월에 어떤 자산에 투자할지 결정한 시그널 데이터프레임
    """
    
    for key, value in regime_asset_dict.items():
        ma_regime_df.loc[ma_regime_df[key] == 1, value] =1
        
    regime_signal = (ma_regime_df == 1) * 1 
    regime_signal.drop(regime_asset_dict.keys(), axis=1, inplace=True)
    regime_signal['CASH'] = (regime_signal.sum(axis=1) == 0).astype(int)
    
    return regime_signal