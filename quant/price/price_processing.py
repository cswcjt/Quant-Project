# !pip install yfinance --quiet
import time
import yfinance as yf
import pandas as pd
import numpy as np
from functools import reduce

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[3]
sys.path.append(str(PJT_PATH))

from scaling import convert_freq, annualize_scaler

def get_price(tickers: list, period: str, interval: str, start_date: str=None) -> pd.DataFrame:
    """Download Price Data

    Args:
        tickers (list): 
            - list -> 원하는 종목의 티커 리스트
        period (str): 
            - str -> 데이터를 다운받을 기간을 설정. 기본값은 한달(1mo). (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): 
            - str -> 데이터의 주기를 설정. 주기를 일별보다 낮은 장중으로 설정할 경우 최대 60일간의 데이터만 제공
                    기본값은 하루(1d)입니다. (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        start_date (str, optional): 
            - str -> period 인자를 사용하지 않을 경우 데이터의 시작일을 설정합니다. 'YYYY-MM-DD' 문자열을 사용하거나 datetime 형식의 날짜를 사용
            - None -> 종목의 역사적 시작날짜를 기본값으로 사용 
        financial_infos (str):
            - str -> 재무정보 필요하면 yf의 history 메서드 사용
            - None -> 종가만 필요한 경우 yf의 download 메서드 사용

    Returns:
        pd.DataFrame -> 금융상품 가격정보를 담고있는 df
    """
    
    temp = []
    
    for ticker in tickers:
        #start = time.time()
        name = f'{ticker}'
        name = yf.Ticker(ticker)
        temp_df = name.history(start=start_date, period=period, interval=interval)['Close']
        temp.append(temp_df)
        #end = time.time()
        #print(f'loop time: {end - start}s')
            
    price_df = pd.concat(temp, axis = 1)
    price_df.columns = tickers

    if interval in ['1d', '5d', '1wk', '1mo', '3mo']:
        price_df.index = pd.to_datetime(price_df.index).date
    
    #price_df.dropna(inplace=True, axis=0)
    nan_cols = price_df.columns[price_df.isna().any()]
    #print(nan_cols)
    price_df.dropna(axis=0, inplace=True)

    return price_df

def add_cash(price: pd.DataFrame, num_day_in_year:int, yearly_rfr: int) -> pd.DataFrame:
    """Add Cash Column 

    Args:
        price (pd.DataFrame):
            - DataFrame -> 타겟 상품들의 종가 df
        num_day_in_year (int): 
            - int -> 1년 중 실제 비즈니스 일 수(임의의 상수값). 주로 252 사용
        yearly_rfr (int): 
            - int -> 무위험 수익률(임의의 상수값). 주로 0.04 사용

    Returns:
        pd.DataFrame -> 금융상품 가격정보에 예금을 추가한 df 
    """

    temp_df = price.copy()

    temp_df['CASH'] = yearly_rfr/num_day_in_year
    temp_df['CASH'] = (1 + temp_df['CASH']).cumprod()
    temp_df.dropna(inplace = True)
    temp_df.index.name = "date_time"

    return temp_df

def rebal_dates(price: pd.DataFrame, period: str,
                include_first_date: bool=False) -> list:
    """Select Rebalancing Period 

    Args:
        price (pd.DataFrame): 
            - DataFrame -> 타겟 상품들의 종가 df
        period (str): 
            - str -> 리밸런싱 주기 설정. (month, quarter, halfyear, year)

    Returns:
        list -> 리밸날짜를 담은 datetimeindex 
    """
    period = convert_freq(period)
    price.index = pd.to_datetime(price.index)
    
    first_date = price.index[0]
    last_date = price.index[-1]
    
    _price = price.reset_index()
    colname = _price.columns[0]
    
    if period == "month":
        groupby = [_price[colname].dt.year, _price[colname].dt.month]
        
    elif period == "quarter":
        groupby = [_price[colname].dt.year, _price[colname].dt.quarter]
        
    elif period == "halfyear":
        groupby = [_price[colname].dt.year, _price[colname].dt.month // 7]
        
    elif period == "year":
        groupby = [_price[colname].dt.year, _price[colname].dt.year]
    
    rebal_dates = pd.to_datetime(_price.groupby(groupby)[colname].last().values)
    
    if include_first_date:
        rebal_dates = rebal_dates.append(pd.to_datetime([first_date]))
        rebal_dates = rebal_dates.sort_values()
    
    if rebal_dates[-1] > last_date:
        return rebal_dates[:-1]
    else:
        return rebal_dates
    

def price_on_rebal(price: pd.DataFrame, rebal_dates: list) -> pd.DataFrame:
    """Prince Info on Rebalancing Date

    Args:
        price (pd.DataFrame):
            - DataFrame -> 타겟 상품들의 종가 df
        rebal_dates (list): 
            - list -> 리밸런싱 날짜 정보

    Returns:
        pd.DataFrame -> 리밸런싱 날짜의 타겟 상품들 종가 df
    """

    price_on_rebal = price.loc[rebal_dates, :]
    return price_on_rebal

def calculate_portvals(price_df: pd.DataFrame, weight_df: pd.DataFrame, signal_df: pd.DataFrame, long_only: str) -> pd.DataFrame:
    cum_rtn_up_until_now = 1 
    individual_port_val_df_list = []
    prev_end_day = weight_df.index[0]
    
    if long_only: 
        for end_day in weight_df.index[1:]:
            sub_price_df = price_df.loc[prev_end_day:end_day]
            sub_asset_flow_df = sub_price_df / sub_price_df.iloc[0]

            weight_series = weight_df.loc[prev_end_day]
            indi_port_cum_rtn_series = (sub_asset_flow_df * weight_series) * cum_rtn_up_until_now
        
            individual_port_val_df_list.append(indi_port_cum_rtn_series)

            total_port_cum_rtn_series = indi_port_cum_rtn_series.sum(axis=1)
            cum_rtn_up_until_now = total_port_cum_rtn_series.iloc[-1]

            prev_end_day = end_day

        individual_port_val_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
        return individual_port_val_df
    
    else:
        for end_day in weight_df.index[1:]:
            sub_price_df = price_df.loc[prev_end_day:end_day]
            signal_series = signal_df.loc[prev_end_day]

            long_signal = signal_series.replace({-1: 0})
            short_signal = signal_series.replace({1: 0, -1: 1})

            sub_price_df_reverse = sub_price_df.sort_index(ascending=False)
            
            sub_asset_flow_df = sub_price_df/sub_price_df.iloc[0]
            sub_asset_flow_df_reverse = sub_price_df_reverse/sub_price_df_reverse.iloc[0]

            weight_series = weight_df.loc[prev_end_day]
            long_weight_series = weight_series * long_signal
            short_weight_series = weight_series * short_signal
            
            indi_port_cum_rtn_series = (sub_asset_flow_df * long_weight_series) * cum_rtn_up_until_now\
                + (sub_asset_flow_df_reverse * short_weight_series) * cum_rtn_up_until_now
        
            individual_port_val_df_list.append(indi_port_cum_rtn_series)

            total_port_cum_rtn_series = indi_port_cum_rtn_series.sum(axis=1)
            cum_rtn_up_until_now = total_port_cum_rtn_series.iloc[-1]

            prev_end_day = end_day 

        individual_port_val_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
        return individual_port_val_df

def port_rets(portvals_df: pd.DataFrame, cumulative: bool=True) -> pd.Series:
    """Calculate Cumulative Returns
    
    portvals_df (pd.DataFrame):
        - DataFrame -> 포트폴리오의 일별 가치변화를 담은 df
    cumulative (bool):
        - bool -> 누적수익률을 계산할지, 일별 수익률을 계산할지 설정
        
    return (pd.Series):
        - Series -> 누적수익률 or 일별 수익률
    """
    
    if cumulative:
        return portvals_df.sum(axis=1) 
    
    else: 
        return portvals_df.sum(axis=1).pct_change().fillna(0)
