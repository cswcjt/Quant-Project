# !pip install yfinance --quiet
import time
import yfinance as yf
import pandas as pd
import numpy as np
from functools import reduce

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

def rebal_dates(price: pd.DataFrame, period: str) -> list:
    """Select Rebalancing Period 

    Args:
        price (pd.DataFrame): 
            - DataFrame -> 타겟 상품들의 종가 df
        period (str): 
            - str -> 리밸런싱 주기 설정. (month, quarter, halfyear, year)

    Returns:
        list -> 리밸날짜를 담은 datetimeindex 
    """
    
    _price = price.reset_index()
    
    if period == "month":
        groupby = [_price['date_time'].dt.year, _price['date_time'].dt.month]
        
    elif period == "quarter":
        groupby = [_price['date_time'].dt.year, _price['date_time'].dt.quarter]
        
    elif period == "halfyear":
        groupby = [_price['date_time'].dt.year, _price['date_time'].dt.month // 7]
        
    elif period == "year":
        groupby = [_price['date_time'].dt.year, _price['date_time'].dt.year]
        
    rebal_dates = pd.to_datetime(_price.groupby(groupby)['date_time'].last().values)
    
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

def port_daily_rets(price_df: pd.DataFrame, weight_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    """calculate_portvals

    Args:
        price_df (pd.DataFrame): 
        - DataFrame -> 일별 종가를 담고 있는 df
        weight_df (pd.DataFrame): 
        - DataFrame -> 팩터, 최적화가 끝난 최종 투자비중 df
        signal_df (pd.DataFrame):
        - DataFrame -> 투자 자산과 롱숏인지 구분하기 위한 시그널 df

    Returns:
        pd.DataFrame -> 일별 가격의 변동에 따른 최종 투자비중의 일별 변동 => 포트폴리오에 담긴 자산별 가치의 변화를 보여줌 
    """

    individual_port_val_df_list = []
    prev_end_day = weight_df.index[0]
    
    for end_day in weight_df.index[1:]:
        sub_price_df = price_df.loc[prev_end_day:end_day]
        signal_series = signal_df.loc[prev_end_day]
        sub_rets_df = sub_price_df.pct_change() * signal_series

        weight_series = weight_df.loc[prev_end_day]
        port_rets_series = (sub_rets_df * weight_series)
    
        individual_port_val_df_list.append(port_rets_series)

        prev_end_day = end_day 

    individual_port_val_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
    return individual_port_val_df

def port_cum_rets(port_daily_rets):
    return (1 + port_daily_rets.sum(axis=1)).cumprod()