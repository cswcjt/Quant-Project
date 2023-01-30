import yfinance as yf
import pandas as pd
from fredapi import Fred

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

    Returns:
        pd.DataFrame -> 금융상품 가격정보를 담고있는 df
    """
    
    temp = []
    for ticker in tickers:
        name = f'{ticker}'
        name = yf.Ticker(ticker)
        temp_df = name.history(start=start_date, period=period, interval=interval)['Close']
        temp.append(temp_df)
    
    price_df = pd.concat(temp, axis = 1)
    price_df.columns = tickers
    
    if interval in ['1d', '5d', '1wk', '1mo', '3mo']:
        price_df.index = pd.to_datetime(price_df.index.date)
    
    price_df.dropna(inplace=True)

    return price_df

def rebal_dates(price: pd.DataFrame, period: str) -> list: 
    '''
    기능: 포트폴리오 리밸런싱 날을 구한다.

    price: get_price()의 결과값
    period: 리밸런싱의 주기를 설정
    '''
    
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

def get_econ_info(indicators) : 
    fred = Fred(api_key='78b31c929aa00cef888d17a4c63cb823')
    temp = []
    for indicator in indicators : 
        target_df = pd.DataFrame({f'{indicator}': fred.get_series(f'{indicator}')})
        temp.append(target_df)

    econ_df = pd.concat(temp, axis = 1)
    econ_df.columns = indicators

    return econ_df
