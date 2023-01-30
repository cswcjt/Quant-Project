import pandas as pd
import numpy as np
from functools import reduce

def calculate_portvals(price_df: pd.DataFrame, weight_df: pd.DataFrame) -> pd.DataFrame:
    """calculate_portvals

    Args:
        price_df (pd.DataFrame): 
        - DataFrame -> 일별 종가를 담고 있는 df
        weight_df (pd.DataFrame): 
        - DataFrame -> 팩터, 최적화가 끝난 최종 투자비중 df

    Returns:
        pd.DataFrame -> 일별 가격의 변동에 따른 최종 투자비중의 일별 변동 => 포트폴리오에 담긴 자산별 가치의 변화를 보여줌 
    """

    cum_rtn_up_until_now = 1 
    individual_port_val_df_list = []
    prev_end_day = weight_df.index[0]
    
    for end_day in weight_df.index[1:]:
        sub_price_df = price_df.loc[prev_end_day:end_day]
        sub_asset_flow_df = sub_price_df/sub_price_df.iloc[0]

        weight_series = weight_df.loc[prev_end_day]
        indi_port_cum_rtn_series = (sub_asset_flow_df*weight_series)*cum_rtn_up_until_now
    
        individual_port_val_df_list.append(indi_port_cum_rtn_series)

        total_port_cum_rtn_series = indi_port_cum_rtn_series.sum(axis=1)
        cum_rtn_up_until_now = total_port_cum_rtn_series.iloc[-1]

        prev_end_day = end_day 

    individual_port_val_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
    return individual_port_val_df


def get_daily_rets(individual_port_val_df: pd.DataFrame, N: int=1, log: bool=False) -> pd.DataFrame:
    """get_daily_rets

    Args:
        individual_port_val_df (pd.DataFrame): 
            - DataFrame -> culate_portvals의 리턴 값으로 일별 투자비중의 변동을 나타내는 df
        N (int): 
            - int: 수익률을 구하기 위한 look-back window -> default=1
        log (bool): 
            - bool: 로그 수익률을 사용할지 결정 -> default=False

    Returns:
        pd.DataFrame -> 포트폴리오의 일별 수익룰 df
    """

    portval_df = individual_port_val_df.sum(axis=1)
    
    if log:
        return np.log(portval_df/portval_df.shift(N)).iloc[N-1:].fillna(0)
    
    else:
        return portval_df.pct_change(N, fill_method=None).iloc[N-1:].fillna(0)


def get_cum_rets(daily_return_df: pd.DataFrame, log: bool=False) -> pd.DataFrame:
    """get_cum_returns

    Args:
        daily_return_df (pd.DataFrame): 
            - DataFrame -> get_returns의 리턴 값으로 포트폴리오의 일별 수익률 df
        log (bool): 
            - bool: 로그 수익률을 사용할지 결정 -> default=False

    Returns:
        pd.DataFrame -> 포트폴리오의 누적수익룰 df
    """
    
    if log:
        return np.exp(daily_return_df.cumsum())
    
    else:
        
        # same with (return_df.cumsum() + 1)
        return (1 + daily_return_df).cumprod()   


def get_CAGR_series(cum_rtn_df: pd.DataFrame, num_day_in_year: int) -> pd.Series:
    '''
    Compound Annual Growth Rate(CAGR)
    usually, num_day_in_year would be 252
    '''
    
    cagr_series = cum_rtn_df.iloc[-1]**(num_day_in_year/(len(cum_rtn_df))) - 1
    return cagr_series

def get_sharpe_ratio(log_rtn_df: pd.DataFrame, num_day_in_year:int, yearly_rfr: int) -> pd.Series:
    '''
    Sharpe Ratio
    yearly_rfr stands for yearly risk free rate
    '''
    
    excess_rtns = log_rtn_df.mean()*num_day_in_year - yearly_rfr
    return excess_rtns / (log_rtn_df.std() * np.sqrt(num_day_in_year))

def get_drawdown_infos(cum_returns_df: pd.DataFrame) -> tuple: 
    '''
    drawdown infos: drawdown, maximum drawdown, longest drawdown period
    drawdown: pd.DataFrame
    maximum drawdown: pd.Series
    longest drawdown period: pd.DataFrame
    '''
    
    # 1. Drawdown
    cummax_df = cum_returns_df.cummax()
    dd_df = cum_returns_df/cummax_df - 1
 
    # 2. Maximum drawdown
    mdd_series = dd_df.min()

    # 3. longest_dd_period
    dd_duration_info_list = list()
    max_point_df = dd_df[dd_df == 0]
    
    for col in max_point_df:
        _df = max_point_df[col]
        _df.loc[dd_df[col].last_valid_index()] = 0
        _df = _df.dropna()

        periods = _df.index[1:] - _df.index[:-1]

        days = periods.days
        max_idx = days.argmax()

        longest_dd_period = days.max()
        dd_mean = int(np.mean(days))
        dd_std = int(np.std(days))

        dd_duration_info_list.append(
            [
                dd_mean,
                dd_std,
                longest_dd_period,
                "{} ~ {}".format(_df.index[:-1][max_idx].date(), _df.index[1:][max_idx].date())
            ]
        )

    dd_duration_info_df = pd.DataFrame(
        dd_duration_info_list,
        index=dd_df.columns,
        columns=['drawdown mean', 'drawdown std', 'longest days', 'longest period']
    )
    return dd_df, mdd_series, dd_duration_info_df