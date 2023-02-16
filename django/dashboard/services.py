import itertools
import os
import pickle
import sys
import pandas as pd

from pathlib import Path

PJT_PATH = Path(__file__).parents[2]
sys.path.append(str(PJT_PATH))

from scaling import convert_freq
from quant.price.price_processing import rebal_dates
from quant.backtest.factor_backtest import FactorBacktest

def daily_to_period(df, period: str, include_first_date: bool=True):
    rebal_rates = rebal_dates(df, period=period,
                              include_first_date=include_first_date)
    
    if isinstance(df, pd.DataFrame):
        rets = df.loc[rebal_rates, :]
    elif isinstance(df, pd.Series):
        rets = df.loc[rebal_rates]
    else: 
        raise TypeError('Invalid Type')
    
    return rets

def set_checkboxs_info():
    return [
        {
            'group_name': 'Factor', 
            'input_name': ['Beta', 'Momentum', 'Volatility', 'AI Forecasting'],
            'is_multiple_check': True,
        },
        {
            'group_name': 'Weights', 
            'input_name': ['Equal Weight', 'Equally Modified Variance', 'Maximize Sharpe Ratio', 
                'Global Minimum Variance', 'Most Diversified Portfolio', 'Risk Parity'],
            'is_multiple_check': False,
        },
        {
            'group_name': 'Risk Tolerance', 
            'input_name': ['Conservative', 'Moderate', 'Aggressive'],
            'is_multiple_check': False,
        },
        {
            'group_name': 'Rebalancing Period', 
            'input_name': ['Month', 'Quarter', 'Half Year', 'Year'],
            'is_multiple_check': False,
        }
    ]


def date_transform(request_date: str, rebal_date: list):
    req_year = int(request_date.split('-')[0])
    req_month = int(request_date.split('-')[1])
    
    for date in rebal_date:
        if req_year == date.year and req_month == date.month:
            res = date
        else:
            try:
                return res.strftime('%Y-%m-%d')
            except:
                raise ValueError('Invalid Date')


## request data를 받아서 backtest에 필요한 데이터로 변환
def request_transform(request: dict):
    data = request.data
    try:
        freq = convert_freq[data['rebalancing_period']]
    except Exception as e:
        freq = 'month'
    
    try:
        factors = []
        for factor in data.getlist('factor'):
            if factor == 'momentum':
                factors.append('mom')
            elif factor == 'volatility':
                factors.append('vol')
            elif factor == 'ai_forecasting':
                factors.append('prophet')
            else:
                factors.append(factor)
    except Exception as e:
        factors = ['beta', 'mom', 'vol', 'prophet']
    
    try:
        risk = data['risk_tolerance']
    except Exception as e:
        risk = 'aggressive'
    
    # convert_weights = {
    #     'equal_weight' : 'ew',
    #     'equally_modified_variance' : 'emv',
    #     'maximize_sharpe_ratio' : 'msr',
    #     'global_minimum_variance' : 'gmw',
    #     'most_diversified_portfolio' : 'mdp',
    #     'risk_parity' : 'rp',
    # }
    
    try:
        if data['start_date'] == '' or data['end_date'] == '':
            raise Exception()
        
        res = {
            "start_date" : data['start_date'],
            "end_date" : data['end_date'],
            "factor" : factors,
            "cs_model" : 'ew',
            "risk_tolerance" : risk,
            "rebal_freq" : freq,
        }
        
    except Exception as e:
        res = {
            "start_date" : '2011-01-03',
            "end_date" : '2022-12-30',
            "factor" : factors,
            "cs_model" : 'ew',
            "risk_tolerance" : risk,
            "rebal_freq" : freq,
        }
        
    return res

def get_factor_returns(param):
    path = PJT_PATH / 'quant'
    factors = param['factor']

    if 'factor' in param:
        del param['factor']
    
    all_assets_df = pd.read_csv(path / 'alter_with_equity.csv', index_col=0)
    all_assets_df.index = pd.to_datetime(all_assets_df.index)
    all_assets_df = all_assets_df.loc[param['start_date']:,].dropna(axis=1)
    
    bs_df = pd.read_csv(path / 'business_cycle.csv', index_col=0)
    bs_df.index = pd.to_datetime(bs_df.index)

    test = FactorBacktest(all_assets=all_assets_df,
                          business_cycle=bs_df,
                          **param)
    
    rets = test.factor_rets(factors=factors) #.dropna()
    cum_rets = (1 + rets).cumprod()
    print('Before')
    print(cum_rets)
    cum_rets = daily_to_period(cum_rets, period=param['rebal_freq'])
    return cum_rets.fillna(1).iloc[1:, :]

## backtest 결과를 받아서 chart에 필요한 컬러로 변환
def color_pick(returns):
    if returns > 0:
        return '#ea5050'
    else:
        return '#0D6EFD'

def make_all_params():
    param_grid = {
        "start_date": ['2011-01-03'],
        "end_date": ['2022-12-30'],
        "factor": [['beta', 'mom', 'vol', 'prophet']],
        "cs_model": ['ew'], #, 'emv', 'msr', 'gmv', 'mdp', 'rp'
        "risk_tolerance": ['conservative', 'moderate', 'aggressive'],
        "rebal_freq": ['month'],
    }
    return [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
def save_pickle():
    path = PJT_PATH / 'django' / 'dashboard' / 'pickle' / 'factor'
    
    for idx, param in enumerate(make_all_params()):
        fname = f"factor_returns_{idx}.pickle"
        rets = get_factor_returns(param)
        print('After')
        print(rets)
        
        with open(path / fname, 'wb') as f:
            pickle.dump(rets, f)
            
        print(f"Risk_tolerance: {param['risk_tolerance']}")
        print(f"{idx + 1}번째 pickle 저장 완료")
        print('#' * (idx + 1) + ' ' * (len(make_all_params()) - idx - 1) + f" ({idx + 1}/{len(make_all_params())})")

def check_param(param1, param2):
    if param1['cs_model'] == param2['cs_model'] and \
        param1['risk_tolerance'] == param2['risk_tolerance'] and \
        param1['rebal_freq'] == param2['rebal_freq']:
        return True
    else:
        return False

def load_pickle(param):
    path = PJT_PATH / 'django' / 'dashboard' / 'pickle' / 'factor'
    start = '-'.join(param['start_date'].split('-')[:2])
    end = '-'.join(param['end_date'].split('-')[:2])
    for idx, p in enumerate(make_all_params()):
        
        if check_param(p, param):
            fname = f"factor_returns_{idx}.pickle"
            
            with open(path / fname, 'rb') as f:
                rets = pickle.load(f)
                
                return rets.loc[start:end, param['factor']]
            
    return get_factor_returns(param)

def save_sp500():
    sp500 = yf.download('SPY', start='2011-01-03', end='2022-12-30', progress=False)['Adj Close']
    path = PJT_PATH / 'django' / 'dashboard' / 'pickle'
    with open(path / 'sp500.pickle', 'wb') as f:
        pickle.dump(sp500, f)

def load_sp500(param):
    path = PJT_PATH / 'django' / 'dashboard' / 'pickle'
    
    with open(path / 'sp500.pickle', 'rb') as f:
        sp500 = pickle.load(f)
        
    sp500 = daily_to_period(sp500, param['rebal_freq'], include_first_date=False)
    return sp500

import yfinance as yf
import time

if __name__ == '__main__':
    param = {
        "start_date" : '2013-01-03',
        "end_date" : '2022-12-30',
        "factor" : ['beta', 'vol'],
        "cs_model" : 'ew',
        "risk_tolerance" : 'aggressive',
        "rebal_freq" : 'month',
    }
    # start = time.time()
    # rets = get_factor_returns(param)
    # end = time.time()
    # print(end - start)
    # print(make_all_params())
    # sp500 = load_sp500(param)
    save_pickle()
    # print(sp500.loc[:'2020-02'])
    portfolio = load_pickle(param)
    print(portfolio)