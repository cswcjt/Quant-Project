import pickle
import sys
import pandas as pd

from pathlib import Path

PJT_PATH = Path(__file__).parents[2]
sys.path.append(str(PJT_PATH))

from scaling import convert_freq
from quant.price.price_processing import rebal_dates
from quant.backtest.factor_backtest import FactorBacktest


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
    except:
        freq = 'month'
    
    try:
        factors = []
        for factor in data.getlist('factor'):
            if factor == 'momentum':
                factors.append('mom')
            elif factor == 'volatility':
                factors.append('vol')
            else:
                factors.append(factor)
    except:
        factors = ['beta', 'mom', 'vol', 'prophet']
        
    
    convert_weights = {
        'equal_weights' : 'ew',
        'equally_modified_variance' : 'emv',
        'maximize_sharpe_ratio' : 'msr',
        'global_minimum_variance' : 'gmw',
        'most_diversified_portfolio' : 'mdp',
        'risk_parity' : 'rp',
    }
    
    with open(PJT_PATH / 'django' / 'dashboard' / 'pickle' / f"rebal_dates_{freq}.pkl", 'rb') as f:
        rebal_dates = pickle.load(f)
        
    try:
        res = {
            "start_date" : data['start_date'],
            "end_date" : date_transform(data['end_date'], rebal_dates),
            "factor" : factors,
            "cs_model" : convert_weights[data['weights']],
            "risk_tolerance" : data['risk_tolerance'],
            "rebal_freq" : freq,
        }
    except:
        res = {
            "start_date" : '2013-01-02',
            "end_date" : '2022-12-30',
            "factor" : factors,
            "cs_model" : 'ew',
            "risk_tolerance" : 'aggressive',
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
    all_assets_df = all_assets_df.loc['2011':,].dropna(axis=1)
    
    bs_df = pd.read_csv(path / 'business_cycle.csv', index_col=0)
    bs_df.index = pd.to_datetime(bs_df.index)

    test = FactorBacktest(all_assets=all_assets_df,
                          business_cycle=bs_df,
                          **param)
    
    rets = test.factor_rets(factors=factors).dropna()
    cum_rets = (1 + rets).cumprod()
    
    rebal_date = rebal_dates(cum_rets, period=param['rebal_freq'],
                             include_first_date=True)
    
    if isinstance(cum_rets, pd.DataFrame):
        cum_rets = cum_rets.loc[rebal_date, :]
    elif isinstance(cum_rets, pd.Series):
        cum_rets = cum_rets.loc[rebal_date]
    else: 
        raise TypeError('Invalid Type')
    
    return cum_rets

## backtest 결과를 받아서 chart에 필요한 컬러로 변환
def color_pick(returns):
    if returns > 0:
        return '#ea5050'
    else:
        return '#5050ea'
    
if __name__ == '__main__':
    param = {
        "start_date" : '2011-01-03',
        "end_date" : '2022-12-30',
        "factor" : ['beta', 'mom', 'vol', 'prophet'],
        "cs_model" : 'ew',
        "risk_tolerance" : 'aggressive',
        "rebal_freq" : 'month',
    }
    rets = get_factor_returns(param)
    print(rets)