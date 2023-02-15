import sys
import random

from django.http import JsonResponse
from django.urls import reverse

from rest_framework import status

## Project Path 추가
import json
import pickle
import sys

import pandas as pd
import yfinance as yf

from pathlib import Path

PJT_PATH = Path(__file__).parents[2]
sys.path.append(str(PJT_PATH))

from quant.backtest.metric import Metric
from quant.backtest.factor_backtest import FactorBacktest
from quant.price.price_processing import rebal_dates
from scaling import convert_freq, annualize_scaler
"""
request.GET.get 구조 예시
key -> {'start_date', 'end_date', 'period', 'risk tolerance', 'factor weights', 'factor'}
value ->
'start' : {'2019-01-01'}
"""

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
            "start_date" : '2011-01-03',
            "end_date" : '2022-12-30',
            "factor" : factors,
            "cs_model" : 'ew',
            "risk_tolerance" : 'aggressive',
            "rebal_freq" : freq,
        }
    return res

def factor_api(request):
    factor_name = ['beta', 'mom', 'vol', 'prophet']
    
    # signals_name = ['beta', 'momentum', 'volatility', 'ai_forecast']
    # signals = [signal_beta, signal_momentum, signal_vol, signal_prophet]
    
    # num_assets = len(df.columns)
    # final_rets = (1 + (signal * df)).cumprod()
    # print(final_rets)

    # final_rets = final_rets.sum(axis=1) / num_assets
    
    data = {
        'metrics': [{'name': factor_name, 'data': []}]
    }
    
    return JsonResponse(data=data, status=status.HTTP_200_OK)

def market_api(request):
    '''
    APEXCHARTS 형식의 맞는 임의 값 생성해서 반환중 
    대충 테스트를 위해서 만든거라 코드 개떡같음 
    백테스팅해서 넘길 json 값을 APEXCHARTS 형식에 맞게 수정이 필요할 수도 있음
    '''
    with open(PJT_PATH / 'quant' / 'price' / 'tickers.json') as f:
        indices_dict = json.load(f)

    print(type(indices_dict))
    indices_name = list(indices_dict.keys())
    print(indices_name)
    indices_tickers = list(indices_dict.values())
    print(indices_tickers)
    
    data = {
        'regime_clustering': {
            'data': [{'name': indices_name, 'data': [[random.randint(1, 11), random.randint(1, 11)] for j in range(1, 21)]} for name in ['setosa', 'versicolor', 'virginica']],
            'type': 'scatter',
            'height': 350,
            'colors': ['#a9a0fc', '#FF6384', '#008FFB']
        },

        'index_chart': {
            'data': [{'name': indices_name, 'data': random.sample(list(range(0, 101)), 11)} for name in ['AAPL', 'AMZN', 'GOOG', 'IMB', 'MSFT']],
            'type': 'area',
            'height': 190,
            'colors': ['#a9a0fc', '#FF6384', '#008FFB', '#fdfd96', '#bfff00']
        },

        'index_forecasting': {
            'data': [{'name': indices_name, 'data': random.sample(list(range(0, 51)), 10)} for name in ['AAPL', 'AMZN', 'GOOG', 'IMB', 'MSFT']],
            'type': 'area',
            'height': 190,
            'colors': ['#a9a0fc', '#FF6384', '#008FFB', '#fdfd96', '#bfff00']
        },

        'market_regime': {
            'data': [{'name': f'Metric{i}', 'data': [{'x': f'w{j}', 'y': random.randint(0, 101)} for j in range(1, 11)]} for i in range(1, 6)],
            'type': 'heatmap',
            'height': 190,
            'colors': ['#008FFB']
        },

        'economic_growth': {
            'data': [{'name': f'Metric{i}', 'data': [{'x': f'w{j}', 'y': random.randint(0, 101)} for j in range(1, 11)]} for i in range(1, 6)],
            'type': 'heatmap',
            'height': 190,
            'colors': ['#008FFB']
        },
    }

    return JsonResponse(data=data, status=status.HTTP_200_OK)

def portfolio_api(request):
    ''' 
    APEXCHARTS 형식의 맞는 임의 값 생성해서 반환중 
    대충 테스트를 위해서 만든거라 코드 개떡같음 
    백테스팅해서 넘길 json 값을 APEXCHARTS 형식에 맞게 수정이 필요할 수도 있음
    
    [['beta', 'mom', 'vol', 'prophet'], ['beta', 'mom', 'vol'],
     ['beta', 'mom', 'prophet'], ['beta', 'vol', 'prophet'],
     ['mom', 'vol', 'prophet'], ['beta', 'mom'], ['beta', 'vol'], ['beta', 'prophet'],
     ['mom', 'vol'], ['mom', 'prophet'], ['vol', 'prophet'],
     ['beta'], ['mom'], ['vol'], ['prophet']]]
     
    ['ew', 'emv', 'msr', 'gmv', 'mdp', 'rp']
    ['aggressive', 'moderate', 'conservative']
    
    param = {
        "start_date" : '2013-01-01',
        "end_date" : '2022-12-31',
        "factor" : ['beta', 'mom', 'vol', 'prophet'],
        "cs_model" : 'gmw',
        "risk_tolerance" : 'assertive',
        "rebal_freq" : 'month',
    }
    '''
    
    # start = request.GET.get['start']
    # end = request.GET.get['end']
    
    # request.start = '2019-01-01'
    # request.end = '2020-01-01'
    
    
    path = PJT_PATH / 'quant'
    param = request_transform(request)
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
    
    names = ['Portfolilo', 'S&P500']
    
    sp500 = yf.download('SPY', start=param['start_date'], end=param['end_date'], progress=False)
    sp500_report = Metric(portfolio=sp500, freq=param['rebal_freq'])
    
    portfolio = test.factor_rets(factors=factors)
    portfolio_report = Metric(portfolio=portfolio, freq=param['rebal_freq'])
    
    method_dict = {
        'Portfolilo': portfolio_report,
        'S&P500': sp500_report,
    }
    
    report_dict = {
        'Portfolilo': portfolio_report.numeric_metric(),
        'S&P500': sp500_report.numeric_metric(),
    }
    
    data = {
        'cumulative': {
            'data': [{'name': name,
                      'data': [{'x': time, 'y': cum_rets} \
                          for time, cum_rets \
                              in zip(method_dict[name].cum_rets.index, method_dict[name].cum_rets.values)]
                      } for name in names],
            'type': 'area',
            'height': 350,
            'colors': ['#FF6384', '#00B1E4']
        },

        'mdd': {
            'data': [{'name': name,
                      'data': [{'x': time, 'y': cum_rets} \
                          for time, cum_rets \
                              in zip(method_dict[name].drawdown().index, method_dict[name].drawdown().values)]
                      } for name in names],
            'type': 'area',
            'height': 190,
            'colors': ['#a9a0fc', '#e2e2e2']
        },

        'rolling_sharp_ratio': {
            'data': [{'name': name,
                      'data': [{'x': time, 'y': cum_rets} \
                          for time, cum_rets \
                              in zip(method_dict[name].sharp_ratio(rolling=True, lookback=1).dropna().index,
                                     method_dict[name].sharp_ratio(rolling=True, lookback=1).dropna().values)
                              ]
                      } for name in names],
            'type': 'area',
            'height': 190,
            'colors': ['#a9a0fc', '#e2e2e2']
        },
        
        'metric': {},
    }
    
    # metric key list
    key_dict = {
        'returns': 'Total Returns',
        'CAGR': 'CAGR',
        'MDD': 'MDD',
        'MDD_duration': 'Underwater Period',
        'volatility': 'Anuallized Volatility',
        'sharp': 'Sharpe Ratio',
        'sortino': 'Sortino Ratio',
        'calmar': 'Calmar Ratio',
        'CVaR_ratio': 'CVaR Ratio',
        'hit': 'Hit Ratio',
        'GtP': 'Gain-to-Pain'
    }
    
    key_list = ['returns', 'CAGR', 'MDD', 'MDD_duration',
                'volatility', 'sharp', 'sortino', 'calmar',
                'CVaR_ratio', 'hit', 'GtP']

    for key in key_list:
        data['metric'][key] = [{'name': name,
                                'title': key_dict[key],
                                'data': report_dict[name][key],
                                'color': color_pick(float(report_dict[name][key])),
                                } for name in names]
        
    """
    data['metric'] = {
        'returns': [{'name': 'Portfolilo',
                     'title': 'Total Returns',
                     'data': 0.1,
                     'color': '#ea5050'},
                    {'name': 'S&P500',
                     'title': 'Total Returns',
                     'data': -0.1,
                     'color': '#5050ea'}],
        
        'CAGR': [{'name': 'Portfolilo', 'title': 'CAGR', 'data': 0.1, 'color': '#ea5050'},
                 {'name': 'S&P500', 'title': 'CAGR', 'data': -0.1, 'color': '#5050ea'}],
        ...
    }
    """
        
    print(data)
    
    return JsonResponse(data=data, status=status.HTTP_200_OK)

def color_pick(returns):
    if returns > 0:
        return '#ea5050'
    else:
        return '#5050ea'

if __name__ == '__main__':
    df = pd.read_csv(PJT_PATH / 'asset_universe.csv', index_col=0)