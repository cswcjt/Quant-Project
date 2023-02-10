import sys
import random

from django.http import JsonResponse
from django.urls import reverse

from rest_framework import status

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[2]
sys.path.append(str(PJT_PATH))

from quant.backtest.metric import Metric
import yfinance as yf

def factor_api(request):
    name = ['beta', 'momentum', 'volatility', 'prophet', 'xgboost', 'lstm', 'transformer']
    data = {
        'metrics': [{'name': name, 'data': []}]
    }
    
    return JsonResponse(data=data, status=status.HTTP_200_OK)

def market_api(request):
    '''
    APEXCHARTS 형식의 맞는 임의 값 생성해서 반환중 
    대충 테스트를 위해서 만든거라 코드 개떡같음 
    백테스팅해서 넘길 json 값을 APEXCHARTS 형식에 맞게 수정이 필요할 수도 있음
    '''

    data = {
        'regime_clustering': {
            'data': [{'name': name, 'data': [[random.randint(1, 11), random.randint(1, 11)] for j in range(1, 21)]} for name in ['setosa', 'versicolor', 'virginica']],
            'type': 'scatter',
            'height': 350,
            'colors': ['#a9a0fc', '#FF6384', '#008FFB']
        },

        'index_chart': {
            'data': [{'name': name, 'data': random.sample(list(range(0, 101)), 11)} for name in ['AAPL', 'AMZN', 'GOOG', 'IMB', 'MSFT']],
            'type': 'area',
            'height': 190,
            'colors': ['#a9a0fc', '#FF6384', '#008FFB', '#fdfd96', '#bfff00']
        },

        'index_forecasting': {
            'data': [{'name': name, 'data': random.sample(list(range(0, 51)), 10)} for name in ['AAPL', 'AMZN', 'GOOG', 'IMB', 'MSFT']],
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
    '''
    # start = request.GET.get['start']
    # end = request.GET.get['end']
    
    start = '2019-01-01'
    end = '2020-01-01'
    
    names = ['Portfolilo', 'S&P500']
    sp500 = yf.download('SPY', start=start, end=end, progress=False)
    portfolio = yf.download('AAPL', start=start, end=end, progress=False)
    sp500_report = Metric(portfolio=sp500, freq='D')
    portfolio_report = Metric(portfolio=portfolio, freq='daily')
    
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
    key_list = ['returns', 'CAGR', 'MDD', 'MDD_duration',
                'volatility', 'sharp', 'sortino', 'calmar',
                'VaR_ratio', 'hit', 'GtP']

    for key in key_list:
        data['metric'][key] = [{'name': name,
                                'data': report_dict[name].numeric_metric()[key],
                                'color': color_pick(report_dict[name][key]),
                                } for name in names]
        
    print(data)
    
    return JsonResponse(data=data, status=status.HTTP_200_OK)

def color_pick(returns):
    if returns > 0:
        return '#ea5050'
    else:
        return '#5050ea'

if __name__ == '__main__':
    print(Path(__file__).parents[2]) # /Users/john/Quant-Project
    print(random.sample(list(range(0, 101)), 11))
    
    start = '2017-01-01'
    end = '2020-01-01'
    
    names = ['Portfolilo', 'S&P500']
    sp500 = yf.download('SPY', start=start, end=end, progress=False)
    portfolio = yf.download('AAPL', start=start, end=end, progress=False)
    sp500_report = Metric(portfolio=sp500, freq='D')
    portfolio_report = Metric(portfolio=portfolio, freq='daily')
    report_dict = {
        'Portfolilo': portfolio_report.numeric_metric(delta=0.01),
        'S&P500': sp500_report.numeric_metric(delta=0.01),
    }
    print(report_dict['S&P500']['sharp'])
    portfolio_api(end)