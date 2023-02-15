import random
import pandas as pd
import yfinance as yf

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from dashboard.services import (
    PJT_PATH,
    get_factor_returns, 
    color_pick,
    request_transform
)

from quant.backtest.metric import Metric

class FactorAPIView(APIView):
    def get_data(self, request):
        param = request_transform(request)
        portfolio = get_factor_returns(param)
    
    def get(self, request, *args, **kwargs):
        pass

    def post(self, request, *args, **kwargs):
        pass


class MarketAPIView(APIView):
    def get_data(self, request):
        return {
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
    def get(self, request, *args, **kwargs):
        data = self.get_data(request)
        print(data)
        return Response(data=data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        print(request.data)
        data = self.get_data(request)
        return Response(data=data, status=status.HTTP_200_OK)

class PortfolioAPIView(APIView):
    def get_data(self, request):
        param = request_transform(request)
        names = ['Portfolilo', 'S&P500']
        
        sp500 = yf.download('SPY', start=param['start_date'], end=param['end_date'], progress=False)
        sp500_report = Metric(portfolio=sp500, freq=param['rebal_freq'])
        portfolio = get_factor_returns(param)
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
                        'data': [{'x': time.strftime('%Y-%m'), 'y': f'{cum_rets: .2f}'} \
                            for time, cum_rets \
                                in zip(method_dict[name].cum_rets.index, method_dict[name].cum_rets.values)]
                        } for name in names],
                'type': 'area',
                'height': 350,
                'colors': ['#FF6384', '#00B1E4']
            },

            'mdd': {
                'data': [{'name': name,
                        'data': [{'x': time.strftime('%Y-%m'), 'y': f'{cum_rets: .2f}'} \
                            for time, cum_rets \
                                in zip(method_dict[name].drawdown().index, method_dict[name].drawdown().values)]
                        } for name in names],
                'type': 'area',
                'height': 190,
                'colors': ['#a9a0fc', '#e2e2e2']
            },

            'rolling_sharp_ratio': {
                'data': [{'name': name,
                        'data': [{'x': time.strftime('%Y-%m'), 'y': f'{cum_rets: .2f}'} \
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
                                    'data': report_dict[name][key],
                                    'color': color_pick(float(report_dict[name][key])),
                                    } for name in names]

        return data

        # names = ['Portfolilo', 'S&P500']
        # return {
        #     'cumulative': {
        #         'data': [{'name': name, 'data': random.sample(list(range(0, 101)), 11)} for name in names],
        #         'type': 'area',
        #         'height': 350,
        #         'colors': ['#FF6384', '#00B1E4']
        #     },

        #     'mdd': {
        #         'data': [{'name': name, 'data': random.sample(list(range(0, 51)), 10)} for name in names],
        #         'type': 'area',
        #         'height': 190,
        #         'colors': ['#a9a0fc', '#e2e2e2']
        #     },

        #     'rolling_sharp_ratio': {
        #         'data': [{'name': name, 'data': random.sample(list(range(0, 51)), 10)} for name in names],
        #         'type': 'area',
        #         'height': 190,
        #         'colors': ['#a9a0fc', '#e2e2e2']
        #     },
        # }

    def get(self, request, *args, **kwargs):
        data = self.get_data(request)
        print(data)
        return Response(data=data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        print(request.data)
        data = self.get_data(request)
        return Response(data=data, status=status.HTTP_200_OK)