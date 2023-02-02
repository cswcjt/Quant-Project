import random

from django.http import JsonResponse
from django.urls import reverse

from rest_framework import status


def factor_api(request):
    pass


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
    
    names = ['Portfolilo', 'S&P500']

    [{'name': name, 'data': random.sample(list(range(0, 101)), 11)} for name in names]
    data = {
        'cumulative': {
            'data': [{'name': name, 'data': random.sample(list(range(0, 101)), 11)} for name in names],
            'type': 'area',
            'height': 350,
            'colors': ['#FF6384', '#00B1E4']
        },

        'mdd': {
            'data': [{'name': name, 'data': random.sample(list(range(0, 51)), 10)} for name in names],
            'type': 'area',
            'height': 190,
            'colors': ['#a9a0fc', '#e2e2e2']
        },

        'rolling_sharp_ratio': {
            'data': [{'name': name, 'data': random.sample(list(range(0, 51)), 10)} for name in names],
            'type': 'area',
            'height': 190,
            'colors': ['#a9a0fc', '#e2e2e2']
        },
    }

    return JsonResponse(data=data, status=status.HTTP_200_OK)