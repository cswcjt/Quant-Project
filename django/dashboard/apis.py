import random

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


class FactorAPIView(APIView):
    def get(self, request, *args, **kwargs):
        pass

    def post(self, request, *args, **kwargs):
        pass


class MarketAPIView(APIView):
    def get_data(self):
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
        data = self.get_data()
        return Response(data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        print(request.data)
        data = self.get_data()
        return Response(data=data, status=status.HTTP_200_OK)


class PortfolioAPIView(APIView):
    def get_data(self):
        names = ['Portfolilo', 'S&P500']
        return {
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

    def get(self, request, *args, **kwargs):
        data = self.get_data()
        return Response(data=data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        print(request.data)
        data = self.get_data()
        return Response(data=data, status=status.HTTP_200_OK)
        