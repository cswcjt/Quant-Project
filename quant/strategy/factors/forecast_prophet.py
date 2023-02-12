import holidays
import itertools
import json
import numpy as np
import os
import pandas as pd

from dask.distributed import Client
from numba import njit, vectorize
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

import warnings
warnings.filterwarnings('ignore')

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[3]
sys.path.append(str(PJT_PATH))

from scaling import convert_freq, annualize_scaler
from quant.price.price_processing import rebal_dates, get_price

class ProphetFactor:
    def __init__(self, price: pd.DataFrame,
                 freq: str, n_sel: int,
                 lookback: int=1,
                 long_only: bool=True):
        """초기화 함수

        Args:
            rebal_price (pd.DataFrame): 
                - DataFrame -> price_on_rebal()의 리턴 값. 리밸런싱 날짜의 타켓 상품들 종가 df
            lookback_window (int):
                - int -> 모멘텀(추세)를 확인할 기간 설정
            n_sel (int):
                - int -> 몇 개의 금융상품을 고를지 결정
            long_only (bool, optional):
                - bool -> 매수만 가능한지 아님 공매도까지 가능한지 결정. Defaults to True.
        """
        self.save_path = PJT_PATH / 'quant' / 'strategy' / 'factors' / 'models'
        
        self.freq = convert_freq(freq)
        self.param = annualize_scaler(self.freq)
        
        self.price = price
        if not isinstance(self.price.index[0], pd.Timestamp):
            self.price.index = pd.to_datetime(self.price.index)
        
        self.rebal_dates = rebal_dates(self.price, self.freq)
        self.rebal_price = self.price.loc[self.rebal_dates, :]
        
        self.lookback = lookback
        self.n_sel = n_sel
        self.long_only = long_only
        self.model = None
    
    def __getattr__(self, __name: str) -> any:
        return __name + ' is not defined'
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}'
    
    def preprocessing(self, asset_price: pd.Series, date: pd.Timestamp):
        """모델에 적합한 데이터로 가공하는 함수

        Returns:
            pd.DataFrame:
                - DataFrame -> Prophet 모델에 적합한 데이터로 가공한 df
        """
        # Prophet 모델에 적합한 데이터로 가공
        df = asset_price.copy()
        
        if not isinstance(df.index[0], pd.Timestamp):
            df.index = pd.to_datetime(df.index)
           
        df = df.asfreq('D').interpolate(method='linear')
        df = df.loc[:date]
        
        lookback = 365 * self.lookback
        if (lookback > 0) and (len(df) >= lookback):
            df = df.iloc[-lookback:, :]
        
        df = df.reset_index(drop=False)
        df.columns = ['ds', 'y']
        return df
    
    def find_best_params(self, df: pd.DataFrame, metric: str) -> dict:
        if len(df) < 180:
            return {'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10,
                    'seasonality_mode': 'additive'}
        elif len(df) < 365:
            intial_days = '90 days'
        else:
            intial_days = '180 days'
        
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) \
            for v in itertools.product(*param_grid.values())]
        scores = []  # Store the MAEs(or other metric) for each params here

        client = Client()
        for params in all_params:
            model = Prophet(**params).fit(df)
            df_cv = cross_validation(model, 
                                     initial=intial_days,
                                     period='30 days',
                                     horizon='30 days',
                                     parallel="dask",
                                     disable_tqdm=True)
            
            df_p = performance_metrics(df_cv)
            scores.append(df_p['mae'].values[-1])
            
        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results[metric] = scores
        
        # Sorted by metric values and Pick the best parameters
        tuning_results = tuning_results.sort_values(by=[metric])
        final_params = tuning_results.iloc[0, :-1].to_dict()
        return final_params
        
    def make_full_params(self, metric: str) -> None:
        best_params = {}
        for date in self.rebal_dates:
            best_params[date] = {}
            for asset in self.price.columns:
                df = self.preprocessing(self.price.loc[:, asset], date)
                params = self.find_best_params(df, metric)
                best_params[date][asset] = params
                
        return best_params
    
    def save_params(self, metric: str) -> None:
        best_params = self.make_full_params(metric=metric)
        
        start = self.rebal_dates[0].strftime('%Y%m')
        end = self.rebal_dates[-1].strftime('%Y%m')
        
        fname = f'prophet_params_{metric}_{start}_{end}.json'
        with open(self.save_path / 'prophet_params.json', 'w') as f:
            json.dump(best_params, f)
    
    def load_params(self) -> dict:
        start = self.rebal_dates[0].strftime('%Y%m')
        end = self.rebal_dates[-1].strftime('%Y%m')
        load_file = ''
        
        for fname in os.listdir(self.save_path):
            fstart = fname.split('_')[3]
            fend = fname.split('_')[4].split('.')[0]
            if start >= fstart and end <= fend:
                load_file = fname
                break
        
        if load_file == '':
            raise FileNotFoundError('There is no parameter file')
        else:
            with open(self.save_path / load_file, 'r') as f:
                params = json.load(f)
            
        return params

    def calc_returns(self) -> pd.DataFrame:
        # Prophet 모델을 통해 예측한 파라미터를 불러옴
        try:
            best_params = self.load_params()
        except FileNotFoundError:
            best_params = self.make_full_params(metric='mae')
            
        returns = {}
        
        # 각 날짜별, 자산별로 예측 수익률 계산
        for date in self.rebal_dates:
            returns[date] = 0
            
            for asset in self.price.columns:
                df = self.preprocessing(self.price[asset], date)
                param = self.load_params()[date][asset]
                
                model = Prophet(**param).fit(df)
                future = model.make_future_dataframe(periods=30, freq='D')
                forecast = model.predict(future)
                
                returns[date][asset] = (forecast['yhat'].iloc[-1] - df.loc[-1,asset]) \
                                        / df.iloc[-1, asset]

        return pd.DataFrame(returns)
    
    def calc_signal(self):
        """ 시그널 계산 함수 """
        rets = self.calc_returns()
        
        if self.long_only:
            # 수익률 순위화
            rank = rets.rank(axis=1, ascending=False)
            # 롱 시그널
            long_signal = (rank <= self.n_sel) * 1
            signal = long_signal
        
        else:
            # 수익률 절대값 순위화
            abs_rank = rets.applymap(abs).rank(axis=1, ascending=False)
            
            # 시그널
            abs_signal = (abs_rank <= self.n_sel) * 1
            
            # 롱 시그널
            long_signal = (rets > 0) * 1
            
            # 숏 시그널
            short_signal = (rets < 0) * -1
            
            # 롱/숏 시그널 합산
            signal = (abs_signal * long_signal.values) \
                + (abs_signal * short_signal.values)
            
        return signal
    
import yfinance as yf

if __name__ == '__main__':
    df = pd.read_csv(PJT_PATH / 'asset_universe.csv', index_col=0)
    prophet1 = ProphetFactor(df, freq='month', n_sel=20,
                             lookback=1, long_only=True)
    prophet1.save_params(metric='mae')
    params = prophet1.load_params()
    print(params)