import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

import warnings
warnings.filterwarnings('ignore')

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[3]
sys.path.append(str(PJT_PATH))

from scaling import convert_freq, annualize_scaler
from quant.price.price_processing import rebal_dates, get_price
from typing import Union

class ProphetFactor:
    def __init__(self, price: pd.DataFrame,
                 freq: str, lookback: Union[int, float]=1,
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
        
        self.rebal_dates = rebal_dates(self.price, self.freq, include_first_date=True)
        self.rebal_price = self.price.loc[self.rebal_dates, :]
        
        self.lookback = int(360 * lookback)
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
           
        df = df.asfreq('D').interpolate(method='spline', order=2)
        df = df.dropna()
        df = df.loc[:date]
        
        if (self.lookback > 0) and (len(df) >= self.lookback):
            df = df.iloc[-self.lookback:]
        
        df = df.reset_index(drop=False)
        df.columns = ['ds', 'y']
        return df
    
    def find_best_params(self, df: pd.DataFrame, metric: str) -> dict:
        if len(df) < int(self.lookback / 2):
            return {'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10,
                    'seasonality_mode': 'multiplicative'}
        elif len(df) < self.lookback:
            intial_days = f'{int(self.lookback / 4)} days'
        else:
            intial_days = f'{int(self.lookback / 2)} days'
        
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5], #4
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],  #4
            'seasonality_mode': ['additive', 'multiplicative'] #2
        }
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) \
            for v in itertools.product(*param_grid.values())]
        scores = []  # Store the MAEs(or other metric) for each params here

        # client = Client()
        for params in all_params:
            model = Prophet(**params).fit(df)
            
            # df_cv = cross_validation(model, 
            #                          initial=intial_days,
            #                          period='30 days',
            #                          horizon='30 days',
            #                          parallel="dask",
            #                          disable_tqdm=True)
            
            df_cv = cross_validation(model, 
                                     initial=intial_days,
                                     period='30 days',
                                     horizon='30 days',
                                     parallel=None,
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
            strdate = date.strftime('%Y-%m-%d')
            best_params.update({strdate: {}})
            
            for asset in self.price.columns:
                df = self.preprocessing(self.price.loc[:, asset], date)
                params = self.find_best_params(df, metric)
                best_params[strdate].update({asset: params})
                
        return best_params
    
    def save_params(self, metric: str) -> None:
        best_params = self.make_full_params(metric=metric)
        
        start = self.rebal_dates[0].strftime('%Y%m')
        end = self.rebal_dates[-1].strftime('%Y%m')
        
        fname = f'prophet_params_{metric}_{start}_{end}.json'
        with open(self.save_path / fname, 'w') as f:
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
            raise FileNotFoundError('There is no saved file.')
        else:
            with open(self.save_path / load_file, 'r') as f:
                params = json.load(f)
            
        return params

    def calc_returns(self) -> pd.DataFrame:
        # Prophet 모델을 통해 다음달 가격 예측
        try:
            params_exist = True
            best_params = self.load_params()
        except FileNotFoundError:
            params_exist = False
            param = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'seasonality_mode': 'additive',
            }
            
        returns = {}        
        # 각 날짜별, 자산별로 예측 수익률 계산
        for date in self.rebal_dates[1:]:
            strdate = date.strftime('%Y-%m-%d')
            returns.update({strdate: {}})
            for asset in self.price.columns:
                df = self.preprocessing(self.price[asset], date)
                if params_exist:
                    param = best_params[strdate][asset]
                
                model = Prophet(**param).fit(df)
                future = model.make_future_dataframe(periods=30, freq='D')
                forecast = model.predict(future)
                
                ret = (forecast['yhat'].iloc[-1] - df.iloc[-1,1]) / df.iloc[-1,1]
                returns[strdate].update({
                    asset: ret
                })

        return pd.DataFrame(returns)
    
    def save_returns(self, num: int):
        """ 시계열 예측팩터 수익률 테이블 저장 함수 """
        retruns = self.calc_returns()
        
        save_path = PJT_PATH / 'quant' / 'strategy' / 'factors' / 'data'
        fname = f'prophet_returns_{num}.json'
        
        retruns.to_json(save_path / 'prophet_returns.json')
        
    def load_returns(self, num: int):
        """ 시계열 예측팩터 수익률 테이블 불러오기 함수 """
        load_path = PJT_PATH / 'quant' / 'strategy' / 'factors' / 'data'
        fname = f'prophet_returns_{num}.json'
        rets = pd.read_json(load_path / fname).T
        rets.index = pd.to_datetime(rets.index)
        return rets
    
    def calc_signal(self, returns: pd.DataFrame=None, n_sel: int=20):
        """ 시그널 계산 함수 """
        if returns is None:
            rets = self.load_returns()
        elif not isinstance(returns, pd.DataFrame):
            raise TypeError('returns must be pd.DataFrame')
        else:
            rets = returns
        
        if self.long_only:
            # 수익률 순위화
            rank = rets.rank(axis=1, ascending=False)
            # 롱 시그널
            long_signal = (rank <= n_sel) * 1
            signal = long_signal
        
        else:
            # 수익률 절대값 순위화
            abs_rank = rets.applymap(abs).rank(axis=1, ascending=False)
            
            # 시그널
            abs_signal = (abs_rank <= n_sel) * 1
            
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
    load_path = PJT_PATH / 'quant' / 'strategy' / 'factors' / 'data'
    
    df = pd.read_csv(PJT_PATH / 'asset_universe.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    prophet = ProphetFactor(df, freq='month',
                            lookback=0.5, long_only=True)
    
    rets = prophet.load_returns(num=1).drop(columns=['SPY', 'TLT', 'GSG', 'VNQ', 'UUP'])
    signal = prophet.calc_signal(rets, n_sel=20)
    print(signal)
    with open(PJT_PATH / 'django' / 'dashboard' / 'pickle' / 'prophet_signal.pickle', 'wb') as f:
        pickle.dump(signal, f)
    # rets = prophet.rebal_price.pct_change().dropna()
    # # print(rets)
    
    # predicted_rets = prophet.load_returns()
    # signal = prophet.calc_signal(predicted_rets, n_sel=20)
    # print(signal.sum(axis=1))
    
    # num_assets = len(rets.columns)
    # final_rets = (1 + (signal * rets)).cumprod()
    # print(final_rets)
    
    # final_rets = final_rets.sum(axis=1) / num_assets
    # print(final_rets)

    # # print(signal)
    # # signal.to_json('./data/prophet_signal.json')