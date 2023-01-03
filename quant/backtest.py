import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs

from scipy.optimize import minimize
from scipy.stats import norm

from typing import *

import yfinance as yf

class PortOptimPy:
    # 초기화 함수
    def __init__(self, price: pd.DataFrame, param: Union[int, str]='weekly'):
        # 주기에 따른 연율화 패러미터
        annualize_scale_dict = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }
        
        # 연율화 패러미터 (Manual + Periodical)
        self.param: int = annualize_scale_dict[param] if type(param) is str else param

        # 일별 수익률
        self.rets: pd.Series = price.pct_change().dropna()

        # 기대수익률
        self.er = np.array(self.rets * self.param)

        # 변동성
        self.vol = np.array(self.rets.rolling(self.param).std() * np.sqrt(self.param))

        # 공분산행렬
        cov = self.rets.rolling(self.param).cov().dropna() * self.param
        self.cov = cov.values.reshape(int(cov.shape[0]/cov.shape[1]), cov.shape[1], cov.shape[1])

        # 거래비용
        self.cost = 0.0005
    
    # 거래비용 함수
    def transaction_cost(self, weights_df, rets_df, cost=0.0005):
        # 이전 기의 투자 가중치
        prev_weights_df = (weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])) \
        .div((weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])).sum(axis=1), axis=0)

        # 거래비용 데이터프레임
        cost_df = abs(weights_df - prev_weights_df) * cost
        cost_df.fillna(0, inplace=True)

        return cost_df

    # 백테스팅 실행 함수
    def run(self, cs_model, ts_model, cost):
        # 빈 딕셔너리
        backtest_dict = {}
        
        # 일별 수익률 데이터프레임
        rets = self.rets
        
        # 횡적 배분 모델 선택 및 실행
        for i, index in enumerate(rets.index[self.param-1:]):
            if cs_model == 'EW':
                backtest_dict[index] = self.CrossSectional().ew(self.er[i])
            elif cs_model == 'MSR':
                backtest_dict[index] = self.CrossSectional().msr(self.er[i], self.cov[i])
            elif cs_model == 'GMV':
                backtest_dict[index] = self.CrossSectional().gmv(self.cov[i])
            elif cs_model == 'MDP':
                backtest_dict[index] = self.CrossSectional().mdp(self.vol[i], self.cov[i])
            elif cs_model == 'EMV':
                backtest_dict[index] = self.CrossSectional().emv(self.vol[i])
            elif cs_model == 'RP':
                backtest_dict[index] = self.CrossSectional().rp(self.cov[i])
        
        # 횡적 가중치 데이터프레임
        cs_weights = pd.DataFrame(list(backtest_dict.values()), index=backtest_dict.keys(), columns=rets.columns)
        cs_weights.fillna(0, inplace=True)

        # 횡적 배분 모델 자산 수익률
        cs_rets = cs_weights.shift(1) * rets.iloc[self.param-1:,:]

        # 횡적 배분 모델 포트폴리오 수익률
        cs_port_rets = cs_rets.sum(axis=1)
        
        # 종적 배분 모델 선택 및 실행
        if ts_model == 'VT':
            ts_weights = (self.TimeSeries().vt(cs_port_rets, self.param))
        elif ts_model == 'CVT':
            ts_weights = (self.TimeSeries().cvt(cs_port_rets, self.param))
        elif ts_model == 'KL':
            ts_weights = (self.TimeSeries().kl(cs_port_rets, self.param))
        elif ts_model == 'CPPI':
            ts_weights = (self.TimeSeries().cppi(cs_port_rets))
        elif ts_model == None:
            ts_weights = 1

        # 최종 포트폴리오 투자 가중치
        port_weights = cs_weights.multiply(ts_weights, axis=0)

        # 거래비용 데이터프레임
        cost = self.transaction_cost(port_weights, rets)

        # 최종 포트폴리오 자산별 수익률
        port_asset_rets = port_weights.shift() * rets - cost

        # 최종 포트폴리오 수익률 
        port_rets = port_asset_rets.sum(axis=1)
        port_rets.index = pd.to_datetime(port_rets.index).strftime("%Y-%m-%d")

        return port_weights, port_asset_rets, port_rets

    # 성과분석 수행 함수
    def performance_analytics(self, port_weights, port_asset_rets, port_rets, qs_report=False):
        
        # 자산별 투자 가중치
        plt.figure(figsize=(12, 7))
        port_weights['Cash'] = 1 - port_weights.sum(axis=1)
        plt.stackplot(port_weights.index, port_weights.T, labels=port_weights.columns)
        plt.title('Portfolio Weights')
        plt.xlabel('Date')
        plt.ylabel('Weights')
        plt.legend(loc='upper left')
        plt.show()

        # 자산별 누적 수익률
        plt.figure(figsize=(12, 7))
        plt.plot((1 + port_asset_rets).cumprod() - 1)
        plt.title('Underlying Asset Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend(port_asset_rets.columns, loc='upper left')
        plt.show()

        # 포트폴리오 누적 수익률
        plt.figure(figsize=(12, 7))
        plt.plot((1 + port_rets).cumprod() - 1)
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.show()

        # QuantStats 성과분석 리포트 작성
        if qs_report == True:
            port_rets.index = pd.to_datetime(port_rets.index)
            qs.reports.html(port_rets, output='./file-name.html')

if __name__ == '__main__':
    pass