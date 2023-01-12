# 패키지 임포트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from scipy.optimize import minimize
from scipy.stats import norm


class CrossSectional:
    # 횡적 배분 모델 클래스

    def __init__(self, rebal_price: pd.DataFrame, param: int):

        # 연율화 패러미터
        self.param = param

        # 일별 수익률
        self.rets = rebal_price.pct_change().dropna()

        # 기대수익률
        self.er = np.array(self.rets * self.param)

        # 변동성
        self.vol = np.array(self.rets.rolling(
            self.param).std() * np.sqrt(self.param))

        # 공분산행렬
        cov = self.rets.rolling(self.param).cov().dropna() * self.param
        self.cov = cov.values.reshape(
            int(cov.shape[0]/cov.shape[1]), cov.shape[1], cov.shape[1])

    # EW(동일 비중 가중치 계산 함수)

    def ew(self, er):
        noa = er.shape[0]
        weights = np.ones_like(er) * (1/noa)
        return weights

    # MSR(샤프비율 최대화)
    def msr(self, er, cov):
        noa = er.shape[0]
        init_guess = np.repeat(1/noa, noa)

        # 제약조건 및 상하한값
        bounds = ((0.0, 1.0), ) * noa
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}

        # 목적함수 : 마이너스 샤프비율
        def neg_sharpe(weights, er, cov):
            r = weights.T @ er
            vol = np.sqrt(weights.T @ cov @ weights)
            return - r / vol

        weights = minimize(neg_sharpe,
                           init_guess,
                           args=(er, cov),
                           method='SLSQP',
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)

        return weights.x

    # GMV(최소 변동성)
    def gmv(self, cov):
        noa = cov.shape[0]
        init_guess = np.repeat(1/noa, noa)

        # 제약조건 및 상하한값
        bounds = ((0.0, 1.0), ) * noa
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}

        # 목적함수 : 포트폴리오 변동성
        def port_vol(weights, cov):
            vol = np.sqrt(weights.T @ cov @ weights)
            return vol

        weights = minimize(port_vol, init_guess,
                           args=(cov),
                           method='SLSQP',
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)

        return weights.x

    # MDP(최대 분산비율)
    def mdp(self, vol, cov):
        noa = vol.shape[0]
        init_guess = np.repeat(1/noa, noa)

        # 제약조건 및 상하한값
        bounds = ((0.0, 1.0), ) * noa
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}

        # 목적함수 : 마이너스 분산비율
        def neg_div_ratio(weights, vol, cov):
            weighted_vol = weights.T @ vol
            port_vol = np.sqrt(weights.T @ cov @ weights)
            return - weighted_vol / port_vol

        weights = minimize(neg_div_ratio,
                           init_guess,
                           args=(vol, cov),
                           method='SLSQP',
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)

        return weights.x

    # RP(리스크 패리티)
    def rp(self, cov):
        noa = cov.shape[0]
        init_guess = np.repeat(1/noa, noa)

        # 목표 위험기여도 : 동등 위험 기여
        target_risk = np.repeat(1/noa, noa)

        # 제약조건 및 상하한값
        bounds = ((0.0, 1.0), ) * noa
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}

        # 목적 함수 : 목표 위험 기여도와의 오차 최소화
        def msd_risk(weights, target_risk, cov):

            port_var = weights.T @ cov @ weights
            marginal_contribs = cov @ weights

            risk_contribs = np.multiply(
                marginal_contribs, weights.T) / port_var

            w_contribs = risk_contribs
            return ((w_contribs - target_risk)**2).sum()

        weights = minimize(msd_risk,
                           init_guess,
                           args=(target_risk, cov),
                           method='SLSQP',
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x

    # EMV(역변동성)
    def emv(self, vol):
        inv_vol = 1 / vol
        weights = inv_vol / inv_vol.sum()

        return weights

    # 백테스팅 실행 함수

    def run(self, cs_model):
        # 빈 딕셔너리
        backtest_dict = {}

        # 일별 수익률 데이터프레임
        rets = self.rets

        # 횡적 배분 모델 선택 및 실행
        for i, index in enumerate(rets.index[self.param-1:]):
            if cs_model == 'EW':
                backtest_dict[index] = self.ew(self.er[i])
            elif cs_model == 'MSR':
                backtest_dict[index] = self.msr(self.er[i], self.cov[i])
            elif cs_model == 'GMV':
                backtest_dict[index] = self.gmv(self.cov[i])
            elif cs_model == 'MDP':
                backtest_dict[index] = self.mdp(self.vol[i], self.cov[i])
            elif cs_model == 'EMV':
                backtest_dict[index] = self.emv(self.vol[i])
            elif cs_model == 'RP':
                backtest_dict[index] = self.rp(self.cov[i])

        # 횡적 가중치 데이터프레임
        cs_weights = pd.DataFrame(
            list(backtest_dict.values()), index=backtest_dict.keys(), columns=rets.columns)
        cs_weights.fillna(0, inplace=True)

        # 횡적 배분 모델 자산 수익률
        cs_rets = cs_weights.shift(1) * rets.iloc[self.param-1:, :]

        # 횡적 배분 모델 포트폴리오 수익률
        cs_port_rets = cs_rets.sum(axis=1)

        return cs_port_rets, cs_weights
