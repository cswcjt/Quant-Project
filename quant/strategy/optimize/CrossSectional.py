# 패키지 임포트
import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import *

from scipy.optimize import minimize
from scipy.stats import norm


class Equalizer:
    # 횡적 배분 모델 클래스
    def __init__(self, signal: pd.DataFrame, rebal_price: pd.DataFrame, param: int) -> pd.DataFrame:

        # 팩터의 시그널
        self.signal = signal

        # 연율화 패러미터
        self.param = param

        # 총 투자자산의 개수
        self.noa = len(rebal_price.columns)

        # 리밸런싱별 수익률
        self.rets = self.signal * rebal_price.pct_change().dropna()

        # 기대수익률(연율화 수익률): 12개월
        self.er = np.array(self.rets.mean() * self.param)

        # 연율화 변동성: 12개월 
        self.vol = np.array(self.rets.std() * np.sqrt(self.param))

        # 연율화 공분산행렬: 12개월
        self.cov = self.rets.cov().dropna() * self.param

    # BETA(buy and hold 가중 함수)
    def beta(self):
        weights = self.signal.copy()
        weights.iloc[:] = 1 / self.noa

        return weights
    
    # EW(동일 비중 가중치 계산 함수)
    def ew(self):
        weights = self.signal.apply(lambda series: series / series.sum(), axis=1)

        return weights

    # EMV(역변동성)
    def emv(self):
        temp_weights = self.vol * self.signal
        weights = temp_weights.apply(lambda series: series / series.sum(), axis=1)

        return weights


class Optimization(Equalizer):
    def __init__(self, signal: pd.DataFrame, rebal_price: pd.DataFrame, param: int, call_method: str):
        super().__init__(signal, rebal_price, param)

        # 최적화 기법을 호출하면 목적함수를 자동으로 부를 수 있게 딕셔너리로 관리한다.
        # self.odjective_dict = {
        #                     'msr': [self.msr, self.optimizing_weights(self.neg_sharpe)], 
        #                     'gmv': [self.gmv, self.optimizing_weights(self.port_vol)], 
        #                     'mdp': [self.mdp, self.optimizing_weights(self.neg_div_ratio)], 
        #                     'rp': [self.rp, self.optimizing_weights(self.msd_risk)]
        #                     }

        # # scipy.optimize의 minimize 사용하기 위한 변수들
        # # 1. 목적함수: 목적함수에 따라 최적화 기법의 이름이 정해진다고 보면 된다.
        # # 목적함수에 공통으로 필요한 속성 정의
        # self.temp_weight = np.repeat(1/self.noa, self.noa)
        # self.target_risk = np.repeat(1/self.noa, self.noa)

        # # 2. 최적화의 초기갑
        # self.init_guess = np.repeat(1/self.noa, self.noa)

        # # 3. method : 사용할 알고리즘(솔버)를 나타내는 문자열
        # self.method = 'SLSQP'

        # # 4. bounds, constraints : 구속최적화 문제에서의 구속조건을 부과하
        # self.weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        # self.bounds = ((0.0, 1.0)) * self.noa
        
        ###### test
        self.call_method = call_method
        self.rebal_price = rebal_price
        
        add_total_signal = self.signal.copy()
        add_total_signal['total_signal'] = add_total_signal.sum(axis=1)
        self.add_total_signal = add_total_signal
        
    # opt_processing
    def opt_processing(self, target_assets, present_date):
        
        method = self.call_method
        past = pd.Timestamp(present_date) - DateOffset(years = 1)
        present = pd.Timestamp(present_date) 
        
        rets = self.rebal_price.loc[past : present, target_assets].pct_change()
        er = np.array(rets.mean() * self.param)
        vol = np.array(rets.std() * np.sqrt(self.param))
        cov = rets.cov() * self.param
        
        init_guess = np.repeat(1/len(target_assets), len(target_assets))
        bounds = ((0,1) for i in range(len(target_assets)))
        weights_sum_to_1 = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}) 

        if method == 'msr':
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
        
        elif method == 'gmv':
            def port_vol(weights, cov):
                return np.sqrt(weights.T @ cov @ weights)
            
            weights = minimize(port_vol, 
                                init_guess,
                                args=(cov), 
                                method='SLSQP', 
                                constraints=(weights_sum_to_1,), 
                                bounds=bounds
                                )

            return weights.x
        
        elif method == 'mdp':
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

        elif method == 'rp':
            target_risk = np.repeat(1/len(target_assets), len(target_assets))
            
            def msd_risk(weights, target_risk, cov):
                port_var = weights.T @ cov @ weights
                marginal_contribs = cov @ weights
                risk_contribs = np.multiply(marginal_contribs, weights.T) / port_var
                w_contribs = risk_contribs
                return ((w_contribs - target_risk)**2).sum()
            
            weights = minimize(msd_risk, 
                                init_guess,
                                args=(target_risk, cov), 
                                method='SLSQP',
                                constraints=(weights_sum_to_1,),
                                bounds=bounds)
            return weights.x
        
    def target_assets(self, series): 
        signal = series.loc["total_signal"]
        present_date = series["index"]
        tickers = self.add_total_signal.columns.drop('total_signal')
        #print(tickers)
    
        if signal:
            target_assets = series.loc[tickers].sort_values().iloc[-int(signal):].index.tolist()
            result = self.opt_processing(target_assets, present_date)
            return pd.Series(result, index = target_assets)

    def run(self):
        final_weight = self.add_total_signal.reset_index().apply(self.target_assets, axis=1).fillna(0)
        final_weight.index = self.add_total_signal.index
        final_weight = final_weight.iloc[self.param:,]
        return final_weight
    
# target_weight.index = test_rel_signal.index
    # # optimizing_weights: minimize의 결과를 weight으로 받는다.
    # def optimizing_weights(self, objective_func) -> pd.DataFrame:
    #     """_summary_

    #     Args:
    #         objective_func (function): odjective_dict의 value의 2번째 값에 저장된 함수

    #     Returns:
    #         pd.DataFrame: 최적화가 완료 된 투자비중 df
    #     """
        
    #     optimized_weights = minimize(
    #                                 objective_func,
    #                                 self.init_guess,
    #                                 args=(self.er, self.cov),
    #                                 method=self.method,
    #                                 constraints=(self.weights_sum_to_1),
    #                                 bounds=self.bounds
    #                                 )
    #     return optimized_weights.x
    
    # # 목적함수 종류: 마이너스 샤프비율, 포트폴리오 변동성, 마이너스 분산비율, 목표 위험 기여도와의 오차 최소화
    # # 1.1) MSR(샤프비율 최대화)의 목적함수: 마이너스 샤프비율 
    # def neg_sharpe(self):
    #     # portfolio returns
    #     port_returns = self.temp_weight.T @ self.er

    #     # portfolio volatility
    #     port_vol = np.sqrt(self.temp_weight.T @ self.cov @ self.temp_weight)

    #     # portfolio sharpe ratio
    #     port_sharpe = port_returns / port_vol

    #     return port_sharpe

    # # 1.2) GMV(최소 변동성)의 목적함수: 포트폴리오 변동성
    # def port_vol(self):
    #     # portfolio volatility
    #     port_vol = np.sqrt(self.temp_weight.T @ self.cov @ self.temp_weight)

    #     return port_vol

    # # 1.3) MDP(최대 분산비율)의 목적함수: 마이너스 분산비율
    # def neg_div_ratio(self):
    #     # weighted_vol
    #     weighted_vol = self.temp_weight.T @ self.vol

    #     # portfolio volatility
    #     port_vol = np.sqrt(self.temp_weight.T @ self.cov @ self.temp_weight)

    #     return - weighted_vol / port_vol

    # # 1.4) RP(리스크 패리티)의 목적함수: 목표 위험 기여도와의 오차 최소화
    # def msd_risk(self):

    #     port_var = self.temp_weight.T @ self.cov @ self.temp_weight
    #     marginal_contribs = self.cov @ self.temp_weight

    #     risk_contribs = np.multiply(
    #         marginal_contribs, self.temp_weight.T) / port_var

    #     w_contribs = risk_contribs
    #     return ((w_contribs - self.target_risk)**2).sum()

    # # MSR(샤프비율 최대화)
    # def msr(self, optimizing_weights) -> pd.DataFrame:
        
    #     optimized_weights = optimizing_weights

    #     return optimized_weights

    # # GMV(최소 변동성)
    # def gmv(self):

    #     weights = self.optimized_weights

    #     return weights.x

    # # MDP(최대 분산비율)
    # def mdp(self):

    #     weights = self.optimized_weights

    #     return weights.x

    # # RP(리스크 패리티)의 목적함수: 목표 위험 기여도와의 오차 최소화
    # def rp(self):

    #     weights = self.optimized_weights

    #     return weights.x
    

    # def run(self, series): 
    #     signal = series.loc["_signal"]
    #     present_date = series["index"]
    #     tickers = self.rebal_price.columns.drop('_signal')

    #     if signal:
    #         target_assets = series.loc[tickers].sort_values().iloc[-int(signal):].index.tolist()
    #         weights = self.run(target_assets, present_date)
    #         return pd.Series(weights, index = target_assets)


# from pandas.tseries.offsets import *

# def res_cal(target_assets, present_date) : 
#     past = pd.Timestamp(present_date) - DateOffset(years = 1)
#     present = pd.Timestamp(present_date)
#     print(past)
#     print(present)
#     ret_daily = test_rebal_price_df.loc[past : present, target_assets].pct_change()
#     cov_daily = ret_daily.cov()
        
#     n_assets = len(target_assets)
#     covmat=cov_daily*250
#     weights =np.ones(n_assets)/n_assets
#     bnds = ((0,1) for i in range(n_assets))
#     cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}) 

#     def obj_variance(weights, covmat):
#         return np.sqrt(weights.T @ covmat @ weights)
#     res = minimize(obj_variance, weights,(covmat), method='SLSQP', bounds=bnds, constraints=cons)
#     return res.x


