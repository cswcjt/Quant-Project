# 패키지 임포트

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import *

from scipy.optimize import minimize
from scipy.stats import norm


class Equalizer:
    """
    횡적 배분 모델들 중 동일비중 방법론에 기반한 모델들
    
    사용 예시:
    test_ew_weight = Equalizer(signal=test_rel_signal, rebal_price=test_rebal_price_df, param=12)
    ew_weight = test_ew_weight.ew()
    """
    
    def __init__(self, signal: pd.DataFrame, rebal_price: pd.DataFrame, param: int) -> pd.DataFrame:
        """__init__
        
        Args:
            signal (pd.DataFrame): 팩터를 적용한 결과로 각 자산에 대한 투자 여부를 알려주는 df
            rebal_price (pd.DataFrame): 리밸런싱 날의 각 자산의 종가 정보를 갖고 있는 df
            param (int): 연율화를 위한 상수값 설정

        Returns:
            pd.DataFrame: 각 자산의 투자비중을 알려주는 df
        """

        # 팩터의 시그널
        self.signal = abs(signal)

        # 총 투자자산의 개수
        self.noa = len(rebal_price.columns)

        # 리밸런싱별 수익률
        self.rets = self.signal * rebal_price.pct_change().dropna()
        
        # 연율화 상수
        self.param = param

        # 기대수익률(연율화 수익률): 12개월
        self.er = np.array(self.rets.mean() * self.param)

        # 연율화 변동성: 12개월 
        self.vol = np.array(self.rets.std() * np.sqrt(self.param))

        # 연율화 공분산행렬: 12개월
        self.cov = self.rets.cov().dropna() * self.param

    def eps(self, weights, eps=1e-6):
        data = weights.copy()
        eliminate = lambda x: 0 if abs(x)<eps else x
        if isinstance(data, pd.DataFrame):
            data = data.applymap(eliminate)
        elif isinstance(data, pd.Series):
            data = data.apply(eliminate)
        return data

    # BETA(buy and hold 가중 함수)
    def beta(self) -> pd.DataFrame:
        """beta

        Returns:
            weights: 시그널을 무시하고 모든 자산에 동일한 비중으로 투자할 때의 weight df -> 벤치마크로 사용하기 위해 만듬
            밑에서 부터는 시그널이 존재하는 자산에만 비중을 산출하는 방법론들임
        """
    
        weights = self.signal.replace({0: 1/self.noa, 1: 1/self.noa})
    
        return weights
    
    # EW(동일 비중 가중치 계산 함수)
    def ew(self) -> pd.DataFrame:
        """ew

        Returns:
            weights: 투자 시그널이 존재하는 종목에만 동일한 비중으로 투자할 때의 weight df
        """
        weights = self.signal.apply(lambda series: series / series.sum(), axis=1)
        
        weights = self.eps(weights)
        
        return weights

    # EMV(역변동성)
    def emv(self) -> pd.DataFrame:
        """emv

        Returns:
            weights: 투자 시그널이 존재하는 종목들의 변동성의 역가중으로 투자 비중을 산출한 weight df
        """
        temp_weights = self.vol * self.signal
        weights = temp_weights.apply(lambda series: series / series.sum(), axis=1)
        
        weights = self.eps(weights)

        return weights


class Optimization(Equalizer):
    """
    횡적 배분 모델들 중 최적화 기법을 사용하는 모델들
    scipy.optimize의 minimize 함수를 사용해야 하기 때문에 Equalizer 클래스와 다르게 만들었음
    (refactoring 환영합니다..)
    
    사용 예시: 
    msr = Optimization(signal=test_rel_signal, rebal_price=test_rebal_price_df, param=12, call_method='msr')
    msr.weight = msr.run()
    
    동작 순서: run() -> target_assets() -> opt_processing() -> minimize()
    """
    
    def __init__(self, signal: pd.DataFrame, rebal_price: pd.DataFrame, param: int, call_method: str):
        super().__init__(signal, rebal_price, param)
        """__init__
        call_method: 사용할 모델의 이름
        나머지는 Equalizer와 동일
        """

        self.call_method = call_method
        self.rebal_price = rebal_price
        
        # 각 리밸런싱 날에 시그널이 몇 개가 존재하는 알려주는 컬럼 추가
        self.signal = abs(signal)
        add_total_signal = self.signal.copy()
        add_total_signal['total_signal'] = add_total_signal.sum(axis=1)
        self.add_total_signal = add_total_signal
        
    # opt_processing
    def opt_processing(self, target_assets: list, present_date: pd.DatetimeIndex) -> pd.DataFrame:
        """opt_processing: 최적화 함수들이 작동하는 파트

        Args:
            target_assets (_type_): 시그널이 존재하는 티커들
            present_date (_type_): 현재 날짜를 기준으로 1년(12개월) 기대수익률, 변동성, 공분산을 구하기 위해 필요

        Returns:
            weights: 최적화의 결과물인 투자비중 df
        """
        
        # 사용할 모델 이름
        method = self.call_method
        
        # 1년(12개월) 수익률, 기대수익률, 변동성, 공분산을 구하기 위한 시간정보 
        past = pd.Timestamp(present_date) - DateOffset(years = 1)
        present = pd.Timestamp(present_date) 
        
        # 1년간의 수익률을 기준으로 기대수익률, 변동성, 공분산을 구한다  
        rets = self.rebal_price.loc[past : present, target_assets].pct_change()
        er = np.array(rets.mean() * self.param)
        vol = np.array(rets.std() * np.sqrt(self.param))
        cov = rets.cov() * self.param
        
        # minimize 함수 사용을 위한 초기 설정들: init_guess, bounds, weights_sum_to_1
        # init_guess: 초기 비중 설정 -> 최적화의 대상
        # bounds: 최적화할 비중이 몇 개인지 알려줘야한다
        # weights_sum_to_1: 최적화가 완료된 각각의 비중들의 합이 1이 되도록 제한 설정
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
        
    def target_assets(self, series: pd.Series) -> pd.Series: 
        """target_assets: 투자 자산의 티커를 찾는 함수

        Args:
            series: apply 함수에 사용하기 위해 필요한 변수

        Returns:
            pd.Series(result, index = target_assets): 
            result: 최적화에 따른 투자비중
            index: 투자종목의 티커명
        """
        signal = series.loc["total_signal"]
        present_date = series["index"]
        tickers = self.add_total_signal.columns.drop('total_signal')
    
        if signal:
            target_assets = series.loc[tickers].sort_values().iloc[-int(signal):].index.tolist()
            result = self.opt_processing(target_assets, present_date)
            return pd.Series(result, index = target_assets)

    def run(self) -> pd.DataFrame:
        """run: apply함수에 target_assets 함수를 적용하는 파트

        Returns:
            cs_weight: 최적화가 끝난 투자 비중 df
        """
        cs_weight = self.add_total_signal.reset_index().apply(self.target_assets, axis=1).fillna(0)
        cs_weight.index = self.add_total_signal.index
        cs_weight = cs_weight.iloc[self.param:,]
        final_weight = self.eps(cs_weight)
        
        return final_weight

