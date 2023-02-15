# 패키지 임포트
import sys
import numpy as np
import pandas as pd
import yfinance as yf

from pandas.tseries.offsets import *
from pathlib import Path

PJT_PATH = Path(__file__).parents[3]
sys.path.append(str(PJT_PATH))

from scaling import convert_freq, annualize_scaler
from quant.price.price_processing import rebal_dates

from scipy.optimize import minimize
from scipy.stats import norm

class Equalizer:
    """
    횡적 배분 모델들 중 동일비중 방법론에 기반한 모델들
    
    사용 예시:

    """
    
    def __init__(self, signal: pd.DataFrame, 
                price_df: pd.DataFrame, 
                select_period: str, 
                call_method: str
                ) -> pd.DataFrame:
        """__init__
        
        Args:
            signal (pd.DataFrame): 팩터를 적용한 결과로 각 자산에 대한 투자 여부를 알려주는 df
            price_df (pd.DataFrame): 종가 정보를 갖고 있는 df
            select_period (int): 리밸런싱 주기 정보
            call_method (str): 최적화 방법론 선택

        Returns:
            pd.DataFrame: 각 자산의 투자비중을 알려주는 df
        """
        
        # 팩터의 시그널
        self.signal = abs(signal)
        self.signal.index = pd.to_datetime(self.signal.index)
        
        # 벤치마크와 벤치마크에 포함된 종목들의 가격 데이터프레임
        self.price_df = price_df
        self.daily_rets_df = self.price_df.pct_change().fillna(0)
        
        # 유동적인 리밸런싱 주기를 위한 인덱스
        self.select_period = select_period
        self.price_df.index.name = 'date_time'
        self.rebal_date_list = rebal_dates(self.signal, self.select_period)
        self.signal_on_rebal = self.signal.loc[self.rebal_date_list, :]

        # variable_setting()의 결과물
        self.er_df = self.variable_setting()[0]
        self.vol_df = self.variable_setting()[1]
        self.cov_df = self.variable_setting()[2]
        
        # call_method
        self.call_method = call_method
        
    def variable_setting(self):
        
        er_list = []
        std_list = []
        cov_list = []
        for index in self.rebal_date_list:

            rets = self.daily_rets_df.loc[:index, :]
            rets = rets.iloc[-252:, :] if len(rets) >= 252 else rets

            er = rets.mean() * 252
            er_list.append(er)

            std = rets.std() * np.sqrt(252)
            std_list.append(std)

            cov = rets.cov() * 252
            cov_list.append(cov)
            
        er_df = pd.concat(er_list, axis=1).T 
        er_df.index = self.signal_on_rebal.index

        vol_df = pd.concat(std_list, axis=1).T 
        vol_df.index = self.signal_on_rebal.index

        cov_df = pd.concat(cov_list, axis=1).T

        return er_df, vol_df, cov_df
        

    def eps(self, weights, eps=1e-6):
        data = weights.copy()
        eliminate = lambda x: 0 if abs(x)<eps else x
        if isinstance(data, pd.DataFrame):
            data = data.applymap(eliminate)
        elif isinstance(data, pd.Series):
            data = data.apply(eliminate)
        return data
    
    # BETA(비교를 위해 계산)
    def beta(self) -> pd.DataFrame:
        """beta

        Returns:
            weights: 모든 종목에만 동일한 비중으로 투자할 때의 weight df
        """
        beta_signal = self.signal_on_rebal.copy()
        beta_signal = beta_signal.replace({0:1})
        
        weights = beta_signal.apply(lambda series: series / series.sum(), axis=1)
        weights = self.eps(weights)
        
        return weights
        
    # EW(동일 비중 가중치 계산 함수)
    def ew(self) -> pd.DataFrame:
        """ew

        Returns:
            weights: 투자 시그널이 존재하는 종목에만 동일한 비중으로 투자할 때의 weight df
        """
        weights = self.signal_on_rebal.apply(lambda series: series / series.sum(), axis=1)
        weights = self.eps(weights)
        
        return weights

    # EMV(역변동성)
    def emv(self) -> pd.DataFrame:
        """emv

        Returns:
            weights: 투자 시그널이 존재하는 종목들의 변동성의 역가중으로 투자 비중을 산출한 weight df
        """
        temp_weights = self.vol_df * self.signal_on_rebal
        
        weights = temp_weights.apply(lambda series: series / series.sum(), axis=1)
        weights = self.eps(weights)

        return weights
    
    def weight(self):
        
        if self.call_method == 'beta':
            return self.beta()
        
        elif self.call_method == 'ew':
            return self.ew()
        
        elif self.call_method == 'emv':
            return self.emv()

class Optimization(Equalizer):
    """
    횡적 배분 모델들 중 최적화 기법을 사용하는 모델들
    scipy.optimize의 minimize 함수를 사용해야 하기 때문에 Equalizer 클래스와 다르게 만들었음
    (refactoring 환영합니다..)
    
    사용 예시: 
    msr = Optimization(signal=test_rel_signal, rebal_price=test_rebal_price_df, param=12, call_method='msr')
    msr.weight = msr.weight()
    
    동작 순서: weight() -> target_assets() -> opt_processing() -> minimize()
    """
    
    def __init__(self, signal: pd.DataFrame, 
                price_df: pd.DataFrame, 
                select_period: str, 
                call_method: str):
        super().__init__(signal, price_df, select_period, call_method)
        """__init__
        call_method: 사용할 모델의 이름
        나머지는 Equalizer와 동일
        """

        # 사용할 모델의 이름
        self.call_method = call_method

        # 각 리밸런싱 날에 시그널이 몇 개가 존재하는 알려주는 컬럼 추가        
        add_total_signal = self.signal.copy()
        add_total_signal['total_signal'] = add_total_signal.sum(axis=1)
        self.add_total_signal = add_total_signal
        self.signal_on_rebal = self.add_total_signal.loc[self.rebal_date_list]
        
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
        
        # # # 1년간의 수익률을 기준으로 기대수익률, 변동성, 공분산을 구한다  
        rets = self.daily_rets_df.loc[past : present, target_assets]
        er = np.array(rets.mean() * 252)
        vol = np.array(rets.std() * np.sqrt(252))
        cov = rets.cov() * 252

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

    def weight(self) -> pd.DataFrame:
        """run: apply함수에 target_assets 함수를 적용하는 파트

        Returns:
            cs_weight: 최적화가 끝난 투자 비중 df
        """
        
        cs_weight = self.signal_on_rebal.reset_index().apply(self.target_assets, axis=1).fillna(0)
        cs_weight.index = self.signal_on_rebal.index
        final_weight = self.eps(cs_weight)
        
        return final_weight


import yfinance as yf

if __name__ == '__main__':
    path = '/Users/jtchoi/Library/CloudStorage/GoogleDrive-jungtaek0227@gmail.com/My Drive/quant/Quant-Project/quant'
    equity_df = pd.read_csv(path + '/equity_universe.csv', index_col=0)
    equity_df.index = pd.to_datetime(equity_df.index)
    equity_universe = equity_df.loc['2011':,].dropna(axis=1)
    print(equity_universe)
    signal = pd.read_csv(path + '/result/mom_signal.csv', index_col=0)
    print(signal)
    ew_weight = Equalizer(signal=signal, price_df=equity_universe, select_period='quarter', call_method='ew').weight()
    #test = Optimization(signal, equity_universe, 'quarter', call_method='msr').weight()
    print(ew_weight)
    
