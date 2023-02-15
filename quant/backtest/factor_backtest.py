import numpy as np
import pandas as pd
import yfinance as yf

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[2]
sys.path.append(str(PJT_PATH))
#from backtest.metric import Metric
#from metric import Metric
from scaling import annualize_scaler
from quant.price.price_processing import convert_freq, add_cash, rebal_dates, price_on_rebal, calculate_portvals, port_rets
from quant.strategy.factors.beta import BetaFactor
from quant.strategy.factors.momentum import MomentumFactor
#from quant.strategy.factors.forecast_prophet import ProphetFactor
from quant.strategy.factors.volatility import VolatilityFactor
from quant.strategy.optimize.cross_sectional import Equalizer, Optimization
from quant.strategy.optimize.time_series import TimeSeries

class FactorBacktest:
    # 초기화 함수
    def __init__(self, 
                start_date: str, 
                end_date: str, 
                rebal_freq: str,
                cs_model: str,
                risk_tolerance: str, 
                all_assets: pd.DataFrame, 
                business_cycle: pd.DataFrame,
                alter_asset_list: list=['TLT', 'GSG', 'VNQ', 'UUP'], 
                benchmark_name: str='SPY', 
                ts_model: str='ew'
                ): 
        """팩터의 성과를 분석하는 파트_
            - 대시보드에 표시될 데이터를 생성하는 파트
            - 대시보드에 필요한 변수들:
                - Time Range: start_date, end_date
                - Rebalancing Period: rebal_freq
                - Factor: factor
                - Weight: cs_model
                - Risk Tolerance: ts_model, risk_tolerance
        
            - 대시보드와 상관없는 변수들: 
                - 참고: 데이터정보와 관련된 변수명을 제외하면 모두 대시보드에 필요한 변수들
                - all_assets(자산+대체자산 데이터프레임), business_cycle(시황 데잌터프레임)
                - alter_asset_list(대체자산 이름들), benchmark_name(벤치마크 이름)
        
        Args:
            start_date (str): 투자 시작날
            end_date (str): 투자 종료날
            rebal_freq (str): 리밸런싱 주기
                - 'month', 'quarter', 'halfyear', 'year'
            all_assets (pd.DataFrame): 전체자산 데이터프레임 -> 벤치마크+대체자산+투자대상자산
            alter_asset_list (list): 대체자산 이름들
            benchmark_name (str): 벤치마크 이름
            business_cycle (pd.DataFrame): 시황 데이터프레임
            factor (str): 사용할 팩터 이름
                - 'beta', 'mom', 'prophet', 'vol' 
                - prophet은 사용법 몰라서 생략해 놓음
            cs_model (str): 자산간 투자 비중을 결정하는 모델
                - 'ew', 'emv', 'msr', 'gmv', 'mdp', 'rp' 
            ts_model (str): 현금보유비율 설정
                - 'ew' 
            risk_tolerance (str): 위험선호도
                - 'aggressive': 30, 'moderate': 50, 'conservative': 70
        """
        self.start_date = start_date
        self.end_date = end_date
        
        # 가격데이터 날짜, 리밸 설정
        self.rebal_freq = convert_freq(rebal_freq)
        
        self.business_cycle = business_cycle
        
        self.cs_model = cs_model
        self.ts_model = ts_model
        
        self.risk_tolerance = risk_tolerance
        
        self.all_assets = all_assets
        self.bs_df = business_cycle
        
        # 데이터 슬라이싱
        self.all_assets_df = self.all_assets.loc[self.start_date:self.end_date, :]
        # 시황 데이터
        self.business_cycle = self.business_cycle.loc[self.start_date:self.end_date,]
        
        # all_assets_df를 컬럼으로 슬라이싱하기 위한 파트 
        self.benchmark_name = benchmark_name 
        self.alter_asset_list = alter_asset_list
        
        # 리밸런싱 날짜 정보
        self.rebal_dates_list = rebal_dates(self.all_assets_df, self.rebal_freq,
                                            include_first_date=True)
        # 리밸런싱 날의 가격 정보
        self.price_on_rebal_df = price_on_rebal(self.all_assets_df, self.rebal_dates_list)
        
    def run(self, factor: str='mom'):
        # 팩터 선택
        self.factor = factor
        self.daily_price_df = self.all_assets_df.drop(columns=self.alter_asset_list)
        self.signal = self.factor_signal(self.factor)
            
        # 최적화(cs) 선택 및 횡적 비중 계산
        cs_dict = {'ew': Equalizer,
                   'emv': Equalizer,
                   'msr': Optimization,
                   'gmv': Optimization,
                   'mdp': Optimization,
                   'rp': Optimization
                   }
        self.cs_instance = cs_dict[self.cs_model]
        self.cs_weight = self.cross_weight()
        
        # 최적화(cs)가 끝난 포트폴리오의 수익률 계산 결과
        self.cs_port_cum_rets = self.port_return(cumulative=True)
        self.cs_port_daily_rets = self.port_return(cumulative=False)
        
        # 최적화(ts) 선택
        ts_dict = {'ew': TimeSeries}
        self.ts_instance = ts_dict[self.ts_model]
        self.ts_weight, self.ts_cs_weight = self.time_weight()
        
        # 최적화(ts)가 끝난 포트폴리오의 수익률 계산 결과
        self.ts_port_cum_rets = self.port_return('ts_weight', cumulative=True)
        self.ts_port_daily_rets = self.port_return('ts_weight', cumulative=False)
        
        self.business_cycle = self.business_cycle.loc[self.start_date:self.end_date,]
        
        return self.ts_port_daily_rets

    def factor_signal(self, factor) -> pd.DataFrame:
        """월별 시그널 생성 함수
        
        Returns:
            pd.DataFrame: 월별 시그널 데이터프레임
        """
        factor_dict = {'beta': BetaFactor,
                        'mom': MomentumFactor,
                        'prophet': 'load_csv',
                        'vol': VolatilityFactor,
                        }
        
        class_instance = factor_dict[factor]
        price_df = self.daily_price_df
        
        if self.factor == 'beta':
            factor_signal = class_instance(equity_with_benchmark=price_df,
                                           benchmark_ticker=self.benchmark_name
                                           ).signal()
            
            return factor_signal
        
        elif self.factor == 'mom':
            factor_signal = class_instance(price_df).signal()
            
            return factor_signal

        elif self.factor == 'vol':
            factor_signal = class_instance(price_df).signal()
            
            return factor_signal 
        
        elif self.factor == 'prophet':
            factor_signal = pd.read_csv(PJT_PATH / 'quant'/ 'strategy' / 'factors' / 'data' / 'prophet_signal.csv',
                                        index_col=0,
                                        parse_dates=True)
            
            return factor_signal
        
    def cross_weight(self) -> pd.DataFrame:
        """월별 포트폴리오의 횡적 가중치 계산 함수
        
        Returns:
            pd.DataFrame: 월별 포트폴리오 가중치 데이터프레임
        """        
        return self.cs_instance(self.signal, self.daily_price_df, 
                                self.rebal_freq, self.cs_model
                                ).weight()
        
    def time_weight(self) -> pd.Series:
        return self.ts_instance(self.cs_port_cum_rets, 
                                self.cs_weight, 
                                self.risk_tolerance, 
                                self.ts_model
                                ).weight()

    def port_return(self, weight_name: str='cs_weight',
                    freq: str='D', cumulative: bool=True, 
                    long_only: bool=True, yearly_rfr: float=0.03
                    ) -> pd.Series:
        """_summary_

        Args:
            cumulative (bool, optional): 
                - port_rets의 변수. True: 누적수익률, False: 일별 수익률. Defaults to True.
            long_only (bool, optional): 
                - calculate_portvals의 변수. Defaults to True.

        Returns:
            pd.Series: 수익률 pd.Series
        """
        if weight_name == 'cs_weight':
            weight = self.cross_weight()
            port_value = calculate_portvals(self.daily_price_df, 
                                            weight, 
                                            self.signal, 
                                            long_only)

        elif weight_name == 'ts_weight':
            _, ts_cs_weight = self.time_weight()
            weight = ts_cs_weight
            price_df = add_cash(self.daily_price_df, 252, 0.03)
            port_value = calculate_portvals(price_df, ts_cs_weight,
                                            self.signal, long_only)
            
        return port_rets(port_value, cumulative)
        
    def factor_rets(self, factors: list) -> pd.DataFrame:
        port_rets_dict = {}
        
        for factor in factors:
            port_rets_dict[factor] = self.run(factor=factor)
            
        df = pd.DataFrame(port_rets_dict)
        return df
    
    def mutually_exclusive(self, factors: list) -> pd.DataFrame:
        return self.factor_rets(factors=factors).corr(numeric_only=True)
    
if __name__ == '__main__':
    path = PJT_PATH / 'quant'
    
    all_assets_df = pd.read_csv(path / 'alter_with_equity.csv', index_col=0)
    all_assets_df.index = pd.to_datetime(all_assets_df.index)
    all_assets_df = all_assets_df.loc['2011':,].dropna(axis=1)
    
    alter_asset_list=['TLT', 'GSG', 'VNQ', 'UUP']
    
    bs_df = pd.read_csv(path / 'business_cycle.csv', index_col=0)
    bs_df.index = pd.to_datetime(bs_df.index)

    #prophet_signal = pd.read_csv(path + '/prophet_signal.csv', index_col=0)
    #print(prophet_signal)
    #print(all_assets_df.drop(columns=alter_asset_list))
    
    test = FactorBacktest(start_date='2011-01-01', 
                          end_date='2022-12-31', 
                          rebal_freq='month', 
                          cs_model='ew', 
                          risk_tolerance='moderate',
                          all_assets=all_assets_df, 
                          business_cycle=bs_df
                          )
    
    #print(test.factor_signal())
    #print(test.cross_weight())
    #print(test.port_return('cs_weight'))
    print(test.factor_rets(['mom', 'beta', 'vol', 'prophet']))
    # print(FactorBacktest.mutually_exclusive(FactorBacktest, 
    #                                         factors=['mom', 
    #                                                 'beta', 
    #                                                 'vol', 
    #                                                 'prophet']))


