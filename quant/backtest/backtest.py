import pandas as pd

import sys
sys.path.append('/Users/jtchoi/Library/CloudStorage/GoogleDrive-jungtaek0227@gmail.com/My Drive/quant/Quant-Project/quant')

#from backtest.metric import Metric
from price.price_processing import convert_freq, get_price, add_cash, rebal_dates, price_on_rebal, calculate_portvals, port_rets
from strategy.factors.beta import BetaFactor
from strategy.factors.momentum import MomentumFactor
#from strategy.factors.forecast_prophet import ProphetFactor
from strategy.factors.volatility import VolatilityFactor
from strategy.optimize.cross_sectional import Equalizer, Optimization
from strategy.optimize.time_series import TimeSeries

class BackTest:
    # 초기화 함수
    def __init__(self, start_date: str, 
                end_date: str, rebal_freq: str,
                all_assets: pd.DataFrame, alter_asset_list: list,
                benchmark_name: str, business_cycle: pd.DataFrame,
                factor: str, cs_model: str,
                ts_model: str, risk_tolerance: str
                ): 
        
        # 가격데이터 날짜, 리밸 설정
        self.start_date = start_date
        self.end_date = end_date
        self.rebal_freq = convert_freq(rebal_freq)
        
        # 데이터 슬라이싱 
        self.all_assets_df = all_assets.loc[self.start_date:self.end_date,]
        self.rebal_dates_list = rebal_dates(self.all_assets_df, self.rebal_freq)
        
        # 리밸런싱 날의 가격 정보
        self.price_on_rebal_df = price_on_rebal(self.all_assets_df, self.rebal_dates_list)
        
        # all_assets_df를 컬럼으로 슬라이싱하기 위한 파트 
        self.benchmark_name = benchmark_name 
        self.alter_asset_list = alter_asset_list
        
        # 시황 데이터
        self.business_cycle = business_cycle.loc[self.start_date:self.end_date,]
        
        # 팩터 선택
        self.factor = factor
        factor_dict = {'beta': BetaFactor,
                        'mom': MomentumFactor,
                        # 'prophet': ProphetFactor(),
                        'vol': VolatilityFactor,
                        }
        self.factor_instance = factor_dict[self.factor]  
        self.daily_price_df = self.all_assets_df.drop(columns=self.alter_asset_list)
        self.signal = self.factor_signal()
        
        # 최적화(cs) 선택 및 횡적 비중 계산
        cs_dict = {'ew': Equalizer,
                    'emv': Equalizer,
                    'msr': Optimization,
                    'gmv': Optimization,
                    'mdp': Optimization,
                    'rp': Optimization
                    }
        self.cs_model = cs_model
        self.cs_instance = cs_dict[self.cs_model]
        self.cs_weight = self.cross_weight()
        
        # 최적화(cs)가 끝난 포트폴리오의 수익률 계산 결과
        self.cs_port_cum_rets = self.port_return('cs_weight', cumulative=False)
        self.cs_port_daily_rets = self.port_return('cs_weight', cumulative=False)
        
        # 최적화(ts) 선택
        ts_dict = {'ew': TimeSeries}
        self.ts_model = ts_model
        self.ts_instance = ts_dict[self.ts_model]
        self.risk_tolerance = risk_tolerance
        self.ts_weight, self.ts_cs_weight = self.time_weight()
        
        # 최적화(ts)가 끝난 포트폴리오의 수익률 계산 결과
        self.ts_port_cum_rets = self.port_return('ts_weight', cumulative=False)
        self.ts_port_daily_rets = self.port_return('ts_weight', cumulative=False)
        
        self.business_cycle = business_cycle.loc[self.start_date:self.end_date,]

    def factor_signal(self) -> pd.DataFrame:
        """월별 시그널 생성 함수
        참고: 리밸런싱 날짜에만 생성하는게 아니다.
        
        Returns:
            pd.DataFrame: 월별 시그널 데이터프레임
        """
        class_instance = self.factor_instance
        price_df = self.daily_price_df
        
        if class_instance == BetaFactor:
            factor_signal = class_instance(equity_with_benchmark=price_df, 
                                        benchmark_ticker=self.benchmark_name)\
                                        .signal()
            
            return factor_signal
        
        elif class_instance == MomentumFactor:
            factor_signal = class_instance(price_df).signal()
            
            return factor_signal
        
        # elif class_instance == ProphetFactor:
        #     pass
        
        elif class_instance == VolatilityFactor:
            factor_signal = class_instance(price_df).signal()
            
            return factor_signal 
        
    def cross_weight(self) -> pd.DataFrame:
        """월별 포트폴리오의 횡적 가중치 계산 함수
        
        Returns:
            pd.DataFrame: 월별 포트폴리오 가중치 데이터프레임
        """
        class_instance = self.cs_instance
        factor_signal = self.signal
        price_df = self.daily_price_df
        rebal_period = self.rebal_freq
        cs_model = self.cs_model 
        
        if class_instance == Equalizer:
            weight = class_instance(factor_signal, price_df, rebal_period, cs_model).weight()
            
            return weight
        
        elif class_instance == Optimization:
            weight = class_instance(factor_signal, price_df, rebal_period, cs_model).weight()
            
            return weight
        
    def time_weight(self) -> pd.Series:
        class_instance = self.ts_instance
        cs_port_cum_rets = self.cs_port_cum_rets
        cs_weight = self.cs_weight
        risk_tol = self.risk_tolerance
        call_method = self.ts_model

        if class_instance == TimeSeries:
            ts_weight, ts_cs_weight = class_instance(cs_port_cum_rets, cs_weight, risk_tol, call_method).weight()
            
            return ts_weight, ts_cs_weight

    def port_return(self, weight_name: str, 
                    cumulative: bool = True, 
                    long_only: bool = True) -> pd.Series:
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
            port_value = calculate_portvals(self.daily_price_df, weight, self.signal, long_only)
            port_returns = port_rets(port_value, cumulative)
            
            return port_returns
        
        elif weight_name == 'ts_weight':
            _, ts_cs_weight = self.time_weight()
            weight = ts_cs_weight
            price_df = add_cash(self.daily_price_df, 252, 0.03)
            port_value = calculate_portvals(price_df, ts_cs_weight, self.signal, long_only)
            port_returns = port_rets(port_value, cumulative)
            
            return port_returns
        
        # elif weight_name == 'test_weight':
        #     weight = self.cross_weight(self.signal, 
        #                                 self.daily_price_df, 
        #                                 self.rebal_freq, 
        #                                 cs_model='beta').weight()
            
        port_value = calculate_portvals(self.daily_price_df, weight, self.signal, long_only)
        port_returns = port_rets(port_value, cumulative)
        
        return port_returns

class RegimeCheck(BackTest):
    def __init__(self, start_date: str, 
                end_date: str, rebal_freq: str,
                all_assets: pd.DataFrame, alter_asset_list: list,
                benchmark_name: str, business_cycle: pd.DataFrame,
                factor: str, cs_model: str,
                ts_model: str, risk_tolerance: str
                ):
        super.__init__(start_date, end_date, 
                        rebal_freq, all_assets, 
                        alter_asset_list, benchmark_name, 
                        business_cycle, factor, 
                        cs_model, ts_model, 
                        risk_tolerance
                        )
    





if __name__ == '__main__':
    path = '/Users/jtchoi/Library/CloudStorage/GoogleDrive-jungtaek0227@gmail.com/My Drive/quant/Quant-Project/quant'
    all_assets_df = pd.read_csv(path + '/alter_with_equity.csv', index_col=0)
    all_assets_df.index = pd.to_datetime(all_assets_df.index)
    all_assets_df = all_assets_df.loc['2011':,].dropna(axis=1)
    alter_asset_list=['TLT', 'GSG', 'VNQ', 'UUP']
    bs_df = pd.read_csv(path + '/business_cycle.csv', index_col=0)
    bs_df.index = pd.to_datetime(bs_df.index)

    #print(all_assets_df.drop(columns=alter_asset_list))
    
    test = BackTest(start_date='2011-01-01', end_date='2022-12-31', 
                    rebal_freq='quarter', all_assets=all_assets_df, 
                    benchmark_name='SPY', alter_asset_list=['TLT', 'GSG', 'VNQ', 'UUP'],
                    business_cycle=bs_df, factor='mom', 
                    cs_model='emv', ts_model='ew', 
                    risk_tolerance='aggressive')
    
    
    #print(test.factor_signal())
    #print(test.cs_weight())
    #print(test.port_return())
    #print(test.time_weight())
    print(test.port_return('ts_weight'))


