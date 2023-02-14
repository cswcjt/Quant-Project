import pandas as pd
from pandas.tseries.offsets import DateOffset

import sys
sys.path.append('/Users/jtchoi/Library/CloudStorage/GoogleDrive-jungtaek0227@gmail.com/My Drive/quant/Quant-Project/quant')
from backtest.metric import Metric
from backtest.factor_backtest import FactorBacktest
from price.price_processing import add_cash, rebal_dates, calculate_portvals, port_rets

class MarketBacktest(FactorBacktest):
    def __init__(self, 
                start_date: str, 
                end_date: str, 
                rebal_freq: str,
                factor: str, 
                cs_model: str,
                risk_tolerance: str, 
                all_assets: pd.DataFrame, 
                business_cycle: pd.DataFrame,
                alter_asset_list: list=['TLT', 'GSG', 'VNQ', 'UUP'], 
                benchmark_name: str='SPY', 
                ts_model: str='ew'
                ):
        super().__init__(start_date, 
                        end_date, 
                        rebal_freq, 
                        factor, 
                        cs_model,
                        risk_tolerance, 
                        all_assets, 
                        business_cycle,
                        alter_asset_list, 
                        benchmark_name, 
                        ts_model 
                        )
        
        # 시황 데이터프레임
        self.business_cycle_df = business_cycle
        
        # FactorBacktest에서 선택한 변수에 따른 팩터의 누적 수익률
        self.factor_rets_df = FactorBacktest(start_date=self.start_date,
                                            end_date=self.end_date,
                                            rebal_freq=self.rebal_freq,
                                            factor=self.factor,
                                            cs_model=self.cs_model,
                                            risk_tolerance=self.risk_tolerance,
                                            all_assets=self.all_assets_df,
                                            business_cycle=self.business_cycle).port_return('cs_weight')
        
        # self.rets_with_regime: 팩터의 월별 수익률과 시황 지표를 합친 데이터프레임
        rebal_dates_list = rebal_dates(self.factor_rets_df, period='month')
        self.month_rets_df = self.factor_rets_df.resample('M').last().pct_change()
        self.month_rets_df.index = rebal_dates_list
        
        self.business_cycle_df = self.business_cycle_df.loc[rebal_dates_list[0] - DateOffset(day=1):]
        self.business_cycle_df.index = rebal_dates_list
        
        self.rets_with_regime = pd.concat([self.month_rets_df, self.business_cycle_df], axis=1)
        self.rets_with_regime.fillna(0, inplace=True)
        self.rets_with_regime.columns = [f'{self.factor}', 
                                        'deflation', 
                                        'inflation', 
                                        'recovery',  
                                        'expansion'
                                        ]
        
        self.inspect_factor_dict = self.inspect_factor()
        
        # self.multi_asset_df: 대체자산의 월별 수익률과 rets_with_regime 합친 데이터프레임
        target_list = (self.alter_asset_list + [self.benchmark_name])
        self.temp = self.all_assets_df[target_list]
        self.temp = self.temp.resample('M').last().pct_change()
        
        self.multi_asset_df = pd.merge(self.rets_with_regime, 
                                self.temp.loc[self.rets_with_regime.iloc[0].name:], 
                                left_index=True, 
                                right_index=True, 
                                how='left')\
                                .dropna()

        self.inspect_multi_asset_dict = self.inspect_multi_asset()
        
        # self.target_price_df: 시황별 샤프비율이 높은 투자를 가정한 포트폴리오 생성에 씌이는 데이터프레임
        # regime_signal 생성에 사용
        self.target_price_df = add_cash(self.multi_asset_df, 252, 0.04)
        self.target_price_df.index.name = 'date_time'
        
    def inspect_factor(self) -> dict:
        """시황별 팩터의 성과지표

        Returns:
            dict: 시황을 key로 갖고 팩터의 성과지표를 value로 갖는 딕셔너리
        """
        each_regime = self.business_cycle_df.columns
        
        metric_result = {}
        for regime in each_regime:
            metric_result[regime] = Metric(portfolio=(1 + 
                                                        self.rets_with_regime[self.factor]\
                                                        .mul(self.rets_with_regime[regime], 
                                                        axis=0)
                                                        )
                                                        .cumprod(), 
                                                        freq='month')\
                                                        .numeric_metric()
        
        return metric_result
    
    def inspect_multi_asset(self) -> dict:
        """시황별 대체자산들의 성과지표

        Returns:
            dict: 시황을 key로 갖고 대체자산들의 성과지표를 value로 갖는 딕셔너리
        """
        each_regime = self.business_cycle_df.columns
        tickers = self.multi_asset_df.columns.difference(each_regime)
        
        metric_dict = {}
        for regime in each_regime:
            
            temp_dict = {}
            for ticker in tickers:
                temp_dict[ticker]\
                    = Metric(portfolio=(1 + 
                                        self.multi_asset_df[ticker]\
                                        * self.multi_asset_df[regime]
                                        )\
                                        .cumprod(), 
                                        freq='month')\
                                        .numeric_metric()
            metric_dict[regime] = temp_dict
            
        return metric_dict
    
    def returns_stats(self) -> dict:
        """수익률의 기초통계량과 샤프지수

        Returns:
            dict: 시황을 key로 갖고 자산들의 수익률의 기초통계량과 샤프지수를 value로 갖는 딕셔너리
        """
        each_regime = self.business_cycle_df.columns
        tickers = self.multi_asset_df.columns.difference(each_regime)

        metric_dict = {}
        for regime in each_regime:
            
            temp_dict = {}
            for ticker in tickers:
                df = self.multi_asset_df.loc[self.multi_asset_df[regime] == 1, ticker]
                stats = df.describe().to_dict()
                stats['Sharpe'] = stats['mean'] / stats['std']
                temp_dict[ticker] = stats
                
            metric_dict[regime] = temp_dict
        
        return metric_dict
    
    def best_sharpe(self) -> pd.DataFrame:
        """
        시황별 sharpe ratio가 가장 높은 factor를 찾기 위한 데이터프레임
        """
        each_regime = self.business_cycle_df.columns
        tickers = self.multi_asset_df.columns.difference(each_regime)
        best_sharpe = self.returns_stats()
        
        best_sharpe_dict = {}
        for regime in each_regime:
            
            temp_dict = {}
            for factor in tickers:
                temp_dict[factor] = best_sharpe[regime][factor]['Sharpe']
                
            best_sharpe_dict[regime] = temp_dict
            
        best_sharpe_df = pd.DataFrame(best_sharpe_dict)
        best_sharpe_df = best_sharpe_df.T
        best_sharpe_df['best_sharpe'] = best_sharpe_df.idxmax(axis=1)
        best_sharpe_df['best_sharpe_ratio'] = best_sharpe_df.max(axis=1)
    
        return best_sharpe_df
    
    def target_assets(self) -> dict:
        """best_sharpe()의 결과에 따라 시황별 투자자산을 정함

        Returns:
            dict: 시황을 key로 갖고 투자자산을 value로 갖는 딕셔너리
        """
        sharpe_df = self.best_sharpe()
        target_assets = sharpe_df['best_sharpe'].to_dict()
        return target_assets
    
    def regime_signal(self) -> pd.DataFrame:
        """샤프비율에 기반해 시황별 투자자산에 대한 시그널을 생성

        Returns:
            pd.DataFrame: 시황별 투자자산에 대한 시그널을 담은 데이터프레임
        """
        regime_asset_dict = self.target_assets()
        ma_regime_df = self.multi_asset_df
        
        for key, value in regime_asset_dict.items():
            ma_regime_df.loc[ma_regime_df[key] == 1, value] =1
            
        regime_signal = (ma_regime_df == 1) * 1 
        regime_signal.drop(regime_asset_dict.keys(), axis=1, inplace=True)
        regime_signal['CASH'] = (regime_signal.sum(axis=1) == 0).astype(int)
        
        return regime_signal
    
    def cross_weight(self) -> pd.DataFrame:
        return super().cross_weight()
    
    def time_weight(self) -> pd.Series:
        return super().time_weight()
    
    def port_return(self) -> pd.DataFrame:
        """포트폴리오의 월별 수익률

        Returns:
            pd.DataFrame: 포트폴리오의 월별 수익률을 담은 데이터프레임
        """
        return super().port_return()

if __name__ == '__main__':
    path = '/Users/jtchoi/Library/CloudStorage/GoogleDrive-jungtaek0227@gmail.com/My Drive/quant/Quant-Project/quant'
    all_assets_df = pd.read_csv(path + '/alter_with_equity.csv', index_col=0)
    all_assets_df.index = pd.to_datetime(all_assets_df.index)
    all_assets_df = all_assets_df.loc['2011':,].dropna(axis=1)
    alter_asset_list=['TLT', 'GSG', 'VNQ', 'UUP']
    bs_df = pd.read_csv(path + '/business_cycle.csv', index_col=0)
    bs_df.index = pd.to_datetime(bs_df.index)
    
    test = MarketBacktest(start_date='2011-01-01', 
                        end_date='2022-12-31', 
                        rebal_freq='quarter', 
                        factor='mom',
                        cs_model='emv',
                        risk_tolerance='aggressive',
                        all_assets=all_assets_df, 
                        business_cycle=bs_df
                        )
    
    print(test.port_return())