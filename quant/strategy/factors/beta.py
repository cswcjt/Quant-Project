import pandas as pd
import statsmodels.api as sm

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[3]
sys.path.append(str(PJT_PATH))
from quant.price.price_processing import rebal_dates
from scaling import annualize_scaler

class BetaFactor:
    """_summary_
    자산의 베타값을 계산하는 클래스
    각 자산군의 벤치마크를 설정하고, 해당 벤치마크에 포함된 종목들의 가격 데이터프레임을 입력하면 베타값을 계산한다.
    """
    
    def __init__(self, equity_with_benchmark: pd.DataFrame, 
                 freq: str='month',
                benchmark_ticker: str='SPY', 
                intercept: int=1, 
                n_sel: int=20,
                lookback_window: int=1) -> pd.DataFrame:
        """_summary_
        Args:
            equity_with_benchmark (pd.DataFrame): 
                - 벤치마크와 벤치마크에 포한된 종목들의 일별 가격 데이터프레임
            benchmark_ticker (str): 
                - 벤치마크의 티커
            intercept (int, optional): 
                - 선형회귀의 y절편 값. Defaults to 1.
            n_sel (int, optional):
                - 베타 상위 n개 종목에 투자. Defaults to 20.
            lookback_window (int, optional): 
                - lookback window. Defaults to 12.
        Returns:
            pd.DataFrame: 종목별 베타값
        """
        
        # 벤치마크와 벤치마크에 포함된 종목들의 가격 데이터프레임
        self.price_df = equity_with_benchmark
        self.first_date = self.price_df.iloc[0].name
        self.last_date = self.price_df.iloc[-1].name
        
        self.benchmart_ticker = benchmark_ticker
        self.benchmark_df = pd.DataFrame({f'{self.benchmart_ticker}': 
                                            self.price_df[self.benchmart_ticker]})
        
        # 한달마다의 마지막 날짜 & lookback window = 1년
        monthly_index = rebal_dates(self.price_df, period=freq)
        self.lookback_window = lookback_window * annualize_scaler(freq)
        self.monthly_index = monthly_index[(self.lookback_window - 1):]
        
        # 수익률 데이터프레임
        self.rets = self.price_df.pct_change().dropna()
        
        # 선형회귀의 y절편 값
        self.intercept = intercept
        
        # 베타 상위 n개 종목에 투자
        self.n_sel = n_sel
        
    def cal_beta(self) -> pd.DataFrame:
        """_summary_
        Returns:
            pd.DataFrame: 종목별 베타값 계산
        """
        
        rets = self.rets
        rets['intercept'] = self.intercept
        
        beta_dict = {}
        for col in rets:
            
            beta_dict[col] = sm.OLS(rets[col], 
                                    rets[[self.benchmart_ticker, 'intercept']]
                                    ).fit().params[0]
            
        beta_df = pd.DataFrame({ticker: data 
                                for ticker, data 
                                in beta_dict.items()}, 
                                index=['beta']).T
        
        beta_df.sort_values(by='beta', ascending=False)

        return beta_df
    
    def beta(self) -> pd.DataFrame:
        """_summary_
        Returns:
            pd.DataFrame: 베타 시그널 df
        """
        
        def assign_value(series):
            isin_series_index = series.loc[beta_index].index
            series1 = pd.Series([1] * len(isin_series_index), index=isin_series_index.tolist())
        
            not_isin_beta_index = series.index[~series.index.isin(beta_index)]
            series2 = pd.Series([0] * len(not_isin_beta_index), index=not_isin_beta_index.tolist())
        
            return pd.concat([series1, series2]).reindex(series.index).sort_index()
        
        signal_list = []

        for index in self.monthly_index:
            df = self.price_df.loc[:index, :]
            df = df.iloc[-252:, :] if len(df) >= 252 else df
            
            beta_index = BetaFactor(equity_with_benchmark=df, 
                                    benchmark_ticker=self.benchmart_ticker)\
                                    .cal_beta()\
                                    .sort_values(by='beta', ascending=False)\
                                    .head(self.n_sel)\
                                    .index

            signal_list.append(df.resample('M').last().apply(assign_value, axis=1).iloc[-1])

        try: 
            signal_df = pd.concat(signal_list, axis=1).T 
            signal_df.index = self.monthly_index
            return pd.concat(signal_df, axis=1).T
        
        except TypeError:
                signal_df = pd.concat(signal_list, axis=1).T
                signal_df.index = self.monthly_index
                return signal_df
    
    def signal(self):
        return self.beta()
    
    
if __name__ == '__main__':
    path = '/Users/jtchoi/Library/CloudStorage/GoogleDrive-jungtaek0227@gmail.com/My Drive/quant/Quant-Project/quant'
    equity_df = pd.read_csv(path + '/alter_with_equity.csv', index_col=0)
    print(equity_df.tail())
    equity_df.index = pd.to_datetime(equity_df.index)
    equity_universe = equity_df.loc['2011':,].dropna(axis=1)
    
    signal = BetaFactor(equity_universe, 'quarter').signal()
    print(signal.sum(axis=1))