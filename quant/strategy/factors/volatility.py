## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[3]
sys.path.append(str(PJT_PATH))
from quant.price.price_processing import rebal_dates
from scaling import annualize_scaler

#from quant.backtest.metric import Metric
import numpy as np
import pandas as pd

class VolatilityFactor:

    def __init__(self, price_df: pd.DataFrame,
                 freq: str='month',
                 n_sel: int=20,
                 lookback_window: int=1) -> pd.DataFrame:
        """_summary_
        
        Args:
        price_df (pd.DataFrame): 
            - DataFrame -> 벤치마크에 포한된 종목들의 가격 데이터프레임
        n_sel (int, optional):
            - int -> 베타 상위 n개 종목에 투자. Defaults to 20.
        lookback_window (int, optional): 
            - int ->lookback window. Defaults to 12.
        
        Returns:
            pd.DataFrame: 변동성 시그널 df
        """   

        # 벤치마크와 벤치마크에 포함된 종목들의 가격 데이터프레임
        self.price_df = price_df
        self.first_date = self.price_df.iloc[0].name
        self.last_date = self.price_df.iloc[-1].name
        
        # 한달마다의 마지막 날짜 & lookback window = 1년
        self.lookback_window = lookback_window * annualize_scaler(freq)
        monthly_index = rebal_dates(self.price_df, period=freq)
        self.monthly_index = monthly_index[(self.lookback_window - 1):]
        
        # 수익률 데이터프레임   
        self.rets = self.price_df.pct_change().dropna()
        
        # 베타 상위 n개 종목에 투자
        self.n_sel = n_sel
        
    def volatility(self) -> pd.DataFrame:
        """_summary_
        Returns:
            pd.DataFrame: 변동성 시그널 df
        """
        def assign_value(series):
            isin_series_index = series.loc[vol_index].index
            series1 = pd.Series([1] * len(isin_series_index), index=isin_series_index.tolist())
        
            not_isin_beta_index = series.index[~series.index.isin(vol_index)]
            series2 = pd.Series([0] * len(not_isin_beta_index), index=not_isin_beta_index.tolist())
        
            return pd.concat([series1, series2]).reindex(series.index).sort_index()
            
        signal_list = []

        for index in self.monthly_index:
            df = self.price_df.loc[:index, :]
            df = df.iloc[-252:, :] if len(df) >= 252 else df
            
            rets = df.pct_change().fillna(0)
            
            weights = np.array([i for i in range(1, len(df)+1)])
            weights = weights / weights.mean()
    
            rets = rets.apply(lambda x: x * weights, axis=0)
            vol_index = rets.std().sort_values(ascending=False).tail(self.n_sel).index
            
            signal_list.append(df.resample('M').last().apply(assign_value, axis=1).iloc[-1])
            
        signal_df = pd.concat(signal_list, axis=1).T 
        signal_df.index = self.monthly_index
        
        return signal_df
    
    def signal(self):
        return self.volatility()
    
    
if __name__ == '__main__':
    path = '/Users/jtchoi/Library/CloudStorage/GoogleDrive-jungtaek0227@gmail.com/My Drive/quant/Quant-Project/quant'
    equity_df = pd.read_csv(path + '/alter_with_equity.csv', index_col=0)
    print(equity_df.tail())
    equity_df.index = pd.to_datetime(equity_df.index)
    equity_universe = equity_df.loc['2011':,].dropna(axis=1)
    
    signal = VolatilityFactor(equity_universe, 'quarter').signal()
    print(signal.sum(axis=1))