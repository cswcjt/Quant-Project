import pandas as pd
import numpy as np

class ValueFactor:
    def __init__(self, rebal_price: pd.DataFrame, 
                lookback_window: int=12, long_only: bool=True) -> pd.DataFrame:
        """_summary_

        Args:
            rebal_price (pd.DataFrame): 
                - DataFrame -> price_on_rebal()의 리턴 값. 리밸런싱 날짜의 타켓 상품들 종가 df
            lookback_window (int):
                - int -> 모멘텀(추세)를 확인할 기간 설정
            long_only (bool, optional): 
                - bool -> 매수만 가능한지 아님 공매도까지 가능한지 결정. Defaults to True.

        Returns:
            pd.DataFrame: 투자 시그널을 담고있는 df
        """
        self.rebal_price = rebal_price
        self.lookback_window = lookback_window
        self.long_only = long_only    
        
    def commoditiy_value(self, year: int=1):
        log_price_df = np.log(self.rebal_price)
        value_df = log_price_df.shift(self.lookback_window * year).div(self.rebal_price, axis=1)
        value_df.dropna(inplace=True)
        
        high_threshold = value_df.quantile(2/3)
        long_signal = (value_df > high_threshold) * 1

        low_threshold = value_df.quantile(1/3)
        short_signal = (value_df < low_threshold) * -1

        total_signal = long_signal + short_signal
        
        if self.long_only == True:
            return long_signal
        
        else:
            return total_signal
        
        
        