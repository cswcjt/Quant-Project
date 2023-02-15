import pandas as pd
import numpy as np

## Project Path 추가
import sys
from pathlib import Path

PJT_PATH = Path(__file__).parents[3]
sys.path.append(str(PJT_PATH))
from quant.price.price_processing import rebal_dates, price_on_rebal
from scaling import annualize_scaler

# 주식 모멘텀 클래스
class MomentumFactor:
    """Momentum 전략을 관리할 클래스
    Returns:
        pd.DataFrame -> 거래 시그널을 알려주는 df
    """
    
    def __init__(self, price_df: pd.DataFrame,
                 freq: str='M', lookback_window: int=12,
                 n_sel: int=20, long_only: bool=True):
        """초기화 함수
        Args:
            rebal_price (pd.DataFrame): 
                - DataFrame -> 일별 종가 데이터프레임
            lookback_window (int):
                - int -> 모멘텀(추세)를 확인할 기간 설정
            n_sel (int):
                - int -> 몇 개의 금융상품을 고를지 결정
            long_only (bool, optional): 
                - bool -> 매수만 가능한지 아님 공매도까지 가능한지 결정. Defaults to True.
        """
        
        self.lookback_window = lookback_window
        self.rebal_dates_list = rebal_dates(price_df, 
                                            period=freq,
                                            include_first_date=True)
        
        self.rets = price_df.loc[self.rebal_dates_list, :].pct_change(self.lookback_window).dropna()
        self.n_sel = n_sel
        self.long_only = long_only

    # 절대 모멘텀 시그널 계산 함수
    def absolute_momentum(self) -> pd.DataFrame:
        """absolute_momentum
        Args:
            long_only (bool, optional): 
                - bool -> 매수만 가능한지 결정. Defaults to True.
        Returns:
            pd.DataFrame -> 투자 시그널 정보를 담고있는 df
        """

        returns = self.rets

        # 롱 시그널
        long_signal = (returns > 0) * 1

        # 숏 시그널
        short_signal = (returns < 0) * -1

        # 토탈 시그널
        if self.long_only == True:
            signal = long_signal

        else:
            signal = long_signal + short_signal
        
        return signal.iloc[self.lookback_window:,]#.dropna(inplace=True)
    
    # 상대 모멘텀 시그널 계산 함수
    def relative_momentum(self) -> pd.DataFrame:
        """relative_momentum
        Args:
            long_only (bool, optional): 
                - bool -> 매수만 가능한지 결정. Defaults to True.
        Returns:
            pd.DataFrame -> 투자 시그널 정보를 담고있는 df
        """

        # 수익률
        returns = self.rets

        # 자산 개수 설정
        n_sel = self.n_sel

        # 수익률 순위화
        rank = returns.rank(axis=1, ascending=False)

        # 롱 시그널
        long_signal = (rank <= n_sel) * 1

        # 숏 시그널
        short_signal = (rank >= len(rank.columns) - n_sel + 1) * -1

        # 토탈 시그널
        if self.long_only == True:
            signal = long_signal

        else:
            signal = long_signal + short_signal

        return signal.iloc[self.lookback_window:,]#.dropna(inplace=True)
    
    # 듀얼 모멘텀 시그널 계산 함수
    def dual_momentum(self) -> pd.DataFrame:
        """dual_momentum
        Args:
            long_only (bool, optional): 
                - bool -> 매수만 가능한지 결정. Defaults to True.
        Returns:
            pd.DataFrame -> 투자 시그널 정보를 담고있는 df
        """

        # 절대 모멘텀 시그널
        abs_signal = self.absolute_momentum()

        # 상대 모멘텀 시그널
        rel_signal = self.relative_momentum()

        # 듀얼 모멘텀 시그널
        signal = (abs_signal == rel_signal) * abs_signal

        # 절대 모멘텀과 상대 모멘텀의 시그널을 받을 때 이미 signal.iloc[self.lookback_window:,] 반영되어 있음
        return signal

    def signal(self):
        return self.dual_momentum()
    
    
if __name__ == '__main__':
    path = '/Users/jtchoi/Library/CloudStorage/GoogleDrive-jungtaek0227@gmail.com/My Drive/quant/Quant-Project/quant'
    equity_df = pd.read_csv(path + '/equity_universe.csv', index_col=0)
    equity_df.index = pd.to_datetime(equity_df.index)
    equity_universe = equity_df.loc['2011':,].dropna(axis=1)
    
    signal = MomentumFactor(equity_universe, 'quarter', 12).signal()
    print(signal.sum(axis=1))