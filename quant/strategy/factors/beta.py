import pandas_datareader as web
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

class Beta():
    """_summary_

    자산의 베타값을 계산하는 클래스
    각 자산군의 벤치마크를 설정하고, 해당 벤치마크에 포함된 종목들의 가격 데이터프레임을 입력하면 베타값을 계산한다.
    """
    
    def __init__(self, price_df: pd.DataFrame, benchmark_ticker: str, intercept: int=1) -> pd.DataFrame:
        """_summary_

        Args:
            price_df (pd.DataFrame): 벤치마크에 포한된 종목들의 가격 데이터프레임
            benchmark_ticker (str): 벤치마크의 티커
            intercept (int, optional): 선형회귀의 y절편 값. Defaults to 1.

        Returns:
            pd.DataFrame: 종목별 베타값
        """
        
        # 벤치마크와 벤치마크에 포함된 종목들의 가격 데이터프레임
        self.price_df = price_df
        self.benchmart_ticker = benchmark_ticker
        self.benchmark_df = pd.DataFrame({f'{self.benchmart_ticker}': yf.download(self.benchmart_ticker)['Adj Close']})
        self.universe = pd.concat([self.price_df, self.benchmark_df], axis=1)
        
        # 수익률 데이터프레임   
        self.rets = self.universe.pct_change().dropna()
        
        # 선형회귀의 y절편 값
        self.intercept = intercept
        
    def get_beta(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: 종목별 베타값
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
        