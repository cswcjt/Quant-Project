import FinanceDataReader as fdr
import fredpy as fp

class Carry:
    def __init__(self, price, cost=0.002):
        # with open('./fredapikey.txt', 'r') as f:
        #     apikey = f.read()
            
        # fp.api_key = apikey
        self.price = price
        
        self.rets = self.price.pct_change()
        
        self.cost = cost
        
        self.weight
    
    def get_vix(self, start_date, end_date):
        vix = fdr.DataReader('VIX', start_date, end_date)
        vix = vix['Close']
        return vix
    
    def get_vix_slope(self, start_date, end_date):
        vix = self.get_vix(start_date, end_date)
        vix_slope = vix.shift(1) / vix
        return vix_slope
    
    def calculate_weights(self, p):
        slope = self.price.diff().fillna(0)
        
        # 롱 포지션
        long_weights = (slope < 0) * 1

        # 숏 포지션
        short_weights = (slope > 0) * -1

        # 토탈 포지션
        total_weights = long_weights + short_weights

        return total_weights
    
    def calculate_returns(self, rets, weights, cost):
        port_rets = weights.shift() * rets - abs(weights.diff()) * cost

        return port_rets
    
if __name__ == '__main__':
    ca = Carry()
    # print(ca.get_vix('2010-01-04', '2022-10-26'))
    # print(ca.get_vixfuture('2010-01-04', '2022-10-26'))
    print(ca.get_vix_slope('2010-01-04', '2022-10-27'))