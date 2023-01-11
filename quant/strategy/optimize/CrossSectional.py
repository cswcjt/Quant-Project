import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# 횡적 배분 모델 클래스 
class CrossSectional:
    def __init__(self, vol, er, cov):
        self.vol = vol
        self.er = er
        self.cov = cov

    # EW
    def ew(self):
        noa = self.er.shape[0]
        weights = np.ones_like(self.er) * (1/noa)
        return weights
    
    # MSR
    def msr(self):
        noa = self.er.shape[0]
        init_guess = np.repeat(1/noa, noa)

        bounds = ((0.0, 1.0), ) * noa
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}

        def neg_sharpe(weights, er, cov):
            r = weights.T @ er
            vol = np.sqrt(weights.T @ cov @ weights)
            return - r / vol

        weights = minimize(neg_sharpe,
                           init_guess,
                           args=(self.er, self.cov),
                           method='SLSQP',
                           constraints=(weights_sum_to_1,), 
                           bounds=bounds)

        return weights.x
    
    # GMV
    def gmv(self):
        noa = self.cov.shape[0]
        init_guess = np.repeat(1/noa, noa)

        bounds = ((0.0, 1.0), ) * noa
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}

        def port_vol(weights, cov):
            vol = np.sqrt(weights.T @ cov @ weights)
            return vol

        weights = minimize(port_vol, init_guess, args=(self.cov), method='SLSQP', constraints=(weights_sum_to_1,), bounds=bounds)

        return weights.x
    
    # MDP
    def mdp(self):
        noa = self.vol.shape[0]
        init_guess = np.repeat(1/noa, noa)
        bounds = ((0.0, 1.0), ) * noa
        
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}
        
        def neg_div_ratio(weights, vol, cov):
            weighted_vol = weights.T @ vol
            port_vol = np.sqrt(weights.T @ cov @ weights)
            return - weighted_vol / port_vol
        
        weights = minimize(neg_div_ratio,
                           init_guess, 
                           args=(self.vol, self.cov),
                           method='SLSQP',
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        
        return weights.x
    
    # RP
    def rp(self):
        noa = self.cov.shape[0]
        init_guess = np.repeat(1/noa, noa)
        bounds = ((0.0, 1.0), ) * noa
        target_risk = np.repeat(1/noa, noa)
        
        weights_sum_to_1 = {'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1}
        
        def msd_risk(weights, target_risk, cov):
            
            port_var = weights.T @ cov @ weights
            marginal_contribs = cov @ weights
            
            risk_contribs = np.multiply(marginal_contribs, weights.T) / port_var
            
            w_contribs = risk_contribs
            return ((w_contribs - target_risk)**2).sum()
        
        weights = minimize(msd_risk,
                           init_guess,
                           args=(target_risk, self.cov),
                           method='SLSQP',
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x
    
    # EMV
    def emv(self):
        inv_vol = 1 / self.vol
        weights = inv_vol / inv_vol.sum()

        return weights
