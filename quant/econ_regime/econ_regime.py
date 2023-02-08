import pandas as pd
import numpy as np
from datetime import date, timedelta

import yfinance as yf
from fredapi import Fred

def business_cycle(starting_date: str='1992-12-01', recession: bool=False) -> pd.DataFrame:
    with open('../fredapikey.txt', 'r') as f:
        fred_key = f.read()
        
    fred = Fred(api_key=fred_key)
    regime_list = []
    
    us_cli = fred.get_series('USALOLITONOSTSAM', observation_start=starting_date).dropna()
    us_cli = pd.concat([us_cli.iloc[1:,], us_cli.pct_change().dropna()], axis=1)
    
    core_pce = fred.get_series('PCEPILFE', observation_start=starting_date)
    core_pce = core_pce.pct_change(12).dropna()
    
    business_cycle = pd.concat([us_cli, core_pce], axis=1)
    business_cycle.columns = ['cli', 'cli_change', 'pce']
    
    business_cycle = business_cycle.shift()
    business_cycle = business_cycle.dropna()
    
    deflation = ((business_cycle['cli'] < 100) & (business_cycle['cli_change'] < 0) & (business_cycle['pce'] < 0.025)) * 1
    inflation = ((business_cycle['cli'] < 100) & (business_cycle['cli_change'] < 0) & (business_cycle['pce'] >= 0.025)) * 1
    recovery = ((business_cycle['cli'] < 100) & (business_cycle['cli_change'] >= 0)) * 1
    expansion = ((business_cycle['cli'] >= 100) & (business_cycle['cli_change'] >= 0)) * 1
    
    regime_df = pd.concat([deflation, inflation, recovery, expansion], axis=1).dropna()
    regime_df.columns = ['deflation', 'inflation', 'recovery', 'expansion']
    
    if recession:
        sahm_rule = fred.get_series('SAHMREALTIME')
        sahm_rule = pd.DataFrame(sahm_rule)
        sahm_rule = sahm_rule.rename(columns={0:'sahm'})

        sahm_rule['3m'] = sahm_rule.sahm.rolling(window=3).mean()
        sahm_rule['min'] = sahm_rule.sahm.rolling(window=12).min()
        sahm_rule['recession'] = (sahm_rule['3m'] - sahm_rule['min'])
        sahm_rule['recession'] = (sahm_rule['recession'] >= 0.5) * 1
        
        regime_df = pd.concat([regime_df, sahm_rule['recession']], axis=1)
        
    return regime_df
        
    