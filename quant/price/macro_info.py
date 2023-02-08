import pandas as pd
from fredapi import Fred

def get_econ_info(indicators: dict={'cli': 'USALOLITONOSTSAM', 
                                    'core_pce': 'PCEPILFE', 
                                    'sahm': 'SAHMREALTIME', 
                                    'fed_rate':'FEDFUNDS'
                                    }
                ): 
    
    with open('../fredapikey.txt', 'r') as f:
        fred_key = f.read()
    
    fred = Fred(api_key=fred_key)
    temp = []
    for key, value in indicators.items() : 
        target_df = pd.DataFrame({f'{key}': fred.get_series(f'{value}')})
        temp.append(target_df)

    econ_df = pd.concat(temp, axis = 1)
    econ_df.columns = indicators

    return econ_df


