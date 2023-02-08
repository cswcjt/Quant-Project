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
    for indicator in indicators : 
        target_df = pd.DataFrame({f'{indicator}': fred.get_series(f'{indicator}')})
        temp.append(target_df)

    econ_df = pd.concat(temp, axis = 1)
    econ_df.columns = indicators

    return econ_df


