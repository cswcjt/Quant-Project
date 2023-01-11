import pandas as pd
from fredapi import Fred

def get_econ_info(indicators) : 
    fred = Fred(api_key='78b31c929aa00cef888d17a4c63cb823')
    temp = []
    for indicator in indicators : 
        target_df = pd.DataFrame({f'{indicator}': fred.get_series(f'{indicator}')})
        temp.append(target_df)

    econ_df = pd.concat(temp, axis = 1)
    econ_df.columns = indicators

    return econ_df
  
  