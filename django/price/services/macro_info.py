import pandas as pd
from fredapi import Fred

from django.conf import settings


def get_econ_info(indicators: dict): 
    fred = Fred(api_key=settings.FRED_API_KEY)
    temp = []

    for key, value in indicators.items(): 
        target_df = pd.DataFrame(
            {key: fred.get_series(
                value,
                observation_start='1980-01-01',
                observation_end='2022-12-31'
        )})

        temp.append(target_df)

    econ_df = pd.concat(temp, axis=1)
    econ_df.index = pd.to_datetime(econ_df.index)
    econ_df.columns = indicators.keys()
    
    econ_df['year'] = econ_df.index.year
    econ_df['month'] = econ_df.index.month

    return econ_df

if __name__ == '__main__':
    indicators = {
        'cli': 'USALOLITONOSTSAM',
        'core_pce': 'PCEPILFE',
        'sahm': 'SAHMREALTIME',
        'fed_rate':'FEDFUNDS',
        'cpi': 'CPIAUCSL',
        'ppi': 'PPIACO',
        'unemployment': 'UNRATE',
        'gdp': 'GDP',
        'csi': 'UMCSENT',
        'yield_spread': 'T10Y2Y',
        'house': 'HSN1F',
        'cpi_core': 'CPILFESL',
    }
    
    print(get_econ_info(indicators))