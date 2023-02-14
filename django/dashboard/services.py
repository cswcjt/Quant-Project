def set_checkboxs_info():
    return [
        {
            'group_name': 'Factor', 
            'input_name': ['Momentum', 'Value', 'Carry', 'Volatility', 'AI Forcasting'],
            'is_multiple_check': True,
        },
        {
            'group_name': 'Factor Weights', 
            'input_name': ['Equal Weights', 'Market Analysis', 'Return Inc', 'AI Market Forecasting'],
            'is_multiple_check': False,
        },
        # {
        #     'group_name': 'Asset Type Priority', 
        #     'input_name': ['Equal Weights', 'Cap Desc', 'Cap Asc', 'ROI Desc', 'OPM Desc'],
        #     'is_multiple_check': False,
        # },
        {
            'group_name': 'Risk Tolerance', 
            'input_name': ['Stable', 'Moderate', 'Risky'],
            'is_multiple_check': False,
        },
        {
            'group_name': 'Rebalancing Period', 
            'input_name': ['Month', 'Quarter (3m)', 'Half Year (6m)', 'Year (12m)'],
            'is_multiple_check': False,
        }
    ]