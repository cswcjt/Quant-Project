def set_checkboxs_info():
    return [
        {
            'group_name': 'Factor', 
            'input_name': ['Beta', 'Momentum', 'Volatility', 'AI Forcasting'],
            'is_multiple_check': True,
        },
        {
            'group_name': 'Weights', 
            'input_name': ['Equal Weights', 'Equal Volatility Weight', 'Maximize Sharpe Ratio', 
                'Global Mimimum Variance', 'Most Diversified Portfolio', 'Risk Parity'],
            'is_multiple_check': False,
        },
        {
            'group_name': 'Risk Tolerance', 
            'input_name': ['Conservative', 'Moderate', 'Aggressive'],
            'is_multiple_check': False,
        },
        {
            'group_name': 'Rebalancing Period', 
            'input_name': ['Month', 'Quarter', 'Half Year', 'Year'],
            'is_multiple_check': False,
        }
    ]