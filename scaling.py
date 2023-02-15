"""
사용 예시

self.freq = convert_freq(freq)
self.param = annualize_scaler(self.freq)
"""
def convert_freq(freq: str) -> str:
    convert = {
        'day': 'day',
        'D': 'day',
        'daily': 'day',
        'week': 'week',
        'W': 'week',
        'weekly': 'week',
        'month': 'month',
        'M': 'month',
        'monthly': 'month',
        'quarter': 'quarter',
        'Q': 'quarter',
        'quarterly': 'quarter',
        'halfyear': 'halfyear',
        'half-year': 'halfyear',
        'half_year': 'halfyear',
        'HY': 'halfyear',
        'half-yearly': 'halfyear',
        'half_yearly': 'halfyear',
        'year': 'year',
        'Y': 'year',
        'yearly': 'year',
    }
    return convert[freq]

def annualize_scaler(freq: str) -> int:
    # 주기에 따른 연율화 파라미터 반환해주는 함수
    annualize_scale_dict = {
        'day': 252,
        'week': 52,
        'month': 12,
        'quarter': 4,
        'halfyear': 2,
        'year': 1
    }
    try:
        scale: int = annualize_scale_dict[freq]
    except:
        raise Exception("freq is only ['day', 'week', 'month', \
            'quarter', 'half-year', 'year']")
    
    return scale