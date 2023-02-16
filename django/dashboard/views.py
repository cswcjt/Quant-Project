from config.settings.base import BASE_DIR
from django.http import HttpResponse
from django.shortcuts import render
from django.templatetags.static import static
from django.urls import reverse

from dashboard.services import set_checkboxs_info

# Create your views here.
def factor(request):
	return render(request, 'dashboard/container/factor.html', {
        'app_name': 'factor',
        'checkboxs_info': set_checkboxs_info(),
    })


def market(request):
	return render(request, 'dashboard/container/market.html', {
        'app_name': 'market',
        'api_url': reverse('dashboard:market_api'),
        'checkboxs_info': set_checkboxs_info(),
    })


def portfolio(request):
    metric_first = {
        'returns': 'Total Returns',
        'cagr': 'CAGR',
    }

    metric_second = {
        'MDD': 'MDD',
        'MDD_duration': 'Underwater Period',
        'volatility': 'Anuallized Volatility',
    } 

    metric_third = {
        'sharp': 'Sharpe Ratio',
        'sortino': 'Sortino Ratio',
        'calmar': 'Calmar Ratio',
    }

    metric_last = {
        'CVaR_ratio': 'CVaR Ratio',
        'hit': 'Hit Ratio',
        'GtP': 'Gain-to-Pain'
    }
    
    return render(request, 'dashboard/container/portfolio.html', {
        'app_name': 'portfolio',
        'api_url': reverse('dashboard:portfolio_api'),
        'checkboxs_info': set_checkboxs_info(),
        'metric_first': metric_first,
        'metric_second': metric_second,
        'metric_third': metric_third,
        'metric_last': metric_last,
    })


def presentation(request):
    print(BASE_DIR)
    with open(BASE_DIR / 'static' / 'img' / 'presentation.pdf', 'rb') as pdf:
        response = HttpResponse(pdf.read(), content_type='application/pdf')
        response['Content-Disposition'] = 'filename=presentation..pdf'
        return response
    #  return render(request, 'dashboard/container/presentation.html')