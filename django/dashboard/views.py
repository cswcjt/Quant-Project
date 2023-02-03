from django.shortcuts import render
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
	return render(request, 'dashboard/container/portfolio.html', {
        'app_name': 'portfolio',
        'api_url': reverse('dashboard:portfolio_api'),
        'checkboxs_info': set_checkboxs_info(),
    })