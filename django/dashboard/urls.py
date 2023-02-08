from django.urls import path
from dashboard import apis, views

app_name = 'dashboard'
urlpatterns = [
	path('api/factor/', apis.factor_api, name='factor_api'),
	path('api/market/', apis.market_api, name='market_api'),
	path('api/portfolio/', apis.portfolio_api, name='portfolio_api'),

	path('factor/', views.factor, name='factor'),
	path('market/', views.market, name='market'),
	path('portfolio/', views.portfolio, name='portfolio'),
]