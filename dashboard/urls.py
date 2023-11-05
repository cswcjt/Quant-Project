from django.urls import path
from dashboard import apis, views

app_name = 'dashboard'
urlpatterns = [
	path('api/factor/', apis.FactorAPIView.as_view(), name='factor_api'),
	path('api/market/', apis.MarketAPIView.as_view(), name='market_api'),
	path('api/portfolio/', apis.PortfolioAPIView.as_view(), name='portfolio_api'),

	path('factor/', views.factor, name='factor'),
	path('market/', views.market, name='market'),
	path('portfolio/', views.portfolio, name='portfolio'),
	path('presentation/', views.presentation, name='presentation'),
]