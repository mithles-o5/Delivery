from django.urls import path
from .views import PredictDelay,map_view,login

urlpatterns = [
    path('predict/', PredictDelay.as_view(), name='predict-delay'),
    path('map/', map_view, name='map'),
    path('login/', login, name='login'),
]