from django.urls import path
from .views import PredictDelay,map_view

urlpatterns = [
    path('predict/', PredictDelay.as_view(), name='predict-delay'),
    path('map/', map_view, name='map'),
]