from django.urls import path
from .views import PredictDelay

urlpatterns = [
    path('predict/', PredictDelay.as_view(), name='predict-delay'),
]