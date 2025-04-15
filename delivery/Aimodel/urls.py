from django.urls import path
from .views import predict_transport_mode,map_view,login,register

urlpatterns = [
    path('predict/', predict_transport_mode, name='predict-transport-mode'),
    path('map/', map_view, name='map'),
    path('login/', login, name='login'),
    path('register/', register, name='register')
]