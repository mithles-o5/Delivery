"""
URL configuration for delivery project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from Aimodel.views import login,register,map_view,get_transport_data,home,profile
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('Aimodel.urls')),  # Backend API
    path('', TemplateView.as_view(template_name='index.html'), name='frontend'),  # Frontend
    path('login/', login, name='login'),
    path('map/', map_view, name='map'),
    path('profile/', profile, name='profile'),
    path('home/', home, name='home'),
    path('register/', register, name='register'),
    path('api/get-transport-data/', get_transport_data, name='get_transport_data'),
    # static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

urlpatterns += staticfiles_urlpatterns()