from django.urls import path, include
from . import views
urlpatterns = [
    path('', views.home_page, name='index'),
    path('cartoon/home/', views.home_page, name='home'),
    path('cartoon/result/', views.ResultPage.as_view(), name='result')
]