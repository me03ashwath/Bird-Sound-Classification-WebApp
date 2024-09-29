# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_audio, name='upload_audio'),
    path('', views.upload_audio_page, name='upload_audio_page'),
]
