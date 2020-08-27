from django.urls import path
from . import views

app_name = 'app_to_predict'

urlpatterns = [
    path('showall/', views.showall, name='showall'),
    path('upload/', views.upload, name='upload'),
]
