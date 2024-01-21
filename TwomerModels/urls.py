from django.urls import path
from . import views

urlpatterns = [
    path('random-forest/', views.random_forest, name='random-forest'),
    path('upload/', views.UploadApkView.as_view(), name='upload_apk'),
    path('process/', views.ProcessApkView.as_view(), name='process_uploaded_apk'),
]