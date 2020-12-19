from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
     path('',views.IndexViewSet),
     path('upload',views.uploadImage, name = "uploadImage"),
     path('ingredients',views.model_call, name = "ingredients"),
]


