from django.urls import path
from . import views

urlpatterns = [
    path('classify/', views.classify_review, name='classify_review'),
]
