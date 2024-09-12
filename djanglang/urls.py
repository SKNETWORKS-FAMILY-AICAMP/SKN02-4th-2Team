from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('walkerhill.urls')),  # walkerhill 앱의 urls.py를 루트 경로에 포함
]