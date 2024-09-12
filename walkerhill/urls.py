from django.urls import path
from .views import chatbot_view, index_view, db_status, display_crawled_text

urlpatterns = [
    path('', index_view, name='index'),  # 루트 경로
    path('chatbot/', chatbot_view, name='chatbot'),  # 챗봇 API
    path('db_status/', db_status, name='db_status'),  # 데이터베이스 상태 확인 API
    path('display/', display_crawled_text, name='display_crawled_text'),  # 크롤링된 텍스트를 보여주는 경로
]