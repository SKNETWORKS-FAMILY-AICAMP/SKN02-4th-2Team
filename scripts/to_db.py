import os
import django
import sys

# Django 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['DJANGO_SETTINGS_MODULE'] = 'djanglang.settings'
django.setup()

from walkerhill.models import CrawledText

def save_to_db(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
        document = CrawledText(content=text_content, title="워커힐 리워즈 멤버십 약관")
        document.save()
        print("텍스트가 DB에 성공적으로 저장되었습니다.")

# 크롤링 결과 파일 경로 지정
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crawler_output.txt')
save_to_db(file_path)