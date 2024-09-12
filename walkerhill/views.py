from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import CrawledText
from .main import process_question  # main.py에서 LLM 로직을 가져오기
import os  

# POST 요청을 처리하는 챗봇 
@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        question = request.POST.get('question', '')
        if question:
            answer = process_question(question)  # main.py에서 처리한 답변을 가져오기
            return JsonResponse({'answer': answer})
        return JsonResponse({'error': '질문을 입력해주세요.'})
    return JsonResponse({'error': 'POST 요청만 지원합니다.'})

# 크롤링된 텍스트를 데이터베이스에서 불러와서 display.html에 출력
def display_crawled_text(request):
    crawled_texts = CrawledText.objects.all()  # 모든 크롤링된 텍스트를 가져오기
    context = {'crawled_texts': crawled_texts}
    return render(request, 'base/display.html', context)

# GET 요청을 처리하여 index.html을 렌더링
def index_view(request):
    return render(request, 'base/index.html')

# 데이터베이스 상태를 확인
def db_status(request):
    vectorstore_path = "chroma_store"  # Chroma 벡터스토어가 저장되는 경로
    if os.path.exists(vectorstore_path):
        return JsonResponse({'exists': True})
    else:
        return JsonResponse({'exists': False})