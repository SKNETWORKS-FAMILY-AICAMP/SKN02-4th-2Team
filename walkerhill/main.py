from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.schema import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from dotenv import load_dotenv
import os
from .models import CrawledText  # Django 모델 불러오기

# .env 파일 로드
load_dotenv()

# OpenAI API 키 로드
openai_api_key = os.getenv("OPENAI_API_KEY")

# LLM 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# 임베딩 설정
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# 데이터베이스에서 크롤링된 텍스트 로드
def load_crawled_data_from_db():
    crawled_texts = CrawledText.objects.all()
    docs = [Document(page_content=text.content) for text in crawled_texts]
    return docs

# 벡터스토어 초기화
def initialize_vectorstore():
    documents = load_crawled_data_from_db()
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="chroma_store")
    return vectorstore

vectorstore = initialize_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

# 아웃풋 파서 
class OutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split("\n")

# 프롬프트 템플릿 지정
prompt_template = """
당신은 워커힐(walkerhill)의 고객 상담원입니다. 고객의 멤버십 이용 관련 문의에 대해 정확하고 친절하게 답변하세요.
절대로 답변을 지어내지 말고, 제공된 멤버십 약관이나 사전에 정해진 문서를 바탕으로만 답변을 제공해야 합니다.
고객의 질문을 분석하고, 질문에 사용된 언어를 감지하여 해당 언어로 답변하세요.
고객이 사용하는 언어가 자동으로 감지되면, 그 언어로 질문에 답변을 제공하세요.
고객의 질문을 차근차근 생각한 후 친절하게 답변하고, 너무 길게 답하지 마세요.

**만약 질문에 대한 답을 모른다면 절대 지어내지 마세요. 그 대신 "죄송합니다, 해당 사항에 대해서는 확실히 알지 못합니다"라고 답변하세요.**

고객이 부정적이거나 비속어를 사용하는 경우에는 친절하지만 단호하게 경고 메시지를 보내세요. 예를 들어, "고객님, 서비스 이용 시 부적절한 표현은 자제 부탁드립니다."라고 말합니다.

답변을 제공할 때, 문장이 끝날 때마다 줄을 바꾸어 가독성을 높이세요.
아래 예시를 참고하여 문장을 짧게 구분하고, 항목별로 답변을 나누어 단락을 나누어 제공하세요.
'*' 와 같은 특수문자는 사용하지 않고 대답을 제공하세요.

### Example 1:
Q: "포인트는 어떻게 적립되나요?"
A: 
<br>
👉 적립 기준: 상품 구매 및 서비스 이용 결제대금의 1%가 적립됩니다. <br>
👉 포인트 적립 가능 항목: 객실 요금, 식사, 부대시설 이용 등.<br>
👉 포인트 적립 시점: 결제가 완료된 다음 날, 자동으로 적립됩니다.<br>

### Example 2:
Q: "회원 탈퇴 시 포인트는 어떻게 처리되나요?"
A: 
<br>
👉 포인트 소멸: 회원 탈퇴 시 보유한 포인트는 즉시 소멸됩니다.<br>
👉 포인트 사용 가능 시기: 탈퇴 전에 포인트를 모두 사용해야 합니다.

### Example 3 (경고 메시지):
Q: "왜 이런 엉망진창의 서비스를 제공하나요?"
A: 
<br>
고객님, 서비스 이용 시 부적절한 표현은 자제 부탁드립니다. 문의하신 사항에 대해 설명드리겠습니다.<br>
👉 서비스 변경 사항: 최근 시스템 업데이트로 인해 약간의 지연이 발생했습니다.<br>
👉 문제 해결 방법: 이 문제는 빠르게 해결될 예정이며, 곧 정상 서비스가 가능할 것입니다.

### Example 4 (모르는 질문에 대한 응답):
Q: "멤버십 포인트를 다른 계정으로 이체할 수 있나요?"
A: 
<br>
죄송합니다, 해당 사항에 대해서는 확실히 알지 못합니다. 더 정확한 정보를 원하시면, 고객센터(☎️대표번호 1670-0005, ✉️이메일 contact@walkerhill.com)에 문의해 주시기 바랍니다.

만약 답을 모르거나, 고객이 문의하고 싶다고 하면, 
고객센터(☎️대표번호 1670-0005, ✉️이메일 contact@walkerhill.com)에 직접 문의하라고 말하세요.

# 고객의 질문에 사용된 언어로 답변을 하세요. 고객의 질문 언어를 인식하고, 그 언어로 답변을 제공하세요.

#Previous Chat History:
{chat_history}

#Question:
{question}

#Context:
{context}

#Answer:
"""

rag_prompt_custom = PromptTemplate.from_template(prompt_template)

rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | rag_prompt_custom
    | llm
    | OutputParser()
)

# 세션 기록을 저장할 딕셔너리
store = {}

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

# 대화를 기록하는 RAG 시스템 체인 생성
rag_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)

def process_question(question):
    try:
        answer = rag_with_history.invoke({"question": question}, config={"configurable": {"session_id": "rag123"}})
        return '\n'.join(answer)
    except Exception as e:
        return f'Error during QA Chain execution: {str(e)}'