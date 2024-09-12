from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os

# 크롬 브라우저 설정
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# 웹페이지 열기
url = "https://www.walkerhill.com/customer/RewardsAgreement"
driver.get(url)

# 페이지가 완전히 로드될 때까지 대기
driver.implicitly_wait(10)

# 페이지 소스 가져오기
page_source = driver.page_source

# BeautifulSoup로 파싱
soup = BeautifulSoup(page_source, 'html.parser')

# 원하는 섹션 크롤링
rule_area = soup.find('div', {'id': 'ruleArea'})

# 텍스트 추출
if rule_area:
    rules_text = rule_area.get_text(separator='\n', strip=True)

    # 크롤링한 텍스트를 파일로 저장할 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
    output_path = os.path.join(script_dir, 'crawler_output.txt')

    # crawler_output.txt 파일에 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rules_text)

    print(f"크롤링된 텍스트가 {output_path}에 저장되었습니다.")
else:
    print("크롤링할 데이터가 없습니다.")

# 브라우저 종료
driver.quit()