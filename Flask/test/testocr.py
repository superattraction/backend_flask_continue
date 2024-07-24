import PyPDF2
import re

# PDF 파일 경로
pdf_path = 'test.pdf'  # 여기에 PDF 파일의 경로를 입력하세요

# PDF 파일 열기
with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

# 텍스트를 '■' 기준으로 분할
sections = re.split(r'■', text)

# 데이터 저장을 위한 딕셔너리 초기화
data = {"자격": "", "훈련": "", "경력": ""}

# 각 섹션을 해당 키에 할당
for section in sections:
    if section.startswith("자격"):
        data["자격"] = section.strip()
    elif section.startswith("훈련"):
        data["훈련"] = section.strip()
    elif section.startswith("경력"):
        data["경력"] = section.strip()

# 결과 딕셔너리 출력
print(data)