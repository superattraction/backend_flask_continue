from PyPDF2 import PdfReader
import re
def ocrpdf(pdf_file_path):
# PDF 파일 열기
    with open(pdf_file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    # 텍스트를 '■' 기준으로 분할
    sections = re.split(r'■', text)

    # 데이터 저장을 위한 딕셔너리 초기화
    data = {1: "", 2: "", 3: ""}

    # 각 섹션을 해당 키에 할당
    for section in sections:
        if section.startswith("자격"):
            data[1] = section.strip()
        elif section.startswith("훈련"):
            data[2] = section.strip()
        elif section.startswith("경력"):
            data[3] = section.strip()

    # 결과 딕셔너리 출력
    return data

