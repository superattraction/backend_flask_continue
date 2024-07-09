from PyPDF2 import PdfReader
def ocrpdf(PDF_FILE_PATH):
    reader = PdfReader(PDF_FILE_PATH)
    pages = reader.pages
    text=""
    for page in pages:
        sub = page.extract_text()
        text +=sub
    data=text.replace("\n"," ").split("■",4)

    자격=data[1].split('(능력단위코드)')[1]
    split_data=data[2].split('(능력단위코드)')
    훈련 = split_data[1].strip() + ' ' + split_data[2].strip()
    경력=data[3].split('※')[0]
    input = {1:자격, 2:훈련, 3:경력}
    return input