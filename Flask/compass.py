from flask import Flask, session, request, jsonify
from flask_cors import CORS
import logging
from cachetools import TTLCache
from cryptography.fernet import Fernet
import os
import json
from flask_sqlalchemy import SQLAlchemy
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from review_summary.preprocess import parse_review3
from chat_account.chat_accountOFF import pdfOff
from chat_account.chat_accountON import pdfOn
from ocr.pdf_om import om
app = Flask(__name__)
CORS(app)

app.secret_key = 'oj_super_key'

# 데이터베이스 설정
username = 'hrd'
password = '1234'
hostname = '10.125.121.212'  # MySQL 서버의 IP 주소
port = '3306'  # MySQL 기본 포트
database_name = 'job'

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{username}:{password}@{hostname}:{port}/{database_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class FinalData(db.Model):
    __tablename__ = 'final_data'
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, nullable=False)
    course_name = db.Column(db.String(255), nullable=False)
    reviews = db.Column(db.Text, nullable=False)
    goals = db.Column(db.Text, nullable=False)

class SummaryReview(db.Model):
    __tablename__ = 'summary_review2'
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, nullable=False)
    course_name = db.Column(db.String(500), nullable=False)
    summary_review = db.Column(db.String(5000), nullable=False)

# Prompt
prompt = ChatPromptTemplate.from_template(
    ''' 
    다음 문장은 직업훈련강의에 대한 수강후기들이다. 이 문장의 주요 특징을 잘 선별해서 요약하여 소감을 한 문장으로 나타낸다.
    주요 특징에는 직업, 훈련내용, 강의내용, 교사, 배운 점, 배울 점, 앞으로의 방향들 같은 직업과 미래에 관련된 내용들이 있을 것이다.
    한글을 준수하자. 기호는 , 이거랑 . 이거 ! 이거말고 들어가면 안돼. 너 이모지 만들지마. 텍스트만 적어야 돼.
    : {Sentence}'''
)

# 모델
model = ChatOllama(model="gemma2:9b", temperature=0)
chain = prompt | model | StrOutputParser()  # 문자열 출력 파서를 사용합니다.

# 암호화 키 생성
encryption_key = os.getenv('ENCRYPTION_KEY')
if not encryption_key:
    encryption_key = Fernet.generate_key()
else:
    encryption_key = encryption_key.encode()

# 초기화
cipher = Fernet(encryption_key)

# TTLCache 초기화 (캐시의 최대 크기와 TTL 설정)
chatbot_data = TTLCache(maxsize=100, ttl=3600)  # 최대 100개 항목, 각 항목은 1시간 동안 유효

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/process_apply/<num>', methods=['POST'])
def pdfapply(num):
    result = om(num)
    encrypted_result = cipher.encrypt(json.dumps(result).encode())
    latch = str(num)
    chatbot_data[latch] = encrypted_result
    return "success"

@app.route('/process_chatbot/<num>/<key>', methods=['POST'])
def process_chatbot_pdfOn(num, key):
    if str(num) in chatbot_data:
        encrypted_result = chatbot_data[str(num)]
        ocr_data = json.loads(cipher.decrypt(encrypted_result).decode())  # 결과를 복호화하여 사용
        print("key", key)
        print("ocr_data확인", ocr_data)
        return pdfOn(key, ocr_data)
    else:
        return "에러났어요", 404

@app.route('/process_chatbot/', methods=['POST'])
def process_chatbot_pdfOff():
    return pdfOff

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
