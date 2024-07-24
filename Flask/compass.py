from flask import Flask, jsonify
from flask_cors import CORS
import logging
from cachetools import TTLCache
from cryptography.fernet import Fernet
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chat_account.chat_accountOFF import pdfOff
from chat_account.chat_accountON import pdfOn
from ocr.pdf_om import om
from chatbot.chatbot_summary import chatbot_s
from chatbot.chatbot_hello import chatbot_h
import os

app = Flask(__name__)
CORS(app)

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
app.secret_key = 'oj_super_key'



@app.route('/api/chat/<content>', methods=['GET'])
def chat(content):
    return chatbot_s(content)

##직업능력계좌##
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
        return pdfOn(key, ocr_data)
    else:
        return "에러났어요", 404

@app.route('/process_chatbot/<content>', methods=['POST'])
def process_chatbot_pdfOff(content):
    return pdfOff(content)

@app.route('/compass/hello', methods=['GET'])
def process_chatbot_hello():
     return chatbot_h()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
