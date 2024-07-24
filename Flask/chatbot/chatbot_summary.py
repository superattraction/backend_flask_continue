
from langchain_community.vectorstores import Chroma
from flask import jsonify
import time
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import os
os.environ['CUDA_DEVICE_ORDER']="FASTEST_FIRST"
os.environ['CUDA_VISIBLE_DEVICES']='0'

def load_vector_store():
    start_time = time.time()


    print("챗봇 서머리 벡터 스토어 로드를 시작합니다...")

    embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
    )

    # 벡터 스토어 로드
    db = Chroma(
        collection_name='LLM_output',
        persist_directory='./db_summary/chromadb',
        embedding_function= embeddings_model
    )
    print(f"챗봇 서머리 벡터 스토어 로드 완료. 소요 시간: {time.time() - start_time:.2f} 초")

    return db


def normalize_content(content):
    return content.replace('\ufeff', '').strip()

def chatbot_s(content):
    vector_store_instance = load_vector_store()
    docs = vector_store_instance.similarity_search(content, k=30)
        # Using a set to track seen contents
    print(docs)
    seen = set()
    unique_documents = []

    for doc in docs:
        content = normalize_content(doc.page_content)
        if content not in seen:
            unique_documents.append({'page_content': content})
            seen.add(content)
    result=[]
    # 결과 출력
    for doc in unique_documents:
        result.append(doc['page_content'])
    print(result)
    return jsonify({"reply": result})



