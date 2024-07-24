import os
import time
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

# CUDA 설정
os.environ['CUDA_DEVICE_ORDER'] = "FASTEST_FIRST"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def init_model(file_paths, persist_dir):
    start_time = time.time()
    print("Starting model initialization...")

    # CSV 파일 로드
    data_list = []
    for file_path in file_paths:
        loader = CSVLoader(file_path=file_path, encoding='utf-8')
        data = loader.load()
        data_list.extend(data)
    print(f"CSV files loaded. Time taken: {time.time() - start_time:.2f} seconds")

    # 텍스트 분할기 설정
    text_splitter_start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=50,
        encoding_name='cl100k_base'
    )
    print(f"Text splitter configured. Time taken: {time.time() - text_splitter_start_time:.2f} seconds")

    # 텍스트 분할
    text_split_start_time = time.time()
    final_text = []
    for data in data_list:
        final_text.append(text_splitter.split_text(data.page_content)[0])
    print(f"Text split completed. Time taken: {time.time() - text_split_start_time:.2f} seconds")

    print("직능계 벡터 스토어 로드를 시작합니다...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('GPU로드', device)
    # print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    # GPU 사용 가능 -> True, GPU 사용 불가 -> False
    print(torch.cuda.is_available())

    # 임베딩 모델 설정
    embeddings_model_start_time = time.time()
    embeddings_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-nli',
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True},
    )
    print(f"Embeddings model configured. Time taken: {time.time() - embeddings_model_start_time:.2f} seconds")

    # 벡터 스토어 생성 및 저장
    vector_store_start_time = time.time()
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    
    try:
        db = Chroma.from_texts(
            embedding=embeddings_model,
            texts=final_text,
            collection_name='combined_output',
            persist_directory=persist_dir
        )
        db.persist()
        print(f"Vector store created and saved. Time taken: {time.time() - vector_store_start_time:.2f} seconds")
    except Exception as e:
        print(f"Error occurred during vector store creation: {e}")
        raise  # Optional: re-raise the exception for further debugging

    total_time = time.time() - start_time
    print(f"Model initialization completed. Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    file_paths = ['./LLM학습 전처리0709.csv', './combined_df.csv']
    persist_dir = '../db_job/chromadb'
    init_model(file_paths, persist_dir)