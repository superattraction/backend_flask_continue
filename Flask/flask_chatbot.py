from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def chatbot(input):
    # CSV 파일 로드
    loader = CSVLoader(file_path='combined_df.csv', encoding='utf-8')
    data = loader.load()

    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=50,
        encoding_name='cl100k_base'
    )

    # 텍스트 분할
    final_text = []
    for i in range(len(data)):
        final_text.append(text_splitter.split_text(data[i].page_content)[0])

    # 임베딩 모델 설정
    embeddings_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-nli',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    # 벡터 스토어 생성 및 저장
    db = Chroma.from_texts(
        final_text,
        embeddings_model,
        collection_name='LLM_output',
        persist_directory='./db/chromadb',
        collection_metadata={'hnsw:space': 'cosine'},  # l2 is the default
    )
    # 메모리 초기화 후 ChatOllama 모델 설정
    model = ChatOllama(model='gemma2:9b', temperature=0)

    # 프롬프트 설정
    template = '''당신은 AI 잡 컨설턴드입니다. 
    제공되는 상담자의 자격, 직업훈련, 경력에 관한 정보를 바탕으로 경력개발을 추천해주세요
    상담자에게 직업, 적성, 훈련 등 직업에 대한 정보를 상세하게 제공해주세요: {context}
    질문: {question}
    '''

    prompt = ChatPromptTemplate.from_template(template)

    # 리트리버 설정
    retriever = db.as_retriever()

    # 검색된 문서 포맷팅 함수
    def format_docs(docs):
        # 검색된 문서를 연결하여 컨텍스트를 제공
        return '\n\n'.join(doc.page_content for doc in docs)

    # RAG 체인 설정
    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # 특정 질문으로 체인 호출
    output = rag_chain.invoke(input) ##여기에 인풋


# 상세한 답변 출력
    return output