
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import time
import os

os.environ['CUDA_DEVICE_ORDER']="FASTEST_FIRST"
os.environ['CUDA_VISIBLE_DEVICES']='0'
   

def load_vector_store():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('GPU로드', device)
    # print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    # GPU 사용 가능 -> True, GPU 사용 불가 -> False
    print(torch.cuda.is_available())
  
    print("직능계 벡터 스토어 로드를 시작합니다...")
    embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device':device},
    encode_kwargs={'normalize_embeddings':True},
    )

    # 벡터 스토어 로드
    db = Chroma(
        collection_name='combined_output',
        persist_directory='./db_job/chromadb',
        embedding_function= embeddings_model
    )
    print(f"직능계 벡터 스토어 로드 완료. 소요 시간: {time.time() - start_time:.2f} 초")

    return db



def chatbot_j(input):
    vector_store_instance = load_vector_store()
    model = ChatOllama(model='gemma2:9b', temperature=0)

    # 프롬프트 설정
    template = '''당신은 Ai Job 컨설턴트입니다. 컨설턴트의 자세를 가져주세요. 인사는 필수입니다.
    인사는 유동적이게 위트있고 센스있게 부탁해요.
    제공되는 정보제공자의 자격, 직업훈련, 경력에 관한 정보를 바탕으로 커리어 개발을 상세히 추천해주십시오.
    정보제공자에게 직업, 적성, 교육 등 직업에 대한 정보를 상세하게 제공해주십시오.
    또한, 국비/훈련 등의 단어는 제외해주시고, 정보제공자의 배경지식과 관련없이 잘 이해되도록 쉬운 문장을 구사해 주십시오.
    답변의 대분류는 3개야
    정보제공자의 강점과 장점, 직업 적성에 관해 이해하고  상세하고 자세하고 친절하게 이야기해주세요.
    정보에 포함된 직업 적성과 전망에 관해 이야기하고 경력개발을 위해 발전시킬 능력들을 설명해주세요.
    맞춤형 추천 교육을 제시하고  해당 교육의 hrd-net url을 보여줘.
    '저희는 당신의 꿈을 응원합니다.' 문장을 추가하면서 마무리해주세요.
    문단으로 나눠서 해주세요.
    
    번호는 표출할 때 빼주세요.
    내가 말한 걸 답변만들어주시고 그대로 이야기하지 말아주세요.
    
    : {context}
    질문: 
    {question}
    '''
    template1 = '''
    너가 대답한 {context1}을 토대로 hrdnet 훈련 교육명, 교육 한문장요약, 그리고 해당 교육들의 url을 찾아서 나타내어 상세하게 정보를 제공해라.
    마지막으로 'Super 이끌림은 당신의 꿈을 응원합니다.' 문장을 추가해주십시오.

    상세한 답변들의 포맷을 정확히 구분해주십시오.
    : 
    질문: 
    {question}
    '''

   

    prompt = ChatPromptTemplate.from_template(template)
    # prompt1 = ChatPromptTemplate.from_template(template1)
    # 리트리버 설정
    retriever = vector_store_instance.as_retriever()

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
    # rag_chain2 = (
    #     {'context1': rag_chain, 'question': RunnablePassthrough()}
    #     | prompt1
    #     | model
    #     | StrOutputParser()
    # )

    # 특정 질문으로 체인 호출
    output = rag_chain.invoke(input) ##여기에 인풋
    # output2  = rag_chain2.invoke(input)
    # output=[output1,output2]
  

# 상세한 답변 출력
    return output