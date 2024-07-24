
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['CUDA_DEVICE_ORDER']="FASTEST_FIRST"
os.environ['CUDA_VISIBLE_DEVICES']='0'


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
   


def chatbot_h():
    # 모델 로드 (여기서는 예시로 Gemma2:9b 모델을 사용한다고 가정)
    model = ChatOllama(model='gemma2:9b',format='json', temperature=0)

    # 프롬프트 설정
    template = '''{Sentence} 포맷은 message를 키값으로 한문장으로 해야해. 
    인사는 센스있고 정중하게 그리고 참신하게 매번 다양하게 제공하세요. 참신하고 멋진 인사를 기대합니다.
    resonse in JSON format.'''
    hello = "당신은 AI job컨설턴트입니다. 이용자는 여러 커리어에 관한 고민 및 개발을 하려는 사람. "
    prompt = ChatPromptTemplate.from_template(template)
    # 체인 생성
    chain = prompt | model | StrOutputParser()
    # answer = chain.stream({"topic": "deep learning"})
    response = chain.invoke({"Sentence": hello})
    # 인사말 생성 호출
    output = response
    return output
