#라이브러리 임포트
import tensorflow as tf
import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import time
import datetime


# 데이터 전처리
import re

# model = torch.load("modelHRDreview_sentiment_model.pt")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


model = torch.load("modelHRDreview_sentiment_model.pt")
model.to(device)
model.eval()
# torch.save(model.state_dict(), "modelHRDreview_sentiment_model.pt")


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
# 입력 데이터 변환
def convert_input_data(sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # 어텐션 마스크 초기화
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids,dtype=torch.long)
    masks = torch.tensor(attention_masks, dtype=torch.float)

    return inputs, masks
#잠깐만 ㅠㅠㅠ
def test_sentences(sentences):

    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 출력 로짓 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    return logits

# def test_sentences(sentences):
#     model.eval()  # 모델을 평가 모드로 설정

#     inputs, masks = convert_input_data(sentences)  # 입력 데이터 변환
#     b_input_ids = inputs.to(device)  # 입력 데이터를 디바이스로 이동
#     b_input_mask = masks.to(device)

#     with torch.no_grad():  # 그래디언트 계산 없이 진행
#         outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
#     logits = outputs.logits  # 모델의 로짓 값 구하기
#     print("logits:", logits)

#     # return logits

#     # NaN 값 처리 (필요한 경우)
#     logits = torch.nan_to_num(logits, nan=0.0)  # NaN 값을 0으로 대체
#     print("logits after NaN handling:", logits)

#     # # 로짓 값을 소프트맥스 함수로 확률로 변환
#     probs = F.softmax(logits, dim=-1)
#     print("probs:", probs)

#     # 가장 높은 확률을 가지는 클래스 인덱스 구하기
#     predicted_class = torch.argmax(logits)
#     print("predicted_class:", predicted_class)
    
#     return predicted_class.cpu().numpy()

# def test_sentence(sentence):
#     model.eval()  # 모델을 평가 모드로 설정

#     inputs, masks = convert_input_data([sentence])  # 입력 데이터 변환
#     print("inputs:", inputs)
#     print("masks:", masks)

#     b_input_ids = inputs.to(device)  # 입력 데이터를 디바이스로 이동
#     b_input_mask = masks.to(device)

#     with torch.no_grad():  # 그래디언트 계산 없이 진행
#         outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
#     logits = outputs.logits  # SequenceClassifierOutput 객체에서 logits 추출
#     print("logits:", logits)

#     # NaN 값 처리
#     logits = torch.nan_to_num(logits, nan=0.0)  # NaN 값을 0으로 대체
#     print("logits after NaN handling:", logits)

#     # 로짓 값을 소프트맥스 함수로 확률로 변환
#     probs = F.softmax(logits, dim=-1)
#     print("probs:", probs)

#     # 임계값을 0.5로 설정하여 예측
#     if probs[0][1] > 0.6:
#         predicted_class = torch.tensor([1])
#     else:
#         predicted_class = torch.tensor([0])
    
#     print("predicted_class:", predicted_class)
    
#     return predicted_class.cpu().numpy()
