#라이브러리 임포트
import torch
from transformers import BertTokenizer
import tensorflow as tf
from torch.nn.utils.rnn  import pad_sequence
# 데이터 전처리

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('GPU로드', device)
model = torch.load("modelHRDreview_sentiment_model.pt")
model.to(device)
model.eval()

# 입력 데이터 변환
def convert_input_data(sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 문장을 텐서로 변환
    input_ids = [torch.tensor(seq) for seq in input_ids]

    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    # 길이가 MAX_LEN을 넘는 경우 잘라냄
    input_ids = input_ids[:, :MAX_LEN]

    # 어텐션 마스크 초기화
    attention_masks = torch.tensor([[float(i > 0) for i in seq] for seq in input_ids], dtype=torch.float)

    return input_ids, attention_masks

def test_sentences(sentences):
   

    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    input_ids, attention_masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = input_ids.to(device)
    b_input_mask = attention_masks.to(device)

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
