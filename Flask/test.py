from review_senti.preprocess import parse_review3
from review_senti.model_0709 import test_sentences
import numpy as np


a = parse_review3(str({1: ['1회차 수강후기\n나름 바쁘게 시간쪼개서 수업을 들었던터라 합격하고나니 더 뿌듯하다\n요리의 기본적인 것에 대해 배울 수 있었으며 나아가 한식을 조리하는 과정까지 체계적으로 배울 수 있어 좋았습니다.\n칼질을 해본 적이 없었는데 자세히 배울 기회가 되었습니다.'], 2: ['2회차 수강후기\n기초적인 부분 실기 연습시간이 없어서 좀\n아쉬웠다. \n한식 조리사 실기및 필기시험에  좋은 결과가 있도록 효과적으로 수업이 진행되었다\n선생님이 친절하게 잘 알려주셔서 좋았습니다\n수강을 통해 한번에 자격증 취득ㅇ삼'], 3: ['3회차 수강후기\n강사님의 오랜 지식으로 쉽게 알려주시고 시험대비 잘할수 있도록 알려주셔서 도움이 많이 되었어요']}))


for round_num, reviews in a.items():
     for review in reviews:
        print("[review]:", review)
        prediction = test_sentences([review])
        print("result", prediction)
        prediction = np.argmax(prediction)
        print(f"회차: {round_num}, 리뷰: {review}, 예측값: {prediction}")

print(a) 