from sqlalchemy import create_engine, Column, Integer, String, Text, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from review_senti.preprocess import parse_review4
from review_senti.model_0709 import test_sentences
from review_senti.preprocess import parse_review3
import numpy as np

# 데이터베이스 설정
username = 'hrd'
password = '1234'
hostname = '10.125.121.212'
port = '3306'
database_name = 'job2'

# SQLAlchemy 설정
DATABASE_URI = f'mysql+pymysql://{username}:{password}@{hostname}:{port}/{database_name}'
engine = create_engine(DATABASE_URI)
Base = declarative_base()

# 데이터베이스 모델 정의
class FinalData(Base):
    __tablename__ = 'maindata'
    # id = Column(Integer, primary_key=True)
    # ncs_num = Column(String(50), nullable=False)
    course_id = Column(Integer, primary_key=True)
    course_name = Column(String(255), nullable=False)
    reviews = Column(Text, nullable=False)
    # goals = Column(Text, nullable=False)

class SummaryReview(Base):
    __tablename__ = 'summary_review'
    course_id = Column(Integer, ForeignKey('maindata.course_id'),primary_key=True,nullable=False)
    course_name = Column(String(500), nullable=False)
    summary_review = Column(String(5000), nullable=False)

# class CourseSenti(Base):
#     __tablename__ = 'courses_senti'
#     id = Column(Integer, primary_key=True)
#     ncs_num = Column(String(50), nullable=False)
#     course_id = Column(String(50), nullable=False, unique=True)
#     course_name = Column(String(255), nullable=False)

# class Review(Base):
#     __tablename__ = 'reviews_senti'
#     id = Column(Integer, primary_key=True)
#     course_id = Column(String(50), ForeignKey('courses_senti.course_id'), nullable=False)
#     round = Column(Integer, nullable=False)
#     review = Column(Text, nullable=False)
#     result = Column(Numeric(10, 5))

# 세션 생성
Session = sessionmaker(bind=engine)
session = Session()

# Prompt
prompt = ChatPromptTemplate.from_template(
    ''' 
    다음 문장은 직업훈련강의에 대한 수강후기들이다. 이문장의 주요특징을 잘 선별해서 요약하여 소감을 한문장으로 나타낸다.
    주요특징에는 직업, 훈련내용, 강의내용, 교사, 배운점, 배울점, 앞으로의 방향들 같은 직업과 미래에 관련된 내용들이 있을 것이다.
    한글을 준수하자. 특성을 세밀히 파악하고 그것을 적절히 요약해서 후기를 남기는 어투여야해. 한문장으로 요약이야.
    : {Sentence}'''
)

# 모델 초기화
model = ChatOllama(model="gemma2:9b", temperature=0)
chain = prompt | model | StrOutputParser()  # 문자열 출력 파서를 사용합니다.

# 요약 처리 함수
def process_summaries():
    print("요약시작")
    # SummaryReview 테이블에서 마지막으로 처리된 course_id를 가져옵니다.
    last_processed = session.query(SummaryReview.course_id).order_by(SummaryReview.course_id.desc()).first()
    last_processed_id = last_processed if last_processed else None

    # 마지막으로 처리된 course_id 이후의 모든 강의를 가져옵니다.
    query = session.query(FinalData)
    if last_processed_id:
        query = query.filter(FinalData.course_id > last_processed_id)
    courses = query.all()

    for course in courses:
        reviews_text = course.reviews
        if reviews_text.strip() == "{}":
            summary_result = SummaryReview(
                course_name=course.course_name,
                course_id=course.course_id,
                summary_review='수강후기가 없습니다.'
            )
        else:
            try:
                allreview = parse_review3(reviews_text)
                print("AI에 input될 문장: ", allreview)
                if not allreview:
                    summary_result = SummaryReview(
                        course_name=course.course_name,
                        course_id=course.course_id,
                        summary_review='수강후기가 없습니다.'
                    )
                else:
                    response = chain.invoke({"Sentence": allreview})
                    print("AI의 답변: ", response)
                    summary_result = SummaryReview(
                        course_name=course.course_name,
                        course_id=course.course_id,
                        summary_review=response
                    )
            except Exception as e:
                print("에러 발생:", str(e))
                summary_result = SummaryReview(
                    course_name=course.course_name,
                    course_id=course.course_id,
                    summary_review='수강후기가 없습니다.'
                )
        session.add(summary_result)
        # 각 강의를 처리한 후에 커밋하여 진행 상황을 저장합니다.
        session.commit()
    print("요약 처리 완료!")

# 감성 분석 처리 함수
# def process_sentiments():
#     courses = session.query(FinalData).all()
#     for course in courses:
#         existing_course = session.query(CourseSenti).filter_by(course_id=course.course_id).first()
#         if not existing_course:
#             course_entry = CourseSenti(
#                 ncs_num=course.ncs_num,
#                 course_id=course.course_id,
#                 course_name=course.course_name,
#             )
#             session.add(course_entry)

#         reviews_text = course.reviews
#         if reviews_text.strip() == "{}":
#             review_entry = Review(
#                 course_id=course.course_id,
#                 round=0,
#                 review='수강후기가 없습니다.',
#                 result=None
#             )
#             session.add(review_entry)
#             print(f"{course.course_name}에 수강후기가 없습니다.")
#         else:
#             try:
#                 all_reviews = parse_review4(reviews_text)
#                 if not all_reviews:
#                     review_entry = Review(
#                         course_id=course.course_id,
#                         round=0,
#                         review='수강후기가 없습니다.',
#                         result=None
#                     )
#                     session.add(review_entry)
#                     print(f"{course.course_name}에 수강후기가 없습니다.")
#                 else:
#                     for round_num, reviews in all_reviews.items():
#                         for review in reviews:
#                             print("[review]:", review)
#                             prediction = test_sentences([review])
#                             print("result", prediction)
#                             prediction = np.argmax(prediction)
#                             print(f"회차: {round_num}, 리뷰: {review}, 예측값: {prediction}")
#                             review_entry = Review(
#                                 course_id=course.course_id,
#                                 round=int(round_num),
#                                 review=review,
#                                 result=prediction
#                             )
#                             session.add(review_entry)
#             except Exception as e:
#                 print(f"리뷰 파싱 중 에러 발생: {str(e)}")
#                 print(f"{course.course_name}에 수강후기가 없습니다.")
#                 session.rollback()
#         session.commit()
#         print(f"{course.course_name}에 입력완.")
#     print("감성 분석 처리 완료!")

if __name__ == '__main__':
    process_summaries()
    # process_sentiments()
