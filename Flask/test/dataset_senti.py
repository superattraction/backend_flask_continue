from sqlalchemy import create_engine, Column, Integer, String, Text, Numeric,ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from review_senti.model_0709 import test_sentences
from review_senti.preprocess import parse_review3
import numpy as np
from flask import Flask

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
    course_id = Column(Integer, primary_key=True)
    course_name = Column(String(255), nullable=False)
    ncs_num = Column(String(50), nullable=False)
    edu_institute = Column(String(255),nullable=False)
    training_type = Column(String(255),nullable=False)
    reviews = Column(Text, nullable=False)
    address2 = Column(String(255),nullable=False)
    # goals = Column(Text, nullable=False)

# class SummaryReview(Base):
#     __tablename__ = 'summary_review'
#     course_id = Column(Integer, ForeignKey('maindata.course_id'),primary_key=True,nullable=False)
#     course_name = Column(String(500), nullable=False)
#     summary_review = Column(String(5000), nullable=False)

class Review(Base):
    __tablename__ = 'sentiment_result'
    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, nullable=False)
    course_name = Column(String(255), nullable=False)
    ncs_num = Column(String(8), nullable=False)
    address = Column(String(255),nullable=False)
    training_type = Column(String(255),nullable=False)
    edu_institute = Column(String(255),nullable=False)
    round = Column(Integer, nullable=False)
    review = Column(Text, nullable=False)
    result = Column(Numeric(10, 1))

# 세션 생성
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


## 감성 분석 처리 함수
def process_sentiments():
    session = Session()

    # 마지막으로 처리된 course_id를 가져오기
    last_processed_course = session.query(Review.course_id).order_by(Review.course_id.desc()).first()
    last_course_id = last_processed_course[0] if last_processed_course else None

    # 마지막으로 처리된 course_id 다음의 course들을 가져오기
    if last_course_id:
        courses = session.query(FinalData).filter(FinalData.course_id > last_course_id).all()
    else:
        courses = session.query(FinalData).all()

    for course in courses:
        existing_course = session.query(Review).filter_by(course_id=course.course_id).first()
        if not existing_course:
            course_entry = Review(
                ncs_num=course.ncs_num,
                course_id=course.course_id,
                course_name=course.course_name,
                training_type = course.training_type,
                edu_institute = course.edu_institute,
                address = course.address2
            )
            session.add(course_entry)

        reviews_text = course.reviews
        if reviews_text.strip() == "{}":
            review_entry = Review(
                course_id=course.course_id,
                course_name=course.course_name,
                ncs_num=course.ncs_num,
                training_type=course.training_type,  
                edu_institute=course.edu_institute,
                address = course.address2,
                round=0,
                review='수강후기가 없습니다.',
                result=None
            )
            session.add(review_entry)
            print(f"{course.course_name}에 수강후기가 없습니다.")
        else:
            try:
                all_reviews = parse_review3(reviews_text)
                if not all_reviews:
                    review_entry = Review(
                        course_id=course.course_id,
                        course_name=course.course_name,  
                        ncs_num=course.ncs_num,  
                        training_type=course.training_type, 
                        edu_institute=course.edu_institute,  
                        address = course.address2,
                        round=0,
                        review='수강후기가 없습니다.',
                        result=None
                    )
                    session.add(review_entry)
                    print(f"{course.course_name}에 수강후기가 없습니다.")
                else:
                    for round_num, reviews in all_reviews.items():
                        for review in reviews:
                            print("[review]:", review)
                            prediction = test_sentences([review])
                            print("result", prediction)
                            prediction = np.argmax(prediction)
                            print(f"회차: {round_num}, 리뷰: {review}, 예측값: {prediction}")
                            review_entry = Review(
                                course_id=course.course_id,
                                course_name=course.course_name,  
                                ncs_num=course.ncs_num,  
                                training_type=course.training_type,  
                                edu_institute=course.edu_institute,  
                                address = course.address2,
                                round=int(round_num),
                                review=review,
                                result=prediction
                            )
                            session.add(review_entry)
            except Exception as e:
                print(f"리뷰 파싱 중 에러 발생: {str(e)}")
                print(f"{course.course_name}에 수강후기가 없습니다.")
                session.rollback()
        session.commit()
        print(f"{course.course_name}에 입력완.")
    print("감성 분석 처리 완료!")

if __name__ == '__main__':
    print("Starting sentiment analysis process...")
    process_sentiments()
    print("Process completed.")
    
