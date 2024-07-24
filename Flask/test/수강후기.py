from flask import Flask,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from review_senti.preprocess import  parse_review3

app = Flask(__name__)
CORS(app)
app.secret_key = 'oj_super_key'

# 데이터베이스 설정
username = 'hrd'
password = '1234'
hostname = '10.125.121.212'  # MySQL 서버의 IP 주소
port = '3306'  # MySQL 기본 포트
database_name = 'job'


app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{username}:{password}@{hostname}:{port}/{database_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class FinalData(db.Model):
    __tablename__ = 'final_data'
    id = db.Column(db.Integer, primary_key=True)
    course_name = db.Column(db.String(255), nullable=False)
    reviews = db.Column(db.Text, nullable=False)
    goals = db.Column(db.Text, nullable=False)

class SummaryReview(db.Model):
    __tablename__ = 'summary_review_copy'
    id = db.Column(db.Integer, primary_key=True)
    course_id=db.Column(db.Integer, nullable=False)
    course_name = db.Column(db.String(500), nullable=False)
    summary_review = db.Column(db.String(5000), nullable=False)


# Prompt
prompt = ChatPromptTemplate.from_template(
    ''' 
    다음 문장은 직업훈련강의에 대한 수강후기들이다. 이문장의 주요특징을 잘 선별해서 요약하여 소감을 한문장으로 나타낸다.
    주요특징에는 직업, 훈련내용, 강의내용, 교사, 배운점, 배울점, 앞으로의 방향들 같은 직업과 미래에 관련된 내용들이 있을 것이다.
    한글을 준수하자.
    : {Sentence}'''
    # output_variables=['summary']
)

##모9
model = ChatOllama(model="gemma2:9b",temperature=0)
chain = prompt | model | StrOutputParser()  # 문자열 출력 파서를 사용합니다.


@app.route('/api/chat/reviewsummary', methods=['POST'])
def chat_summary():
    courses = FinalData.query.all()
    try:
        for course in courses:
            # Parse the reviews text to extract individual reviews
            reviews_text = course.reviews
            #리뷰 없을 때
            if reviews_text.strip() == "{}":
                summary_result = SummaryReview(
                    course_name=course.course_name,
                    course_id = course.id,
                    summary_review = '수강후기가 없습니다.'
                )
            #리뷰 있을 때
            else:
                try:
                    allreview= parse_review3(reviews_text)
                    # allreview = ''.join(allreview1)
                    print("ai에 input될 문장: ", allreview)
                    if not allreview:
                        summary_result = SummaryReview(
                        course_name=course.course_name,
                        course_id = course.id,
                        summary_review = '수강후기가 없습니다.'
                        )
                    else:
                        # chain 실행
                        response=chain.invoke({"Sentence":
                                            allreview})
                        print
                        print("ai의 답변: ", response)
                        summary_result = SummaryReview(
                            course_name=course.course_name,
                            course_id = course.id,
                            summary_review = response
                        )
                except:
                    print("에러발생")
                    summary_result = SummaryReview(
                        course_name=course.course_name,
                        course_id = course.id,
                        summary_review = '수강후기가 없습니다.'
                    )
            db.session.add(summary_result)        
        db.session.commit()
        print("데이터베이스 커밋 완료!")
        return jsonify({"message": "Sentiments processed and stored successfully"})
    except Exception as e:
        print(f"데이터베이스 커밋 중 에러 발생: {str(e)}")
        return jsonify({"error": "Database commit failed"}), 500
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
