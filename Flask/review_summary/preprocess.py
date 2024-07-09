import re

def preprocess_text(text):
    # Remove non-Korean characters and specific unwanted phrases
    new_text = re.sub(r'[^\가-힣\s]', '', text)
    new_text = re.sub("회차수강후기", '', new_text)
    new_text = re.sub("등록된후기가없습니다", '', new_text)
    new_text = re.sub(r'회차\s*수강후기', '', new_text)
    new_text = re.sub(r'등록된\s*후기가\s*없습니다', '', new_text)
    return new_text.strip()

def parse_review1(reviews_text):
    filtered_list = []
    if isinstance(reviews_text, dict):
        for key, value in reviews_text.items():
            review = value[0]  # 각 값은 리스트로 되어 있으며, 첫 번째 요소를 가져옴
            split_reviews = re.split(r'\n', review)  # '\n'으로 분리하여 각 줄을 처리
            for text in split_reviews:
                processed_text = preprocess_text(text)
                if processed_text:  # 빈 문자열이 아닌 경우에만 추가
                    filtered_list.append(processed_text)
    else:
        if reviews_text[5] == '"':
            pattern = re.compile(r'\d+: \[\"(.*?)\"\]', re.DOTALL)
        else:
            pattern = re.compile(r'\d+: \[\'(.*?)\'\]', re.DOTALL)
        matches = pattern.findall(reviews_text)
        split_matches = [re.split(r'\\n\s*', match.strip('\'')) for match in matches][0]
        filtered_list = [preprocess_text(text) for text in split_matches if not preprocess_text(text) == '']
    return filtered_list

def parse_review3(reviews_text):
    filtered_list = parse_review1(reviews_text)
    reviews = []
    for review in filtered_list:
        # Skip reviews containing unwanted phrases
        if review and "회차수강후기" not in review and "등록된후기가없습니다" not in review:
            reviews.append(review)
    
    return reviews