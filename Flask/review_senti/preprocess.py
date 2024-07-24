import re
import ast

def preprocess_text(text):
    new_text = re.sub(r'[^\가-힣\s]', '', text)
    new_text = re.sub("회차수강후기", '', new_text)
    new_text = re.sub("등록된후기가없습니다", '', new_text)
    new_text = re.sub(r'회차\s*수강후기', '', new_text)
    new_text = re.sub(r'등록된\s*후기가\s*없습니다', '', new_text)
    return new_text.strip()

def parse_review1(reviews_text):
    filtered_dict = {}
    if isinstance(reviews_text, dict):
        for key, value in reviews_text.items():
            review = value[0]
            split_reviews = re.split(r'\n', review)
            processed_reviews = [preprocess_text(text) for text in split_reviews if preprocess_text(text)]
            if processed_reviews:
                filtered_dict[key] = processed_reviews
    else:
        if reviews_text[5] == '"':
            pattern = re.compile(r'\d+: \[\"(.*?)\"\]', re.DOTALL)
        else:
            pattern = re.compile(r'\d+: \[\'(.*?)\'\]', re.DOTALL)
        matches = pattern.findall(reviews_text)
        split_matches = [re.split(r'\\n\s*', match.strip('\'')) for match in matches]
        for i, match in enumerate(split_matches):
            processed_reviews = [preprocess_text(text) for text in match if preprocess_text(text)]
            if processed_reviews:
                filtered_dict[i + 1] = processed_reviews
    return filtered_dict

def parse_review3(reviews_text):
    reviews_text1 = ast.literal_eval(reviews_text)
    filtered_dict = parse_review1(reviews_text1)
    reviews = {}
    for key, review_list in filtered_dict.items():
        reviews[key] = [review for review in review_list if "회차수강후기" not in review and "등록된후기가없습니다" not in review]
    return reviews
