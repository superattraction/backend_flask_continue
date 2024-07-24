from flask import jsonify, request, make_response
from chatbot.chatbot_job import chatbot_j
import logging
def pdfOn(key,ocr_data):
    key = int(key)
    user_input = ocr_data[str(key)]
    print("유저 인풋이야:" ,user_input)

    if user_input is None:
        logging.error(f"No OCR data found for key {key}")
        return jsonify({"error": f"No OCR data found for key {key}"}), 404
    

    try:
        chat_responses = chatbot_j(user_input)
        # 세션에서 해당 key를 제거
        logging.info(f"Analysis successful for key {key}")
        print("chat 내용:",chat_responses)
        accept_header = request.headers.get('Accept')
        
        if accept_header == 'application/json':
            response_json = jsonify({"response": chat_responses})
            return response_json, 200
        else:
            response = make_response(chat_responses)
            response.headers['Content-Type'] = 'text/markdown'
            return response, 200
        
        # return jsonify({"response": chat_responses}), 200
        
    except Exception as e:
        logging.error(f"Error processing key {key}: {e}")
        return jsonify({"error": str(e)}), 500