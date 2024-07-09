from flask import jsonify, session
from flask_chatbot import chatbot
import logging
def pdfOn(key,ocr_data):
    key = int(key)
    user_input = ocr_data[str(key)]
  

    if user_input is None:
        logging.error(f"No OCR data found for key {key}")
        return jsonify({"error": f"No OCR data found for key {key}"}), 404
    

    try:
        chat_responses = chatbot(user_input)
        # 세션에서 해당 key를 제거
        logging.info(f"Analysis successful for key {key}")
        print("chat 내용:",chat_responses)
        return jsonify({"response": chat_responses}), 200
        
    except Exception as e:
        logging.error(f"Error processing key {key}: {e}")
        return jsonify({"error": str(e)}), 500