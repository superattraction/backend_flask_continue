from flask import jsonify
from flask_chatbot import chatbot
import logging
import requests
def pdfOff(): 
    try:
        data = requests.get_json()
        if data is None or not isinstance(data, str):
            logging.error("No input")
            return jsonify({"Result: False"}),400
        
        user_input = data
        print(f"user_input내용: {user_input}")
        chat_responses = chatbot(user_input)
        logging.info("Analysis successful for str")
        return jsonify({"response": chat_responses}),200
    except Exception as e:
        logging.error(f"Error processing:{e}")
        return jsonify({"error": str(e)}), 500