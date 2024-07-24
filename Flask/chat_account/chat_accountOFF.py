from flask import jsonify
from chatbot.chatbot_job import chatbot_j
import logging
from urllib.parse import unquote
def pdfOff(data): 
    try:
        user_input = unquote(data)
        print(f"user_input내용: {user_input}")
        chat_responses = chatbot_j(user_input)
        logging.info("Analysis successful for str")
        return jsonify({"response": chat_responses}),200
    except Exception as e:
        logging.error(f"Error processing:{e}")
        return jsonify({"error": str(e)}), 500