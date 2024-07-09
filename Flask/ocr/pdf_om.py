
from ocr.pdf import ocrpdf
from flask import  jsonify, session
import os
import requests
def om(num):
    url_num = (f"http://10.125.121.212:9090/api/account/data/{num}")
    print(num)
    response = requests.get(url_num)
    print(response.json)
    print(url_num)
    # 응답 상태 코드 확인
    if response.status_code != 200:
        print("Request failed with status code:", response.status_code)
        return jsonify({'error': 'Failed to fetch uniquecode'}),404
    # 상태 코드가 200이면 응답 본문을 JSON으로 파싱

    filename = response.json().get("fileName")
    print("FileName:", filename)


    url_name = (f'http://10.125.121.212:9090/api/account/view/{filename}')
    response = requests.get(url_name)
    print(f"Request to {url_name} returned status code {response.status_code}")

    if response.status_code != 200:
        print("Request failed with status code:", response.status_code)
        return jsonify({'error': 'Failed to fetch filename'}), 404

    # Ensure the response content is saved as a PDF file
    pdf_file_path = f'{filename}'
    with open(pdf_file_path, 'wb') as f:
        f.write(response.content)

    # Call the ocrpdf function
    data = ocrpdf(pdf_file_path)

    # Clean up the downloaded PDF file
    os.remove(pdf_file_path)

    
    print("om data :",data)
    return data
 