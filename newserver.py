from flask import Flask, request, jsonify
import joblib
import re
from pyvi import ViTokenizer
import pandas as pd
from datetime import datetime
import requests 
from collections import OrderedDict
import json

app = Flask(__name__)

# Load mô hình và các đối tượng đã huấn luyện
word_vectorizer = joblib.load('./tfidf_vectorizer.pkl')
model_rf_device = joblib.load('./rf_device.pkl')
model_rf_room = joblib.load('./rf_room.pkl')
model_rf_action = joblib.load('./rf_action.pkl')

# Load các bộ mã hóa nhãn
label_encoder_device = joblib.load('./label_encoder_device.pkl')
label_encoder_room = joblib.load('./label_encoder_room.pkl')
label_encoder_action = joblib.load('./label_encoder_action.pkl')

# Tiền xử lý văn bản
vietnamese_punctuations = "0123456789…""''–" + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@[^\s]+', ' ', text) 
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', text) 
    translator = str.maketrans('', '', vietnamese_punctuations)
    text = text.translate(translator)
    text = ViTokenizer.tokenize(text)  
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text")

    processed_text = preprocess_text(text)
    text_vect = pd.DataFrame(word_vectorizer.transform([processed_text]).toarray(), columns=word_vectorizer.get_feature_names_out())

    device_pred = model_rf_device.predict(text_vect)[0]
    room_pred = model_rf_room.predict(text_vect)[0]
    action_pred = model_rf_action.predict(text_vect)[0]

    device_label = label_encoder_device.inverse_transform([device_pred])[0]
    room_label = label_encoder_room.inverse_transform([room_pred])[0]
    action_label = label_encoder_action.inverse_transform([action_pred])[0]

    if device_label == "thời gian" and room_label == "giờ":
        current_time = datetime.now().strftime("%H:%M")
        return jsonify({
            'message': f'Bây giờ là {current_time}'
        })
    
    if device_label == "thời gian" and room_label == "ngày":
        current_date = datetime.now().strftime("%d/%m/%Y")
        return jsonify({
            'message': f'Hôm nay là {current_date}'
        })
    
    if device_label == "thời tiết":
        location = "Ho Chi Minh"
        api_key = "5be629da4c48b6ff8b51911437540e8b"
        weather_url = f"http://api.weatherstack.com/current?access_key={api_key}&query={location}"

        try:
            weather_response = requests.get(weather_url)
            if weather_response.status_code == 200:
                weather_data = weather_response.json()
                if 'current' in weather_data:
                    temperature = weather_data['current']['temperature']
                    weather_descriptions = weather_data['current']['weather_descriptions'][0]

                    if room_label == "nhiệt độ":
                        return jsonify({
                            'message': f"Nhiệt độ là {temperature}°C"
                        })
                    elif room_label == "thời tiết":
                        return jsonify({
                            'message': f"Thời tiết hôm nay là: {weather_descriptions}"
                        })
                else:
                    return jsonify({"error": "Không tìm thấy thông tin thời tiết cho địa điểm này."}), 404
            else:
                return jsonify({"error": "Không thể kết nối đến API thời tiết."}), 500
        except Exception as e:
            return jsonify({"error": f"Đã xảy ra lỗi: {str(e)}"}), 500

    if(room_label=="phòng khách"):
        room_label="room_0001"
    if(room_label=="phòng ngủ"):
        room_label="room_0002"
    if(device_label=="đèn"):
        device_label= "device_0001"
    if(device_label=="quạt"):
        device_label= "device_0002"
    if(action_label=="on"):
        action_label="1"
    if(action_label=="off"):
        action_label="0"
    if device_label=="device_0001" and action_label=="1" and room_label=="không xác định":
        return jsonify({"question": "Bạn muốn bật đèn ở phòng nào"}), 200
    if device_label=="device_0001" and action_label=="0" and room_label=="không xác định":
        return jsonify({"question": "Bạn muốn tắt đèn ở phòng nào"}), 200
    if device_label=="device_0002" and action_label=="1" and room_label=="không xác định":
        return jsonify({"question": "Bạn muốn bật quạt ở phòng nào"}), 200
    if device_label=="device_0002" and action_label=="0" and room_label=="không xác định":
        return jsonify({"question": "Bạn muốn tắt quạt ở phòng nào"}), 200          
    
    response_data = {
        'station_id': room_label,
        'device_id': device_label,
        'device_value': action_label
    }
    return response_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
