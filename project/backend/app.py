# === backend/app.py ===
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

from preprocess import clean_text
app = Flask(__name__)
CORS(app)

# Load model
with open('sms_spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)




@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    message = clean_text(data.get('message', ''))
    prediction = model.predict([message])[0]
    return jsonify({"classification": prediction})

if __name__ == '__main__':
    app.run(debug=True)