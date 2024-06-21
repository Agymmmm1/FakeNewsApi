from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
from bs4 import BeautifulSoup
import requests
from process import process_text as pt
import __main__
__main__.process_text = pt

app = Flask(__name__)
CORS(app)

@app.before_request
def load_model():
    global model
    model = joblib.load(open('fakenewsmei152.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('home.html')

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

@app.route('/result', methods=['POST'])
def result():
    message = request.form.get('message')
    url = request.form.get('url')
    
    if url:
        message = extract_text_from_url(url)
    
    if not message:
        return jsonify({'error': 'No message or URL provided'}), 400
    
    data = [message]
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0]
    prob_fake = round((prob[1] * 100), 2)
    prob_true = round((prob[0] * 100), 2)
    
    result_text = 'Hoax' if prob_fake > prob_true else 'Fakta'
    
    return jsonify({
        'text': message,
        'prediction': result_text,
        'prob_fake': prob_fake,
        'prob_true': prob_true
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
