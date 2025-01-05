from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import subprocess
from pathlib import Path
import sys
import os

# تحديد المسار الأساسي للمشروع
BASE_DIR = Path(__file__).resolve().parent.parent

# إنشاء تطبيق Flask
app = Flask(__name__, 
    template_folder=str(BASE_DIR / 'templates'),
    static_folder=str(BASE_DIR / 'static'))

# تحميل النموذج والمحولات
def load_model():
    model = tf.keras.models.load_model(str(BASE_DIR / 'models' / 'news_classifier_model.keras'))
    
    with open(str(BASE_DIR / 'models' / 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open(str(BASE_DIR / 'models' / 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder

# تحميل النموذج عند بدء التطبيق
model, tokenizer, label_encoder = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        # تحويل النص إلى تسلسل رقمي
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        
        # التنبؤ بالتصنيفات
        predictions = model.predict(padded_sequence)
        
        # تحويل التنبؤات إلى تصنيفات مع النسب
        categories = label_encoder.classes_
        predictions_with_labels = [
            {"category": cat, "probability": float(prob)}
            for cat, prob in zip(categories, predictions[0])
        ]
        
        # ترتيب التصنيفات حسب النسبة
        predictions_with_labels.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            "tahmin": predictions_with_labels[0]["category"],
            "güven": float(predictions_with_labels[0]["probability"]) * 100,
            "tüm_olasılıklar": {pred["category"]: float(pred["probability"]) * 100 for pred in predictions_with_labels}
        })
        
    except Exception as e:
        return jsonify({"hata": str(e)}), 500

@app.route('/add_article', methods=['POST'])
def add_article():
    try:
        data = request.get_json()
        text = data['text']
        category = data['category']
        
        # تنظيف النص: إزالة علامات السطر الجديد والفواصل
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.replace(',', ' ')  # استبدال الفواصل لتجنب مشاكل CSV
        text = ' '.join(text.split())  # إزالة المسافات الزائدة
        
        # إضافة المقال إلى ملف البيانات التدريبية في سطر واحد
        train_file = str(BASE_DIR / 'data' / 'train_dataset.csv')
        with open(train_file, 'a', newline='', encoding='utf-8') as f:
            f.write(f'{category},{text}\n')
        
        return jsonify({"mesaj": "Makale başarıyla eklendi"})
    except Exception as e:
        return jsonify({"hata": str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # تأكد من وجود الملف
        progress_file = BASE_DIR / 'training_progress.txt'
        if not progress_file.exists():
            progress_file.write_text('0')
        
        # تشغيل سكربت التدريب في عملية منفصلة
        train_script = str(BASE_DIR / 'src' / 'train_enhanced_model.py')
        subprocess.Popen([sys.executable, train_script], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        return jsonify({"mesaj": "Model yeniden eğitimi başlatıldı"})
    except Exception as e:
        return jsonify({"hata": str(e)}), 500

@app.route('/get_training_progress')
def get_training_progress():
    try:
        progress_file = BASE_DIR / 'training_progress.txt'
        if not progress_file.exists():
            return jsonify({"ilerleme": 0})
        
        progress = progress_file.read_text().strip()
        return jsonify({"ilerleme": float(progress) if progress else 0})
    except Exception as e:
        return jsonify({"hata": str(e)}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    global model, tokenizer, label_encoder
    model, tokenizer, label_encoder = load_model()
    return jsonify({"mesaj": "Model başarıyla yeniden yüklendi"})

if __name__ == '__main__':
    app.run(debug=True)
