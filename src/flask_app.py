from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import subprocess
from pathlib import Path
import sys
import os
import pandas as pd

# Proje kök dizinini belirle
BASE_DIR = Path(__file__).resolve().parent.parent

# Flask uygulamasını oluştur
app = Flask(__name__, 
    template_folder=str(BASE_DIR / 'templates'),
    static_folder=str(BASE_DIR / 'static'))

# Model ve dönüştürücüleri yükle
def load_model():
    try:
        model = tf.keras.models.load_model(BASE_DIR / 'models' / 'news_classifier_model.keras')
        
        with open(BASE_DIR / 'models' / 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open(BASE_DIR / 'models' / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        return None, None, None
    
    return model, tokenizer, label_encoder

# Başlangıçta modeli yükle
model, tokenizer, label_encoder = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text')
        if not text:
            return jsonify({'error': 'Metin girilmedi'}), 400
        
        # Metni sayısal diziye dönüştür
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        
        # Sınıflandırma yap
        predictions = model.predict(padded_sequence)
        
        # Softmax uygula
        predictions = tf.nn.softmax(predictions).numpy()
        
        # Tüm kategoriler için olasılıkları hesapla
        categories = label_encoder.classes_
        probabilities = {}
        total_prob = 0
        
        # Her kategori için olasılıkları hesapla
        for cat, prob in zip(categories, predictions[0]):
            prob_value = float(prob)
            probabilities[cat] = prob_value
            total_prob += prob_value
        
        # Olasılıkları normalize et
        if total_prob > 0:
            for cat in probabilities:
                probabilities[cat] = (probabilities[cat] / total_prob) * 100
        
        # En yüksek olasılıklı sınıfı bul
        predicted_class = max(probabilities.items(), key=lambda x: x[1])[0]
        max_probability = probabilities[predicted_class]
        
        # Türkçe kategori isimleri
        category_names = {
            'business': 'İş',
            'entertainment': 'Eğlence',
            'politics': 'Siyaset',
            'sport': 'Spor',
            'tech': 'Teknoloji'
        }
        
        # Sonuçları hazırla
        formatted_probabilities = {
            category_names.get(cat, cat): f"%{prob:.1f}"
            for cat, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        }
        
        # Güven seviyesi kategorileri
        confidence_level = "Yüksek" if max_probability >= 70 else "Orta" if max_probability >= 40 else "Düşük"
        
        return jsonify({
            'prediction': f'En Yüksek Olasılık: {category_names.get(predicted_class, predicted_class)}',
            'confidence': f'Güven: {confidence_level} (%{max_probability:.1f})',
            'probabilities': formatted_probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_article', methods=['POST'])
def add_article():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'category' not in data:
            return jsonify({'error': 'Gerekli veriler eksik'}), 400

        text = data['text']
        category = data['category']
        
        # Metni temizle
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.replace(',', ' ')
        text = ' '.join(text.split())
        
        # Veri dosyasını kontrol et
        train_file = BASE_DIR / 'data' / 'train_dataset.csv'
        if not train_file.exists():
            # Yeni dosya oluştur
            pd.DataFrame(columns=['category', 'text']).to_csv(train_file, index=False)
        
        # Yeni veriyi ekle
        new_data = pd.DataFrame({'category': [category], 'text': [text]})
        new_data.to_csv(train_file, mode='a', header=False, index=False)
        
        return jsonify({'message': 'Haber başarıyla eklendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Eğitim sürecini başlat
        subprocess.Popen([sys.executable, str(BASE_DIR / 'src' / 'train_enhanced_model.py')],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        return jsonify({'message': 'Model eğitimi başlatıldı'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training_progress')
def training_progress():
    try:
        progress_file = BASE_DIR / 'training_progress.txt'
        if progress_file.exists():
            progress = progress_file.read_text().strip()
            return jsonify({'progress': float(progress)})
        return jsonify({'progress': 0})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_training_progress')
def get_training_progress():
    try:
        progress_file = BASE_DIR / 'training_progress.txt'
        if not progress_file.exists():
            return jsonify({'progress': 0})
        
        progress = progress_file.read_text().strip()
        return jsonify({'progress': float(progress) if progress else 0})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    try:
        global model, tokenizer, label_encoder
        model, tokenizer, label_encoder = load_model()
        return jsonify({'message': 'Model başarıyla yeniden yüklendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
