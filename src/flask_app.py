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
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحديد المسار الأساسي للمشروع
BASE_DIR = Path(__file__).resolve().parent.parent

# إنشاء تطبيق Flask
app = Flask(__name__, 
    template_folder=str(BASE_DIR / 'templates'),
    static_folder=str(BASE_DIR / 'static'))

# تحميل النموذج والمحولات
def load_model():
    try:
        model_path = BASE_DIR / 'models' / 'news_classifier_model.keras'
        tokenizer_path = BASE_DIR / 'models' / 'tokenizer.pkl'
        label_encoder_path = BASE_DIR / 'models' / 'label_encoder.pkl'
        
        # التحقق من وجود الملفات
        if not model_path.exists():
            raise FileNotFoundError(f"ملف النموذج غير موجود: {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"ملف tokenizer غير موجود: {tokenizer_path}")
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"ملف label_encoder غير موجود: {label_encoder_path}")
        
        # تحميل النموذج
        logger.info("جاري تحميل النموذج...")
        model = tf.keras.models.load_model(str(model_path))
        
        # تحميل tokenizer
        logger.info("جاري تحميل tokenizer...")
        with open(str(tokenizer_path), 'rb') as f:
            tokenizer = pickle.load(f)
        
        # تحميل label_encoder
        logger.info("جاري تحميل label_encoder...")
        with open(str(label_encoder_path), 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info("تم تحميل جميع المكونات بنجاح!")
        return model, tokenizer, label_encoder
        
    except Exception as e:
        logger.error(f"خطأ في تحميل النموذج: {str(e)}")
        return None, None, None

# تحميل النموذج عند بدء التطبيق
model, tokenizer, label_encoder = load_model()

# التحقق من تحميل النموذج
if None in (model, tokenizer, label_encoder):
    logger.error("فشل في تحميل النموذج أو المحولات!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # التحقق من تحميل النموذج
        if None in (model, tokenizer, label_encoder):
            return jsonify({
                'error': 'لم يتم تحميل النموذج بشكل صحيح. يرجى التحقق من وجود الملفات المطلوبة.'
            }), 500
        
        # التحقق من البيانات المدخلة
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'لم يتم تقديم نص للتصنيف'
            }), 400
        
        text = data['text']
        if not text or not isinstance(text, str):
            return jsonify({
                'error': 'النص فارغ أو غير صالح'
            }), 400
        
        # تحويل النص إلى تسلسل رقمي
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        
        # التنبؤ
        predictions = model.predict(padded_sequence)
        
        # تطبيق softmax
        predictions = tf.nn.softmax(predictions).numpy()
        
        # حساب الاحتمالات لكل الفئات
        categories = label_encoder.classes_
        probabilities = {}
        total_prob = 0
        
        # حساب الاحتمالات لكل فئة
        for cat, prob in zip(categories, predictions[0]):
            prob_value = float(prob)
            probabilities[cat] = prob_value
            total_prob += prob_value
        
        # تطبيع الاحتمالات
        if total_prob > 0:
            for cat in probabilities:
                probabilities[cat] = (probabilities[cat] / total_prob) * 100
        
        # العثور على الفئة الأكثر احتمالاً
        predicted_class = max(probabilities.items(), key=lambda x: x[1])[0]
        max_probability = probabilities[predicted_class]
        
        # أسماء الفئات
        category_names = {
            'business': 'أعمال',
            'entertainment': 'ترفيه',
            'politics': 'سياسة',
            'sport': 'رياضة',
            'tech': 'تكنولوجيا'
        }
        
        # تنسيق النتائج
        formatted_probabilities = {
            category_names.get(cat, cat): f"{prob:.1f}%"
            for cat, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        }
        
        # مستوى الثقة
        if max_probability >= 70:
            confidence_level = "عالي"
        elif max_probability >= 40:
            confidence_level = "متوسط"
        else:
            confidence_level = "منخفض"
        
        return jsonify({
            'tahmin': category_names.get(predicted_class, predicted_class),
            'güven': f"{confidence_level} ({max_probability:.1f}%)",
            'tüm_olasılıklar': formatted_probabilities
        })
        
    except Exception as e:
        logger.error(f"خطأ في التنبؤ: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_article', methods=['POST'])
def add_article():
    try:
        data = request.get_json()
        text = data['text']
        category = data['category']
        
        # حفظ المقال الجديد في ملف البيانات
        with open(str(BASE_DIR / 'data' / 'temp_article.csv'), 'a', encoding='utf-8') as f:
            f.write(f"{category},{text}\n")
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # تشغيل سكربت التدريب
        subprocess.Popen([sys.executable, str(BASE_DIR / 'src' / 'train_enhanced_model.py')])
        return jsonify({'success': True, 'message': 'بدأت عملية إعادة التدريب'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/training_progress')
def get_training_progress():
    try:
        progress_file = BASE_DIR / 'training_progress.txt'
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = f.read().strip()
            return jsonify({'success': True, 'progress': progress})
        return jsonify({'success': True, 'progress': 'لا توجد عملية تدريب جارية'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reload_model', methods=['POST'])
def reload_model():
    try:
        global model, tokenizer, label_encoder
        model, tokenizer, label_encoder = load_model()
        return jsonify({'success': True, 'message': 'تم إعادة تحميل النموذج بنجاح'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
