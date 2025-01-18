"""
هذا الملف يحتوي على اختبارات للنموذج القديم المبني باستخدام TensorFlow
يقوم باختبار 3 مقالات من فئات مختلفة:
- الأعمال (business)
- الترفيه (entertainment)
- السياسة (politics)

ملاحظة: هذا النموذج لم يعد مستخدماً، تم استبداله بنموذج BERT الجديد.
النموذج الجديد موجود في ملف bert_model.py
"""

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TensorFlowNewsClassifier:
    def __init__(self):
        # تحميل النموذج
        self.model = tf.keras.models.load_model('news_classifier_model.keras')
        
        # تحميل Tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # تحميل Label Encoder
        with open('label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.max_len = 200  # نفس الطول المستخدم في التدريب
    
    def predict(self, text):
        # تحويل النص إلى vector
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=self.max_len)
        
        # التنبؤ
        prediction = self.model.predict(padded)
        
        # الحصول على الفئة المتوقعة
        predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
        
        # حساب نسبة الثقة
        confidence = float(np.max(prediction[0]))
        
        return {
            'category': predicted_class,
            'confidence': confidence,
            'probabilities': {
                category: float(prob)
                for category, prob in zip(self.label_encoder.classes_, prediction[0])
            }
        }

def test_articles():
    classifier = TensorFlowNewsClassifier()
    
    # مقالات اختبار من مختلف الفئات
    test_articles = [
        {
            'category': 'business',
            'text': """
            Apple's stock surges to record high as AI optimism drives tech rally. 
            The iPhone maker's shares jumped 3% on Monday, pushing its market value 
            above $3 trillion as investors bet on the company's artificial intelligence potential.
            """
        },
        {
            'category': 'entertainment',
            'text': """
            Taylor Swift's 'Eras Tour' becomes highest-grossing concert tour of all time.
            The global pop sensation has broken records with her spectacular show, 
            surpassing $1 billion in ticket sales worldwide.
            """
        },
        {
            'category': 'politics',
            'text': """
            UN Security Council calls for immediate ceasefire in ongoing conflict.
            World leaders gather for emergency session to address escalating tensions
            and humanitarian crisis in the region.
            """
        }
    ]
    
    print("اختبار النموذج على مقالات جديدة:\n")
    
    for article in test_articles:
        print(f"النص: {article['text'].strip()}")
        print(f"الفئة الحقيقية: {article['category']}")
        
        result = classifier.predict(article['text'])
        
        print(f"الفئة المتوقعة: {result['category']}")
        print(f"نسبة الثقة: {result['confidence']:.2%}")
        print("\nالاحتمالات لكل فئة:")
        for category, prob in result['probabilities'].items():
            print(f"- {category}: {prob:.2%}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_articles()
