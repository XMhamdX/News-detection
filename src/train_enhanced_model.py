"""
تدريب نموذج تصنيف الأخبار المحسن
يقوم هذا السكربت بتدريب نموذج LSTM ثنائي الاتجاه لتصنيف النصوص الإخبارية
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os

# تحديد المسار الأساسي للمشروع
BASE_DIR = Path(__file__).resolve().parent.parent

def load_data():
    """
    تحميل وتجهيز البيانات للتدريب
    Returns:
        DataFrame: بيانات التدريب المجهزة
    """
    # قراءة ملف البيانات الرئيسي
    train_data = pd.read_csv(BASE_DIR / 'data' / 'train_dataset.csv')
    
    # إضافة المقالات الجديدة إذا وجدت
    temp_file = BASE_DIR / 'data' / 'temp_article.csv'
    if temp_file.exists():
        temp_data = pd.read_csv(temp_file)
        train_data = pd.concat([train_data, temp_data], ignore_index=True)
        # حذف الملف المؤقت بعد الإضافة
        temp_file.unlink()
    
    return train_data

def preprocess_data(data):
    """
    تجهيز النصوص والفئات للتدريب
    Args:
        data (DataFrame): البيانات الخام
    Returns:
        tuple: النصوص المجهزة، الفئات المشفرة، المحول النصي، محول الفئات
    """
    # تجهيز النصوص
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    
    # تشفير الفئات
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['category'])
    
    return padded_sequences, encoded_labels, tokenizer, label_encoder

def build_model(vocab_size, num_classes):
    """
    بناء نموذج LSTM ثنائي الاتجاه
    Args:
        vocab_size (int): حجم المفردات
        num_classes (int): عدد الفئات
    Returns:
        Model: نموذج Keras المجهز
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_training_progress(progress):
    """
    حفظ تقدم التدريب في ملف
    Args:
        progress (float): نسبة التقدم
    """
    progress_file = BASE_DIR / 'training_progress.txt'
    progress_file.write_text(str(progress))

def plot_training_history(history):
    """
    رسم مخطط لتاريخ التدريب
    Args:
        history: تاريخ تدريب النموذج
    """
    plt.figure(figsize=(12, 4))
    
    # رسم الدقة
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # رسم الخسارة
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'static' / 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    """
    رسم مصفوفة الارتباك
    Args:
        y_true: القيم الحقيقية
        y_pred: التنبؤات
        label_encoder: محول الفئات
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'static' / 'confusion_matrix.png'))
    plt.close()

def main():
    """
    الدالة الرئيسية لتدريب النموذج
    """
    print("بدء تحميل البيانات...")
    data = load_data()
    
    print("تجهيز البيانات...")
    X, y, tokenizer, label_encoder = preprocess_data(data)
    
    # تقسيم البيانات
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("بناء النموذج...")
    model = build_model(10000, len(np.unique(y)))
    
    # تدريب النموذج
    print("بدء التدريب...")
    total_epochs = 10
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=total_epochs,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: save_training_progress(
                    ((epoch + 1) / total_epochs) * 100
                )
            )
        ]
    )
    
    print("حفظ النموذج والمحولات...")
    model.save(str(BASE_DIR / 'models' / 'news_classifier_model.keras'))
    
    with open(str(BASE_DIR / 'models' / 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open(str(BASE_DIR / 'models' / 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("إنشاء الرسوم البيانية...")
    plot_training_history(history)
    
    # إنشاء مصفوفة الارتباك
    y_pred = np.argmax(model.predict(X_val), axis=1)
    plot_confusion_matrix(y_val, y_pred, label_encoder)
    
    print("اكتمل التدريب!")
    save_training_progress(100)

if __name__ == '__main__':
    main()
