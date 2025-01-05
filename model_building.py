import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def build_and_train_model():
    print("بدء عملية بناء وتدريب النموذج...")
    
    # قراءة البيانات المعالجة
    data = pd.read_csv('cleaned_bbc_text.csv')
    
    # تحويل التصنيفات إلى أرقام
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['category'])
    
    # حفظ Label Encoder للاستخدام لاحقاً
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # تجهيز النصوص
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(data['cleaned_text'])
    
    # حفظ Tokenizer للاستخدام لاحقاً
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # تحويل النصوص إلى مصفوفات متساوية الطول
    max_length = 200
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, 
        encoded_labels, 
        test_size=0.2, 
        random_state=42
    )
    
    # بناء النموذج
    model = Sequential([
        Embedding(5000, 128, input_length=max_length),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    # تجميع النموذج
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # طباعة ملخص النموذج
    print("\nملخص النموذج:")
    model.summary()
    
    # تدريب النموذج مع Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    print("\nبدء تدريب النموذج...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # تقييم النموذج
    print("\nتقييم النموذج على بيانات الاختبار:")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"الدقة على بيانات الاختبار: {accuracy:.4f}")
    
    # التنبؤ بالتصنيفات
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # تقرير التصنيف
    print("\nتقرير التصنيف:")
    print(classification_report(
        y_test, 
        y_pred_classes, 
        target_names=label_encoder.classes_
    ))
    
    # رسم مصفوفة الارتباك
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix(y_test, y_pred_classes),
        annot=True,
        fmt='d',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('مصفوفة الارتباك')
    plt.xlabel('التنبؤات')
    plt.ylabel('القيم الحقيقية')
    plt.savefig('confusion_matrix.png')
    
    # رسم منحنيات التدريب
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='تدريب')
    plt.plot(history.history['val_loss'], label='تحقق')
    plt.title('دالة الخسارة')
    plt.xlabel('Epoch')
    plt.ylabel('الخسارة')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='تدريب')
    plt.plot(history.history['val_accuracy'], label='تحقق')
    plt.title('الدقة')
    plt.xlabel('Epoch')
    plt.ylabel('الدقة')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # حفظ النموذج
    model.save('news_classifier_model.h5')
    print("\nتم حفظ النموذج والرسومات البيانية")

if __name__ == "__main__":
    build_and_train_model()
