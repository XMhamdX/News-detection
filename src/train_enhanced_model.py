import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import tensorflow as tf

def load_and_preprocess_data():
    print("تحميل البيانات...")
    train_df = pd.read_csv('train_dataset.csv')
    test_df = pd.read_csv('test_dataset.csv')
    
    # تحديد الفئات
    classes = sorted(train_df['category'].unique())
    print("\nالفئات المتوفرة:", classes)
    
    # تحويل الفئات إلى أرقام
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    train_labels = label_encoder.transform(train_df['category'])
    test_labels = label_encoder.transform(test_df['category'])
    
    # حفظ Label Encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # تحويل النصوص إلى vectors
    max_words = 10000  # حجم المفردات
    max_len = 200      # طول النص
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_df['text'])
    
    # حفظ Tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # تحويل النصوص
    X_train = tokenizer.texts_to_sequences(train_df['text'])
    X_test = tokenizer.texts_to_sequences(test_df['text'])
    
    # Padding
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    return X_train, X_test, train_labels, test_labels, classes

def create_model(max_words=10000, max_len=200, num_classes=5):
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # رسم منحنى الدقة
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # رسم منحنى الخسارة
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    print("بدء عملية التدريب...")
    
    # إنشاء ملف للتقدم
    with open('training_progress.txt', 'w', encoding='utf-8') as f:
        f.write('0')
    
    X_train, X_test, train_labels, test_labels, classes = load_and_preprocess_data()
    
    with open('training_progress.txt', 'w', encoding='utf-8') as f:
        f.write('10')  # 10% بعد تحميل البيانات
    
    model = create_model(num_classes=len(classes))
    
    with open('training_progress.txt', 'w', encoding='utf-8') as f:
        f.write('20')  # 20% بعد إنشاء النموذج
    
    # تكوين callbacks
    checkpoint = ModelCheckpoint(
        'news_classifier_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # حساب النسبة المئوية (20-90%)
            progress = 20 + int((epoch + 1) / self.params['epochs'] * 70)
            with open('training_progress.txt', 'w', encoding='utf-8') as f:
                f.write(str(progress))
    
    # تدريب النموذج
    history = model.fit(
        X_train, train_labels,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping, ProgressCallback()]
    )
    
    # تقييم النموذج
    test_loss, test_accuracy = model.evaluate(X_test, test_labels)
    print(f"\nدقة النموذج على بيانات الاختبار: {test_accuracy:.2f}")
    
    # رسم النتائج
    plot_training_history(history)
    
    # رسم مصفوفة الارتباك
    y_pred = np.argmax(model.predict(X_test), axis=1)
    plot_confusion_matrix(test_labels, y_pred, classes)
    
    with open('training_progress.txt', 'w', encoding='utf-8') as f:
        f.write('100')  # 100% عند الانتهاء
    
    print("اكتمل التدريب!")

if __name__ == "__main__":
    main()
