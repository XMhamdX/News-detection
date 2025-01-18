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
from pathlib import Path
import pickle
import tensorflow as tf

# تحديد المسار الأساسي للمشروع
BASE_DIR = Path(__file__).resolve().parent.parent

def update_progress(progress):
    """تحديث تقدم التدريب في الملف"""
    progress_file = BASE_DIR / 'training_progress.txt'
    progress_file.write_text(str(progress))

class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """تحديث التقدم في نهاية كل epoch"""
        total_epochs = self.params['epochs']
        progress = ((epoch + 1) / total_epochs) * 100
        update_progress(progress)

def load_and_preprocess_data():
    """تحميل ومعالجة البيانات"""
    # تحميل مجموعة البيانات
    train_file = BASE_DIR / 'data' / 'train_dataset.csv'
    if not train_file.exists():
        raise FileNotFoundError("ملف التدريب غير موجود")
    
    df = pd.read_csv(train_file)
    
    # ترميز التصنيفات
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    
    # معالجة النصوص
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(X, maxlen=100, padding='post', truncating='post')
    
    # حفظ المحولات
    with open(BASE_DIR / 'models' / 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open(BASE_DIR / 'models' / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return X, y, label_encoder.classes_

def create_model(vocab_size=10000, max_len=100, num_classes=5):
    """إنشاء نموذج LSTM ثنائي الاتجاه"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
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
    
    # رسم دقة التدريب
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='دقة التدريب')
    plt.plot(history.history['val_accuracy'], label='دقة التحقق')
    plt.title('دقة النموذج')
    plt.xlabel('Epoch')
    plt.ylabel('الدقة')
    plt.legend()
    
    # رسم خسارة التدريب
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='خسارة التدريب')
    plt.plot(history.history['val_loss'], label='خسارة التحقق')
    plt.title('خسارة النموذج')
    plt.xlabel('Epoch')
    plt.ylabel('الخسارة')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('مصفوفة الارتباك')
    plt.ylabel('التصنيف الفعلي')
    plt.xlabel('التصنيف المتنبأ')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    try:
        # إعادة تعيين التقدم
        update_progress(0)
        
        # تحميل ومعالجة البيانات
        X, y, classes = load_and_preprocess_data()
        
        # إنشاء النموذج
        model = create_model(vocab_size=10000, max_len=100, num_classes=len(classes))
        
        # تحضير الدوال الخلفية للتدريب
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(
                str(BASE_DIR / 'models' / 'news_classifier_model.keras'),
                save_best_only=True
            ),
            ProgressCallback()
        ]
        
        # تدريب النموذج
        history = model.fit(
            X, y,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            callbacks=callbacks
        )
        
        # رسم تاريخ التدريب
        plot_training_history(history)
        
        # تقييم النموذج
        test_loss, test_accuracy = model.evaluate(X, y)
        print(f"\nدقة النموذج على مجموعة الاختبار: {test_accuracy:.2f}")
        
        # رسم مصفوفة الارتباك
        y_pred = np.argmax(model.predict(X), axis=1)
        plot_confusion_matrix(y, y_pred, classes)
        
        # إكمال التدريب
        update_progress(100)
        
    except Exception as e:
        print(f"خطأ: {str(e)}")
        update_progress(-1)

if __name__ == "__main__":
    main()
