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

# Proje kök dizinini belirle
BASE_DIR = Path(__file__).resolve().parent.parent

def update_progress(progress):
    """Eğitim ilerlemesini dosyaya yaz"""
    progress_file = BASE_DIR / 'training_progress.txt'
    progress_file.write_text(str(progress))

class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Her epoch sonunda ilerlemeyi güncelle"""
        total_epochs = self.params['epochs']
        progress = ((epoch + 1) / total_epochs) * 100
        update_progress(progress)

def load_and_preprocess_data():
    """Veriyi yükle ve ön işle"""
    # Veri setini yükle
    train_file = BASE_DIR / 'data' / 'train_dataset.csv'
    if not train_file.exists():
        raise FileNotFoundError("Eğitim veri seti bulunamadı")
    
    df = pd.read_csv(train_file)
    
    # Etiketleri kodla
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    
    # Metinleri tokenize et
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(X, maxlen=100)
    
    # Model ve dönüştürücüleri kaydet
    with open(BASE_DIR / 'models' / 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open(BASE_DIR / 'models' / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return X, y, label_encoder.classes_

def create_model(vocab_size=10000, max_len=100, num_classes=5):
    """LSTM tabanlı model oluştur"""
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
    
    # Eğitim doğruluğunu çizme
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    
    # Eğitim kaybını çizme
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Karışıklık Matrisi')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    try:
        # İlerlemeyi sıfırla
        update_progress(0)
        
        # Veriyi yükle ve ön işle
        X, y, classes = load_and_preprocess_data()
        
        # Modeli oluştur
        model = create_model(vocab_size=10000, max_len=100, num_classes=len(classes))
        
        # Eğitim için callback'leri hazırla
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(
                str(BASE_DIR / 'models' / 'news_classifier_model.keras'),
                save_best_only=True
            ),
            ProgressCallback()
        ]
        
        # Modeli eğit
        history = model.fit(
            X, y,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            callbacks=callbacks
        )
        
        # Eğitim sonuçlarını çizme
        plot_training_history(history)
        
        # Modeli değerlendirme
        test_loss, test_accuracy = model.evaluate(X, y)
        print(f"\nModelin test doğruluğu: {test_accuracy:.2f}")
        
        # Karışıklık matrisini çizme
        y_pred = np.argmax(model.predict(X), axis=1)
        plot_confusion_matrix(y, y_pred, classes)
        
        # Eğitimi tamamla
        update_progress(100)
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        update_progress(-1)

if __name__ == "__main__":
    main()
