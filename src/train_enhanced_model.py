"""
Geliştirilmiş haber sınıflandırma modelini eğitme
Bu modül, haber sınıflandırma modelini eğitir
"""

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

# Proje dizinini belirleme
BASE_DIR = Path(__file__).resolve().parent.parent

def ilerleme_guncelle(ilerleme):
    """Eğitim ilerlemesini güncelleme"""
    ilerleme_dosyası = BASE_DIR / 'training_progress.txt'
    ilerleme_dosyası.write_text(str(ilerleme))

class IlerlemeCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Epoch sonunda ilerlemeyi güncelleme"""
        toplam_epoch = self.params['epochs']
        ilerleme = ((epoch + 1) / toplam_epoch) * 100
        ilerleme_guncelle(ilerleme)

def veri_yukle_ve_on_isleme():
    """Verileri yükleme ve ön işleme"""
    # Eğitim verilerini yükleme
    egitim_dosyası = BASE_DIR / 'data' / 'train_dataset.csv'
    if not egitim_dosyası.exists():
        raise FileNotFoundError("Eğitim dosyası bulunamadı")
    
    df = pd.read_csv(egitim_dosyası)
    
    # Sınıfları kodlama
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    
    # Metinleri işleme
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(X, maxlen=100, padding='post', truncating='post')
    
    # Dönüşümleri kaydetme
    with open(BASE_DIR / 'models' / 'tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open(BASE_DIR / 'models' / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return X, y, label_encoder.classes_

def model_olustur(sozcuk_sayisi=10000, max_uzunluk=100, sinif_sayisi=5):
    """Çift yönlü LSTM modeli oluşturma"""
    model = Sequential([
        Embedding(sozcuk_sayisi, 128, input_length=max_uzunluk),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(sinif_sayisi, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def egitim_tarihcesini_goster(tarihce):
    plt.figure(figsize=(12, 4))
    
    # Eğitim doğruluğunu gösterme
    plt.subplot(1, 2, 1)
    plt.plot(tarihce.history['accuracy'], label='Eğitim doğruluğu')
    plt.plot(tarihce.history['val_accuracy'], label='Doğrulama doğruluğu')
    plt.title('Model doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    
    # Eğitim kaybını gösterme
    plt.subplot(1, 2, 2)
    plt.plot(tarihce.history['loss'], label='Eğitim kaybı')
    plt.plot(tarihce.history['val_loss'], label='Doğrulama kaybı')
    plt.title('Model kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def karisiklik_matrisini_goster(y_true, y_pred, siniflar):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=siniflar, yticklabels=siniflar)
    plt.title('Karışıklık matrisi')
    plt.ylabel('Gerçek sınıf')
    plt.xlabel('Tahmin edilen sınıf')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    try:
        # İlerlemeyi sıfırlama
        ilerleme_guncelle(0)
        
        # Verileri yükleme ve ön işleme
        X, y, siniflar = veri_yukle_ve_on_isleme()
        
        # Modeli oluşturma
        model = model_olustur(sozcuk_sayisi=10000, max_uzunluk=100, sinif_sayisi=len(siniflar))
        
        # Eğitim için geri çağırma fonksiyonlarını hazırlama
        geri_cagirma_fonksiyonlari = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(
                str(BASE_DIR / 'models' / 'news_classifier_model.keras'),
                save_best_only=True
            ),
            IlerlemeCallback()
        ]
        
        # Modeli eğitme
        tarihce = model.fit(
            X, y,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            callbacks=geri_cagirma_fonksiyonlari
        )
        
        # Eğitim tarihçesini gösterme
        egitim_tarihcesini_goster(tarihce)
        
        # Modeli değerlendirme
        test_kaybi, test_dogrulugu = model.evaluate(X, y)
        print(f"\nModelin test doğruluğu: {test_dogrulugu:.2f}")
        
        # Karışıklık matrisini gösterme
        y_pred = np.argmax(model.predict(X), axis=1)
        karisiklik_matrisini_goster(y, y_pred, siniflar)
        
        # İlerlemeyi tamamlama
        ilerleme_guncelle(100)
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        ilerleme_guncelle(-1)

if __name__ == "__main__":
    main()
