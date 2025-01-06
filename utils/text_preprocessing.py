import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK'den gerekli verileri indirme
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # HTML etiketlerini kaldırma
    text = re.sub('<[^>]*>', '', str(text))
    
    # Küçük harfe dönüştürme
    text = text.lower()
    
    # Özel karakterleri kaldırma
    text = re.sub('[^a-zA-z\s]', '', text)
    
    # Metni kelimelere ayırma
    words = word_tokenize(text)
    
    # Stop words'leri kaldırma
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Metni yeniden birleştirme
    text = ' '.join(words)
    
    # Fazla boşlukları kaldırma
    text = re.sub('\s+', ' ', text).strip()
    
    return text

def preprocess_data():
    print("Veri ön işleme başlatıldı...")
    
    # Veri setini yükleme
    data = pd.read_csv('bbc-text.csv')
    
    # Veri seti hakkında bilgi verme
    print("\nVeri seti hakkında bilgi:")
    print(data.info())
    
    # Metinleri temizleme
    print("\nMetinleri temizleme...")
    data['cleaned_text'] = data['text'].apply(clean_text)
    
    # İşlenmiş veriyi kaydetme
    data.to_csv('cleaned_bbc_text.csv', index=False)
    print("\nİşlenmiş veri cleaned_bbc_text.csv olarak kaydedildi")
    
    # İşlenmiş veri setinden örnek verme
    print("\nİşlenmiş veri setinden örnek:")
    print(data[['category', 'cleaned_text']].head())

if __name__ == "__main__":
    preprocess_data()
