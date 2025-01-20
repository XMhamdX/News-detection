"""
وحدة معالجة النصوص الإخبارية
توفر وظائف لتنظيف وتجهيز النصوص للتصنيف

Haber metinleri işleme modülü
Metinleri sınıflandırma için temizleme ve hazırlama işlevleri sağlar
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# تحميل الموارد اللازمة
# Gerekli kaynakları yükle
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """
    تنظيف النص من الأحرف الخاصة والأرقام والرموز
    المعلمات:
        text: النص المراد تنظيفه
    
    Metni özel karakterlerden, rakamlardan ve sembollerden temizler
    Parametreler:
        text: Temizlenecek metin
    """
    # إزالة علامات HTML
    # HTML etiketlerini kaldır
    text = re.sub('<[^>]*>', '', str(text))
    
    # تحويل النص إلى أحرف صغيرة
    # Metni küçük harflere dönüştür
    text = text.lower()
    
    # إزالة الأرقام والرموز الخاصة
    # Rakamları ve özel sembolleri kaldır
    text = re.sub('[^a-zA-z\s]', '', text)
    
    # تقسيم النص إلى كلمات
    # Metni kelimelere ayır
    words = word_tokenize(text)
    
    # إزالة الكلمات الشائعة
    # Yaygın kelimeleri kaldır
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # إعادة تجميع النص
    # Metni yeniden birleştir
    text = ' '.join(words)
    
    # إزالة المسافات الزائدة
    # Fazla boşlukları kaldır
    text = re.sub('\s+', ' ', text).strip()
    
    return text

def preprocess_data():
    """
    معالجة البيانات الإخبارية
    المعلمات:
        لا يوجد
    
    Haber verilerini işler
    Parametreler:
        Yok
    """
    print("بدء معالجة البيانات...")
    
    # قراءة البيانات
    # Verileri oku
    data = pd.read_csv('bbc-text.csv')
    
    # عرض معلومات عن البيانات قبل المعالجة
    # Verilerin işlenmeden önceki bilgilerini göster
    print("\nمعلومات البيانات قبل المعالجة:")
    print(data.info())
    
    # تطبيق التنظيف على النصوص
    # Metinleri temizle
    print("\nجاري معالجة النصوص...")
    data['cleaned_text'] = data['text'].apply(clean_text)
    
    # حفظ البيانات المعالجة
    # İşlenen verileri kaydet
    data.to_csv('cleaned_bbc_text.csv', index=False)
    print("\nتم حفظ البيانات المعالجة في cleaned_bbc_text.csv")
    
    # عرض عينة من البيانات بعد المعالجة
    # İşlenen verilerin bir örneğini göster
    print("\nعينة من البيانات بعد المعالجة:")
    print(data[['category', 'cleaned_text']].head())

if __name__ == "__main__":
    preprocess_data()
