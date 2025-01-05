import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# تحميل البيانات المطلوبة من NLTK
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # إزالة علامات HTML
    text = re.sub('<[^>]*>', '', str(text))
    
    # تحويل النص إلى أحرف صغيرة
    text = text.lower()
    
    # إزالة الأرقام والرموز الخاصة
    text = re.sub('[^a-zA-z\s]', '', text)
    
    # تقسيم النص إلى كلمات
    words = word_tokenize(text)
    
    # إزالة الكلمات الشائعة
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # إعادة تجميع النص
    text = ' '.join(words)
    
    # إزالة المسافات الزائدة
    text = re.sub('\s+', ' ', text).strip()
    
    return text

def preprocess_data():
    print("بدء معالجة البيانات...")
    
    # قراءة البيانات
    data = pd.read_csv('bbc-text.csv')
    
    # عرض معلومات عن البيانات قبل المعالجة
    print("\nمعلومات البيانات قبل المعالجة:")
    print(data.info())
    
    # تطبيق التنظيف على النصوص
    print("\nجاري معالجة النصوص...")
    data['cleaned_text'] = data['text'].apply(clean_text)
    
    # حفظ البيانات المعالجة
    data.to_csv('cleaned_bbc_text.csv', index=False)
    print("\nتم حفظ البيانات المعالجة في cleaned_bbc_text.csv")
    
    # عرض عينة من البيانات بعد المعالجة
    print("\nعينة من البيانات بعد المعالجة:")
    print(data[['category', 'cleaned_text']].head())

if __name__ == "__main__":
    preprocess_data()
