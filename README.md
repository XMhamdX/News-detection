# Haber Sınıflandırma Sistemi (نظام تصنيف الأخبار)

Bu proje, derin öğrenme kullanarak haber metinlerini kategorilere ayıran bir web uygulamasıdır.
هذا المشروع عبارة عن تطبيق ويب يستخدم التعلم العميق لتصنيف النصوص الإخبارية إلى فئات محددة.

## Yapı (هيكل المشروع)

```
News-detection/
├── src/                    # الملفات المصدرية الرئيسية
│   ├── flask_app.py       # تطبيق Flask للويب
│   ├── model_building.py  # بناء النموذج
│   └── train_enhanced_model.py # تدريب النموذج
├── data/                   # ملفات البيانات
│   ├── train_dataset.csv  # بيانات التدريب
│   ├── test_dataset.csv   # بيانات الاختبار
│   └── temp_article.csv   # المقالات المؤقتة
├── models/                 # النماذج المدربة
│   └── news_classifier_model.keras
├── utils/                  # الأدوات المساعدة
│   ├── text_preprocessing.py
│   ├── data_collection.py
│   └── newsapi_collector.py
├── tests/                  # ملفات الاختبار
│   ├── test_articles.py
│   └── test_model.py
├── static/                 # الملفات الثابتة
│   ├── confusion_matrix.png
│   └── training_history.png
└── templates/              # قوالب HTML
    └── index.html
```

## Dosyaların Detaylı Açıklaması (شرح تفصيلي للملفات)

### 1. الملفات المصدرية (src/)

#### flask_app.py
```python
"""
الوظائف الرئيسية:
- تطبيق الويب الرئيسي باستخدام Flask
- معالجة طلبات المستخدم
- تحميل النموذج وإجراء التنبؤات
- إدارة إضافة المقالات الجديدة
- عرض نتائج التصنيف
"""
```

#### model_building.py
```python
"""
الوظائف الرئيسية:
- تعريف بنية النموذج العصبي
- إعداد طبقات الشبكة:
  * Embedding (128 dimensions)
  * Bidirectional LSTM
  * Dropout layers
  * Dense output layer
- ضبط معلمات التدريب
"""
```

#### train_enhanced_model.py
```python
"""
الوظائف الرئيسية:
- تحميل وتجهيز البيانات
- تدريب النموذج
- حفظ النموذج والمعلمات
- إنشاء الرسوم البيانية
- تتبع تقدم التدريب
"""
```

### 2. الأدوات المساعدة (utils/)

#### text_preprocessing.py
```python
"""
الوظائف الرئيسية:
- تنظيف النصوص
- تقطيع النصوص إلى كلمات
- تحويل النصوص إلى أرقام
- توحيد أطوال النصوص
"""
```

#### data_collection.py
```python
"""
الوظائف الرئيسية:
- جمع المقالات من مصادر مختلفة
- تنظيف وتنسيق البيانات
- حفظ البيانات في CSV
- التحقق من جودة البيانات
"""
```

#### newsapi_collector.py
```python
"""
الوظائف الرئيسية:
- الاتصال بـ NewsAPI
- جمع المقالات حسب الفئة
- تحديث قاعدة البيانات
- إدارة حصص API
"""
```

### 3. ملفات البيانات (data/)

#### train_dataset.csv
- بيانات التدريب الرئيسية
- يحتوي على عمودين:
  * النص (text)
  * الفئة (category)

#### test_dataset.csv
- بيانات الاختبار المنفصلة
- نفس بنية ملف التدريب
- يستخدم لتقييم أداء النموذج

### 4. النماذج المدربة (models/)

#### news_classifier_model.keras
- النموذج المدرب
- يحتوي على:
  * الأوزان المدربة
  * بنية النموذج
  * معلمات التدريب

### 5. واجهة المستخدم (templates/)

#### index.html
```html
<!--
المكونات الرئيسية:
- نموذج إدخال النص
- زر التصنيف
- عرض النتائج
- شريط التقدم
- عرض الرسوم البيانية
-->
```

### 6. ملفات إدارة البيانات والتدريب

#### collect_additional_data.py
```python
"""
الوظائف الرئيسية:
- جمع بيانات إضافية من مصادر مختلفة
- تنظيف وتنسيق البيانات الجديدة
- إضافة البيانات إلى مجموعة التدريب
- التحقق من جودة البيانات المضافة
"""
```

#### download_dataset.py
```python
"""
الوظائف الرئيسية:
- تحميل مجموعات البيانات من المصادر المختلفة
- التحقق من صحة التحميل
- تنظيم الملفات المحملة
- إنشاء نسخ احتياطية
"""
```

#### enhanced_data_collection.py
```python
"""
الوظائف الرئيسية:
- تحسين عملية جمع البيانات
- التحقق من تنوع المصادر
- تصفية المحتوى المكرر
- ضمان جودة البيانات
"""
```

#### merge_datasets.py
```python
"""
الوظائف الرئيسية:
- دمج مجموعات البيانات المختلفة
- إزالة التكرار في البيانات
- توحيد تنسيق البيانات
- التحقق من اتساق البيانات
"""
```

#### split_dataset.py
```python
"""
الوظائف الرئيسية:
- تقسيم البيانات إلى تدريب واختبار
- ضمان التوزيع المتوازن للفئات
- خلط البيانات بشكل عشوائي
- إنشاء ملفات منفصلة للتدريب والاختبار
"""
```

#### update_training_data.py
```python
"""
الوظائف الرئيسية:
- تحديث بيانات التدريب بالمقالات الجديدة
- التحقق من توازن الفئات
- تنظيف وتحديث ملف التدريب
- حفظ سجل التحديثات
"""
```

### 7. ملفات التكوين والإعداد

#### requirements.txt
```
# قائمة المكتبات المطلوبة:
- tensorflow>=2.0.0
- flask>=2.0.0
- numpy>=1.19.0
- pandas>=1.3.0
- scikit-learn>=0.24.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
```

#### .gitignore
```
# الملفات والمجلدات المتجاهلة في Git:
- ملفات النموذج الكبيرة
- البيانات المؤقتة
- ملفات النظام
- البيئات الافتراضية
```

#### training_progress.txt
```
# ملف تتبع تقدم التدريب:
- يحتوي على النسبة المئوية للتقدم
- يتم تحديثه أثناء التدريب
- يستخدم لعرض شريط التقدم
```

### 8. ملفات المراقبة والتقييم

#### news_classifier_metrics.csv
```
# مقاييس أداء النموذج:
- دقة التصنيف لكل فئة
- مصفوفة الارتباك
- تقارير التقييم
- سجل التحسينات
```

## Yapay Zeka ile Haber Sınıflandırma Sistemi

Bu proje, derin öğrenme kullanarak haberleri otomatik olarak kategorilere ayıran bir web uygulamasıdır.

## Özellikler

- 🤖 LSTM tabanlı derin öğrenme modeli
- 📊 Çoklu kategori sınıflandırması (İş, Eğlence, Politika, Spor, Teknoloji)
- 🌐 Çok dilli arayüz (Türkçe, Arapça, İngilizce)
- 📈 Gerçek zamanlı model eğitimi
- 💾 Yeni verilerle model güncelleme imkanı

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri setini indirin:
```bash
python download_dataset.py
```

3. Modeli eğitin:
```bash
python src/train_enhanced_model.py
```

4. Uygulamayı başlatın:
```bash
python src/flask_app.py
```

## Kullanım

1. Web arayüzünü açın (varsayılan: http://localhost:5000)
2. Sınıflandırmak istediğiniz haber metnini girin
3. "Sınıflandır" butonuna tıklayın
4. Sonuçları görüntüleyin

## Model Eğitimi

- "Eğitime Ekle" butonu ile yeni haberler ekleyebilirsiniz
- "Yeniden Eğit" butonu ile modeli güncelleyebilirsiniz
- Eğitim ilerlemesi gerçek zamanlı olarak gösterilir

## Teknik Detaylar

- Framework: Flask
- Model: LSTM (TensorFlow/Keras)
- Veri İşleme: pandas, numpy
- Metin İşleme: NLTK, scikit-learn

## Gereksinimler

- Python 3.8+
- TensorFlow 2.x
- Flask
- pandas
- numpy
- scikit-learn
- NLTK

## Lisans

MIT License

---

[English](README_EN.md) | [العربية](README_AR.md) | [Türkçe](README.md)
