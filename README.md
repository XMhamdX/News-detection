# Haber Sınıflandırma Sistemi

Bu proje, derin öğrenme kullanarak haber metinlerini kategorilere ayıran bir web uygulamasıdır.

## Proje Yapısı

```
News-detection/
├── data/                   # Data directory
│   └── processed/         
│       ├── raw/           # Raw data files
│       └── scripts/       # Data processing scripts
│           ├── collect_additional_data.py
│           ├── collect_turkish_news.py
│           ├── enhanced_data_collection.py
│           ├── merge_datasets.py
│           ├── split_dataset.py
│           └── update_training_data.py
│
├── models/                 # Model files
│   ├── bert/              # BERT model implementation
│   │   └── bert_model.py
│   ├── evaluation/        # Model evaluation
│   ├── label_encoder.pkl  # Label encoder
│   └── tokenizer.pkl      # Tokenizer
│
├── report_generator/       # Report generation
│   ├── ieee_report_generator.py
│   └── requirements.txt
│
├── src/                   # Source code
│   └── train_enhanced_model.py
│
├── tests/                 # Test files
│   ├── test_bert_api.py
│   ├── test_bert_diverse.py
│   ├── test_model.py
│   ├── test_model_directly.py
│   └── test_request.py
│
├── utils/                 # Utility functions
│   ├── data_collection.py
│   ├── newsapi_collector.py
│   └── text_preprocessing.py
│
├── web/                   # Web application
│   ├── app.py
│   └── templates/
│       ├── index.html
│       └── simple.html
│
├── requirements.txt       # Project dependencies
└── CHANGELOG.md          # Project changes log
```

## Dosyaların Detaylı Açıklaması

### 1. Kaynak Dosyalar (src/)

#### `bert_model.py`
```python
"""
Ana işlevler:
- BERT model tanımı
- Metin sınıflandırması için model kullanımı
"""
```

#### `check_gpu.py`
```python
"""
Ana işlevler:
- GPU kullanılabilirlik kontrolü
- GPU ayarları
"""
```

#### `flask_app_bert.py`
```python
"""
Ana işlevler:
- Ana Flask uygulaması
- Kullanıcı taleplerinin işlenmesi
- Model yüklenmesi ve tahminlerin yapılması
- Yeni makalelerin eklenmesi
- Sınıflandırma sonuçlarının gösterilmesi
"""
```

#### `flask_app.py`
```python
"""
Ana işlevler:
- Eski uygulama sürümü
- Şu anda kullanılmıyor
"""
```

#### `model_building.py`
```python
"""
Ana işlevler:
- Sinir ağının yapısının tanımlanması
- Katmanların hazırlanması:
  * Embedding (128 boyut)
  * Bidirectional LSTM
  * Dropout katmanları
  * Yoğun çıkış katmanı
- Eğitim parametrelerinin ayarlanması
"""
```

#### `simple_app.py`
```python
"""
Ana işlevler:
- Basitleştirilmiş uygulama
- Ana Flask uygulaması
- Kullanıcı taleplerinin işlenmesi
- Model yüklenmesi ve tahminlerin yapılması
- Yeni makalelerin eklenmesi
- Sınıflandırma sonuçlarının gösterilmesi
"""
```

#### `train_enhanced_model.py`
```python
"""
Ana işlevler:
- Verilerin yüklenmesi ve hazırlanması
- Modelin eğitilmesi
- Modelin ve parametrelerin kaydedilmesi
- Grafiğin oluşturulması
- Eğitim ilerlemesinin izlenmesi
"""
```

### 2. Grafiğin Dosyaları

#### `confusion_matrix1.png`
- Modelin karışıklık matrisi
- Doğru ve yanlış sınıflandırmaları gösterir
- Modelin performansının analizinde yardımcı olur

#### `training_history.png`
- Eğitim geçmişinin grafiği
- Zaman içinde doğruluğun iyileştirilmesini gösterir
- Eğitim sürecinin analizinde yardımcı olur

### 3. Veri Dosyaları (data/)

#### `train_dataset.csv`
- Eğitim verisi
- İki sütundan oluşur:
  * Metin (text)
  * Kategori (category)

#### `test_dataset.csv`
- Test verisi
- Eğitim verisi ile aynı yapıdadır
- Modelin performansının değerlendirilmesinde kullanılır

### 4. Eğitilmiş Modeller (models/)

#### `news_classifier_model.keras`
- Eğitilmiş model
- Ağırlıkları, yapısı ve eğitim parametrelerini içerir

### 5. Kullanıcı Arayüzü (templates/)

#### `index.html`
```html
<!--
Ana bileşenler:
- Metin girişi
- Sınıflandırma düğmesi
- Sonuçların gösterilmesi
- İlerleme çubuğu
- Grafiğin gösterilmesi
-->
```

### 6. Veri Yönetimi ve Eğitim Dosyaları

#### `collect_additional_data.py`
```python
"""
Ana işlevler:
- Ek veri toplama
- Verilerin temizlenmesi ve formatlanması
- Verilerin eğitim kümesine eklenmesi
- Eklenen verilerin kalitesinin kontrolü
"""
```

#### `download_dataset.py`
```python
"""
Ana işlevler:
- Veri setlerinin indirilmesi
- İndirme işleminin kontrolü
- Dosyaların düzenlenmesi
- Yedeklerin oluşturulması
"""
```

#### `enhanced_data_collection.py`
```python
"""
Ana işlevler:
- Veri toplama sürecinin iyileştirilmesi
- Kaynakların çeşitlendirilmesi
- İçerik tekrarlarının filtrelenmesi
- Veri kalitesinin garantilenmesi
"""
```

#### `merge_datasets.py`
```python
"""
Ana işlevler:
- Farklı veri setlerinin birleştirilmesi
- Veri tekrarlarının kaldırılması
- Veri formatlarının standardlaştırılması
- Veri tutarlılığının kontrolü
"""
```

#### `split_dataset.py`
```python
"""
Ana işlevler:
- Verilerin eğitim ve test kümelerine bölünmesi
- Kategorilerin dengeli dağılımının garantilenmesi
- Verilerin rastgele karıştırılması
- Eğitim ve test dosyalarının oluşturulması
"""
```

#### `update_training_data.py`
```python
"""
Ana işlevler:
- Eğitim verilerinin yeni makalelerle güncellenmesi
- Kategorilerin dengeli dağılımının kontrolü
- Eğitim dosyasının temizlenmesi ve güncellenmesi
- Güncelleme kayıtlarının tutulması
"""
```

### 7. Yapılandırma ve Ayar Dosyaları

#### `requirements.txt`
```
# Gereken kütüphaneler:
- tensorflow>=2.0.0
- flask>=2.0.0
- numpy>=1.19.0
- pandas>=1.3.0
- scikit-learn>=0.24.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
```

#### `.gitignore`
```
# Git'te yoksayılacak dosyalar ve klasörler:
- Büyük model dosyaları
- Geçici veriler
- Sistem dosyaları
- Sanal ortamlar
```

#### `training_progress.txt`
```
# Eğitim ilerleme kaydı:
- İlerlemenin yüzdesini içerir
- Eğitim sırasında güncellenir
- İlerleme çubuğunun gösterilmesinde kullanılır
```

### 8. İzleme ve Değerlendirme Dosyaları

#### `news_classifier_metrics.csv`
```
# Model performans ölçümleri:
- Her kategori için sınıflandırma doğruluğu
- Karışıklık matrisi
- Değerlendirme raporları
- İyileştirme kayıtları
```

## Haber Sınıflandırma Sistemi

Bu proje, BERT modelini kullanarak haberleri çeşitli kategorilere (spor, teknoloji, ekonomi, eğlence, siyaset) sınıflandıran bir web uygulamasıdır.

## Proje Yapısı

```
News detection/
├── src/
│   ├── bert_model.py      # BERT model tanımı
│   ├── flask_app_bert.py  # Ana Flask uygulaması
│   └── simple_app.py      # Basitleştirilmiş uygulama
│
├── templates/
│   ├── index.html         # Ana arayüz şablonu
│   └── simple.html        # Basit arayüz şablonu
│
├── static/               # Statik dosyalar (CSS, JS, resimler)
│
├── bert_tokenizer/      # Eğitilmiş tokenizer dosyaları
│   ├── vocab.txt
│   └── config.json
│
├── bert_news_classifier.pth    # Eğitilmiş model
├── label_encoder_bert.pkl      # Etiket dönüştürücü
├── test_model_directly.py      # Model testi betiği
└── requirements.txt            # Proje gereksinimleri
```

## Dosya Açıklamaları

### 1. Kaynak Dosyalar (src/)

#### `bert_model.py`
Bu dosya, temel model tanımını içerir:
```python
class NewsClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)
```
- Temel BERT modelini kullanır
- Dropout katmanı ekler
- Çıkış katmanını tanımlar
- Metin sınıflandırması için kullanılır

#### `flask_app_bert.py`
Ana uygulamayı sağlar:
```python
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    # Metin işleme ve sınıflandırma
    prediction = model.predict(text)
    return jsonify({'category': prediction})
```
- Modeli yükler ve hazırlar
- `/predict` endpoint'ini sağlar
- Talepleri işler ve yanıtlar
- Sınıflandırma sonuçlarını gösterir

#### `simple_app.py`
Basitleştirilmiş uygulamayı sağlar:
```python
def init_model():
    """Model ve bileşenlerin hazırlanması"""
    model = NewsClassifier(num_classes)
    model.load_state_dict(torch.load('model.pth'))
    return model

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    prediction = model(text)
    return jsonify({'category': prediction})
```
- Aynı modeli kullanır ancak daha basit bir arayüze sahiptir
- 5001 portunda çalışır
- Sınıflandırma işlevine odaklanır
- İyileştirilmiş hata işleme sağlar

### 2. Veri İşleme ve Eğitim

#### `data_preprocessing.py`
Verilerin işlenmesi ve temizlenmesi:
```python
def clean_text(text):
    # HTML etiketlerinin kaldırılması
    # Metin temizleme
    # Format standardlaştırılması
    return cleaned_text

def prepare_data(texts, labels):
    # Metinlerin tokenleştirilmesi
    # Etiketlerin dönüştürülmesi
    return features, targets
```
- Verilerin temizlenmesi ve hazırlanması
- Metinlerin tokenleştirilmesi
- Etiketlerin dönüştürülmesi

#### `model_training.py`
Modelin eğitilmesi ve değerlendirilmesi:
```python
def train_model(model, train_data, valid_data):
    # Eğitim ayarlarının yapılması
    # Eğitim döngüsü
    # En iyi modelin kaydedilmesi
    return trained_model

def evaluate_model(model, test_data):
    # Modelin değerlendirilmesi
    # Ölçütlerin hesaplanması
    return metrics
```
- Modelin eğitilmesi
- Modelin değerlendirilmesi
- En iyi modelin kaydedilmesi
- Parametrelerin ayarlanması

### 3. Yardımcı Fonksiyonlar

#### `utils.py`
Çeşitli yardımcı fonksiyonlar:
```python
def load_config():
    # Uygulama ayarlarının yüklenmesi
    return config

def setup_logging():
    # Kayıt ayarlarının yapılması
    return logger

def save_model(model, path):
    # Modelin ve meta verilerin kaydedilmesi
    pass
```
- Ayarların yüklenmesi
- Kayıt ayarlarının yapılması
- Modelin kaydedilmesi
- Dosya ve yol işlemleri

## Teknik Gereksinimler

```
flask==2.0.1
torch>=1.9.0
transformers>=4.10.0
numpy>=1.19.5
scikit-learn>=0.24.2
```

## Çalıştırma

1. Gereksinimlerin kurulumu:
```bash
pip install -r requirements.txt
```

2. Uygulamanın çalıştırılması:
```bash
python src/simple_app.py
```

3. Tarayıcıda açılması:
```
http://127.0.0.1:5001

```

📰 News Detection Project

تطبيق ويب لتصنيف الأخبار باستخدام نموذج BERT المدرب على اللغة التركية.

## 🚀 التثبيت

1. قم بتنزيل النموذج من هنا:
   - [تحميل النموذج (BERT News Classifier)](https://t.me/tenvsten_bot?start=60c27095)
   - ضع الملف في المسار: `models/bert/bert_news_classifier.pth`

2. قم بتثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

3. شغل التطبيق:
```bash
python web/app.py
```

## 📋 الميزات
- تصنيف الأخبار إلى فئات: الأعمال، الترفيه، السياسة، الرياضة، التكنولوجيا
- واجهة ويب سهلة الاستخدام
- دعم اللغة التركية

## 📁 هيكل المشروع
```
News detection/
├── models/
│   ├── bert/              # نموذج BERT وملفاته
│   └── evaluation/        # نتائج تقييم النموذج
├── web/                   # تطبيق الويب
│   ├── static/
│   ├── templates/
│   └── app.py
├── utils/                 # أدوات مساعدة
└── data/                  # البيانات
    ├── raw/
    ├── processed/
    └── scripts/
```

## 📊 النتائج
يمكنك رؤية نتائج تقييم النموذج في مجلد `models/evaluation/`.
