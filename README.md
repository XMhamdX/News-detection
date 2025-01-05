# Haber Sınıflandırma Sistemi

Bu proje, derin öğrenme kullanarak haber metinlerini kategorilere ayıran bir web uygulamasıdır.

## Proje Yapısı

```
News-detection/
├── src/                    # Ana kaynak kodları
│   ├── flask_app.py       # Flask web uygulaması
│   ├── model_building.py  # Model oluşturma
│   └── train_enhanced_model.py # Model eğitimi
├── data/                   # Veri dosyaları
│   ├── train_dataset.csv  # Eğitim verisi
│   ├── test_dataset.csv   # Test verisi
│   └── temp_article.csv   # Geçici makale verisi
├── models/                 # Eğitilmiş modeller
│   └── news_classifier_model.keras
├── utils/                  # Yardımcı araçlar
│   ├── text_preprocessing.py
│   ├── data_collection.py
│   └── newsapi_collector.py
├── tests/                  # Test dosyaları
│   ├── test_articles.py
│   └── test_model.py
├── static/                 # Statik dosyalar
│   ├── confusion_matrix.png
│   └── training_history.png
└── templates/              # HTML şablonları
    └── index.html
```

## Dosyaların Detaylı Açıklaması

### 1. Kaynak Kodları (src/)

#### flask_app.py
```python
"""
Ana İşlevler:
- Flask web uygulaması
- Kullanıcı isteklerini işleme
- Model yükleme ve tahmin yapma
- Yeni makale ekleme yönetimi
- Sınıflandırma sonuçlarını görüntüleme
"""
```

#### model_building.py
```python
"""
Ana İşlevler:
- Sinir ağı modelinin tanımlanması
- Ağ katmanlarının ayarlanması:
  * Embedding (128 boyut)
  * Bidirectional LSTM
  * Dropout katmanları
  * Dense çıkış katmanı
- Eğitim parametrelerinin ayarlanması
"""
```

#### train_enhanced_model.py
```python
"""
Ana İşlevler:
- Veri yükleme ve hazırlama
- Model eğitimi
- Model ve parametrelerin kaydedilmesi
- Grafiklerin oluşturulması
- Eğitim ilerlemesinin takibi
"""
```

### 2. Yardımcı Araçlar (utils/)

#### text_preprocessing.py
```python
"""
Ana İşlevler:
- Metin temizleme
- Metin tokenizasyonu
- Metni sayısallaştırma
- Metin uzunluklarını standartlaştırma
"""
```

#### data_collection.py
```python
"""
Ana İşlevler:
- Farklı kaynaklardan makale toplama
- Veri temizleme ve formatlama
- CSV'ye veri kaydetme
- Veri kalitesi kontrolü
"""
```

#### newsapi_collector.py
```python
"""
Ana İşlevler:
- NewsAPI bağlantısı
- Kategoriye göre makale toplama
- Veritabanı güncelleme
- API kotası yönetimi
"""
```

## Özellikler

- Haber metinlerini 5 kategoriye sınıflandırma:
  - İş (Business)
  - Eğlence (Entertainment)
  - Siyaset (Politics)
  - Spor (Sport)
  - Teknoloji (Tech)
- Gerçek zamanlı sınıflandırma
- Yeni makaleler ekleme ve model eğitimi
- Eğitim ilerlemesi görüntüleme
- Web arayüzü

## Kullanılan Teknolojiler

### 1. TensorFlow/Keras
- Derin öğrenme framework'ü
- Model oluşturma ve eğitim için kullanılır
- Metin işleme için LSTM katmanları
- Sürüm: 2.x

### 2. Flask
- Hafif web framework'ü
- Kullanıcı arayüzü için kullanılır
- Model ile etkileşim için API sağlar
- Sürüm: 2.x

### 3. NumPy & Pandas
- Sayısal işlemler ve veri manipülasyonu
- Dizi ve tablo işlemleri
- Sürüm: 1.x

## Sorunlar ve Çözümleri

### 1. Dosya Yolu Sorunu
- Sorun: Dosyaların bulunamaması
- Çözüm: Path kullanarak yol yönetimi

### 2. Makale Kaydetme Sorunu
- Sorun: Makalelerin eğitime eklenememesi
- Çözüm: Dosya yolu ve yazma işlemi düzeltmesi

### 3. Bellek Sorunu
- Sorun: Yüksek bellek kullanımı
- Çözüm: Veri işleme optimizasyonu

## Gelecek İyileştirmeler

1. Daha fazla dil desteği
2. Sınıflandırma doğruluğunu artırma
3. Yeni kategoriler ekleme
4. Kullanıcı arayüzü geliştirmeleri

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:
```bash
python src/flask_app.py
```

3. Tarayıcınızda şu adresi açın: `http://localhost:5000`

## Kullanım

1. Ana sayfada metin kutusuna haber metnini girin
2. "Metni Sınıflandır" düğmesine tıklayın
3. Sonuçları görüntüleyin
4. Yeni makaleler eklemek için "Eğitime Ekle" düğmesini kullanın
5. Modeli yeniden eğitmek için "Modeli Yeniden Eğit" düğmesini kullanın

## Projeye Katkıda Bulunma

1. Projeyi forklayın
2. Yeni bir branch oluşturun
3. Değişikliklerinizi yapın
4. Pull request gönderin

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.
