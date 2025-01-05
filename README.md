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

## Teknolojiler

- Python
- TensorFlow
- Flask
- HTML/CSS/JavaScript
