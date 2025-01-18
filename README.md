# Haber Sınıflandırma Projesi

Bu proje, derin öğrenme kullanarak haber metinlerini kategorilere ayıran bir web uygulamasıdır.

## Kurulum

Proje dosyalarını indirin ve gereksinimleri kurun:
```bash
pip install -r requirements.txt
```

Ardından, uygulamayı çalıştırın:
```bash
python src/flask_app.py
```

## Özellikler
- Haberleri kategorilere ayırma: Ekonomi, Eğlence, Siyaset, Spor, Teknoloji
- Kullanıcı dostu web arayüzü
- Türkçe dil desteği

## Proje Yapısı
```
News-detection/
├── src/                    # Kaynak dosyaları
│   ├── flask_app.py       # Flask web uygulaması
│   ├── model_building.py  # Model yapısı
│   └── train_enhanced_model.py # Model eğitimi
├── data/                   # Veri dosyaları
│   ├── train_dataset.csv  # Eğitim verisi
│   ├── test_dataset.csv   # Test verisi
│   └── temp_article.csv   # Geçici makaleler
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

## Sonuçlar
Modelin eğitim sonuçlarını ve performans değerlendirmesini `models/evaluation/` klasöründe bulabilirsiniz.
