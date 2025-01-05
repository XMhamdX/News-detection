# Haber Sınıflandırma Sistemi

Bu proje, derin öğrenme kullanarak haber metinlerini kategorilere ayıran bir web uygulamasıdır.

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
python flask_app.py
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
