# Derin Öğrenme ile Türkçe Haber Sınıflandırma Sistemi

Mohammed Hamdi
Bilgisayar Mühendisliği
İstanbul Teknik Üniversitesi
hamdi@itu.edu.tr

## Öz
Bu çalışmada, Türkçe haber metinlerini otomatik olarak sınıflandıran bir derin öğrenme sistemi geliştirilmiştir. BERT (Bidirectional Encoder Representations from Transformers) tabanlı model kullanılarak, haberler beş farklı kategoriye (İş, Eğlence, Politika, Spor, Teknoloji) ayrılmaktadır. Sistem, web tabanlı bir arayüz üzerinden kullanıcılara hizmet vermektedir. Yapılan testlerde, sistemin %95'in üzerinde doğruluk oranı ile başarılı sınıflandırma yaptığı gözlemlenmiştir.

**Anahtar Kelimeler** – Derin Öğrenme, BERT, Haber Sınıflandırma, Doğal Dil İşleme, Web Uygulaması

## I. GİRİŞ
Günümüzde internet üzerinden yayınlanan haber sayısının hızla artması, bu haberlerin otomatik olarak kategorize edilmesi ihtiyacını doğurmuştur. Bu çalışmada, Türkçe haber metinlerini yapay zeka kullanarak otomatik olarak sınıflandıran bir sistem geliştirilmiştir. Sistem, BERT modeli kullanarak haberleri beş farklı kategoriye ayırmakta ve web tabanlı bir arayüz üzerinden kullanıcılara hizmet vermektedir.

## II. SİSTEM MİMARİSİ
Sistem üç ana bileşenden oluşmaktadır:

A. **Veri Toplama ve İşleme**
- NewsAPI üzerinden Türkçe haber toplama
- Metin temizleme ve önişleme
- Veri setinin eğitim ve test olarak bölünmesi

B. **Derin Öğrenme Modeli**
- BERT tabanlı sınıflandırma modeli
- Çok katmanlı sinir ağı
- Dropout ve regularizasyon teknikleri

C. **Web Arayüzü**
- Flask tabanlı web uygulaması
- Responsive tasarım
- REST API endpoints

## III. KULLANILAN TEKNOLOJİLER

TABLE I
KULLANILAN TEMEL TEKNOLOJİLER

| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| Python | 3.8+ | Ana programlama dili |
| PyTorch | 1.9.0 | Derin öğrenme framework'ü |
| Transformers | 4.10.0 | BERT modeli implementasyonu |
| Flask | 2.0.1 | Web uygulaması |
| scikit-learn | 0.24.2 | Model değerlendirme |

## IV. MODEL PERFORMANSI
A. **Eğitim Süreci**
Model, 100,000'den fazla Türkçe haber metni üzerinde eğitilmiştir. Eğitim süreci, NVIDIA Tesla V100 GPU üzerinde yaklaşık 24 saat sürmüştür.

B. **Değerlendirme Metrikleri**

TABLE II
MODEL PERFORMANS METRİKLERİ

| Kategori | Precision | Recall | F1-Score |
|----------|-----------|--------|-----------|
| İş | 0.96 | 0.95 | 0.95 |
| Eğlence | 0.94 | 0.93 | 0.93 |
| Politika | 0.97 | 0.96 | 0.96 |
| Spor | 0.98 | 0.98 | 0.98 |
| Teknoloji | 0.95 | 0.94 | 0.94 |

## V. SONUÇLAR
Geliştirilen sistem, Türkçe haber metinlerini yüksek doğrulukla sınıflandırabilmektedir. Web tabanlı arayüz sayesinde, sistem kolayca kullanılabilir ve API üzerinden diğer uygulamalara entegre edilebilir durumdadır. Gelecek çalışmalarda, modelin çok dilli desteğinin artırılması ve gerçek zamanlı haber sınıflandırma özelliklerinin eklenmesi planlanmaktadır.

## KAYNAKLAR
[1] J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv preprint arXiv:1810.04805, 2018.

[2] A. Vaswani et al., "Attention Is All You Need," in Advances in Neural Information Processing Systems, 2017, pp. 5998-6008.

[3] Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv preprint arXiv:1907.11692, 2019.

[4] M. Arıcan and A. Akın, "Türkçe Metin Sınıflandırma: Derin Öğrenme Modelleri ile Bir Karşılaştırma," Bilişim Teknolojileri Dergisi, vol. 12, no. 3, pp. 219-228, 2019.

[5] K. Clark et al., "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators," in International Conference on Learning Representations, 2020.
