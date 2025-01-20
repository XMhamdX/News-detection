"""
Eğitim verilerini güncelleme işlemi
Yeni toplanan verileri mevcut eğitim setine ekler ve günceller
"""

import pandas as pd

def merge_all_datasets():
    """
    Tüm veri setlerini birleştirme fonksiyonu
    """
    # Farklı kaynaklardan verileri oku
    bbc_df = pd.read_csv('bbc-text.csv')
    
    # 'tech' kategorisini 'technology' ve 'sport' kategorisini 'sports' olarak güncelle
    bbc_df['category'] = bbc_df['category'].replace({
        'tech': 'technology',
        'sport': 'sports'
    })
    
    # Güncellenmiş verileri kaydet
    bbc_df.to_csv('train_dataset.csv', index=False)
    
    print("Veriler güncellenmiştir!")
    print("\nYeni kategori dağılımı:")
    print(bbc_df['category'].value_counts())
    print(f"\nToplam makale sayısı: {len(bbc_df)}")

if __name__ == "__main__":
    """
    Programı çalıştıran ana fonksiyon
    """
    merge_all_datasets()
