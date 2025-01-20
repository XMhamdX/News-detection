"""
Veri setini bölme işlemi
Birleştirilmiş veri setini eğitim ve test setlerine ayırır
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Veri setini oku
print("Veri setini oku...")
df = pd.read_csv('bbc-text.csv')

# Verileri karıştır ve böl
print("\nVerileri karıştır ve böl...")
train_df, test_df = train_test_split(
    df,
    test_size=0.2,  # 20% lı test seti
    random_state=42,  # Rastgele tohum değeri
    stratify=df['category']  # Kategorilere göre dağılımı koru
)

# Dosyaları kaydet
print("\nDosyaları kaydet...")
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

# Kategorilere göre dağılımı kontrol et
print("\nEğitim seti kategori dağılımı:")
print(train_df['category'].value_counts())
print("\nTest seti kategori dağılımı:")
print(test_df['category'].value_counts())
