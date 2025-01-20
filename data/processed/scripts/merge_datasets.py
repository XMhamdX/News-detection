"""
Veri setlerini birleştirme işlemi
Farklı kaynaklardan toplanan haber verilerini tek bir veri setinde birleştirir
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def merge_datasets():
    """
    Farklı kaynaklardan gelen veri setlerini birleştir
    """
    print("Veri setlerini birleştirme işlemi başlıyor...")
    
    # Eski verileri oku (BBC veri seti)
    print("Eski verileri okuma...")
    bbc_df = pd.read_csv('bbc-text.csv')
    print(f"BBC veri setindeki haber sayısı: {len(bbc_df)}")
    
    # Yeni verileri oku (NewsAPI veri seti)
    print("\nYeni verileri okuma...")
    newsapi_df = pd.read_csv('newsapi_dataset.csv')
    print(f"NewsAPI veri setindeki haber sayısı: {len(newsapi_df)}")
    
    # Sütun isimlerini birleştir
    bbc_df.columns = ['text', 'category']
    newsapi_df = newsapi_df[['text', 'category']]
    
    # Kategori isimlerini birleştir
    category_mapping = {
        'tech': 'technology',
        'entertainment': 'entertainment',
        'business': 'business',
        'sport': 'sports',
        'politics': 'politics'
    }
    
    bbc_df['category'] = bbc_df['category'].map(category_mapping)
    newsapi_df['category'] = newsapi_df['category'].map(category_mapping)
    
    # Veri setlerini birleştir
    merged_df = pd.concat([bbc_df, newsapi_df], ignore_index=True)
    
    # Eksik verileri kaldır
    initial_len = len(merged_df)
    merged_df = merged_df.dropna()
    if initial_len != len(merged_df):
        print(f"\n{initial_len - len(merged_df)} eksik veri kaldırıldı")
    
    # Tekrar eden haberleri kaldır
    initial_len = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['text'])
    if initial_len != len(merged_df):
        print(f"{initial_len - len(merged_df)} tekrar eden haber kaldırıldı")
    
    print(f"\nBirleştirilmiş veri setindeki haber sayısı: {len(merged_df)}")
    
    # Kategorilere göre dağılım
    print("\nKategorilere göre dağılım:")
    print(merged_df['category'].value_counts())
    
    # Birleştirilmiş veriyi kaydet
    output_file = 'merged_dataset.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nBirleştirilmiş veri {output_file} dosyasına kaydedildi")
    
    # Verileri eğitim ve test setlerine ayır
    train_df, test_df = train_test_split(
        merged_df, 
        test_size=0.2, 
        random_state=42,
        stratify=merged_df['category']
    )
    
    # Eğitim ve test setlerini kaydet
    train_df.to_csv('train_dataset.csv', index=False)
    test_df.to_csv('test_dataset.csv', index=False)
    print(f"\nEğitim seti ({len(train_df)} haber) train_dataset.csv dosyasına kaydedildi")
    print(f"Test seti ({len(test_df)} haber) test_dataset.csv dosyasına kaydedildi")
    
    return merged_df

if __name__ == "__main__":
    merged_df = merge_datasets()
