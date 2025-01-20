"""
NewsAPI'den haber toplama
Bu modül, NewsAPI'yi kullanarak haberleri toplar ve kategorilere göre sınıflandırır
"""

from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAPICollector:
    def __init__(self, api_key=None):
        """
        NewsAPI toplayıcısını başlat
        
        Args:
            api_key (str): NewsAPI API anahtarı
        """
        self.api_key = api_key or os.environ.get('NEWS_API_KEY')
        if not self.api_key:
            raise ValueError("NEWS_API_KEY gerekli")
        
        self.api = NewsApiClient(api_key=self.api_key)
        self.categories = {
            'business': ['business', 'economy', 'finance'],
            'technology': ['technology', 'tech', 'innovation'],
            'entertainment': ['entertainment', 'arts', 'culture'],
            'sport': ['sports', 'football', 'basketball'],
            'politics': ['politics', 'government', 'policy']
        }
    
    def collect_articles(self, days_back=7):
        """
        Belirtilen gün sayısı kadar geriye giderek haberleri topla
        
        Args:
            days_back (int): Kaç gün geriye gidileceği
        """
        collected_data = []
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for category, keywords in self.categories.items():
            logger.info(f"{category} kategorisinden haberler toplanıyor...")
            
            for keyword in keywords:
                try:
                    response = self.api.get_everything(
                        q=keyword,
                        from_param=from_date,
                        language='en',
                        sort_by='relevancy',
                        page_size=100
                    )
                    
                    if response['status'] == 'ok':
                        for article in response['articles']:
                            if article['description'] and article['title']:
                                text = f"{article['title']}. {article['description']}"
                                collected_data.append({
                                    'text': text,
                                    'category': category,
                                    'source': article['source']['name'],
                                    'date': article['publishedAt']
                                })
                    
                    time.sleep(1)  # API sınırlamalarını aşmamak için bekle
                
                except Exception as e:
                    logger.error(f"{keyword} için hata: {str(e)}")
                    continue
        
        return pd.DataFrame(collected_data)
    
    def save_articles(self, df, output_dir='data/raw'):
        """
        Toplanan haberleri kaydet
        
        Args:
            df (DataFrame): Kaydedilecek haberler
            output_dir (str): Çıktı dizini
        """
        if df.empty:
            logger.warning("Kaydedilecek haber yok")
            return
        
        # Çıktı dizinini oluştur
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dosya adını oluştur
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'newsapi_articles_{timestamp}.csv'
        
        # Verileri kaydet
        df.to_csv(output_file, index=False)
        logger.info(f"{len(df)} haber kaydedildi: {output_file}")
        
        # Kategori dağılımını göster
        logger.info("\nKategori dağılımı:")
        logger.info(df['category'].value_counts())

def main():
    """
    Ana çalıştırma fonksiyonu
    """
    try:
        collector = NewsAPICollector()
        articles_df = collector.collect_articles()
        collector.save_articles(articles_df)
    
    except Exception as e:
        logger.error(f"Hata: {str(e)}")

if __name__ == '__main__':
    main()
