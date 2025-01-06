from newsapi import NewsApiClient
import pandas as pd
import time
from datetime import datetime, timedelta

class NewsAPICollector:
    def __init__(self, api_key):
        """
        NewsAPI toplayıcısını başlatma
        Args:
            api_key (str): NewsAPI API anahtarı
        """
        self.api = NewsApiClient(api_key=api_key)
        self.collected_data = []
    
    def collect_news(self, category):
        """
        Belirli bir kategoriden haber toplama
        Args:
            category (str): Haber kategorisi
        """
        try:
            # Son 30 günlük haberleri alma
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            response = self.api.get_everything(
                q=category,
                language='en',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                sort_by='relevancy'
            )
            
            # Haberleri işleme
            for article in response['articles']:
                if article['content'] and article['description']:
                    self.collected_data.append({
                        'title': article['title'],
                        'text': f"{article['description']} {article['content']}",
                        'category': category,
                        'source': article['source']['name'],
                        'url': article['url'],
                        'date': article['publishedAt']
                    })
            
            print(f"{category} kategorisinden {len(response['articles'])} haber toplandı")
            time.sleep(1)  # API sınırlamalarını aşmamak için bekleme
            
        except Exception as e:
            print(f"Hata: {category} kategorisinden haber toplanırken bir hata oluştu - {str(e)}")
    
    def collect_all(self):
        """
        Tüm kategorilerden haber toplama
        """
        categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
        
        for category in categories:
            print(f"{category} kategorisinden haberler toplanıyor...")
            self.collect_news(category)
        
        # Toplanan verileri DataFrame'e dönüştürme ve kaydetme
        if self.collected_data:
            df = pd.DataFrame(self.collected_data)
            df.to_csv('newsapi_dataset.csv', index=False)
            print(f"\nToplam {len(df)} haber toplandı ve kaydedildi")
        else:
            print("Hiç haber toplanamadı")

if __name__ == "__main__":
    # API anahtarını buraya ekleyin
    api_key = '8835b0cbaeff45d3abbb74337686b12e'
    collector = NewsAPICollector(api_key)
    collector.collect_all()
