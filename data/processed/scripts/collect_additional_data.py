"""
BBC News'den ek haber verilerini toplama
Bu dosya, BBC News'den haberleri toplamak ve bunları kategorilerine göre sınıflandırmakla sorumludur.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging

# Günlük kaydı yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def collect_bbc_news():
    """
    BBC News'den haberleri topla ve kategorilerine göre sınıflandır
    Bu fonksiyon, BBC News'den haberleri toplar ve bunları kategorilerine göre sınıflandırır.
    """
    categories = {
        'business': 'https://www.bbc.com/news/business',
        'technology': 'https://www.bbc.com/news/technology',
        'entertainment': 'https://www.bbc.com/news/entertainment_and_arts',
        'sport': 'https://www.bbc.com/sport',
        'politics': 'https://www.bbc.com/news/politics'
    }
    
    collected_data = []
    
    for category, url in categories.items():
        logging.info(f"{category} kategorisinden haberler toplanıyor...")
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Haber başlıklarını ve metinlerini topla
            articles = soup.find_all('article')
            for article in articles:
                title = article.find('h3')
                if title:
                    title_text = title.text.strip()
                    collected_data.append({
                        'text': title_text,
                        'category': category,
                        'source': 'BBC News',
                        'date_collected': datetime.now().strftime('%Y-%m-%d')
                    })
        
        except Exception as e:
            logging.error(f"{category} toplanırken hata: {str(e)}")
    
    # Verileri DataFrame'e dönüştür
    df = pd.DataFrame(collected_data)
    
    # Verileri kaydet
    output_file = f'data/raw/bbc_news_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(output_file, index=False)
    logging.info(f"Veriler kaydedildi: {output_file}")
    
    return df

if __name__ == "__main__":
    collect_bbc_news()
