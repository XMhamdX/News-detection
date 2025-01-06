"""
Veri Toplama İyileştirmesi
Bu dosya, çeşitli kaynaklardan veri toplama ve temizleme için gelişmiş işlevler sağlar
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from fake_useragent import UserAgent
import random
from datetime import datetime
import os

class NewsCollector:
    """
    Çeşitli kaynaklardan haber toplayan sınıf
    CSV dosyasına veri toplama ve kaydetme işlevlerini içerir
    """
    
    def __init__(self):
        """
        Haber toplayıcı başlatma
        """
        self.ua = UserAgent()
        self.collected_data = []
        
    def get_random_headers(self):
        """
        Engellemeyi önlemek için rastgele başlıklar oluşturma
        Returns:
            dict: Rastgele başlıklar
        """
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

    def safe_request(self, url, retries=3):
        """
        Yeniden deneme ile güvenli istek yapma
        Args:
            url (str): İstek yapılacak URL
            retries (int): Yeniden deneme sayısı
        Returns:
            Response: İstek yanıtı
        """
        for _ in range(retries):
            try:
                response = requests.get(url, headers=self.get_random_headers(), timeout=10)
                if response.status_code == 200:
                    return response
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                print(f"İstek hatası: {e}")
                time.sleep(random.uniform(2, 5))
        return None

    def collect_reuters(self):
        """
        Reuters sitesinden haber toplama
        """
        categories = {
            'business': 'business',
            'entertainment': 'entertainment',
            'politics': 'politics',
            'sports': 'sports',
            'technology': 'tech'
        }
        
        for category, label in categories.items():
            url = f'https://www.reuters.com/{category}'
            response = self.safe_request(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('article')
                
                for article in articles[:20]:  # Her kategoriden ilk 20 haberi alıyoruz
                    try:
                        title = article.find('h3').text.strip()
                        text = article.find('p').text.strip()
                        if title and text:
                            self.collected_data.append({
                                'text': f"{title}. {text}",
                                'category': label,
                                'source': 'reuters'
                            })
                    except:
                        continue
                    
                time.sleep(random.uniform(1, 3))

    def collect_guardian(self):
        """
        The Guardian sitesinden haber toplama
        """
        categories = {
            'business': 'business',
            'culture': 'entertainment',
            'politics': 'politics',
            'sport': 'sports',
            'technology': 'tech'
        }
        
        for category, label in categories.items():
            url = f'https://www.theguardian.com/{category}'
            response = self.safe_request(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('div', class_='fc-item__container')
                
                for article in articles[:20]:
                    try:
                        title = article.find('span', class_='fc-item__title').text.strip()
                        if title:
                            self.collected_data.append({
                                'text': title,
                                'category': label,
                                'source': 'guardian'
                            })
                    except:
                        continue
                    
                time.sleep(random.uniform(1, 3))

    def collect_cnn(self):
        """
        CNN sitesinden haber toplama
        """
        categories = {
            'business': 'business',
            'entertainment': 'entertainment',
            'politics': 'politics',
            'sport': 'sports',
            'tech': 'tech'
        }
        
        for category, label in categories.items():
            url = f'https://www.cnn.com/{category}'
            response = self.safe_request(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('article')
                
                for article in articles[:20]:
                    try:
                        title = article.find('span', class_='cd__headline-text').text.strip()
                        if title:
                            self.collected_data.append({
                                'text': title,
                                'category': label,
                                'source': 'cnn'
                            })
                    except:
                        continue
                    
                time.sleep(random.uniform(1, 3))

    def save_data(self):
        """
        Toplanan verileri kaydetme
        """
        if self.collected_data:
            df = pd.DataFrame(self.collected_data)
            
            # Mevcut dosyaya yeni verileri ekleme
            output_file = 'enhanced_news_dataset.csv'
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # Verileri kaydetme
            df.to_csv(output_file, index=False)
            print(f"{len(df)} haber {output_file} dosyasına kaydedildi")
        else:
            print("Hiçbir veri toplanmadı")

    def collect_all(self):
        """
        Tüm kaynaklardan verileri toplama
        """
        print("Veri toplama işlemi başladı...")
        
        # Tüm kaynaklardan verileri toplama
        collectors = [self.collect_reuters, self.collect_guardian, self.collect_cnn]
        for collector in collectors:
            try:
                print(f"{collector.__name__} sitesinden veri toplama...")
                collector()
                print(f"{collector.__name__} sitesinden {len([d for d in self.collected_data if d['source'] == collector.__name__.split('_')[1]])} haber toplandı")
            except Exception as e:
                print(f"{collector.__name__} sitesinden veri toplama hatası: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # Toplanan verilerin istatistiklerini yazdırma
        print("\nToplanan verilerin istatistikleri:")
        sources = pd.DataFrame(self.collected_data)['source'].value_counts()
        for source, count in sources.items():
            print(f"{source}: {count} haber")
        
        # Verileri kaydetme
        self.save_data()

if __name__ == "__main__":
    collector = NewsCollector()
    collector.collect_all()
