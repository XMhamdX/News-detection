"""
تحسين جمع البيانات
هذا الملف يوفر وظائف متقدمة لجمع وتنظيف البيانات من مصادر متعددة
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
    فئة لجمع الأخبار من مصادر متعددة
    تتضمن وظائف لجمع البيانات وتخزينها في ملف CSV
    """
    
    def __init__(self):
        """
        تهيئة جامع الأخبار
        """
        self.ua = UserAgent()
        self.collected_data = []
        
    def get_random_headers(self):
        """
        إنشاء ترويسات عشوائية لتجنب الحظر
        Returns:
            dict: الترويسات العشوائية
        """
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

    def safe_request(self, url, retries=3):
        """
        إجراء طلب آمن مع إعادة المحاولة
        Args:
            url (str): الرابط المراد الطلب منه
            retries (int): عدد مرات إعادة المحاولة
        Returns:
            Response: الرد على الطلب
        """
        for _ in range(retries):
            try:
                response = requests.get(url, headers=self.get_random_headers(), timeout=10)
                if response.status_code == 200:
                    return response
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                print(f"خطأ في الطلب: {e}")
                time.sleep(random.uniform(2, 5))
        return None

    def collect_reuters(self):
        """
        جمع الأخبار من موقع Reuters
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
                
                for article in articles[:20]:  # نأخذ أول 20 مقال من كل فئة
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
        جمع الأخبار من موقع The Guardian
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
        جمع الأخبار من موقع CNN
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
        حفظ البيانات المجمعة
        """
        if self.collected_data:
            df = pd.DataFrame(self.collected_data)
            
            # إضافة البيانات الجديدة إلى الملف الموجود إذا كان موجوداً
            output_file = 'enhanced_news_dataset.csv'
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # حفظ البيانات
            df.to_csv(output_file, index=False)
            print(f"تم حفظ {len(df)} مقال في {output_file}")
        else:
            print("لم يتم جمع أي بيانات")

    def collect_all(self):
        """
        جمع البيانات من جميع المصادر
        """
        print("بدء جمع البيانات...")
        
        # جمع البيانات من كل مصدر
        collectors = [self.collect_reuters, self.collect_guardian, self.collect_cnn]
        for collector in collectors:
            try:
                print(f"جمع البيانات من {collector.__name__}...")
                collector()
                print(f"تم جمع {len([d for d in self.collected_data if d['source'] == collector.__name__.split('_')[1]])} مقال من {collector.__name__}")
            except Exception as e:
                print(f"خطأ في {collector.__name__}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # طباعة إحصائيات
        print("\nإحصائيات البيانات المجمعة:")
        sources = pd.DataFrame(self.collected_data)['source'].value_counts()
        for source, count in sources.items():
            print(f"{source}: {count} مقال")
        
        # حفظ البيانات
        self.save_data()

if __name__ == "__main__":
    collector = NewsCollector()
    collector.collect_all()
