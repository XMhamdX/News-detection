from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import time

class NewsAPICollector:
    def __init__(self, api_key):
        self.api = NewsApiClient(api_key=api_key)
        self.collected_data = []
        
    def collect_news(self, category):
        """جمع الأخبار لفئة معينة"""
        try:
            # الحصول على أخبار اليوم
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # آخر 30 يوم
            
            response = self.api.get_everything(
                q=category,
                language='en',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                sort_by='relevancy'
            )
            
            if response['status'] == 'ok':
                articles = response['articles']
                for article in articles:
                    self.collected_data.append({
                        'text': f"{article['title']}. {article['description']}",
                        'category': category,
                        'source': article['source']['name']
                    })
                
                print(f"تم جمع {len(articles)} مقال من فئة {category}")
                time.sleep(1)  # انتظار لتجنب تجاوز حد الطلبات
            
        except Exception as e:
            print(f"خطأ في جمع أخبار {category}: {str(e)}")
    
    def collect_all(self):
        """جمع الأخبار من جميع الفئات"""
        categories = ['business', 'entertainment', 'politics', 'sports', 'technology']
        
        print("بدء جمع البيانات من NewsAPI...")
        for category in categories:
            self.collect_news(category)
        
        # حفظ البيانات
        if self.collected_data:
            df = pd.DataFrame(self.collected_data)
            output_file = 'newsapi_dataset.csv'
            df.to_csv(output_file, index=False)
            print(f"\nتم حفظ {len(df)} مقال في {output_file}")
            
            # طباعة إحصائيات
            print("\nإحصائيات البيانات المجمعة:")
            print("\nعدد المقالات حسب الفئة:")
            print(df['category'].value_counts())
            print("\nعدد المقالات حسب المصدر:")
            print(df['source'].value_counts().head())
        else:
            print("لم يتم جمع أي بيانات")

if __name__ == "__main__":
    # مفتاح API من NewsAPI
    API_KEY = '8835b0cbaeff45d3abbb74337686b12e'  
    collector = NewsAPICollector(API_KEY)
    collector.collect_all()
