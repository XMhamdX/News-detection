"""
جمع أخبار الرياضة والتكنولوجيا
هذا الملف مسؤول عن جمع أخبار الرياضة والتكنولوجيا وإضافتها إلى مجموعة التدريب
"""

import pandas as pd
from newsapi import NewsApiClient
import time
import logging

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def collect_sports_tech_news(api_key):
    """
    جمع أخبار الرياضة والتكنولوجيا
    Args:
        api_key (str): مفتاح API الخاص بخدمة الأخبار
    """
    api = NewsApiClient(api_key=api_key)
    collected_data = []
    
    # تجميع أخبار الرياضة والتقنية
    categories = {
        'sports': ['sports', 'football', 'basketball', 'tennis'],
        'technology': ['technology', 'tech', 'artificial intelligence', 'software']
    }
    
    for category, keywords in categories.items():
        logging.info(f"جمع أخبار {category}...")
        for keyword in keywords:
            try:
                # جمع الأخبار باستخدام مفتاح API
                response = api.get_everything(
                    q=keyword,
                    language='en',
                    sort_by='relevancy',
                    page_size=25  # 25 مقال لكل كلمة مفتاحية
                )
                
                if response['status'] == 'ok':
                    for article in response['articles']:
                        if article['description'] and article['title']:
                            # معالجة النص
                            text = f"{article['title']}. {article['description']}"
                            collected_data.append({
                                'text': text,
                                'category': category
                            })
                    
                    logging.info(f"تم جمع {len(response['articles'])} مقال لـ {keyword}")
                    time.sleep(1)  # تأخير لتجنب تجاوز حد الطلبات
                    
            except Exception as e:
                logging.error(f"خطأ في جمع أخبار {keyword}: {str(e)}")
    
    # تحويل البيانات إلى DataFrame
    df = pd.DataFrame(collected_data)
    
    # إزالة المقالات المكررة
    df = df.drop_duplicates(subset=['text'])
    
    # قراءة البيانات الحالية
    current_df = pd.read_csv('train_dataset.csv')
    
    # دمج البيانات الجديدة مع القديمة
    final_df = pd.concat([current_df, df], ignore_index=True)
    
    # حفظ البيانات المدمجة
    final_df.to_csv('train_dataset.csv', index=False)
    logging.info(f"\nتم إضافة {len(df)} مقال جديد")
    logging.info("\nتوزيع الفئات الجديد:")
    logging.info(final_df['category'].value_counts())

if __name__ == "__main__":
    API_KEY = '8835b0cbaeff45d3abbb74337686b12e'  # مفتاح API الخاص بك
    collect_sports_tech_news(API_KEY)
