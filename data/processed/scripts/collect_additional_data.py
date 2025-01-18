"""
جمع أخبار الرياضة والتكنولوجيا
هذا الملف مسؤول عن جمع أخبار الرياضة والتكنولوجيا وإضافتها إلى مجموعة التدريب
"""

import pandas as pd
from newsapi import NewsApiClient
import time
import logging
import os
from pathlib import Path

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def collect_sports_tech_news():
    """
    جمع أخبار الرياضة والتكنولوجيا
    يتطلب تعيين متغير البيئة NEWS_API_KEY
    """
    api_key = os.environ.get('NEWS_API_KEY')
    if not api_key:
        raise ValueError("الرجاء تعيين متغير البيئة NEWS_API_KEY")
    
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
                            text = text.replace('\n', ' ').replace('\r', ' ')
                            text = ' '.join(text.split())  # إزالة المسافات الزائدة
                            
                            collected_data.append({
                                'text': text,
                                'category': category
                            })
                
                # انتظار لتجنب تجاوز حد معدل API
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"خطأ في جمع الأخبار لـ {keyword}: {str(e)}")
                continue
    
    # حفظ البيانات المجمعة
    if collected_data:
        df = pd.DataFrame(collected_data)
        output_file = Path('data/additional_articles.csv')
        output_file.parent.mkdir(exist_ok=True)
        
        # حفظ البيانات مع الترميز المناسب
        df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"تم حفظ {len(df)} مقال في {output_file}")
    else:
        logging.warning("لم يتم جمع أي بيانات")

if __name__ == "__main__":
    collect_sports_tech_news()
