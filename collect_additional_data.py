import pandas as pd
from newsapi import NewsApiClient
import time

def collect_sports_tech_news(api_key):
    api = NewsApiClient(api_key=api_key)
    collected_data = []
    
    # تجميع أخبار الرياضة والتقنية
    categories = {
        'sports': ['sports', 'football', 'basketball', 'tennis'],
        'technology': ['technology', 'tech', 'artificial intelligence', 'software']
    }
    
    for category, keywords in categories.items():
        print(f"جمع أخبار {category}...")
        for keyword in keywords:
            try:
                response = api.get_everything(
                    q=keyword,
                    language='en',
                    sort_by='relevancy',
                    page_size=25  # 25 مقال لكل كلمة مفتاحية
                )
                
                if response['status'] == 'ok':
                    for article in response['articles']:
                        if article['description'] and article['title']:
                            text = f"{article['title']}. {article['description']}"
                            collected_data.append({
                                'text': text,
                                'category': category
                            })
                    
                    print(f"تم جمع {len(response['articles'])} مقال لـ {keyword}")
                    time.sleep(1)  # تأخير لتجنب تجاوز حد الطلبات
                    
            except Exception as e:
                print(f"خطأ في جمع أخبار {keyword}: {str(e)}")
    
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
    print(f"\nتم إضافة {len(df)} مقال جديد")
    print("\nتوزيع الفئات الجديد:")
    print(final_df['category'].value_counts())

if __name__ == "__main__":
    API_KEY = '8835b0cbaeff45d3abbb74337686b12e'  # مفتاح API الخاص بك
    collect_sports_tech_news(API_KEY)
