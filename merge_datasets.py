import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def merge_datasets():
    print("بدء دمج مجموعات البيانات...")
    
    # قراءة البيانات القديمة (BBC dataset)
    print("قراءة البيانات القديمة...")
    bbc_df = pd.read_csv('bbc-text.csv')
    print(f"عدد المقالات في مجموعة BBC: {len(bbc_df)}")
    
    # قراءة البيانات الجديدة (NewsAPI dataset)
    print("\nقراءة البيانات الجديدة...")
    newsapi_df = pd.read_csv('newsapi_dataset.csv')
    print(f"عدد المقالات في مجموعة NewsAPI: {len(newsapi_df)}")
    
    # توحيد أسماء الأعمدة
    bbc_df.columns = ['text', 'category']
    newsapi_df = newsapi_df[['text', 'category']]
    
    # توحيد تسميات الفئات
    category_mapping = {
        'tech': 'technology',
        'entertainment': 'entertainment',
        'business': 'business',
        'sport': 'sports',
        'politics': 'politics'
    }
    
    bbc_df['category'] = bbc_df['category'].map(category_mapping)
    newsapi_df['category'] = newsapi_df['category'].map(category_mapping)
    
    # دمج البيانات
    merged_df = pd.concat([bbc_df, newsapi_df], ignore_index=True)
    
    # حذف الصفوف التي تحتوي على قيم مفقودة
    initial_len = len(merged_df)
    merged_df = merged_df.dropna()
    if initial_len != len(merged_df):
        print(f"\nتم حذف {initial_len - len(merged_df)} صف يحتوي على قيم مفقودة")
    
    # حذف المقالات المكررة
    initial_len = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['text'])
    if initial_len != len(merged_df):
        print(f"تم حذف {initial_len - len(merged_df)} مقال مكرر")
    
    print(f"\nإجمالي عدد المقالات بعد الدمج: {len(merged_df)}")
    
    # إحصائيات الفئات
    print("\nتوزيع المقالات حسب الفئة:")
    print(merged_df['category'].value_counts())
    
    # حفظ البيانات المدمجة
    output_file = 'merged_dataset.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nتم حفظ البيانات المدمجة في {output_file}")
    
    # تقسيم البيانات إلى تدريب واختبار
    train_df, test_df = train_test_split(
        merged_df, 
        test_size=0.2, 
        random_state=42,
        stratify=merged_df['category']
    )
    
    # حفظ مجموعتي التدريب والاختبار
    train_df.to_csv('train_dataset.csv', index=False)
    test_df.to_csv('test_dataset.csv', index=False)
    print(f"\nتم حفظ بيانات التدريب ({len(train_df)} مقال) في train_dataset.csv")
    print(f"تم حفظ بيانات الاختبار ({len(test_df)} مقال) في test_dataset.csv")
    
    return merged_df

if __name__ == "__main__":
    merged_df = merge_datasets()
