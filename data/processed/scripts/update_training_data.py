"""
تحديث بيانات التدريب
هذا الملف مسؤول عن تحديث وإدارة بيانات التدريب، بما في ذلك إضافة بيانات جديدة وتنظيفها
"""

import pandas as pd

def merge_all_datasets():
    """
    دالة لدمج جميع مجموعات البيانات
    """
    # قراءة البيانات من المصادر المختلفة
    bbc_df = pd.read_csv('bbc-text.csv')
    
    # تحديث اسم الفئة 'tech' إلى 'technology' و 'sport' إلى 'sports' للتوحيد
    bbc_df['category'] = bbc_df['category'].replace({
        'tech': 'technology',
        'sport': 'sports'
    })
    
    # حفظ البيانات المحدثة
    bbc_df.to_csv('train_dataset.csv', index=False)
    
    print("تم تحديث البيانات!")
    print("\nتوزيع الفئات الجديد:")
    print(bbc_df['category'].value_counts())
    print(f"\nإجمالي عدد المقالات: {len(bbc_df)}")

if __name__ == "__main__":
    """
    الدالة الرئيسية لتشغيل البرنامج
    """
    merge_all_datasets()
