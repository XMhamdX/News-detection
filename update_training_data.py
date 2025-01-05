import pandas as pd

def merge_all_datasets():
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
    merge_all_datasets()
