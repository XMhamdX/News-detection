import pandas as pd
from sklearn.model_selection import train_test_split

# قراءة البيانات
print("قراءة البيانات...")
df = pd.read_csv('bbc-text.csv')

# تقسيم البيانات مع الحفاظ على نسبة الفئات
print("\nتقسيم البيانات...")
train_df, test_df = train_test_split(
    df,
    test_size=0.2,  # 20% للاختبار
    random_state=42,  # لضمان إمكانية تكرار النتائج
    stratify=df['category']  # للحفاظ على نسبة الفئات
)

# حفظ البيانات
print("\nحفظ البيانات...")
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

# عرض معلومات عن التقسيم
print("\nإحصائيات مجموعة التدريب:")
print(train_df['category'].value_counts())
print("\nإحصائيات مجموعة الاختبار:")
print(test_df['category'].value_counts())
