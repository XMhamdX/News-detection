import pandas as pd
import requests
from pathlib import Path

def download_bbc_dataset():
    print("جاري تحميل مجموعة بيانات BBC News...")
    
    # رابط مجموعة البيانات
    url = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
    
    try:
        # تحميل البيانات مباشرة إلى DataFrame
        df = pd.read_csv(url)
        
        # حفظ البيانات في ملف محلي
        df.to_csv('bbc-text.csv', index=False)
        print(f"تم تحميل وحفظ {len(df)} مقال في bbc-text.csv")
        print("\nمعلومات عن البيانات:")
        print(df.info())
        print("\nعينة من البيانات:")
        print(df.head())
        
    except Exception as e:
        print(f"حدث خطأ أثناء تحميل البيانات: {str(e)}")

if __name__ == "__main__":
    download_bbc_dataset()
