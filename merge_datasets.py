"""
دمج مجموعات البيانات
هذا السكربت يقوم بدمج مجموعات البيانات المختلفة من مصادر متعددة في مجموعة بيانات واحدة موحدة.
يتضمن:
- تنظيف البيانات المكررة
- توحيد أسماء الفئات
- التحقق من جودة البيانات
- حفظ البيانات المدمجة
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_log.txt'),
        logging.StreamHandler()
    ]
)

class DatasetMerger:
    """
    فئة لدمج وتنظيف مجموعات البيانات
    تتضمن وظائف للتعامل مع البيانات المكررة وتوحيد التنسيق
    """
    
    def __init__(self, base_dir):
        """
        تهيئة فئة دمج البيانات
        Args:
            base_dir (str): المسار الأساسي لمجلد البيانات
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data'
        self.merged_data = None
        self.category_mapping = {
            # تعيين الفئات المتشابهة لنفس الاسم
            'business': 'business',
            'economy': 'business',
            'finance': 'business',
            'entertainment': 'entertainment',
            'culture': 'entertainment',
            'arts': 'entertainment',
            'politics': 'politics',
            'government': 'politics',
            'sport': 'sport',
            'sports': 'sport',
            'technology': 'tech',
            'tech': 'tech',
            'science': 'tech'
        }
        
    def load_datasets(self):
        """
        تحميل جميع ملفات البيانات من المجلد المحدد
        Returns:
            list: قائمة من DataFrames المحملة
        """
        logging.info("بدء تحميل مجموعات البيانات...")
        datasets = []
        
        # البحث عن جميع ملفات CSV
        csv_files = list(self.data_dir.glob('*.csv'))
        if not csv_files:
            logging.warning("لم يتم العثور على ملفات CSV في المجلد")
            return datasets
            
        for file_path in csv_files:
            try:
                # تجاهل ملف البيانات المدمجة إذا كان موجوداً
                if file_path.name == 'merged_dataset.csv':
                    continue
                    
                df = pd.read_csv(file_path)
                logging.info(f"تم تحميل {file_path.name} - عدد الصفوف: {len(df)}")
                datasets.append(df)
            except Exception as e:
                logging.error(f"خطأ في تحميل {file_path.name}: {str(e)}")
                
        return datasets
        
    def standardize_columns(self, df):
        """
        توحيد أسماء الأعمدة وتنسيقها
        Args:
            df (DataFrame): البيانات المراد توحيدها
        Returns:
            DataFrame: البيانات بعد توحيد الأعمدة
        """
        # تعيين أسماء الأعمدة القياسية
        column_mapping = {
            'title': 'title',
            'text': 'text',
            'content': 'text',
            'article': 'text',
            'category': 'category',
            'class': 'category',
            'type': 'category'
        }
        
        # إعادة تسمية الأعمدة
        df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
        
        # التأكد من وجود الأعمدة المطلوبة
        required_columns = ['text', 'category']
        for col in required_columns:
            if col not in df.columns:
                logging.error(f"العمود المطلوب {col} غير موجود")
                raise ValueError(f"العمود المطلوب {col} غير موجود")
                
        return df
        
    def clean_text(self, text):
        """
        تنظيف النص من الأحرف الخاصة والمسافات الزائدة
        Args:
            text (str): النص المراد تنظيفه
        Returns:
            str: النص بعد التنظيف
        """
        if pd.isna(text):
            return ""
            
        # تحويل النص لسلسلة نصية
        text = str(text)
        
        # إزالة المسافات الزائدة
        text = " ".join(text.split())
        
        # إزالة الأحرف الخاصة غير المرغوب فيها
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        return text.strip()
        
    def standardize_categories(self, category):
        """
        توحيد أسماء الفئات
        Args:
            category (str): اسم الفئة
        Returns:
            str: اسم الفئة الموحد
        """
        if pd.isna(category):
            return "other"
            
        category = str(category).lower().strip()
        return self.category_mapping.get(category, "other")
        
    def remove_duplicates(self, df):
        """
        إزالة المقالات المكررة
        Args:
            df (DataFrame): البيانات المراد تنظيفها
        Returns:
            DataFrame: البيانات بعد إزالة التكرار
        """
        # حساب عدد الصفوف قبل إزالة التكرار
        initial_rows = len(df)
        
        # إزالة الصفوف المكررة تماماً
        df = df.drop_duplicates()
        
        # إزالة المقالات المتشابهة جداً (نفس النص مع اختلافات بسيطة)
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        # تسجيل عدد الصفوف المحذوفة
        removed_rows = initial_rows - len(df)
        logging.info(f"تم حذف {removed_rows} صفوف مكررة")
        
        return df
        
    def merge_datasets(self):
        """
        دمج جميع مجموعات البيانات وتنظيفها
        """
        datasets = self.load_datasets()
        if not datasets:
            logging.error("لا توجد بيانات للدمج")
            return
            
        # دمج جميع DataFrames
        logging.info("بدء عملية الدمج...")
        merged_df = pd.concat(datasets, ignore_index=True)
        
        # توحيد الأعمدة
        merged_df = self.standardize_columns(merged_df)
        
        # تنظيف النصوص
        logging.info("تنظيف النصوص...")
        merged_df['text'] = merged_df['text'].apply(self.clean_text)
        
        # توحيد الفئات
        logging.info("توحيد الفئات...")
        merged_df['category'] = merged_df['category'].apply(self.standardize_categories)
        
        # إزالة الصفوف الفارغة
        merged_df = merged_df.dropna(subset=['text', 'category'])
        
        # إزالة التكرار
        merged_df = self.remove_duplicates(merged_df)
        
        # حفظ النتيجة
        self.merged_data = merged_df
        
    def save_merged_dataset(self):
        """
        حفظ البيانات المدمجة مع إضافة طابع زمني
        """
        if self.merged_data is None:
            logging.error("لا توجد بيانات للحفظ")
            return
            
        # إنشاء اسم الملف مع الطابع الزمني
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.data_dir / f'merged_dataset_{timestamp}.csv'
        
        # حفظ البيانات
        self.merged_data.to_csv(output_file, index=False)
        logging.info(f"تم حفظ البيانات المدمجة في {output_file}")
        
        # طباعة إحصائيات
        self.print_statistics()
        
    def print_statistics(self):
        """
        طباعة إحصائيات عن البيانات المدمجة
        """
        if self.merged_data is None:
            return
            
        logging.info("\nإحصائيات البيانات المدمجة:")
        logging.info(f"إجمالي عدد المقالات: {len(self.merged_data)}")
        logging.info("\nتوزيع الفئات:")
        category_counts = self.merged_data['category'].value_counts()
        for category, count in category_counts.items():
            logging.info(f"{category}: {count} مقال")
            
def main():
    """
    الدالة الرئيسية لتشغيل عملية الدمج
    """
    # تحديد المسار الأساسي للمشروع
    base_dir = Path(__file__).resolve().parent
    
    # إنشاء كائن الدمج
    merger = DatasetMerger(base_dir)
    
    try:
        # تنفيذ عملية الدمج
        merger.merge_datasets()
        
        # حفظ النتائج
        merger.save_merged_dataset()
        
        logging.info("اكتملت عملية الدمج بنجاح!")
        
    except Exception as e:
        logging.error(f"حدث خطأ أثناء الدمج: {str(e)}")
        raise

if __name__ == '__main__':
    main()
