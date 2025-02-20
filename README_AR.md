# نظام تصنيف الأخبار

هذا المشروع عبارة عن تطبيق ويب يستخدم التعلم العميق لتصنيف النصوص الإخبارية إلى فئات مختلفة.

## هيكل المشروع

```
News-detection/
├── data/                   # مجلد البيانات
│   └── processed/         
│       ├── raw/           # البيانات الخام
│       └── scripts/       # سكريبتات معالجة البيانات
│           ├── collect_additional_data.py      # جمع بيانات إضافية
│           ├── collect_turkish_news.py         # جمع الأخبار التركية
│           ├── enhanced_data_collection.py     # تحسين جمع البيانات
│           ├── merge_datasets.py               # دمج مجموعات البيانات
│           ├── split_dataset.py                # تقسيم البيانات
│           └── update_training_data.py         # تحديث بيانات التدريب
│
├── models/                 # ملفات النماذج
│   ├── bert/              # تطبيق نموذج BERT
│   │   └── bert_model.py
│   ├── evaluation/        # تقييم النموذج
│   ├── label_encoder.pkl  # مشفر التسميات
│   └── tokenizer.pkl      # المحلل اللغوي
│
├
├── src/                   # الكود المصدري
│   └── train_enhanced_model.py
│
├── tests/                 # ملفات الاختبار
│   ├── test_bert_api.py
│   ├── test_bert_diverse.py
│   ├── test_model.py
│   ├── test_model_directly.py
│   └── test_request.py
│
├── utils/                 # الأدوات المساعدة
│   ├── data_collection.py
│   ├── newsapi_collector.py
│   └── text_preprocessing.py
│
├── web/                   # تطبيق الويب
│   ├── app.py
│   └── templates/
│       ├── index.html
│       └── simple.html
│
├── requirements.txt       # متطلبات المشروع
└── CHANGELOG.md          # سجل التغييرات
```

## شرح المكونات الرئيسية

### 1. مجلد البيانات (data/)
- يحتوي على البيانات الخام والمعالجة
- يتضمن سكريبتات لجمع ومعالجة البيانات
- يدعم جمع الأخبار من مصادر مختلفة

### 2. النماذج (models/)
- يحتوي على نموذج BERT المستخدم للتصنيف
- يتضمن أدوات المعالجة اللغوية
- يحتوي على أدوات تقييم النموذج

### 4. الكود المصدري (src/)
- يحتوي على الكود الرئيسي للتدريب
- يتضمن نماذج محسنة للتصنيف

### 5. الاختبارات (tests/)
- اختبارات شاملة للنموذج والواجهة البرمجية
- اختبارات متنوعة لضمان جودة التصنيف

### 6. الأدوات المساعدة (utils/)
- أدوات لجمع البيانات
- معالجة النصوص
- واجهات برمجية للأخبار

### 7. تطبيق الويب (web/)
- واجهة المستخدم الرسومية
- قوالب HTML
- خدمة الويب الرئيسية

## كيفية البدء

1. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

2. تشغيل التطبيق:
```bash
cd web
python app.py
```

## المميزات الرئيسية
- تصنيف الأخبار باستخدام التعلم العميق
- دعم اللغة التركية والإنجليزية
- واجهة ويب سهلة الاستخدام
- تقارير تقنية احترافية
- اختبارات شاملة للنظام

## المتطلبات
- Python 3.8+
- PyTorch
- Transformers
- Flask
- وباقي المتطلبات في ملف requirements.txt
