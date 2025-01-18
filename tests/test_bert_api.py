"""
هذا الملف يختبر واجهة API الخاصة بنموذج BERT
يقوم بإرسال طلب POST إلى الخادم المحلي مع نص خبري
ويطبع نتيجة التصنيف مع نسبة الثقة

ملاحظة: يجب أن يكون الخادم (flask_app_bert.py) قيد التشغيل قبل تنفيذ هذا الاختبار
"""

import requests
import json

# نص خبري للاختبار
test_text = """
السعودية تستضيف كأس العالم 2034: أعلن الاتحاد الدولي لكرة القدم (فيفا) رسمياً فوز المملكة العربية السعودية بحق استضافة كأس العالم 2034. وتعهدت المملكة بتنظيم بطولة استثنائية تجمع بين التقاليد العربية والتكنولوجيا الحديثة.
"""

def test_bert_api():
    try:
        # إرسال الطلب إلى الخادم
        response = requests.post('http://localhost:5000/predict', json={'text': test_text})

        # طباعة النتيجة
        if response.status_code == 200:
            result = response.json()
            print("\nنتيجة اختبار API نموذج BERT:")
            print("-" * 40)
            print(f"النص: {test_text.strip()}")
            print(f"\nالتصنيف: {result['category']}")
            print(f"نسبة الثقة: {result['confidence']:.2%}")
            print("\nالاحتمالات لكل فئة:")
            for category, prob in result['probabilities'].items():
                print(f"- {category}: {prob:.2%}")
        else:
            print("حدث خطأ:", response.status_code)
            
    except requests.exceptions.ConnectionError:
        print("خطأ: لم يتم العثور على الخادم. تأكد من تشغيل flask_app_bert.py أولاً")

if __name__ == "__main__":
    test_bert_api()
