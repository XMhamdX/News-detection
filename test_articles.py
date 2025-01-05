from test_model import NewsClassifier

# إنشاء نموذج التصنيف
classifier = NewsClassifier()

# مقالات اختبار من مختلف الفئات
test_articles = [
    {
        'category': 'sports',
        'text': """
        Liverpool maintained their lead at the top of the Premier League with a 
        dramatic 2-0 victory over Arsenal at Anfield. Mohamed Salah scored twice 
        in the second half, bringing his season tally to 15 goals. The win keeps 
        Liverpool three points clear at the summit with Manchester City in close pursuit.
        """
    },
    {
        'category': 'technology',
        'text': """
        OpenAI has unveiled its latest breakthrough in artificial intelligence: 
        GPT-5, a more advanced language model that demonstrates unprecedented 
        capabilities in reasoning and problem-solving. The new model shows significant 
        improvements in mathematical computation and code generation, while maintaining 
        strong safeguards against potential misuse.
        """
    },
    {
        'category': 'business',
        'text': """
        Tesla's stock surged 15% today following the announcement of record-breaking 
        quarterly deliveries. The electric vehicle maker reported 405,000 vehicle 
        deliveries in Q4 2024, exceeding Wall Street expectations. CEO Elon Musk 
        attributed the success to improved production efficiency and strong demand 
        in Asian markets.
        """
    },
    {
        'category': 'entertainment',
        'text': """
        The 96th Academy Awards ceremony made history as 'The Moonlight Path' swept 
        the major categories, winning Best Picture, Best Director, and Best Actress. 
        The emotional drama, which tells the story of a young artist's journey through 
        grief and recovery, has been praised for its innovative cinematography and 
        powerful performances.
        """
    },
    {
        'category': 'politics',
        'text': """
        The United Nations Security Council has passed a landmark resolution on 
        climate change, declaring it a threat to global security. The resolution, 
        which received unanimous support from all 15 member states, calls for 
        immediate action to reduce greenhouse gas emissions and provides framework 
        for international cooperation on climate-related security risks.
        """
    }
]

print("اختبار النموذج على مقالات من مختلف الفئات:\n")

for i, article in enumerate(test_articles, 1):
    print(f"مقال {i} (الفئة الحقيقية: {article['category']}):")
    print("-" * 50)
    print(f"النص: {article['text'].strip()}\n")
    
    result = classifier.predict(article['text'])
    
    print(f"التصنيف المتوقع: {result['category']}")
    print(f"نسبة الثقة: {result['confidence']:.2%}")
    print("\nالاحتمالات لكل فئة:")
    
    # ترتيب الاحتمالات تنازلياً
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for category, prob in sorted_probs:
        print(f"- {category}: {prob:.2%}")
    
    print("\n" + "="*70 + "\n")
