"""
Testing BERT model with diverse news articles in English
Testing various categories to verify classification accuracy
"""

import requests
import json
from pprint import pprint

# Test articles from different categories
test_articles = [
    {
        'category': 'sport',
        'text': """
        Manchester City clinches dramatic Premier League title with a 3-2 victory.
        A last-minute goal from Kevin De Bruyne secured the championship on the final
        day of the season, beating their closest rivals Liverpool by just one point.
        The win marks City's fourth title in five years.
        """
    },
    {
        'category': 'sport',
        'text': """
        Roger Federer announces retirement from professional tennis after 24-year career.
        The Swiss maestro, winner of 20 Grand Slam titles, will play his final match
        at the Laver Cup. Tennis legends and fans worldwide pay tribute to one of the
        sport's greatest players.
        """
    },
    {
        'category': 'entertainment',
        'text': """
        'Oppenheimer' dominates Academy Awards with 7 Oscar wins including Best Picture.
        Christopher Nolan's biographical drama about the father of the atomic bomb
        sweeps major categories. Cillian Murphy wins Best Actor for his portrayal
        of J. Robert Oppenheimer.
        """
    },
    {
        'category': 'entertainment',
        'text': """
        Taylor Swift breaks Billboard record with 'Midnights' album success.
        The pop superstar becomes the first artist to occupy all top 10 spots
        on the Billboard Hot 100 simultaneously. The album also sets new streaming
        records on Spotify and Apple Music.
        """
    },
    {
        'category': 'tech',
        'text': """
        Google unveils breakthrough in artificial intelligence with new language model.
        The company claims the model shows significant improvements in understanding context
        and generating human-like responses. The development marks a major step forward
        in natural language processing technology.
        """
    },
    {
        'category': 'business',
        'text': """
        Tesla shares surge 10% as electric car maker reports record deliveries.
        The company exceeded Wall Street expectations with 300,000 vehicles delivered
        in Q4 2024. CEO Elon Musk attributes success to improved production efficiency
        and strong demand in Asian markets.
        """
    },
    {
        'category': 'politics',
        'text': """
        UN Security Council passes landmark climate resolution. The unanimous decision
        declares climate change a threat to global security and calls for immediate
        action. The resolution provides framework for international cooperation on
        climate-related security risks.
        """
    }
]

def test_bert_classifier():
    print("Testing BERT model with diverse news articles:\n")
    print("=" * 80)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nTest {i}:")
        print("-" * 40)
        print(f"Text: {article['text'].strip()}")
        print(f"Expected category: {article['category']}")
        
        try:
            # Send request to server
            response = requests.post('http://localhost:5000/predict', 
                                  json={'text': article['text']})
            
            if response.status_code == 200:
                result = response.json()
                print("\nClassification result:")
                print(f"Category: {result['category']}")
                print(f"Confidence: {result['confidence']:.2%}")
                
                print("\nProbabilities for each category:")
                # Sort probabilities in descending order
                sorted_probs = sorted(result['probabilities'].items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
                for category, prob in sorted_probs:
                    print(f"- {category}: {prob:.2%}")
            else:
                print(f"Request error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("Error: Make sure the server (flask_app_bert.py) is running")
            break
            
        print("\n" + "=" * 80)

if __name__ == "__main__":
    test_bert_classifier()
