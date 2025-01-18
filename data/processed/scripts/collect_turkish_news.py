import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import random
import os

def collect_hurriyet_news():
    categories = {
        'spor': 'sports',
        'teknoloji': 'technology',
        'ekonomi': 'business',
        'magazin': 'entertainment',
        'gundem': 'politics'
    }
    
    news_data = []
    
    for category_tr, category_en in categories.items():
        url = f'https://www.hurriyet.com.tr/{category_tr}'
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Adjust the selectors based on Hurriyet's HTML structure
            news_items = soup.find_all('div', class_='news-item')  # Update this selector
            
            for item in news_items:
                title = item.find('h2').text.strip() if item.find('h2') else ''
                content = item.find('p').text.strip() if item.find('p') else ''
                
                if title and content:
                    news_data.append({
                        'text': f'{title} {content}',
                        'category': category_en
                    })
                
            time.sleep(random.uniform(1, 3))  # Be polite to the server
            
        except Exception as e:
            print(f'Error collecting {category_tr} news:', str(e))
    
    return pd.DataFrame(news_data)

def collect_sabah_news():
    # Similar implementation for Sabah newspaper
    pass

def main():
    # Create data directory if it doesn't exist
    os.makedirs('../data/turkish', exist_ok=True)
    
    # Collect news from different sources
    hurriyet_df = collect_hurriyet_news()
    
    # Save the collected data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hurriyet_df.to_csv(f'../data/turkish/hurriyet_news_{timestamp}.csv', index=False)
    
    print(f'Collected {len(hurriyet_df)} news articles from Hurriyet')

if __name__ == '__main__':
    main()
