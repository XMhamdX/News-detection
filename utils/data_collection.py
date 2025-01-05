import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from fake_useragent import UserAgent

def get_article_content(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # محاولة العثور على عنوان المقال
            title = soup.find('h1')
            if title:
                title = title.text.strip()
            else:
                return None, None
            
            # محاولة العثور على محتوى المقال
            article_body = soup.find('article')
            if article_body:
                paragraphs = article_body.find_all('p')
                content = ' '.join([p.text.strip() for p in paragraphs])
                if content:  # تأكد من أن المحتوى ليس فارغاً
                    return title, content
    except Exception as e:
        print(f"خطأ في جمع المقال: {str(e)}")
    return None, None

def scrape_bbc_news():
    # إنشاء User-Agent عشوائي
    ua = UserAgent()
    
    base_urls = [
        'https://www.bbc.com/news/world',
        'https://www.bbc.com/news/business',
        'https://www.bbc.com/news/technology',
        'https://www.bbc.com/news/science_and_environment',
        'https://www.bbc.com/news/entertainment_and_arts'
    ]
    
    articles_data = []
    
    for url in base_urls:
        print(f'جاري جمع البيانات من: {url}')
        
        # تغيير User-Agent لكل طلب
        headers = {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # البحث عن روابط المقالات
                article_links = soup.find_all('a', class_='gs-c-promo-heading')
                if not article_links:
                    article_links = soup.find_all('a', {'data-link-track': 'article'})
                
                for link in article_links[:5]:  # نأخذ 5 مقالات من كل فئة
                    href = link.get('href', '')
                    if href:
                        if not href.startswith('http'):
                            article_url = 'https://www.bbc.com' + href
                        else:
                            article_url = href
                            
                        print(f'محاولة جمع المقال من: {article_url}')
                        title, content = get_article_content(article_url, headers)
                        
                        if title and content:
                            articles_data.append({
                                'Title': title,
                                'Content': content
                            })
                            print(f'تم جمع المقال: {title}')
                        
                        # انتظار عشوائي بين 2-5 ثواني
                        time.sleep(random.uniform(2, 5))
                        
        except Exception as e:
            print(f'خطأ في جمع البيانات من {url}: {str(e)}')
        
        # انتظار بين الفئات
        time.sleep(random.uniform(3, 7))
    
    if articles_data:
        # حفظ البيانات في ملف CSV
        df = pd.DataFrame(articles_data)
        df.to_csv('bbc-text.csv', index=False, encoding='utf-8')
        print(f'تم حفظ {len(articles_data)} مقال في الملف bbc-text.csv')
    else:
        print('لم يتم جمع أي بيانات!')

if __name__ == "__main__":
    print("بدء عملية جمع البيانات...")
    scrape_bbc_news()
