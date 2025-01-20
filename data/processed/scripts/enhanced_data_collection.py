"""
Gelişmiş veri toplama işlemi
Bu dosya, çeşitli kaynaklardan veri toplamak ve temizlemek için gelişmiş işlevler sağlar
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from fake_useragent import UserAgent
import random
from datetime import datetime
import os

class HaberToplayıcı:
    """
    Haberleri çeşitli kaynaklardan toplamak için sınıf
    Verileri toplamak ve CSV dosyasına kaydetmek için işlevler içerir
    """
    
    def __init__(self):
        """
        Haber toplayıcıyı başlat
        """
        self.ua = UserAgent()
        self.toplanan_veriler = []
        
    def rastgele_başlıklar_oluştur(self):
        """
        Yasaklanmayı önlemek için rastgele başlıklar oluştur
        Returns:
            dict: Rastgele başlıklar
        """
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.5',
            'Connection': 'keep-alive',
        }

    def güvenli_istek(self, url, yeniden_dene=3):
        """
        Güvenli istek ile verileri toplamak
        Args:
            url (str): İstek yapılacak URL
            yeniden_dene (int): Yeniden deneme sayısı
        Returns:
            Response: İstek cevabı
        """
        for _ in range(yeniden_dene):
            try:
                response = requests.get(url, headers=self.rastgele_başlıklar_oluştur(), timeout=10)
                if response.status_code == 200:
                    return response
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                print(f"İstek hatası: {e}")
                time.sleep(random.uniform(2, 5))
        return None

    def reuters_haberleri_topla(self):
        """
        Reuters haberleri toplamak
        """
        kategoriler = {
            'business': 'business',
            'entertainment': 'entertainment',
            'politics': 'politics',
            'sports': 'sports',
            'technology': 'tech'
        }
        
        for kategori, etiket in kategoriler.items():
            url = f'https://www.reuters.com/{kategori}'
            response = self.güvenli_istek(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                haberler = soup.find_all('article')
                
                for haber in haberler[:20]:  # Her kategoriden ilk 20 haberi al
                    try:
                        başlık = haber.find('h3').text.strip()
                        metin = haber.find('p').text.strip()
                        if başlık and metin:
                            self.toplanan_veriler.append({
                                'text': f"{başlık}. {metin}",
                                'kategori': etiket,
                                'kaynak': 'reuters'
                            })
                    except:
                        continue
                    
                time.sleep(random.uniform(1, 3))

    def guardian_haberleri_topla(self):
        """
        The Guardian haberleri toplamak
        """
        kategoriler = {
            'business': 'business',
            'culture': 'entertainment',
            'politics': 'politics',
            'sport': 'sports',
            'technology': 'tech'
        }
        
        for kategori, etiket in kategoriler.items():
            url = f'https://www.theguardian.com/{kategori}'
            response = self.güvenli_istek(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                haberler = soup.find_all('div', class_='fc-item__container')
                
                for haber in haberler[:20]:
                    try:
                        başlık = haber.find('span', class_='fc-item__title').text.strip()
                        if başlık:
                            self.toplanan_veriler.append({
                                'text': başlık,
                                'kategori': etiket,
                                'kaynak': 'guardian'
                            })
                    except:
                        continue
                    
                time.sleep(random.uniform(1, 3))

    def cnn_haberleri_topla(self):
        """
        CNN haberleri toplamak
        """
        kategoriler = {
            'business': 'business',
            'entertainment': 'entertainment',
            'politics': 'politics',
            'sport': 'sports',
            'tech': 'tech'
        }
        
        for kategori, etiket in kategoriler.items():
            url = f'https://www.cnn.com/{kategori}'
            response = self.güvenli_istek(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                haberler = soup.find_all('article')
                
                for haber in haberler[:20]:
                    try:
                        başlık = haber.find('span', class_='cd__headline-text').text.strip()
                        if başlık:
                            self.toplanan_veriler.append({
                                'text': başlık,
                                'kategori': etiket,
                                'kaynak': 'cnn'
                            })
                    except:
                        continue
                    
                time.sleep(random.uniform(1, 3))

    def verileri_kaydet(self):
        """
        Toplanan verileri kaydet
        """
        if self.toplanan_veriler:
            df = pd.DataFrame(self.toplanan_veriler)
            
            # Mevcut dosyaya yeni verileri ekle
            çıktı_dosyası = 'gelişmiş_haber_verileri.csv'
            if os.path.exists(çıktı_dosyası):
                mevcut_df = pd.read_csv(çıktı_dosyası)
                df = pd.concat([mevcut_df, df], ignore_index=True)
            
            # Verileri kaydet
            df.to_csv(çıktı_dosyası, index=False)
            print(f"{len(df)} haber kaydedildi: {çıktı_dosyası}")
        else:
            print("Hiçbir veri toplanmadı")

    def tüm_verileri_topla(self):
        """
        Tüm kaynaklardan verileri toplamak
        """
        print("Veri toplama işlemi başladı...")
        
        # Tüm kaynaklardan verileri toplamak
        toplayıcılar = [self.reuters_haberleri_topla, self.guardian_haberleri_topla, self.cnn_haberleri_topla]
        for toplayıcı in toplayıcılar:
            try:
                print(f"{toplayıcı.__name__} kaynaklarından veriler toplanıyor...")
                toplayıcı()
                print(f"{toplayıcı.__name__} kaynaklarından {len([d for d in self.toplanan_veriler if d['kaynak'] == toplayıcı.__name__.split('_')[1]])} haber toplanmıştır")
            except Exception as e:
                print(f"{toplayıcı.__name__} hatası: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # İstatistikleri yazdır
        print("\nToplanan verilerin istatistikleri:")
        kaynaklar = pd.DataFrame(self.toplanan_veriler)['kaynak'].value_counts()
        for kaynak, sayı in kaynaklar.items():
            print(f"{kaynak}: {sayı} haber")
        
        # Verileri kaydet
        self.verileri_kaydet()

if __name__ == "__main__":
    toplayıcı = HaberToplayıcı()
    toplayıcı.tüm_verileri_topla()
