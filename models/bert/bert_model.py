"""
نموذج BERT لتصنيف النصوص الإخبارية
يقوم بتحليل النص وتصنيفه إلى فئات محددة

Haber metinlerini sınıflandırmak için BERT modeli
Metni analiz eder ve belirli kategorilere sınıflandırır
"""

import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# تعريف Dataset
class NewsDataset(Dataset):
    """
    مجموعة بيانات النصوص الإخبارية
    تحمل النصوص والتصنيفات المرتبطة بها
    
    Haber metinleri veri kümesi
    Metinleri ve bunlarla ilgili sınıflandırmaları taşır
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        تهيئة Dataset مع تحديد النصوص والتصنيفات والtokenizer والحد الأقصى للطول
        المعلمات:
            texts: قائمة النصوص
            labels: قائمة التصنيفات
            tokenizer: tokenizer المستخدم
            max_len: الحد الأقصى للطول
        
        Veri kümesini metinler, sınıflandırmalar, tokenizer ve maksimum uzunlukla başlatır
        Parametreler:
            texts: Metin listesi
            labels: Sınıflandırma listesi
            tokenizer: Kullanılan tokenizer
            max_len: Maksimum uzunluk
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        """
        إرجاع عدد النصوص في Dataset
        
        Veri kümesindeki metin sayısını döndürür
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        إرجاع النص والتصنيف المرتبطين بالفهرس المحدد
        المعلمات:
            idx: الفهرس المحدد
        
        Belirtilen indeksin metnini ve sınıflandırmasını döndürür
        Parametreler:
            idx: Belirtilen indeks
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# تعريف النموذج
class BERTNewsClassifier(nn.Module):
    """
    نموذج BERT لتصنيف النصوص الإخبارية
    يستخدم النموذج الأساسي BERT مع إضافة طبقة تصنيف مخصصة
    
    Haber metinlerini sınıflandırmak için BERT modeli
    Temel BERT modelini özel bir sınıflandırma katmanıyla kullanır
    """
    def __init__(self, n_classes):
        """
        تهيئة النموذج مع تحديد عدد الفئات
        المعلمات:
            n_classes: عدد فئات التصنيف
        
        Modeli belirtilen sınıf sayısıyla başlatır
        Parametreler:
            n_classes: Sınıflandırma kategorilerinin sayısı
        """
        super(BERTNewsClassifier, self).__init__()
        # تحميل نموذج BERT الأساسي
        # Temel BERT modelini yükle
        self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=False)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        معالجة النص وإنتاج التصنيف
        المعلمات:
            input_ids: مصفوفة تمثل النص المدخل
            attention_mask: مصفوفة تمثل الكلمات المهمة
        
        Metni işler ve sınıflandırma üretir
        Parametreler:
            input_ids: Giriş metnini temsil eden dizi
            attention_mask: Önemli kelimeleri temsil eden dizi
        """
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(output.pooler_output)
        return self.fc(output)

def train_model():
    """
    تدريب النموذج على مجموعة البيانات
    Model eğitimi işlevi
    """
    # تحميل البيانات
    print("Veriler yükleniyor...")
    data = pd.read_csv('data/train_dataset.csv')
    
    print("\nVeri seti bilgileri:")
    print(f"Toplam örnek sayısı: {len(data)}")
    print(f"Kategoriler: {data['category'].unique()}")
    
    # تجهيز البيانات
    print("\nVeriler işleniyor...")
    texts = data['text'].values
    labels = data['category'].values
    
    # تحويل التصنيفات
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print("\nKategoriler:", label_encoder.classes_)
    
    # حفظ Label Encoder
    with open('label_encoder_bert.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # تجهيز BERT Tokenizer
    print("\nBERT Tokenizer yükleniyor...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # حفظ Tokenizer
    tokenizer.save_pretrained('bert_tokenizer')
    
    # تقسيم البيانات
    print("Veri seti bölünüyor...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42
    )
    
    # إنشاء مجموعات البيانات
    print("Veri kümeleri oluşturuluyor...")
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_len=512)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_len=512)
    
    # إعداد data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # تهيئة النموذج
    print("Model başlatılıyor...")
    model = BERTNewsClassifier(n_classes=len(np.unique(labels)))
    
    # إعداد المحسن ودالة الخسارة
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # التدريب
    print("\nModel eğitimi başlıyor...")
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # تدريب
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Eğitim"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Ortalama eğitim kaybı: {avg_train_loss:.4f}")
        
        # تقييم
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Doğrulama"):
                input_ids = batch['input_ids'].squeeze(1)
                attention_mask = batch['attention_mask'].squeeze(1)
                labels = batch['labels']
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        print(f"Ortalama doğrulama kaybı: {avg_val_loss:.4f}")
        print(f"Doğruluk: {accuracy:.4f}")
    
    # حفظ النموذج
    print("\nModel kaydediliyor...")
    torch.save(model.state_dict(), 'models/bert/model.pth')
    print("Model başarıyla kaydedildi!")

if __name__ == '__main__':
    train_model()
