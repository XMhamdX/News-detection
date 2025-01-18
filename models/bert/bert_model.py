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

# تعريف Dataset
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
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
            'label': torch.tensor(label, dtype=torch.long)
        }

# تعريف النموذج
class BERTNewsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTNewsClassifier, self).__init__()
        # استخدام نموذج BERT الإنجليزي
        self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=False)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(output.pooler_output)
        return self.fc(output)

def train_model():
    # تحميل البيانات
    print("تحميل البيانات...")
    data = pd.read_csv('data/train_dataset.csv')
    
    print(f"عدد الأمثلة: {len(data)}")
    print("توزيع الفئات:")
    print(data['category'].value_counts())
    
    # تحويل التصنيفات
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['category'])
    
    print("\nالفئات:", label_encoder.classes_)
    
    # حفظ Label Encoder
    with open('label_encoder_bert.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # تجهيز BERT Tokenizer
    print("\nتحميل BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # حفظ Tokenizer
    tokenizer.save_pretrained('bert_tokenizer')
    
    # تقسيم البيانات
    print("\nتقسيم البيانات...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].values, encoded_labels, test_size=0.2, random_state=42
    )
    
    print(f"عدد أمثلة التدريب: {len(train_texts)}")
    print(f"عدد أمثلة التحقق: {len(val_texts)}")
    
    # إنشاء Datasets
    print("\nتجهيز البيانات للتدريب...")
    MAX_LEN = 512
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    # إنشاء DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # إعداد النموذج
    print("\nتجهيز النموذج...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"استخدام الجهاز: {device}")
    
    model = BERTNewsClassifier(len(label_encoder.classes_))
    model = model.to(device)
    
    # تعريف Optimizer و Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # التدريب
    print("\nبدء التدريب...")
    n_epochs = 3
    
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch+1}/{n_epochs}:')
        
        # التدريب
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc='تدريب')
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # تحديث شريط التقدم
            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_accuracy = 100 * train_correct / train_total
        print(f'متوسط خسارة التدريب: {train_loss/len(train_loader):.4f}')
        print(f'دقة التدريب: {train_accuracy:.2f}%')
        
        # التحقق
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        print('\nجاري التحقق...')
        val_bar = tqdm(val_loader, desc='تحقق')
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # تحديث شريط التقدم
                val_bar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_accuracy = 100 * val_correct / val_total
        print(f'متوسط خسارة التحقق: {val_loss/len(val_loader):.4f}')
        print(f'دقة التحقق: {val_accuracy:.2f}%')
    
    # حفظ النموذج
    print("\nحفظ النموذج...")
    torch.save(model.state_dict(), 'bert_news_classifier.pth')
    print("تم حفظ النموذج بنجاح!")

if __name__ == '__main__':
    train_model()
