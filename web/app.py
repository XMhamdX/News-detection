from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer
import pickle
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the model after adding project root to path
from models.bert.bert_model import BERTNewsClassifier
import logging

# Günlük ayarları
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def init_model():
    """Model ve bileşenlerin başlatılması"""
    try:
        models_dir = project_root / 'models' / 'bert'
        
        # Label Encoder'ı yükle
        logger.info("Label encoder yükleniyor...")
        with open(str(models_dir / 'label_encoder_bert.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info(f"Sınıflar: {label_encoder.classes_}")
        
        # Tokenizer'ı yükle
        logger.info("Tokenizer yükleniyor...")
        tokenizer = BertTokenizer.from_pretrained(str(models_dir / 'bert_tokenizer'))
        
        # Modeli yükle
        logger.info("Model yükleniyor...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Kullanılan cihaz: {device}")
        
        model = BERTNewsClassifier(len(label_encoder.classes_))
        state_dict = torch.load(str(models_dir / 'bert_news_classifier.pth'), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Model başarıyla yüklendi!")
        
        return model, tokenizer, label_encoder, device
    
    except Exception as e:
        logger.error(f"Model başlatma hatası: {str(e)}")
        raise

# Uygulama başlangıcında modeli başlat
model, tokenizer, label_encoder, device = init_model()

@app.route('/')
def home():
    return render_template('simple.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json.get('text')
        if not text:
            return jsonify({'error': 'Lütfen bir metin girin'}), 400
        
        logger.info(f"İşlenen metin: {text[:100]}...")
        
        # Metni hazırla
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Verileri uygun cihaza taşı
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        logger.info("Tahmin yapılıyor...")
        
        # Sınıflandırma
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        
        # Sınıflandırma sonucunu kategori adına dönüştür
        predicted_class = label_encoder.inverse_transform([prediction.item()])[0]
        logger.info(f"Tahmin edilen sınıf: {predicted_class}")
        
        return jsonify({
            'text': text,
            'category': predicted_class,
            'confidence': confidence.item()
        })
        
    except Exception as e:
        logger.error(f"Sınıflandırma hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
