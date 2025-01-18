import torch
from transformers import BertTokenizer
import pickle
from src.bert_model import NewsClassifier
from pathlib import Path
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    try:
        BASE_DIR = Path(__file__).resolve().parent
        
        # تحميل Label Encoder
        logger.info("Loading label encoder...")
        with open(str(BASE_DIR / 'label_encoder_bert.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info(f"Classes: {label_encoder.classes_}")
        
        # تحميل Tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(str(BASE_DIR / 'bert_tokenizer'))
        
        # تحميل النموذج
        logger.info("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = NewsClassifier(len(label_encoder.classes_))
        model.load_state_dict(torch.load(str(BASE_DIR / 'bert_news_classifier.pth'), map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully!")
        
        # اختبار النموذج
        test_texts = [
            "Manchester United won the match against Liverpool with a score of 2-1",  # رياضة
            "The new iPhone 15 features advanced AI capabilities",  # تكنولوجيا
            "The government announced new economic policies",  # سياسة
            "Disney's latest movie breaks box office records"  # ترفيه
        ]
        
        logger.info("\nTesting predictions:")
        for text in test_texts:
            # تجهيز النص
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
            
            # التصنيف
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
                _, prediction = torch.max(outputs, dim=1)
                
            predicted_class = label_encoder.inverse_transform([prediction.item()])[0]
            logger.info(f"\nText: {text}")
            logger.info(f"Predicted class: {predicted_class}")
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_model()
