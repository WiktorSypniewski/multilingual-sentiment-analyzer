import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class BGERidgePredictor:
    
    
    def __init__(self, model_path="regresor_ridge_bge.joblib"):
        self.model_path = model_path
        self.model = None
        self.encoder = None

    def load_resources(self):
        
        if self.model is None:
            self.model = joblib.load(self.model_path)
            
        if self.encoder is None:
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"---> Uruchamiam BGE-M3 na: {device.upper()} <---")
            
            
            self.encoder = SentenceTransformer(
                'BAAI/bge-m3', 
                device=device
            )

    def process_and_predict(self, text: str):
       
        if not text:
            return None
            
        if self.encoder is None or self.model is None:
            self.load_resources()
            
        assert self.encoder is not None
        assert self.model is not None
            
       
        embedding_2d = np.array(self.encoder.encode(text)).reshape(1, -1) 
        
       
        prediction = self.model.predict(embedding_2d)[0]
        clipped_prediction = max(0.0, min(10.0, float(prediction)))
        
        return clipped_prediction

    def process_and_predict_batch(self, texts: list[str], batch_size: int = 64):
        
        if not texts:
            return []
            
        if self.encoder is None or self.model is None:
            self.load_resources()
            
        assert self.encoder is not None
        assert self.model is not None
            
        
        embeddings = self.encoder.encode(texts, batch_size=batch_size, show_progress_bar=False)
        
        
        predictions = self.model.predict(embeddings)
        
        
        clipped_predictions = np.clip(predictions, 0.0, 10.0)
        
        return clipped_predictions.tolist()