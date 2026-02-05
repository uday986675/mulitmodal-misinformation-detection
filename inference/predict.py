"""
Inference Module
================
Inference pipeline for making predictions on new data.
Supports single samples, batches, and uncertainty estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Union
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocess_text import TextPreprocessor
from data.preprocess_image import ImagePreprocessor
from utils.logger import Logger


class Predictor:
    """
    Prediction interface for multimodal misinformation detection.
    Handles preprocessing, inference, and output formatting.
    """
    
    def __init__(
        self,
        model: nn.Module,
        text_preprocessor: TextPreprocessor,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        logger: Logger = None,
    ):
        """
        Initialize Predictor.
        
        Args:
            model: Trained multimodal model
            text_preprocessor: Text preprocessor instance
            image_preprocessor: Image preprocessor instance (optional)
            device: Device to use for inference
            logger: Logger instance
        """
        self.model = model.to(device)
        self.model.eval()
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        self.device = device
        self.logger = logger or Logger()
    
    def predict_single(
        self,
        text: str,
        image_path: Optional[str] = None,
        return_embeddings: bool = False,
    ) -> Dict:
        """
        Predict on a single sample.
        
        Args:
            text: Social media post text
            image_path: Optional path to image
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with prediction, confidence, and optional embeddings
        """
        # Preprocess text
        text_tokens = self.text_preprocessor.preprocess(text)
        
        # Preprocess image
        image = None
        if image_path and self.image_preprocessor:
            image = self.image_preprocessor.preprocess(image_path)
        
        # Prepare batch (add batch dimension)
        input_ids = text_tokens['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = text_tokens['attention_mask'].unsqueeze(0).to(self.device)
        token_type_ids = text_tokens['token_type_ids']
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(0).to(self.device)
        
        if image is not None:
            image = image.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image,
                token_type_ids=token_type_ids,
            )
        
        # Extract results
        probabilities = outputs['probabilities'].cpu().numpy()[0]
        prediction = outputs['predictions'].cpu().item()
        
        # Format output
        result = {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                'Real': float(probabilities[0]),
                'Fake': float(probabilities[1]),
            },
        }
        
        # Add embeddings if requested
        if return_embeddings:
            result['embeddings'] = {
                'text': outputs['text_embedding'].cpu().numpy()[0].tolist(),
                'image': outputs['image_embedding'].cpu().numpy()[0].tolist() if outputs['image_embedding'] is not None else None,
                'fused': outputs['fused_embedding'].cpu().numpy()[0].tolist(),
            }
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
    ) -> List[Dict]:
        """
        Predict on multiple samples.
        
        Args:
            texts: List of text samples
            image_paths: Optional list of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        if image_paths is None:
            image_paths = [None] * len(texts)
        
        for text, image_path in zip(texts, image_paths):
            result = self.predict_single(text, image_path)
            results.append(result)
        
        return results
    
    def predict_with_uncertainty(
        self,
        text: str,
        image_path: Optional[str] = None,
        num_mc_samples: int = 10,
    ) -> Dict:
        """
        Predict with uncertainty estimation via Monte Carlo Dropout.
        
        Args:
            text: Text input
            image_path: Optional image path
            num_mc_samples: Number of MC samples
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Preprocess
        text_tokens = self.text_preprocessor.preprocess(text)
        image = None
        if image_path and self.image_preprocessor:
            image = self.image_preprocessor.preprocess(image_path)
        
        input_ids = text_tokens['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = text_tokens['attention_mask'].unsqueeze(0).to(self.device)
        token_type_ids = text_tokens['token_type_ids']
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(0).to(self.device)
        
        if image is not None:
            image = image.unsqueeze(0).to(self.device)
        
        # MC Dropout samples
        predictions_list = []
        self.model.train()  # Enable dropout
        
        with torch.no_grad():
            for _ in range(num_mc_samples):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=image,
                    token_type_ids=token_type_ids,
                )
                probs = outputs['probabilities'].cpu().numpy()[0]
                predictions_list.append(probs)
        
        self.model.eval()
        
        # Aggregate
        predictions_array = np.array(predictions_list)
        mean_probs = predictions_array.mean(axis=0)
        std_probs = predictions_array.std(axis=0)
        
        prediction = np.argmax(mean_probs)
        confidence = mean_probs[prediction]
        uncertainty = std_probs[prediction]
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'probabilities': {
                'Real': float(mean_probs[0]),
                'Fake': float(mean_probs[1]),
            },
            'probabilities_std': {
                'Real': float(std_probs[0]),
                'Fake': float(std_probs[1]),
            },
        }
    
    def predict_from_dict(
        self,
        data: Dict,
        text_key: str = 'text',
        image_key: str = 'image_path',
    ) -> Dict:
        """
        Predict from dictionary (e.g., JSON input).
        
        Args:
            data: Dictionary with text and optional image
            text_key: Key for text field
            image_key: Key for image path field
            
        Returns:
            Prediction dictionary
        """
        text = data.get(text_key, '')
        image_path = data.get(image_key)
        
        return self.predict_single(text, image_path)
    
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to model file
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f"Model loaded from {path}")


class BatchPredictor:
    """
    Efficient batch prediction interface.
    Collects predictions with minimal memory overhead.
    """
    
    def __init__(
        self,
        predictor: Predictor,
        batch_size: int = 32,
    ):
        """
        Initialize BatchPredictor.
        
        Args:
            predictor: Predictor instance
            batch_size: Batch size for processing
        """
        self.predictor = predictor
        self.batch_size = batch_size
    
    def predict_from_file(
        self,
        input_file: str,
        output_file: str,
        text_key: str = 'text',
        image_key: str = 'image_path',
    ):
        """
        Predict from JSONL file and save results.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            text_key: Key for text in input
            image_key: Key for image in input
        """
        results = []
        
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                result = self.predictor.predict_from_dict(data, text_key, image_key)
                results.append(result)
        
        # Save results
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Predictions saved to {output_file}")
