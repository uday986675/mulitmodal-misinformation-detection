"""
Text Preprocessing Module
=========================
Handles text cleaning, tokenization, and encoding for BERT/DistilBERT.
"""

import re
import string
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer


class TextPreprocessor:
    """
    Preprocesses text for model input.
    Uses DistilBERT tokenizer for efficient processing.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
    ):
        """
        Initialize TextPreprocessor.
        
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean text: lowercase, remove punctuation, extra spaces.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags but keep content
        text = re.sub(r'@\w+|#', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using DistilBERT tokenizer.
        
        Args:
            text: Cleaned text string
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', 'token_type_ids'
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0) if 'token_type_ids' in encoding else None,
        }
    
    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Full preprocessing pipeline: clean -> tokenize.
        
        Args:
            text: Raw text string
            
        Returns:
            Dictionary with tokenized outputs
        """
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        return tokens
    
    def batch_preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocess multiple texts at once.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with batched tensors
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        encodings = self.tokenizer(
            cleaned_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'token_type_ids': encodings['token_type_ids'] if 'token_type_ids' in encodings else None,
        }


class TextStatistics:
    """Analyze text statistics for understanding dataset."""
    
    @staticmethod
    def compute_stats(texts: List[str]) -> Dict:
        """
        Compute basic text statistics.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with stats
        """
        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        return {
            'num_texts': len(texts),
            'avg_words': sum(lengths) / len(lengths),
            'max_words': max(lengths),
            'min_words': min(lengths),
            'avg_chars': sum(char_lengths) / len(char_lengths),
            'max_chars': max(char_lengths),
            'min_chars': min(char_lengths),
        }
