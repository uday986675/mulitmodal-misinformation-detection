"""
Text Encoder Module
===================
Encodes text using DistilBERT for efficient transformers-based representation.
Justification: DistilBERT provides 40% smaller model than BERT with 60% speedup
while retaining 97% of BERT's language understanding capabilities.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple


class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT.
    Extracts contextual embeddings from text tokens.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        """
        Initialize TextEncoder.
        
        Args:
            model_name: HuggingFace model name
            hidden_dim: Hidden dimension of DistilBERT
            dropout: Dropout rate
            freeze_backbone: Whether to freeze BERT weights
        """
        super(TextEncoder, self).__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Load pretrained DistilBERT
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Pooling layer: extract [CLS] token representation
        self.pooling = nn.Identity()  # Use first token ([CLS])
        
        # Optional projection layer
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through text encoder.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Token type IDs (batch_size, seq_len), optional
            
        Returns:
            Dictionary with embeddings and attention
        """
        # Get BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Extract representations
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        pooled_output = last_hidden_state[:, 0, :]      # [CLS] token (batch_size, hidden_dim)
        
        # Project and dropout
        text_embedding = self.projection(pooled_output)
        text_embedding = self.dropout(text_embedding)
        
        return {
            'embedding': text_embedding,
            'last_hidden_state': last_hidden_state,
            'attention_scores': outputs.attentions if outputs.attentions else None,
        }
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.hidden_dim


class TextEncoderWithAttention(nn.Module):
    """
    Enhanced text encoder with multi-head attention pooling.
    Learns weighted combination of all token representations.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 768,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize TextEncoderWithAttention.
        
        Args:
            model_name: HuggingFace model name
            hidden_dim: Hidden dimension
            num_attention_heads: Number of attention heads for pooling
            dropout: Dropout rate
        """
        super(TextEncoderWithAttention, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        # Attention pooling
        self.attention_weights = nn.Linear(hidden_dim, num_attention_heads)
        self.attention_head_projection = nn.Linear(hidden_dim, hidden_dim // num_attention_heads)
        self.num_heads = num_attention_heads
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention pooling.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            
        Returns:
            Dictionary with embeddings
        """
        # Get BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention weights
        attn_logits = self.attention_weights(last_hidden_state)  # (batch_size, seq_len, num_heads)
        
        # Apply attention mask
        attn_logits = attn_logits.masked_fill(
            attention_mask.unsqueeze(-1) == 0, float('-inf')
        )
        
        # Softmax attention
        attn_weights = torch.softmax(attn_logits, dim=1)  # (batch_size, seq_len, num_heads)
        
        # Apply attention to get weighted sum
        weighted_output = torch.einsum('bsh,bsd->bhd', attn_weights, last_hidden_state)
        
        # Reshape back to (batch_size, hidden_dim)
        batch_size = weighted_output.shape[0]
        weighted_output = weighted_output.reshape(batch_size, -1)
        
        # Project output
        text_embedding = self.output_projection(weighted_output)
        
        return {
            'embedding': text_embedding,
            'last_hidden_state': last_hidden_state,
            'attention_weights': attn_weights,
        }
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.hidden_dim
