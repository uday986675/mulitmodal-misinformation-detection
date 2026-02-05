"""
Classification Head Module
==========================
Final classification layers that output fake/real predictions with confidence scores.
"""

import torch
import torch.nn as nn
from typing import Dict


class MultimodalClassifier(nn.Module):
    """
    Classification head for multimodal misinformation detection.
    Maps fused embeddings to binary classification with confidence scores.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize classifier head.
        
        Args:
            input_dim: Input embedding dimension (from fusion)
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes (2 for binary classification)
            dropout: Dropout rate
        """
        super(MultimodalClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through classifier.
        
        Args:
            fused_embedding: Fused multimodal embedding (batch_size, input_dim)
            
        Returns:
            Dictionary with logits and probabilities
        """
        logits = self.classifier(fused_embedding)
        
        # Compute probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': torch.argmax(logits, dim=1),
        }


class BinaryClassifier(nn.Module):
    """
    Simplified binary classifier (Fake vs Real).
    Uses sigmoid instead of softmax for binary classification.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        """
        Initialize binary classifier.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(BinaryClassifier, self).__init__()
        
        self.input_dim = input_dim
        
        # Classification network
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            fused_embedding: Fused embedding
            
        Returns:
            Dictionary with predictions and confidence
        """
        # Sigmoid output (probability of being fake)
        fake_prob = self.classifier(fused_embedding).squeeze(1)
        
        # Real probability
        real_prob = 1 - fake_prob
        
        # Stack for consistency with MultimodalClassifier
        probabilities = torch.stack([real_prob, fake_prob], dim=1)
        
        # Predictions
        predictions = (fake_prob > 0.5).long()
        
        return {
            'logits': torch.log(probabilities + 1e-8),
            'probabilities': probabilities,
            'predictions': predictions,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
        }


class ClassifierWithUncertainty(nn.Module):
    """
    Classifier that outputs both predictions and uncertainty estimates.
    Uses Monte Carlo Dropout for uncertainty quantification.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
        num_mc_samples: int = 10,
    ):
        """
        Initialize classifier with uncertainty.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes
            dropout: Dropout rate (used at test time for MC dropout)
            num_mc_samples: Number of MC samples for uncertainty
        """
        super(ClassifierWithUncertainty, self).__init__()
        
        self.num_mc_samples = num_mc_samples
        
        # Classifier with dropout (not disabled during inference)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        fused_embedding: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional uncertainty estimation.
        
        Args:
            fused_embedding: Fused embedding
            return_uncertainty: If True, compute uncertainty via MC dropout
            
        Returns:
            Dictionary with predictions and optional uncertainty
        """
        if not return_uncertainty or not self.training:
            # Single forward pass (deterministic)
            logits = self.classifier(fused_embedding)
            probabilities = torch.softmax(logits, dim=1)
            
            return {
                'logits': logits,
                'probabilities': probabilities,
                'predictions': torch.argmax(logits, dim=1),
                'uncertainty': None,
            }
        
        else:
            # MC Dropout sampling
            probs_list = []
            
            # Ensure dropout is active
            self.train()
            
            for _ in range(self.num_mc_samples):
                logits = self.classifier(fused_embedding)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
            
            # Stack samples
            probs_samples = torch.stack(probs_list, dim=0)  # (num_mc, batch, num_classes)
            
            # Mean prediction
            mean_probs = probs_samples.mean(dim=0)
            
            # Uncertainty as variance
            uncertainty = probs_samples.var(dim=0).mean(dim=1)
            
            return {
                'logits': torch.log(mean_probs + 1e-8),
                'probabilities': mean_probs,
                'predictions': torch.argmax(mean_probs, dim=1),
                'uncertainty': uncertainty,
                'mc_samples': probs_samples,
            }
    
    def mc_dropout_forward(
        self,
        fused_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Explicit Monte Carlo Dropout inference.
        
        Args:
            fused_embedding: Fused embedding
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        return self.forward(fused_embedding, return_uncertainty=True)


class CompleteMultimodalModel(nn.Module):
    """
    Complete end-to-end multimodal model.
    Combines text encoder, image encoder, fusion, and classifier.
    """
    
    def __init__(
        self,
        text_encoder,
        image_encoder,
        fusion_module,
        classifier,
    ):
        """
        Initialize complete model.
        
        Args:
            text_encoder: Text encoder module
            image_encoder: Image encoder module
            fusion_module: Fusion module
            classifier: Classification head
        """
        super(CompleteMultimodalModel, self).__init__()
        
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = fusion_module
        self.classifier = classifier
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through entire model.
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            images: Image tensors (optional)
            token_type_ids: Token type IDs (optional)
            
        Returns:
            Dictionary with predictions and intermediate representations
        """
        # Encode text
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        text_embedding = text_output['embedding']
        
        # Encode image
        image_embedding = None
        if images is not None:
            image_output = self.image_encoder(images)
            image_embedding = image_output['embedding']
        
        # Fuse modalities
        fusion_output = self.fusion(text_embedding, image_embedding)
        fused_embedding = fusion_output['fused']
        
        # Classify
        classifier_output = self.classifier(fused_embedding)
        
        return {
            'logits': classifier_output['logits'],
            'probabilities': classifier_output['probabilities'],
            'predictions': classifier_output['predictions'],
            'text_embedding': text_embedding,
            'image_embedding': image_embedding,
            'fused_embedding': fused_embedding,
        }
