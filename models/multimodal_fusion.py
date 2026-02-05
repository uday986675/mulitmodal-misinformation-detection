"""
Multimodal Fusion Module
========================
Fuses text and image embeddings using multiple fusion strategies.
Supports: concatenation, cross-modal attention, and gating mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class MultimodalFusion(nn.Module):
    """
    Fuses text and image embeddings via concatenation + FC layers.
    Simple yet effective fusion strategy for multimodal learning.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        fusion_method: str = "concat",
    ):
        """
        Initialize MultimodalFusion.
        
        Args:
            text_dim: Dimension of text embeddings
            image_dim: Dimension of image embeddings
            hidden_dim: Hidden dimension for fusion layers
            dropout: Dropout rate
            fusion_method: Type of fusion ("concat", "bilinear", "gating")
        """
        super(MultimodalFusion, self).__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "concat":
            # Simple concatenation + MLP
            fusion_input_dim = text_dim + image_dim
            
            self.fusion_net = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
        elif fusion_method == "bilinear":
            # Bilinear fusion for interaction modeling
            self.fusion_net = nn.Sequential(
                nn.Linear(text_dim + image_dim + text_dim * image_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
        elif fusion_method == "gating":
            # Gating mechanism: learns to weight modalities
            self.text_gate = nn.Sequential(
                nn.Linear(text_dim + image_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, text_dim),
                nn.Sigmoid(),
            )
            self.image_gate = nn.Sequential(
                nn.Linear(text_dim + image_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, image_dim),
                nn.Sigmoid(),
            )
            self.fusion_net = nn.Sequential(
                nn.Linear(text_dim + image_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse text and image embeddings.
        
        Args:
            text_embedding: Text embedding (batch_size, text_dim)
            image_embedding: Image embedding (batch_size, image_dim), optional
            
        Returns:
            Dictionary with fused representation and weights
        """
        batch_size = text_embedding.shape[0]
        
        # Handle missing image modality
        if image_embedding is None:
            # Use zero placeholder
            image_embedding = torch.zeros(
                batch_size, self.image_dim,
                device=text_embedding.device,
                dtype=text_embedding.dtype,
            )
        
        if self.fusion_method == "concat":
            # Concatenate embeddings
            fused = torch.cat([text_embedding, image_embedding], dim=1)
            fused_output = self.fusion_net(fused)
            
            return {
                'fused': fused_output,
                'text_weight': None,
                'image_weight': None,
            }
            
        elif self.fusion_method == "bilinear":
            # Compute bilinear interaction
            bilinear_interaction = text_embedding.unsqueeze(2) * image_embedding.unsqueeze(1)
            bilinear_interaction = bilinear_interaction.view(batch_size, -1)
            
            # Concatenate with original embeddings
            combined = torch.cat([text_embedding, image_embedding, bilinear_interaction], dim=1)
            fused_output = self.fusion_net(combined)
            
            return {
                'fused': fused_output,
                'bilinear_interaction': bilinear_interaction,
            }
            
        elif self.fusion_method == "gating":
            # Gating weights for each modality
            combined = torch.cat([text_embedding, image_embedding], dim=1)
            
            text_gate = self.text_gate(combined)
            image_gate = self.image_gate(combined)
            
            # Apply gates
            gated_text = text_embedding * text_gate
            gated_image = image_embedding * image_gate
            
            # Fuse gated representations
            gated_combined = torch.cat([gated_text, gated_image], dim=1)
            fused_output = self.fusion_net(gated_combined)
            
            return {
                'fused': fused_output,
                'text_gate': text_gate,
                'image_gate': image_gate,
            }
    
    def get_output_dim(self) -> int:
        """Get output dimension after fusion."""
        return self.hidden_dim


class CrossModalAttentionFusion(nn.Module):
    """
    Fuses modalities using cross-modal attention.
    Allows text to attend to image features and vice versa.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize CrossModalAttentionFusion.
        
        Args:
            text_dim: Text embedding dimension
            image_dim: Image embedding dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossModalAttentionFusion, self).__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Project embeddings to same dimension if needed
        if text_dim != image_dim:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.image_proj = nn.Linear(image_dim, hidden_dim)
        else:
            self.text_proj = nn.Identity()
            self.image_proj = nn.Identity()
            hidden_dim = text_dim
        
        # Cross-modal attention
        self.text_to_image_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply cross-modal attention fusion.
        
        Args:
            text_embedding: Text embedding
            image_embedding: Image embedding (optional)
            
        Returns:
            Dictionary with fused representation
        """
        batch_size = text_embedding.shape[0]
        
        # Handle missing image
        if image_embedding is None:
            image_embedding = torch.zeros_like(text_embedding)
        
        # Project to common dimension
        text_proj = self.text_proj(text_embedding).unsqueeze(1)  # (B, 1, D)
        image_proj = self.image_proj(image_embedding).unsqueeze(1)  # (B, 1, D)
        
        # Cross-modal attention
        text_attended, _ = self.text_to_image_attn(
            text_proj, image_proj, image_proj
        )
        image_attended, _ = self.image_to_text_attn(
            image_proj, text_proj, text_proj
        )
        
        # Reshape
        text_attended = text_attended.squeeze(1)
        image_attended = image_attended.squeeze(1)
        
        # Combine
        combined = torch.cat([text_attended, image_attended], dim=1)
        fused = self.output_projection(combined)
        
        return {
            'fused': fused,
            'text_attended': text_attended,
            'image_attended': image_attended,
        }
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_dim


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion with early and late fusion stages.
    Combines multiple levels of fusion for better representation.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize HierarchicalFusion.
        
        Args:
            text_dim: Text embedding dimension
            image_dim: Image embedding dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(HierarchicalFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Early fusion (direct combination)
        self.early_fusion = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Modality-specific transformations
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.image_transform = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Late fusion (after transformation)
        self.late_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Final combination
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical fusion.
        
        Args:
            text_embedding: Text embedding
            image_embedding: Image embedding (optional)
            
        Returns:
            Dictionary with fused representation
        """
        batch_size = text_embedding.shape[0]
        
        if image_embedding is None:
            image_embedding = torch.zeros_like(text_embedding)
        
        # Early fusion
        early = self.early_fusion(torch.cat([text_embedding, image_embedding], dim=1))
        
        # Transform modalities
        text_trans = self.text_transform(text_embedding)
        image_trans = self.image_transform(image_embedding)
        
        # Late fusion
        late = self.late_fusion(torch.cat([text_trans, image_trans], dim=1))
        
        # Combine early and late
        combined = torch.cat([early, late], dim=1)
        fused = self.final_fusion(combined)
        
        return {
            'fused': fused,
            'early_fusion': early,
            'late_fusion': late,
        }
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_dim
