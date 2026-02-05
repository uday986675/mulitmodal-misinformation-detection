"""
Image Encoder Module
====================
Encodes images using EfficientNet for efficient feature extraction.
Justification: EfficientNet provides better accuracy-efficiency tradeoff than ResNet,
achieving state-of-the-art accuracy with fewer parameters and FLOPs.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional


class ImageEncoder(nn.Module):
    """
    Image encoder using EfficientNet-B0.
    Extracts visual features from images.
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        frozen_backbone: bool = False,
        output_dim: int = 768,
        dropout: float = 0.1,
    ):
        """
        Initialize ImageEncoder.
        
        Args:
            model_name: EfficientNet model variant
            pretrained: Use pretrained ImageNet weights
            frozen_backbone: Freeze backbone weights
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(ImageEncoder, self).__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load pretrained model
        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_out_dim = 1280
        elif model_name == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            backbone_out_dim = 1280
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze backbone if requested
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Global average pooling is already in EfficientNet
        # Remove classifier head
        if hasattr(self.backbone, 'classifier'):
            # For EfficientNet
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            # For ResNet
            self.backbone.fc = nn.Identity()
        
        # Projection layer to align with text embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through image encoder.
        
        Args:
            images: Image tensor (batch_size, 3, height, width)
            
        Returns:
            Dictionary with embeddings
        """
        # Extract features
        if isinstance(self.backbone, models.EfficientNet):
            # EfficientNet forward pass
            x = self.backbone.features(images)
            x = self.backbone.avgpool(x)
            x = x.view(x.size(0), -1)
        else:
            # ResNet forward pass
            x = self.backbone.conv1(images)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = x.flatten(1)
        
        # Project to embedding dimension
        image_embedding = self.projection(x)
        
        return {
            'embedding': image_embedding,
            'features': x,
        }
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim


class ImageEncoderResNet(nn.Module):
    """
    Alternative image encoder using ResNet-50.
    Provides good accuracy with more interpretability.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 768,
        dropout: float = 0.1,
    ):
        """
        Initialize ResNet-based image encoder.
        
        Args:
            pretrained: Use ImageNet weights
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(ImageEncoderResNet, self).__init__()
        
        self.output_dim = output_dim
        
        # Load ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
        )
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Image tensor (batch_size, 3, height, width)
            
        Returns:
            Dictionary with embeddings
        """
        image_embedding = self.resnet(images)
        
        return {
            'embedding': image_embedding,
        }
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim


class ImageEncoderWithPooling(nn.Module):
    """
    Image encoder with multiple pooling strategies.
    Combines global average pooling with max pooling.
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        output_dim: int = 768,
        dropout: float = 0.1,
    ):
        """
        Initialize encoder with multiple pooling.
        
        Args:
            model_name: Model variant
            pretrained: Use pretrained weights
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(ImageEncoderWithPooling, self).__init__()
        
        self.output_dim = output_dim
        
        # Base encoder
        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_dim = 1280
        else:
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        
        # Remove classification head
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone.fc = nn.Identity()
        
        # Multiple pooling heads
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        # Projection (combine two pooling strategies)
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
        )
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple pooling strategies.
        
        Args:
            images: Image tensor
            
        Returns:
            Dictionary with embeddings
        """
        # Extract features
        if isinstance(self.backbone, models.EfficientNet):
            features = self.backbone.features(images)
        else:
            features = self.backbone(images)
        
        # Multiple pooling
        avg_pooled = self.avgpool(features).view(features.size(0), -1)
        max_pooled = self.maxpool(features).view(features.size(0), -1)
        
        # Concatenate and project
        combined = torch.cat([avg_pooled, max_pooled], dim=1)
        image_embedding = self.projection(combined)
        
        return {
            'embedding': image_embedding,
            'avg_features': avg_pooled,
            'max_features': max_pooled,
        }
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim
