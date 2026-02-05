"""
Loss Functions Module
=====================
Custom loss functions for multimodal misinformation detection.
Supports focal loss for class imbalance and contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples by downweighting easy examples.
    Reference: Lin et al., Focal Loss for Dense Object Detection (CVPR 2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize FocalLoss.
        
        Args:
            alpha: Weighting factor in range (0,1) to balance classes
            gamma: Exponent of the modulating factor (1-p_t)^gamma
            reduction: Type of reduction ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Get probabilities
        p = torch.softmax(logits, dim=1)
        
        # Get class probabilities
        p_t = p[torch.arange(len(targets)), targets]
        
        # Compute focal loss
        loss = -(1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, (1 - self.alpha))
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss to prevent overconfidence.
    Reduces the confidence of the target class and smooths other classes.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        """
        Initialize LabelSmoothingLoss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
            reduction: Reduction type
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothed loss.
        
        Args:
            logits: Model logits
            targets: Ground truth labels
            
        Returns:
            Label smoothing loss
        """
        # Get probabilities
        log_probs = F.log_softmax(logits, dim=1)
        
        # Create smoothed targets
        batch_size = targets.shape[0]
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        targets_one_hot.fill_diagonal_(self.smoothing / (self.num_classes - 1))
        
        # Compute loss
        loss = -(targets_one_hot * log_probs).sum(dim=1)
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative embeddings.
    Pulls embeddings of same class closer, pushes different classes apart.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 1.0,
    ):
        """
        Initialize ContrastiveLoss.
        
        Args:
            temperature: Temperature for scaling similarities
            margin: Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE).
        
        Args:
            embeddings: Feature embeddings (batch_size, embedding_dim)
            targets: Class labels (batch_size,)
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.T)
        sim_matrix = sim_matrix / self.temperature
        
        # Create positive pairs (same class)
        batch_size = embeddings.shape[0]
        mask = targets.unsqueeze(1) == targets.unsqueeze(0)
        mask = mask.fill_diagonal_(False)
        
        # InfoNCE loss
        pos = sim_matrix[mask].view(batch_size, -1)
        neg = sim_matrix[~mask].view(batch_size, -1)
        
        # Compute loss
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combines multiple losses for joint optimization.
    Example: CE loss for classification + contrastive loss for embeddings.
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,
        contrastive_weight: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize CombinedLoss.
        
        Args:
            ce_weight: Weight for cross-entropy loss
            focal_weight: Weight for focal loss
            contrastive_weight: Weight for contrastive loss
            label_smoothing: Smoothing factor (0 = no smoothing)
        """
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.contrastive_weight = contrastive_weight
        
        # Initialize losses
        if ce_weight > 0:
            if label_smoothing > 0:
                self.ce_loss = LabelSmoothingLoss(smoothing=label_smoothing)
            else:
                self.ce_loss = nn.CrossEntropyLoss()
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss()
        
        if contrastive_weight > 0:
            self.contrastive_loss = ContrastiveLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Classification logits
            targets: Ground truth labels
            embeddings: Feature embeddings (optional, for contrastive loss)
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Classification losses
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(logits, targets)
            total_loss += self.ce_weight * ce_loss
        
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(logits, targets)
            total_loss += self.focal_weight * focal_loss
        
        # Contrastive loss
        if self.contrastive_weight > 0 and embeddings is not None:
            contrastive_loss = self.contrastive_loss(embeddings, targets)
            total_loss += self.contrastive_weight * contrastive_loss
        
        return total_loss
