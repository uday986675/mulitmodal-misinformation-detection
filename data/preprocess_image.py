"""
Image Preprocessing Module
===========================
Handles image loading, resizing, augmentation, and normalization for CNN/ResNet.
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class ImagePreprocessor:
    """
    Preprocesses images for model input.
    Handles loading, resizing, augmentation, and normalization.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        augment: bool = False,
        normalize: bool = True,
    ):
        """
        Initialize ImagePreprocessor.
        
        Args:
            img_size: Target image size (square)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize to ImageNet stats
        """
        self.img_size = img_size
        self.augment = augment
        
        # Base transforms (resize + convert to tensor)
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
        
        # Add normalization if requested
        if normalize:
            # ImageNet normalization
            base_transforms.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        
        # Create transform pipeline
        self.base_transform = transforms.Compose(base_transforms)
        
        # Augmentation transforms (stronger)
        if augment:
            aug_transforms = [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
            
            if normalize:
                aug_transforms.append(
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                )
            
            self.augment_transform = transforms.Compose(aug_transforms)
        else:
            self.augment_transform = None
    
    def load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor or None if loading fails
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                return None
            
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Apply augmentation during training (randomly)
            if self.augment and self.augment_transform and np.random.rand() > 0.5:
                img_tensor = self.augment_transform(img)
            else:
                img_tensor = self.base_transform(img)
            
            return img_tensor
            
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            return None
    
    def preprocess(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess single image (alias for load_image).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        return self.load_image(image_path)
    
    def batch_preprocess(self, image_paths: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Tuple of (stacked tensors, validity mask)
        """
        images = []
        valid_mask = []
        
        for path in image_paths:
            img = self.load_image(path)
            if img is not None:
                images.append(img)
                valid_mask.append(1.0)
            else:
                # Add placeholder zero tensor
                images.append(torch.zeros(3, self.img_size, self.img_size))
                valid_mask.append(0.0)
        
        if images:
            stacked = torch.stack(images)
        else:
            stacked = torch.zeros(len(image_paths), 3, self.img_size, self.img_size)
        
        return stacked, torch.tensor(valid_mask)
    
    @staticmethod
    def create_placeholder_image(size: int = 224) -> torch.Tensor:
        """
        Create a placeholder image tensor (for missing images).
        
        Args:
            size: Image size
            
        Returns:
            Placeholder tensor with shape (3, size, size)
        """
        return torch.zeros(3, size, size)
    
    @staticmethod
    def get_image_stats(img_path: str) -> Optional[dict]:
        """
        Get statistics about an image.
        
        Args:
            img_path: Path to image
            
        Returns:
            Dictionary with image info or None
        """
        try:
            img = Image.open(img_path)
            return {
                'size': img.size,
                'format': img.format,
                'mode': img.mode,
                'width': img.width,
                'height': img.height,
            }
        except:
            return None


class ImageAugmentation:
    """Additional augmentation utilities."""
    
    @staticmethod
    def apply_mixup(img1: torch.Tensor, img2: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
        """
        Apply mixup augmentation between two images.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            alpha: Blending weight
            
        Returns:
            Blended image
        """
        lam = np.random.beta(alpha, alpha)
        return lam * img1 + (1 - lam) * img2
    
    @staticmethod
    def apply_cutout(img: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
        """
        Apply cutout augmentation (random patch removal).
        
        Args:
            img: Image tensor (C, H, W)
            patch_size: Size of patch to remove
            
        Returns:
            Augmented image
        """
        c, h, w = img.shape
        
        # Random position
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        
        # Create copy and apply cutout
        img_aug = img.clone()
        img_aug[:, y:y+patch_size, x:x+patch_size] = 0
        
        return img_aug
