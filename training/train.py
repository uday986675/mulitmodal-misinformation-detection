"""
Training Script
===============
Main training loop for multimodal misinformation detection model.
Supports: distributed training, mixed precision, early stopping, checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Tuple, Optional
import time
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.loss import FocalLoss, LabelSmoothingLoss, CombinedLoss
from training.metrics import MetricComputer, MetricTracker, evaluate_model
from utils.logger import Logger


class Trainer:
    """
    Trainer class for multimodal misinformation detection model.
    Handles training, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        loss_fn: nn.Module = None,
        optimizer: optim.Optimizer = None,
        lr_scheduler=None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision: bool = True,
        checkpoint_dir: str = 'checkpoints',
        logger: Logger = None,
    ):
        """
        Initialize Trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            test_loader: Test dataloader (optional)
            loss_fn: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: AdamW)
            lr_scheduler: Learning rate scheduler (optional)
            device: Device to use
            mixed_precision: Use mixed precision training
            checkpoint_dir: Directory for checkpoints
            logger: Logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logger or Logger()
        
        # Loss and optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.AdamW(model.parameters(), lr=1e-5)
        self.lr_scheduler = lr_scheduler
        
        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Metrics tracking
        self.metric_tracker = MetricTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = float('-inf')
        self.patience = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Extract data
            input_ids = batch.get('text', {}).get('input_ids', batch.get('input_ids')).to(self.device)
            attention_mask = batch.get('text', {}).get('attention_mask', batch.get('attention_mask')).to(self.device)
            targets = batch.get('label', batch.get('labels')).to(self.device)
            
            images = batch.get('image')
            if images is not None and images is not None:
                images = images.to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                    )
                    logits = outputs['logits']
                    loss = self.loss_fn(logits, targets)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                )
                logits = outputs['logits']
                loss = self.loss_fn(logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].detach().cpu().numpy())
            
            if batch_idx % max(1, len(self.train_loader) // 10) == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch + 1} - Batch {batch_idx}/{len(self.train_loader)}: "
                    f"Loss = {loss.item():.4f}"
                )
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        metrics = MetricComputer.compute_metrics(all_predictions, all_targets, all_probabilities)
        metrics['loss'] = total_loss / num_batches
        
        self.metric_tracker.add_train_metrics(metrics)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Extract data
                input_ids = batch.get('text', {}).get('input_ids', batch.get('input_ids')).to(self.device)
                attention_mask = batch.get('text', {}).get('attention_mask', batch.get('attention_mask')).to(self.device)
                targets = batch.get('label', batch.get('labels')).to(self.device)
                
                images = batch.get('image')
                if images is not None:
                    images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                )
                logits = outputs['logits']
                loss = self.loss_fn(logits, targets)
                
                # Track metrics
                total_loss += loss.item()
                num_batches += 1
                
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        metrics = MetricComputer.compute_metrics(all_predictions, all_targets, all_probabilities)
        metrics['loss'] = total_loss / num_batches
        
        self.metric_tracker.add_val_metrics(metrics)
        
        return metrics
    
    def fit(
        self,
        num_epochs: int = 10,
        early_stopping_patience: int = 3,
        metric_to_monitor: str = 'f1',
    ):
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            metric_to_monitor: Metric to monitor for early stopping
        """
        self.logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        self.logger.info(f"Monitoring metric: {metric_to_monitor}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Train: "
                f"Loss={train_metrics['loss']:.4f}, F1={train_metrics['f1']:.4f}, "
                f"Acc={train_metrics['accuracy']:.4f}"
            )
            
            # Validate
            if self.val_loader:
                val_metrics = self.validate()
                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - Val: "
                    f"Loss={val_metrics['loss']:.4f}, F1={val_metrics['f1']:.4f}, "
                    f"Acc={val_metrics['accuracy']:.4f}"
                )
                
                # Early stopping
                current_metric = val_metrics.get(metric_to_monitor, 0)
                if current_metric > self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.patience = 0
                    self.save_checkpoint(tag='best')
                else:
                    self.patience += 1
                    if self.patience >= early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Learning rate scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        self.logger.info("Training complete!")
    
    def save_checkpoint(self, tag: str = 'latest'):
        """
        Save model checkpoint.
        
        Args:
            tag: Tag for checkpoint name
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_metrics_summary(self) -> Dict:
        """Get training metrics summary."""
        return self.metric_tracker.summary()
