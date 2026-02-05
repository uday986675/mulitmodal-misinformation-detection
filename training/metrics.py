"""
Metrics Module
==============
Evaluation metrics for misinformation detection.
Includes: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
"""

import torch
import numpy as np
from typing import Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class MetricComputer:
    """Computes evaluation metrics for binary classification."""
    
    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics.
        
        Args:
            predictions: Predicted labels (0/1)
            targets: Ground truth labels (0/1)
            probabilities: Predicted probabilities for positive class
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, zero_division=0)
        
        # ROC-AUC (if probabilities provided)
        if probabilities is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(targets, probabilities)
                
                # Precision-Recall AUC
                precision_vals, recall_vals, _ = precision_recall_curve(targets, probabilities)
                metrics['pr_auc'] = auc(recall_vals, precision_vals)
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return metrics
    
    @staticmethod
    def compute_per_class_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict:
        """
        Compute per-class metrics using sklearn.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            
        Returns:
            Dictionary with per-class and macro metrics
        """
        report = classification_report(
            targets, predictions,
            labels=[0, 1],
            target_names=['Real', 'Fake'],
            output_dict=True,
            zero_division=0,
        )
        
        metrics = {
            'real_precision': report['Real']['precision'],
            'real_recall': report['Real']['recall'],
            'real_f1': report['Real']['f1-score'],
            'fake_precision': report['Fake']['precision'],
            'fake_recall': report['Fake']['recall'],
            'fake_f1': report['Fake']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
        }
        
        return metrics


class ConfusionMatrixAnalyzer:
    """Analyzes confusion matrix for error patterns."""
    
    @staticmethod
    def analyze_confusion_matrix(
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict:
        """
        Analyze confusion matrix.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            
        Returns:
            Analysis dictionary
        """
        cm = confusion_matrix(targets, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        total = tn + fp + fn + tp
        
        analysis = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': int(total),
            'tn_percentage': 100 * tn / total,
            'fp_percentage': 100 * fp / total,
            'fn_percentage': 100 * fn / total,
            'tp_percentage': 100 * tp / total,
        }
        
        return analysis


class CurveAnalyzer:
    """Analyzes ROC and PR curves."""
    
    @staticmethod
    def compute_roc_curve(
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> Dict:
        """
        Compute ROC curve metrics.
        
        Args:
            probabilities: Predicted probabilities
            targets: Ground truth labels
            
        Returns:
            ROC curve data
        """
        fpr, tpr, thresholds = roc_curve(targets, probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc),
            'optimal_threshold': float(optimal_threshold),
            'optimal_j_statistic': float(j_scores[optimal_idx]),
        }
    
    @staticmethod
    def compute_pr_curve(
        probabilities: np.ndarray,
        targets: np.ndarray,
    ) -> Dict:
        """
        Compute Precision-Recall curve metrics.
        
        Args:
            probabilities: Predicted probabilities
            targets: Ground truth labels
            
        Returns:
            PR curve data
        """
        precision, recall, thresholds = precision_recall_curve(targets, probabilities)
        pr_auc = auc(recall, precision)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(pr_auc),
        }


class MetricTracker:
    """Tracks metrics during training."""
    
    def __init__(self):
        """Initialize metric tracker."""
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_metrics = None
        self.best_epoch = 0
    
    def add_train_metrics(self, metrics: Dict):
        """Add training metrics."""
        self.train_metrics.append(metrics)
    
    def add_val_metrics(self, metrics: Dict):
        """Add validation metrics."""
        self.val_metrics.append(metrics)
        
        # Track best validation metrics
        if self.best_val_metrics is None or metrics.get('f1', 0) > self.best_val_metrics.get('f1', 0):
            self.best_val_metrics = metrics.copy()
            self.best_epoch = len(self.val_metrics) - 1
    
    def get_best_metrics(self) -> Tuple[Dict, int]:
        """
        Get best validation metrics and epoch.
        
        Returns:
            Tuple of (best_metrics_dict, best_epoch_number)
        """
        return self.best_val_metrics, self.best_epoch
    
    def get_metrics_history(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get full training and validation history.
        
        Returns:
            Tuple of (train_metrics_list, val_metrics_list)
        """
        return self.train_metrics, self.val_metrics
    
    def summary(self) -> Dict:
        """Get summary of best metrics."""
        if self.best_val_metrics is None:
            return {}
        
        return {
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_val_metrics,
        }


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            input_ids = batch.get('input_ids', batch.get('text')).to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
            targets = batch.get('label', batch.get('labels')).to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.get('logits', outputs.get('predictions'))
            probabilities = outputs.get('probabilities', torch.softmax(logits, dim=1))
            
            # Get predictions
            predictions = torch.argmax(probabilities, dim=1)
            
            # Move to CPU and convert to numpy
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    # Compute metrics
    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)
    probabilities_np = np.array(all_probabilities)
    
    metrics = MetricComputer.compute_metrics(
        predictions_np, targets_np, probabilities_np
    )
    
    return metrics
