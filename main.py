"""
Main Application Script
=======================
Complete pipeline for multimodal misinformation detection.
Orchestrates data loading, training, evaluation, and inference.

Usage:
    python main.py --config config.json --mode train
    python main.py --mode inference --text "Post text" --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_loader import DatasetLoader
from data.preprocess_text import TextPreprocessor
from data.preprocess_image import ImagePreprocessor
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.multimodal_fusion import MultimodalFusion
from models.classifier import MultimodalClassifier, CompleteMultimodalModel
from training.train import Trainer
from training.loss import FocalLoss, CombinedLoss
from inference.predict import Predictor
from utils.config import Config, DEFAULT_CONFIG
from utils.logger import Logger, setup_logging
from utils.helpers import (
    set_seed, get_device, get_device_info, get_model_info,
    count_parameters, save_json, create_directories
)


def build_model(config: Config) -> CompleteMultimodalModel:
    """
    Build complete multimodal model.
    
    Args:
        config: Configuration object
        
    Returns:
        Complete model
    """
    logger = Logger()
    logger.section("Building Model Architecture")
    
    # Text encoder
    text_encoder = TextEncoder(
        model_name=config.data.text_model,
        hidden_dim=config.model.text_hidden_dim,
        dropout=config.model.text_dropout,
        freeze_backbone=config.model.text_freeze_backbone,
    )
    logger.info(f"✓ Text Encoder: {config.model.text_encoder_type}")
    
    # Image encoder
    image_encoder = ImageEncoder(
        model_name=config.model.image_encoder_type,
        output_dim=config.model.image_hidden_dim,
        dropout=config.model.image_dropout,
    )
    logger.info(f"✓ Image Encoder: {config.model.image_encoder_type}")
    
    # Fusion module
    fusion = MultimodalFusion(
        text_dim=config.model.text_hidden_dim,
        image_dim=config.model.image_hidden_dim,
        hidden_dim=config.model.fusion_hidden_dim,
        dropout=config.model.text_dropout,
        fusion_method=config.model.fusion_type,
    )
    logger.info(f"✓ Fusion: {config.model.fusion_type}")
    
    # Classifier
    classifier = MultimodalClassifier(
        input_dim=config.model.fusion_hidden_dim,
        hidden_dim=config.model.classifier_hidden_dim,
        num_classes=config.model.output_dim,
        dropout=config.model.classifier_dropout,
    )
    logger.info(f"✓ Classifier: Binary Classification (Fake/Real)")
    
    # Complete model
    model = CompleteMultimodalModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_module=fusion,
        classifier=classifier,
    )
    
    # Print model statistics
    total_params, trainable_params = count_parameters(model)
    logger.info(f"\nModel Statistics:")
    logger.info(f"  - Total Parameters: {total_params:,}")
    logger.info(f"  - Trainable Parameters: {trainable_params:,}")
    logger.info(f"  - Model Size: {sum(p.numel() * 4 for p in model.parameters()) / 1e6:.2f} MB")
    
    return model


def train_mode(config: Config, logger: Logger):
    """
    Training mode: load data, train model.
    
    Args:
        config: Configuration
        logger: Logger instance
    """
    logger.section("TRAINING MODE")
    
    # Set seed
    set_seed(config.seed)
    
    # Device
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Device Info: {get_device_info()}")
    
    # Load datasets
    logger.section("Loading Datasets")
    dataset_loader = DatasetLoader(data_dir=config.data.data_dir)
    
    texts, labels = dataset_loader.load_all_datasets()
    logger.info(f"Total samples: {len(texts)}")
    logger.info(f"Fake samples: {sum(labels)}")
    logger.info(f"Real samples: {len(labels) - sum(labels)}")
    
    # Create splits
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = \
        dataset_loader.create_splits(texts, labels)
    
    logger.info(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    
    # Preprocessors
    text_preprocessor = TextPreprocessor(
        model_name=config.data.text_model,
        max_length=config.data.text_max_length,
    )
    
    image_preprocessor = ImagePreprocessor(
        img_size=config.data.image_size,
        augment=config.data.image_augmentation,
    )
    
    # DataLoaders
    dataloaders = dataset_loader.create_dataloaders(
        train_data=(train_texts, train_labels),
        val_data=(val_texts, val_labels),
        test_data=(test_texts, test_labels),
        text_preprocessor=text_preprocessor.preprocess,
        image_preprocessor=image_preprocessor.preprocess,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    
    logger.info("✓ Data loading complete")
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Loss function
    if config.training.loss_function == "focal":
        loss_fn = FocalLoss(alpha=config.training.focal_alpha, gamma=config.training.focal_gamma)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    logger.info(f"Loss Function: {config.training.loss_function}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Learning rate scheduler
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('val'),
        test_loader=dataloaders.get('test'),
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        mixed_precision=config.training.mixed_precision,
        checkpoint_dir=config.training.checkpoint_dir,
        logger=logger,
    )
    
    # Train
    logger.section("Starting Training")
    trainer.fit(
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        metric_to_monitor=config.training.metric_to_monitor,
    )
    
    # Results
    logger.section("Training Complete")
    best_metrics, best_epoch = trainer.metric_tracker.get_best_metrics()
    logger.info(f"Best Epoch: {best_epoch + 1}")
    logger.info(f"Best Metrics: {best_metrics}")
    
    # Save final model
    final_model_path = Path(config.training.checkpoint_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


def inference_mode(config: Config, text: str, image_path: str, logger: Logger):
    """
    Inference mode: load model and make predictions.
    
    Args:
        config: Configuration
        text: Input text
        image_path: Path to image (optional)
        logger: Logger instance
    """
    logger.section("INFERENCE MODE")
    
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Preprocessors
    text_preprocessor = TextPreprocessor(
        model_name=config.data.text_model,
        max_length=config.data.text_max_length,
    )
    
    image_preprocessor = ImagePreprocessor(
        img_size=config.data.image_size,
    )
    
    # Build and load model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint if specified
    if config.inference.model_checkpoint:
        checkpoint = torch.load(config.inference.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint: {config.inference.model_checkpoint}")
    
    # Predictor
    predictor = Predictor(
        model=model,
        text_preprocessor=text_preprocessor,
        image_preprocessor=image_preprocessor,
        device=device,
        logger=logger,
    )
    
    # Make prediction
    logger.section("Making Prediction")
    logger.info(f"Text: {text[:100]}...")
    if image_path:
        logger.info(f"Image: {image_path}")
    
    result = predictor.predict_single(text, image_path)
    
    logger.info(f"\nPrediction Result:")
    logger.info(f"  - Prediction: {result['prediction']}")
    logger.info(f"  - Confidence: {result['confidence']:.4f}")
    logger.info(f"  - Probabilities:")
    logger.info(f"    - Real: {result['probabilities']['Real']:.4f}")
    logger.info(f"    - Fake: {result['probabilities']['Fake']:.4f}")
    
    # With uncertainty
    if config.inference.use_mc_dropout:
        logger.section("Computing Uncertainty (MC Dropout)")
        result_unc = predictor.predict_with_uncertainty(
            text, image_path,
            num_mc_samples=config.inference.mc_samples
        )
        logger.info(f"  - Uncertainty: {result_unc['uncertainty']:.4f}")
        logger.info(f"  - Std (Real): {result_unc['probabilities_std']['Real']:.4f}")
        logger.info(f"  - Std (Fake): {result_unc['probabilities_std']['Fake']:.4f}")
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal Misinformation Detection System"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Operating mode: train or inference"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text for inference"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image for inference"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_json(args.config)
    else:
        config = DEFAULT_CONFIG
    
    # Logger
    logger, exp_logger = setup_logging(
        experiment_name=f"mmd_{args.mode}",
        log_level=20 if config.verbose else 30,
    )
    
    logger.info(f"Configuration:\n{config}")
    
    try:
        if args.mode == "train":
            train_mode(config, logger)
        
        elif args.mode == "inference":
            if not args.text:
                logger.error("Text required for inference mode (--text)")
                sys.exit(1)
            
            if args.checkpoint:
                config.inference.model_checkpoint = args.checkpoint
            
            result = inference_mode(config, args.text, args.image, logger)
            
            # Save result
            output_path = Path("inference_results") / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path.parent.mkdir(exist_ok=True)
            save_json(result, str(output_path))
            logger.info(f"\nResult saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.section("COMPLETE")


if __name__ == "__main__":
    from datetime import datetime
    main()
