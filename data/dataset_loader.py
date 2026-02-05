
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
import pickle
import torch


class MisinformationDataset(Dataset):
    """
    PyTorch Dataset for multimodal misinformation detection.
    
    Supports:
    - Text content (post/news title)
    - Image paths (optional)
    - Metadata (engagement metrics, credibility scores)
    - Labels (Fake/Real)
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        image_paths: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
        text_preprocessor=None,
        image_preprocessor=None,
        image_size: int = 224,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of binary labels (0=Real, 1=Fake)
            image_paths: Optional list of image file paths
            metadata: Optional list of metadata dicts
            text_preprocessor: Function to preprocess text
            image_preprocessor: Function to preprocess images
            image_size: Expected size of processed images for placeholder creation.
        """
        self.texts = texts
        self.labels = labels
        self.image_paths = image_paths if image_paths is not None else [None] * len(texts)
        self.metadata = metadata if metadata is not None else [{} for _ in texts]
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        self.image_size = image_size
        
        assert len(self.texts) == len(self.labels), "Mismatch between texts and labels"
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get single sample.
        
        Returns:
            Dictionary with keys: text, label, image, metadata
        """
        text = self.texts[idx]
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        metadata = self.metadata[idx]
        
        # Preprocess text
        if self.text_preprocessor:
            text = self.text_preprocessor(text)
        
        # Preprocess image with error handling
        # Default to a black tensor if no image or error in loading
        image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        if image_path and self.image_preprocessor:
            try:
                image = self.image_preprocessor(image_path)
            except Exception as e:
                print(f"Warning: Could not load or preprocess image {image_path} at index {idx}. Error: {e}. Using placeholder.")
                # 'image' remains the placeholder tensor initialized above
        
        return {
            "text": text,
            "label": label,
            "image": image,
            "metadata": metadata,
        }


class DatasetLoader:
    """
    Loads and manages misinformation datasets.
    Supports CSV loading, synthetic data generation, and train-test-val splits.
    """
    
    def __init__(self, data_dir: str = "datasets"):
        """
        Initialize DatasetLoader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
    
    def load_csv_dataset(
        self,
        csv_path: str,
        label: int
    ) -> Tuple[List[str], List[int]]:
        """
        Load texts and generate labels from CSV.
        
        Args:
            csv_path: Path to CSV file
            label: Label to assign (0=Real, 1=Fake)
            
        Returns:
            Tuple of (texts, labels)
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Extract title as text (primary content)
        texts = df['title'].tolist()
        
        # Generate labels
        labels = [label] * len(texts)
        
        return texts, labels
    
    def load_all_datasets(self) -> Tuple[List[str], List[int]]:
        """
        Load all available datasets (GossipCop + PolitiFact).
        
        Returns:
            Tuple of combined (texts, labels)
        """
        all_texts = []
        all_labels = []
        
        dataset_files = [
            ("gossipcop_fake.csv", 1),  # Fake news
            ("gossipcop_real.csv", 0),  # Real news
            ("politifact_fake.csv", 1),
            ("politifact_real.csv", 0),
        ]
        
        for filename, label in dataset_files:
            csv_path = os.path.join(self.data_dir, filename)
            if os.path.exists(csv_path):
                texts, labels = self.load_csv_dataset(csv_path, label)
                all_texts.extend(texts)
                all_labels.extend(labels)
                print(f"✓ Loaded {filename}: {len(texts)} samples")
            else:
                print(f"☢ {filename} not found, skipping")
        
        return all_texts, all_labels
    
    def generate_synthetic_data(
        self,
        num_samples: int = 100,
        fake_ratio: float = 0.5
    ) -> Tuple[List[str], List[int]]:
        """
        Generate synthetic misinformation data for testing/demo purposes.
        
        Args:
            num_samples: Total number of samples to generate
            fake_ratio: Proportion of fake samples
            
        Returns:
            Tuple of (texts, labels)
        """
        fake_templates = [
            "BREAKING: {subject} reveals shocking truth about {topic}",
            "Scientists SHOCKED: {subject} causes {effect}",
            "{subject} caught in scandal involving {topic}",
            "Anonymous sources confirm {subject} is {claim}",
            "EXCLUSIVE: {subject} admits to {action}",
        ]
        
        real_templates = [
            "{subject} releases official statement on {topic}",
            "Researchers publish study about {topic}",
            "{subject} discusses {topic} in interview",
            "News report: {topic} shows new findings",
            "Official update: {subject} provides information on {topic}",
        ]
        
        subjects = ["Celebrity", "Politician", "Company", "Scientist", "Organization"]
        topics = ["climate", "economy", "health", "technology", "politics", "social media"]
        effects = ["increases cancer risk", "destroys ecosystem", "ruins industry", "causes controversy"]
        actions = ["fraud", "theft", "conspiracy", "misconduct", "negligence"]
        claims = ["guilty", "innocent", "bankrupt", "powerful", "influential"]
        
        num_fake = int(num_samples * fake_ratio)
        num_real = num_samples - num_fake
        
        texts = []
        labels = []
        
        # Generate fake samples
        for _ in range(num_fake):
            template = np.random.choice(fake_templates)
            text = template.format(
                subject=np.random.choice(subjects),
                topic=np.random.choice(topics),
                effect=np.random.choice(effects),
                action=np.random.choice(actions),
                claim=np.random.choice(claims),
            )
            texts.append(text)
            labels.append(1)
        
        # Generate real samples
        for _ in range(num_real):
            template = np.random.choice(real_templates)
            text = template.format(
                subject=np.random.choice(subjects),
                topic=np.random.choice(topics),
            )
            texts.append(text)
            labels.append(0)
        
        # Shuffle
        indices = np.random.permutation(len(texts))
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        return texts, labels
    
    def create_splits(
        self,
        texts: List[str],
        labels: List[int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[
        Tuple[List[str], List[int]],
        Tuple[List[str], List[int]],
        Tuple[List[str], List[int]]
    ]:
        """
        Create train/val/test splits while maintaining class balance.
        
        Args:
            texts: List of texts
            labels: List of labels
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_state: Random seed
            
        Returns:
            Tuple of ((train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels))
        """
        np.random.seed(random_state)
        
        # Separate by class
        fake_indices = [i for i, label in enumerate(labels) if label == 1]
        real_indices = [i for i, label in enumerate(labels) if label == 0]
        
        # Shuffle each class
        np.random.shuffle(fake_indices)
        np.random.shuffle(real_indices)
        
        # Split each class
        def split_indices(indices, ratios):
            n = len(indices)
            train_n = int(n * ratios[0])
            val_n = int(n * ratios[1])
            return (
                indices[:train_n],
                indices[train_n:train_n+val_n],
                indices[train_n+val_n:]
            )
        
        fake_train, fake_val, fake_test = split_indices(
            fake_indices, [train_ratio, val_ratio, test_ratio]
        )
        real_train, real_val, real_test = split_indices(
            real_indices, [train_ratio, val_ratio, test_ratio]
        )
        
        # Combine
        train_idx = fake_train + real_train
        val_idx = fake_val + real_val
        test_idx = fake_test + real_test
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)
        
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        
        return (
            (train_texts, train_labels),
            (val_texts, val_labels),
            (test_texts, test_labels)
        )
    
    def create_dataloaders(
        self,
        train_data: Tuple[List[str], List[int]],
        val_data: Optional[Tuple[List[str], List[int]]] = None,
        test_data: Optional[Tuple[List[str], List[int]]] = None,
        text_preprocessor=None,
        image_preprocessor=None,
        batch_size: int = 32,
        num_workers: int = 0,
        image_size: int = 224,
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            train_data: Tuple of (texts, labels) for training
            val_data: Optional validation data
            test_data: Optional test data
            text_preprocessor: Text preprocessing function
            image_preprocessor: Image preprocessing function
            batch_size: Batch size for loader
            num_workers: Number of workers for loading
            image_size: Expected size of processed images for placeholder creation.
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        dataloaders = {}
        
        # Training loader
        train_dataset = MisinformationDataset(
            texts=train_data[0],
            labels=train_data[1],
            text_preprocessor=text_preprocessor,
            image_preprocessor=image_preprocessor,
            image_size=image_size,
        )
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        
        # Validation loader
        if val_data:
            val_dataset = MisinformationDataset(
                texts=val_data[0],
                labels=val_data[1],
                text_preprocessor=text_preprocessor,
                image_preprocessor=image_preprocessor,
                image_size=image_size,
            )
            dataloaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        
        # Test loader
        if test_data:
            test_dataset = MisinformationDataset(
                texts=test_data[0],
                labels=test_data[1],
                text_preprocessor=text_preprocessor,
                image_preprocessor=image_preprocessor,
                image_size=image_size,
            )
            dataloaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        
        return dataloaders
