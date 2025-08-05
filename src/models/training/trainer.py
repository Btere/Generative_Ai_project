"""
Training Module for Custom Facial Landmark Detection CNN

This module provides comprehensive training functionality for the custom CNN model,
including data loading, augmentation, training loops, validation, and model checkpointing.

Author: Stran Connect Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import numpy as np
import cv2
from pathlib import Path
import json
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..facial_detection_model import FacialLandmarkCNN, FacialLandmarkLoss, ModelConfig, create_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacialLandmarkDataset(Dataset):
    """
    Custom dataset for facial landmark detection with bounding box annotations.
    
    This dataset handles loading and preprocessing of facial images with their
    corresponding landmark annotations and facial region bounding boxes.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        transform: Optional[Callable] = None,
        input_size: Tuple[int, int] = (224, 224)
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing image files
            annotations_file: JSON file with annotations
            transform: Optional transform to apply to images
            input_size: Target input size for images
        """
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_files = list(self.annotations.keys())
        
        # Face shape mapping
        self.shape_mapping = {
            'oval': 0, 'round': 1, 'square': 2,
            'heart': 3, 'diamond': 4, 'long': 5
        }
        
        logger.info(f"Loaded dataset with {len(self.image_files)} images")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing image and annotations
        """
        try:
            # Load image
            image_file = self.image_files[idx]
            image_path = self.data_dir / image_file
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get annotations
            annotation = self.annotations[image_file]
            
            # Process landmarks
            landmarks = np.array(annotation['landmarks'], dtype=np.float32)
            landmarks = landmarks.reshape(-1, 2)  # Reshape to (num_landmarks, 2)
            
            # Process bounding boxes
            bboxes = np.array(annotation['bboxes'], dtype=np.float32)
            bboxes = bboxes.reshape(-1, 4)  # Reshape to (num_bboxes, 4)
            
            # Get face shape
            face_shape = self.shape_mapping[annotation['face_shape']]
            
            # Resize image and adjust coordinates
            original_height, original_width = image.shape[:2]
            image_resized = cv2.resize(image, self.input_size)
            
            # Scale landmarks and bboxes to new size
            scale_x = self.input_size[0] / original_width
            scale_y = self.input_size[1] / original_height
            
            landmarks[:, 0] *= scale_x
            landmarks[:, 1] *= scale_y
            
            bboxes[:, 0] *= scale_x  # x
            bboxes[:, 1] *= scale_y  # y
            bboxes[:, 2] *= scale_x  # width
            bboxes[:, 3] *= scale_y  # height
            
            # Apply transforms if provided
            if self.transform:
                # Note: You'd need to adjust landmarks/bboxes for geometric transforms
                image_resized = self.transform(image_resized)
            else:
                # Convert to tensor and normalize
                image_resized = image_resized.astype(np.float32) / 255.0
                image_resized = torch.from_numpy(image_resized).permute(2, 0, 1)
            
            return {
                'image': image_resized,
                'landmarks': torch.from_numpy(landmarks.astype(np.float32)),
                'bboxes': torch.from_numpy(bboxes.astype(np.float32)),
                'face_shape': torch.tensor(face_shape, dtype=torch.long),
                'image_path': str(image_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return a dummy sample in case of error
            return self._get_dummy_sample()
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Create a dummy sample for error cases."""
        return {
            'image': torch.zeros(3, *self.input_size),
            'landmarks': torch.zeros(68, 2),
            'bboxes': torch.zeros(5, 4),
            'face_shape': torch.tensor(0, dtype=torch.long),
            'image_path': 'dummy'
        }


class DataAugmentation:
    """
    Data augmentation pipeline for facial landmark detection.
    
    Applies various transformations while preserving landmark and bounding box
    consistency.
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        flip_probability: float = 0.5
    ) -> None:
        """
        Initialize data augmentation.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            flip_probability: Probability of horizontal flipping
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.flip_probability = flip_probability
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply augmentation to a sample.
        
        Args:
            sample: Dictionary containing image and annotations
            
        Returns:
            Augmented sample
        """
        # For now, implement basic augmentations
        # In practice, you'd use libraries like albumentations for geometric consistency
        
        image = sample['image']
        
        # Color augmentations (don't affect landmarks/bboxes)
        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(*self.brightness_range)
            image = torch.clamp(image * brightness_factor, 0, 1)
        
        # Note: For geometric augmentations, you'd need to transform landmarks and bboxes accordingly
        # This is complex and requires careful coordinate transformations
        
        sample['image'] = image
        return sample


class ModelTrainer:
    """
    Comprehensive trainer for facial landmark detection model.
    
    Handles training loop, validation, checkpointing, and monitoring.
    """
    
    def __init__(
        self,
        model: FacialLandmarkCNN,
        config: ModelConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: Facial landmark detection model
            config: Model configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize loss function
        self.criterion = FacialLandmarkLoss(
            landmark_weight=config.landmark_weight,
            bbox_weight=config.bbox_weight,
            shape_weight=config.shape_weight,
            consistency_weight=config.consistency_weight
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(f'runs/{config.model_name}_{int(time.time())}')
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_landmark_loss = 0.0
        total_bbox_loss = 0.0
        total_shape_loss = 0.0
        total_consistency_loss = 0.0
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {self.epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch['image'])
                
                # Compute loss
                targets = {
                    'landmarks': batch['landmarks'],
                    'bboxes': batch['bboxes'],
                    'face_shape': batch['face_shape']
                }
                
                loss_dict = self.criterion(predictions, targets)
                
                # Backward pass
                loss_dict['total_loss'].backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item()
                total_landmark_loss += loss_dict['landmark_loss'].item()
                total_bbox_loss += loss_dict['bbox_loss'].item()
                total_shape_loss += loss_dict['shape_loss'].item()
                total_consistency_loss += loss_dict['consistency_loss'].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to tensorboard
                if batch_idx % 10 == 0:
                    step = self.epoch * num_batches + batch_idx
                    self.writer.add_scalar('Train/Loss', loss_dict['total_loss'].item(), step)
                    self.writer.add_scalar('Train/LandmarkLoss', loss_dict['landmark_loss'].item(), step)
                    self.writer.add_scalar('Train/BBoxLoss', loss_dict['bbox_loss'].item(), step)
                    self.writer.add_scalar('Train/ShapeLoss', loss_dict['shape_loss'].item(), step)
        
        # Compute average losses
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'landmark_loss': total_landmark_loss / num_batches,
            'bbox_loss': total_bbox_loss / num_batches,
            'shape_loss': total_shape_loss / num_batches,
            'consistency_loss': total_consistency_loss / num_batches
        }
        
        return avg_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_landmark_loss = 0.0
        total_bbox_loss = 0.0
        total_shape_loss = 0.0
        correct_shape_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch['image'])
                
                # Compute loss
                targets = {
                    'landmarks': batch['landmarks'],
                    'bboxes': batch['bboxes'],
                    'face_shape': batch['face_shape']
                }
                
                loss_dict = self.criterion(predictions, targets)
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item()
                total_landmark_loss += loss_dict['landmark_loss'].item()
                total_bbox_loss += loss_dict['bbox_loss'].item()
                total_shape_loss += loss_dict['shape_loss'].item()
                
                # Compute accuracy for face shape classification
                shape_predictions = torch.argmax(predictions['face_shape'], dim=1)
                correct_shape_predictions += (shape_predictions == batch['face_shape']).sum().item()
                total_samples += batch['face_shape'].size(0)
        
        num_batches = len(self.val_loader)
        shape_accuracy = correct_shape_predictions / total_samples if total_samples > 0 else 0.0
        
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'landmark_loss': total_landmark_loss / num_batches,
            'bbox_loss': total_bbox_loss / num_batches,
            'shape_loss': total_shape_loss / num_batches,
            'shape_accuracy': shape_accuracy
        }
        
        return avg_metrics
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['total_loss'])
            
            # Update scheduler
            self.scheduler.step(val_metrics['total_loss'])
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}, "
                       f"Val Loss: {val_metrics['total_loss']:.4f}, "
                       f"Shape Acc: {val_metrics['shape_accuracy']:.4f}")
            
            # Tensorboard logging
            self.writer.add_scalar('Epoch/TrainLoss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('Epoch/ShapeAccuracy', val_metrics['shape_accuracy'], epoch)
            self.writer.add_scalar('Epoch/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if epoch % self.config.save_frequency == 0 or val_metrics['total_loss'] < self.best_val_loss:
                self.save_checkpoint(val_metrics['total_loss'], is_best=val_metrics['total_loss'] < self.best_val_loss)
                
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
        
        logger.info("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, val_loss: float, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            val_loss: Current validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.config.model_name}_epoch_{self.epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"{self.config.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            
            logger.info(f"Loaded checkpoint from epoch {self.epoch}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def plot_training_curves(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        
        plt.show()


def create_data_loaders(
    train_data_dir: Union[str, Path],
    train_annotations: Union[str, Path],
    val_data_dir: Union[str, Path],
    val_annotations: Union[str, Path],
    config: ModelConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_data_dir: Training data directory
        train_annotations: Training annotations file
        val_data_dir: Validation data directory
        val_annotations: Validation annotations file
        config: Model configuration
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Data augmentation for training
    if config.use_augmentation:
        train_augmentation = DataAugmentation(
            rotation_range=config.rotation_range,
            brightness_range=config.brightness_range,
            contrast_range=config.contrast_range
        )
    else:
        train_augmentation = None
    
    # Create datasets
    train_dataset = FacialLandmarkDataset(
        data_dir=train_data_dir,
        annotations_file=train_annotations,
        transform=train_augmentation,
        input_size=config.input_size
    )
    
    val_dataset = FacialLandmarkDataset(
        data_dir=val_data_dir,
        annotations_file=val_annotations,
        transform=None,  # No augmentation for validation
        input_size=config.input_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader


def main_training_pipeline(
    train_data_dir: Union[str, Path],
    train_annotations: Union[str, Path],
    val_data_dir: Union[str, Path],
    val_annotations: Union[str, Path],
    config: Optional[ModelConfig] = None
) -> FacialLandmarkCNN:
    """
    Main training pipeline function.
    
    Args:
        train_data_dir: Training data directory
        train_annotations: Training annotations file
        val_data_dir: Validation data directory
        val_annotations: Validation annotations file
        config: Optional model configuration
        
    Returns:
        Trained model
    """
    if config is None:
        config = ModelConfig()
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data_dir, train_annotations,
        val_data_dir, val_annotations,
        config
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Start training
    trainer.train()
    
    # Plot training curves
    trainer.plot_training_curves(
        save_path=Path(config.checkpoint_dir) / f"{config.model_name}_training_curves.png"
    )
    
    return model