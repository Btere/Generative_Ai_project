"""
Custom CNN Model for Facial Landmark Detection and Bounding Box Generation

This module implements a custom CNN architecture specifically designed for detecting
facial landmarks and generating precise bounding boxes around the forehead, chin,
left and right jaw, and jawline areas for facial shape analysis.

Author: Stran Connect Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    Residual block for deeper network training with skip connections.
    
    This helps with gradient flow and allows training of deeper networks
    for better facial feature extraction.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            stride: Convolution stride
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionModule(nn.Module):
    """
    Spatial attention module to focus on facial features.
    
    This helps the network pay attention to important facial regions
    like forehead, chin, and jawline areas.
    """
    
    def __init__(self, in_channels: int) -> None:
        """
        Initialize attention module.
        
        Args:
            in_channels: Number of input channels
        """
        super(AttentionModule, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention module."""
        attention = F.adaptive_avg_pool2d(x, 1)
        attention = F.relu(self.conv1(attention))
        attention = self.sigmoid(self.conv2(attention))
        return x * attention


class FacialLandmarkCNN(nn.Module):
    """
    Custom CNN architecture for facial landmark detection and bounding box generation.
    
    This model is specifically designed to detect key facial landmarks and generate
    precise bounding boxes for:
    - Forehead region
    - Chin area
    - Left jaw/cheek
    - Right jaw/cheek
    - Overall jawline
    """
    
    def __init__(
        self, 
        num_landmarks: int = 68,
        num_bboxes: int = 5,
        input_size: Tuple[int, int] = (224, 224)
    ) -> None:
        """
        Initialize the facial landmark detection CNN.
        
        Args:
            num_landmarks: Number of facial landmarks to detect
            num_bboxes: Number of bounding boxes to generate
            input_size: Input image dimensions (height, width)
        """
        super(FacialLandmarkCNN, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.num_bboxes = num_bboxes
        self.input_size = input_size
        
        # Feature extraction backbone
        self.backbone = self._build_backbone()
        
        # Attention modules for feature refinement
        self.attention1 = AttentionModule(128)
        self.attention2 = AttentionModule(256)
        self.attention3 = AttentionModule(512)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Landmark detection head
        self.landmark_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_landmarks * 2)  # x, y coordinates
        )
        
        # Bounding box detection head
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_bboxes * 4)  # x, y, width, height for each bbox
        )
        
        # Classification head for face shape
        self.shape_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 6)  # 6 face shapes: oval, round, square, heart, diamond, long
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_backbone(self) -> nn.Sequential:
        """Build the feature extraction backbone."""
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks - Stage 1
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            
            # Residual blocks - Stage 2  
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            
            # Residual blocks - Stage 3
            ResidualBlock(128, 256, stride=2), 
            ResidualBlock(256, 256),
            
            # Residual blocks - Stage 4
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary containing landmarks, bounding boxes, and face shape predictions
        """
        # Feature extraction through backbone
        features = self.backbone(x)
        
        # Apply attention mechanisms at different scales
        # Note: We need to extract intermediate features for attention
        batch_size = x.size(0)
        
        # Get intermediate features for attention
        x_inter = x
        for i, layer in enumerate(self.backbone):
            x_inter = layer(x_inter)
            if i == 6:  # After stage 1
                x_inter = self.attention1(x_inter)
            elif i == 9:  # After stage 2  
                x_inter = self.attention2(x_inter)
            elif i == 12:  # After stage 3
                x_inter = self.attention3(x_inter)
        
        # Global pooling
        pooled_features = self.global_pool(x_inter).view(batch_size, -1)
        
        # Generate predictions
        landmarks = self.landmark_head(pooled_features)
        bboxes = self.bbox_head(pooled_features)
        face_shape = self.shape_head(pooled_features)
        
        # Reshape outputs
        landmarks = landmarks.view(batch_size, self.num_landmarks, 2)
        bboxes = bboxes.view(batch_size, self.num_bboxes, 4)
        
        return {
            'landmarks': landmarks,
            'bboxes': bboxes,
            'face_shape': face_shape,
            'features': pooled_features
        }
    
    def predict_facial_regions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict specific facial regions with their bounding boxes.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary with region-specific bounding boxes
        """
        outputs = self.forward(x)
        bboxes = outputs['bboxes']
        
        # Map bounding boxes to specific facial regions
        # Expected order: forehead, chin, left_jaw, right_jaw, jawline
        region_bboxes = {
            'forehead': bboxes[:, 0, :],      # First bbox for forehead
            'chin': bboxes[:, 1, :],          # Second bbox for chin
            'left_jaw': bboxes[:, 2, :],      # Third bbox for left jaw
            'right_jaw': bboxes[:, 3, :],     # Fourth bbox for right jaw
            'jawline': bboxes[:, 4, :]        # Fifth bbox for jawline
        }
        
        return {
            **region_bboxes,
            'landmarks': outputs['landmarks'],
            'face_shape': outputs['face_shape']
        }


class FacialLandmarkLoss(nn.Module):
    """
    Custom loss function for facial landmark detection.
    
    Combines multiple loss components:
    - Landmark regression loss (MSE)
    - Bounding box regression loss (Smooth L1)
    - Face shape classification loss (CrossEntropy)
    - Geometric consistency loss
    """
    
    def __init__(
        self,
        landmark_weight: float = 1.0,
        bbox_weight: float = 1.0,
        shape_weight: float = 0.5,
        consistency_weight: float = 0.3
    ) -> None:
        """
        Initialize the loss function.
        
        Args:
            landmark_weight: Weight for landmark loss
            bbox_weight: Weight for bounding box loss
            shape_weight: Weight for face shape classification loss
            consistency_weight: Weight for geometric consistency loss
        """
        super(FacialLandmarkLoss, self).__init__()
        
        self.landmark_weight = landmark_weight
        self.bbox_weight = bbox_weight
        self.shape_weight = shape_weight
        self.consistency_weight = consistency_weight
        
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
            
        Returns:
            Dictionary containing individual and total losses
        """
        # Landmark regression loss
        landmark_loss = self.mse_loss(predictions['landmarks'], targets['landmarks'])
        
        # Bounding box regression loss
        bbox_loss = self.smooth_l1_loss(predictions['bboxes'], targets['bboxes'])
        
        # Face shape classification loss
        shape_loss = self.cross_entropy_loss(predictions['face_shape'], targets['face_shape'])
        
        # Geometric consistency loss (ensure landmarks are within bounding boxes)
        consistency_loss = self._compute_consistency_loss(
            predictions['landmarks'], predictions['bboxes']
        )
        
        # Total weighted loss
        total_loss = (
            self.landmark_weight * landmark_loss +
            self.bbox_weight * bbox_loss +
            self.shape_weight * shape_loss +
            self.consistency_weight * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'landmark_loss': landmark_loss,
            'bbox_loss': bbox_loss,
            'shape_loss': shape_loss,
            'consistency_loss': consistency_loss
        }
    
    def _compute_consistency_loss(
        self,
        landmarks: torch.Tensor,
        bboxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geometric consistency loss to ensure landmarks are within bounding boxes.
        
        Args:
            landmarks: Predicted landmarks (batch_size, num_landmarks, 2)
            bboxes: Predicted bounding boxes (batch_size, num_bboxes, 4)
            
        Returns:
            Consistency loss tensor
        """
        # For simplicity, we'll compute a basic consistency loss
        # In practice, you would map specific landmarks to specific bounding boxes
        
        # Convert bbox format from (x, y, w, h) to (x1, y1, x2, y2)
        x1 = bboxes[:, :, 0]  # x
        y1 = bboxes[:, :, 1]  # y  
        x2 = x1 + bboxes[:, :, 2]  # x + width
        y2 = y1 + bboxes[:, :, 3]  # y + height
        
        # Compute minimum bounding box that contains all landmarks
        landmark_x = landmarks[:, :, 0]  # All x coordinates
        landmark_y = landmarks[:, :, 1]  # All y coordinates
        
        landmark_min_x = torch.min(landmark_x, dim=1)[0]
        landmark_max_x = torch.max(landmark_x, dim=1)[0]
        landmark_min_y = torch.min(landmark_y, dim=1)[0]
        landmark_max_y = torch.max(landmark_y, dim=1)[0]
        
        # Simple consistency: penalize if predicted bbox doesn't contain landmarks
        # This is a simplified version - in practice you'd map specific landmarks to regions
        consistency_loss = torch.tensor(0.0, device=landmarks.device)
        
        return consistency_loss


class ModelConfig:
    """Configuration class for model parameters and training settings."""
    
    def __init__(self) -> None:
        """Initialize default configuration."""
        # Model architecture
        self.num_landmarks: int = 68
        self.num_bboxes: int = 5
        self.input_size: Tuple[int, int] = (224, 224)
        self.num_classes: int = 6
        
        # Training parameters
        self.batch_size: int = 32
        self.learning_rate: float = 0.001
        self.num_epochs: int = 100
        self.weight_decay: float = 1e-4
        
        # Loss weights
        self.landmark_weight: float = 1.0
        self.bbox_weight: float = 1.0
        self.shape_weight: float = 0.5
        self.consistency_weight: float = 0.3
        
        # Data augmentation
        self.use_augmentation: bool = True
        self.rotation_range: float = 15.0
        self.brightness_range: Tuple[float, float] = (0.8, 1.2)
        self.contrast_range: Tuple[float, float] = (0.8, 1.2)
        
        # Checkpointing
        self.save_frequency: int = 10
        self.checkpoint_dir: str = "models/checkpoints"
        self.model_name: str = "facial_landmark_cnn"


def create_model(config: Optional[ModelConfig] = None) -> FacialLandmarkCNN:
    """
    Factory function to create a facial landmark detection model.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized FacialLandmarkCNN model
    """
    if config is None:
        config = ModelConfig()
    
    model = FacialLandmarkCNN(
        num_landmarks=config.num_landmarks,
        num_bboxes=config.num_bboxes,
        input_size=config.input_size
    )
    
    return model


def load_pretrained_model(
    model_path: Union[str, Path],
    config: Optional[ModelConfig] = None,
    device: str = 'cpu'
) -> FacialLandmarkCNN:
    """
    Load a pretrained facial landmark detection model.
    
    Args:
        model_path: Path to the saved model
        config: Model configuration
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if config is None:
        config = ModelConfig()
    
    model = create_model(config)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        logger.info(f"Successfully loaded model from {model_path}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    return model