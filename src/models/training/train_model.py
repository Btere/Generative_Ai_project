#!/usr/bin/env python3
"""
Training Script for Custom Facial Landmark Detection CNN

This script provides a command-line interface for training the custom CNN model
on facial landmark detection data with comprehensive typing and error handling.

Usage:
    python train_model.py --data_dir datasets/facial_landmarks --epochs 100 --batch_size 32

Author: Stran Connect Team
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, NoReturn, Tuple
import logging
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from models.facial_detection_model import ModelConfig, create_model
    from models.training.trainer import ModelTrainer, create_data_loaders, main_training_pipeline
    from models.data_processing.data_preparation import DatasetCreator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the correct directory and dependencies are installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train Custom Facial Landmark Detection CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py --data_dir datasets/facial_landmarks --epochs 50
  python train_model.py --data_dir datasets/facial_landmarks --config custom_config.json
  python train_model.py --data_dir datasets/facial_landmarks --resume models/checkpoints/best.pth
        """
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to the dataset directory'
    )
    
    parser.add_argument(
        '--train_annotations',
        type=str,
        default='annotations/train_annotations.json',
        help='Path to training annotations file (relative to data_dir)'
    )
    
    parser.add_argument(
        '--val_annotations',
        type=str,
        default='annotations/val_annotations.json',
        help='Path to validation annotations file (relative to data_dir)'
    )
    
    # Model arguments
    parser.add_argument(
        '--config',
        type=str,
        help='Path to model configuration JSON file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for regularization'
    )
    
    # Loss weights
    parser.add_argument(
        '--landmark_weight',
        type=float,
        default=1.0,
        help='Weight for landmark loss'
    )
    
    parser.add_argument(
        '--bbox_weight',
        type=float,
        default=1.0,
        help='Weight for bounding box loss'
    )
    
    parser.add_argument(
        '--shape_weight',
        type=float,
        default=0.5,
        help='Weight for face shape classification loss'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/checkpoints',
        help='Directory to save model checkpoints'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='facial_landmark_cnn',
        help='Name for the model (used in checkpoint filenames)'
    )
    
    # Other arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--save_frequency',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    parser.add_argument(
        '--no_augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> ModelConfig:
    """
    Load model configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ModelConfig object
    """
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig()
        
        # Update config with values from file
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise


def create_config_from_args(args: argparse.Namespace) -> ModelConfig:
    """
    Create model configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ModelConfig object
    """
    config = ModelConfig()
    
    # Update config with command line arguments
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    
    # Loss weights
    config.landmark_weight = args.landmark_weight
    config.bbox_weight = args.bbox_weight
    config.shape_weight = args.shape_weight
    
    # Other settings
    config.checkpoint_dir = args.output_dir
    config.model_name = args.model_name
    config.save_frequency = args.save_frequency
    config.use_augmentation = not args.no_augmentation
    
    return config


def validate_dataset(data_dir: Path, train_annotations: str, val_annotations: str) -> bool:
    """
    Validate that the dataset structure is correct.
    
    Args:
        data_dir: Path to dataset directory
        train_annotations: Path to training annotations
        val_annotations: Path to validation annotations
        
    Returns:
        True if dataset is valid, False otherwise
    """
    try:
        # Check directory structure
        required_dirs = ['images', 'annotations']
        for dir_name in required_dirs:
            dir_path = data_dir / dir_name
            if not dir_path.exists():
                logger.error(f"Required directory missing: {dir_path}")
                return False
        
        # Check annotation files
        train_file = data_dir / train_annotations
        val_file = data_dir / val_annotations
        
        if not train_file.exists():
            logger.error(f"Training annotations file missing: {train_file}")
            return False
        
        if not val_file.exists():
            logger.error(f"Validation annotations file missing: {val_file}")
            return False
        
        # Validate annotation format
        try:
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            
            with open(val_file, 'r') as f:
                val_data = json.load(f)
            
            if not train_data:
                logger.error("Training annotations file is empty")
                return False
            
            if not val_data:
                logger.error("Validation annotations file is empty")
                return False
            
            # Check a sample annotation
            sample_key = list(train_data.keys())[0]
            sample_annotation = train_data[sample_key]
            
            required_fields = ['landmarks', 'bboxes', 'face_shape']
            for field in required_fields:
                if field not in sample_annotation:
                    logger.error(f"Required field '{field}' missing in annotations")
                    return False
            
            logger.info("Dataset validation passed")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in annotation files: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        return False


def setup_logging(debug: bool = False) -> None:
    """
    Setup logging configuration.
    
    Args:
        debug: Whether to enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logging.getLogger().setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Update existing handlers
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)


def main() -> Optional[NoReturn]:
    """
    Main training function.
    
    Returns:
        Does not return normally, exits with status code
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.debug)
        
        logger.info("=" * 60)
        logger.info("Custom Facial Landmark Detection CNN Training")
        logger.info("=" * 60)
        
        # Validate paths
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"Dataset directory does not exist: {data_dir}")
            sys.exit(1)
        
        # Validate dataset
        if not validate_dataset(data_dir, args.train_annotations, args.val_annotations):
            logger.error("Dataset validation failed")
            sys.exit(1)
        
        # Create configuration
        if args.config:
            config = load_config_from_file(args.config)
            # Override with command line arguments
            config.num_epochs = args.epochs
            config.batch_size = args.batch_size
            config.learning_rate = args.learning_rate
        else:
            config = create_config_from_args(args)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_save_path = output_dir / f"{config.model_name}_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_save_path}")
        logger.info(f"Training configuration:")
        for key, value in config.__dict__.items():
            logger.info(f"  {key}: {value}")
        
        # Start training
        logger.info("Starting training pipeline...")
        
        trained_model = main_training_pipeline(
            train_data_dir=data_dir / 'images',
            train_annotations=data_dir / args.train_annotations,
            val_data_dir=data_dir / 'images',  # Same images directory
            val_annotations=data_dir / args.val_annotations,
            config=config
        )
        
        # Save final model
        final_model_path = output_dir / f"{config.model_name}_final.pth"
        import torch
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': config.__dict__
        }, final_model_path)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()