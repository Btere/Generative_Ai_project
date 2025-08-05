"""
Data Preparation Module for Custom CNN Training

This module provides utilities for preparing training data, including data collection,
annotation, augmentation, and format conversion for the facial landmark detection model.

Author: Stran Connect Team
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetCreator:
    """
    Utility class for creating and managing training datasets.
    
    This class helps create datasets from various sources including:
    - Existing facial landmark datasets (300W, WFLW, etc.)
    - Custom annotations
    - Synthetic data generation
    """
    
    def __init__(self, output_dir: Union[str, Path]) -> None:
        """
        Initialize the dataset creator.
        
        Args:
            output_dir: Directory to save processed dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)
        
        self.annotations: Dict[str, Dict] = {}
        
        logger.info(f"Dataset creator initialized with output directory: {output_dir}")
    
    def convert_300w_dataset(
        self, 
        dataset_path: Union[str, Path],
        include_challenging: bool = True
    ) -> None:
        """
        Convert 300W dataset to our format.
        
        The 300W dataset is a standard facial landmark dataset with 68 landmarks.
        
        Args:
            dataset_path: Path to the 300W dataset
            include_challenging: Whether to include challenging subset
        """
        try:
            dataset_path = Path(dataset_path)
            logger.info(f"Converting 300W dataset from {dataset_path}")
            
            # Process different subsets
            subsets = ['01_Indoor', '02_Outdoor']
            if include_challenging:
                subsets.extend(['01_Challenging', '02_Challenging'])
            
            for subset in subsets:
                subset_path = dataset_path / subset
                if subset_path.exists():
                    self._process_300w_subset(subset_path, subset)
            
            logger.info(f"300W dataset conversion completed. Total images: {len(self.annotations)}")
            
        except Exception as e:
            logger.error(f"Error converting 300W dataset: {str(e)}")
            raise
    
    def _process_300w_subset(self, subset_path: Path, subset_name: str) -> None:
        """Process a single 300W subset."""
        try:
            # Find all .pts files (landmark annotations)
            pts_files = list(subset_path.glob('*.pts'))
            
            for pts_file in tqdm(pts_files, desc=f"Processing {subset_name}"):
                # Find corresponding image file
                image_name = pts_file.stem
                possible_extensions = ['.jpg', '.png', '.jpeg']
                image_file = None
                
                for ext in possible_extensions:
                    candidate = subset_path / f"{image_name}{ext}"
                    if candidate.exists():
                        image_file = candidate
                        break
                
                if image_file is None:
                    logger.warning(f"Image file not found for {pts_file}")
                    continue
                
                # Read landmarks
                landmarks = self._read_pts_file(pts_file)
                if landmarks is None:
                    continue
                
                # Copy image to our dataset
                new_image_name = f"{subset_name}_{image_name}{image_file.suffix}"
                new_image_path = self.output_dir / 'images' / new_image_name
                
                # Copy image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                cv2.imwrite(str(new_image_path), image)
                
                # Generate bounding boxes from landmarks
                bboxes = self._generate_bboxes_from_landmarks(landmarks, image.shape)
                
                # Estimate face shape (you would normally have ground truth labels)
                face_shape = self._estimate_face_shape_from_landmarks(landmarks)
                
                # Create annotation
                annotation = {
                    'landmarks': landmarks.flatten().tolist(),
                    'bboxes': bboxes,
                    'face_shape': face_shape,
                    'image_width': image.shape[1],
                    'image_height': image.shape[0],
                    'source_dataset': '300W',
                    'subset': subset_name
                }
                
                self.annotations[new_image_name] = annotation
                
        except Exception as e:
            logger.error(f"Error processing 300W subset {subset_name}: {str(e)}")
    
    def _read_pts_file(self, pts_file: Path) -> Optional[np.ndarray]:
        """Read landmarks from .pts file format."""
        try:
            with open(pts_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines
            start_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('n_points'):
                    num_points = int(line.split(':')[1].strip())
                elif line.strip() == '{':
                    start_idx = i + 1
                    break
            
            if start_idx is None:
                return None
            
            # Read landmarks
            landmarks = []
            for i in range(start_idx, start_idx + num_points):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    landmarks.append([x, y])
            
            return np.array(landmarks, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error reading pts file {pts_file}: {str(e)}")
            return None
    
    def _generate_bboxes_from_landmarks(
        self, 
        landmarks: np.ndarray, 
        image_shape: Tuple[int, int, int]
    ) -> List[List[float]]:
        """
        Generate bounding boxes for facial regions from landmarks.
        
        Args:
            landmarks: Facial landmarks array (68, 2)
            image_shape: Image shape (height, width, channels)
            
        Returns:
            List of bounding boxes [x, y, width, height] for each region
        """
        try:
            height, width = image_shape[:2]
            
            # Define landmark indices for different regions
            # Based on 68-point facial landmark model
            landmark_regions = {
                'jaw_contour': list(range(0, 17)),
                'right_eyebrow': list(range(17, 22)),
                'left_eyebrow': list(range(22, 27)),
                'nose_bridge': list(range(27, 31)),
                'nose_tip': list(range(31, 36)),
                'right_eye': list(range(36, 42)),
                'left_eye': list(range(42, 48)),
                'outer_mouth': list(range(48, 60))
            }
            
            bboxes = []
            
            # Forehead bbox (estimate from eyebrows and image top)
            eyebrow_points = landmarks[landmark_regions['right_eyebrow'] + 
                                     landmark_regions['left_eyebrow']]
            forehead_x_min = np.min(eyebrow_points[:, 0])
            forehead_x_max = np.max(eyebrow_points[:, 0])
            forehead_y_min = max(0, np.min(eyebrow_points[:, 1]) - 50)  # Estimate above eyebrows
            forehead_y_max = np.min(eyebrow_points[:, 1])
            
            forehead_bbox = [
                forehead_x_min,
                forehead_y_min,
                forehead_x_max - forehead_x_min,
                forehead_y_max - forehead_y_min
            ]
            bboxes.append(forehead_bbox)
            
            # Chin bbox
            jaw_points = landmarks[landmark_regions['jaw_contour']]
            chin_center_idx = len(landmark_regions['jaw_contour']) // 2
            chin_point = landmarks[chin_center_idx]
            
            chin_width = 80  # Estimate
            chin_height = 60  # Estimate
            chin_bbox = [
                chin_point[0] - chin_width / 2,
                chin_point[1] - chin_height / 2,
                chin_width,
                chin_height
            ]
            bboxes.append(chin_bbox)
            
            # Left jaw bbox
            left_jaw_points = landmarks[landmark_regions['jaw_contour'][:5]]  # First 5 points
            left_jaw_x_min = np.min(left_jaw_points[:, 0])
            left_jaw_x_max = np.max(left_jaw_points[:, 0])
            left_jaw_y_min = np.min(left_jaw_points[:, 1])
            left_jaw_y_max = np.max(left_jaw_points[:, 1])
            
            left_jaw_bbox = [
                left_jaw_x_min - 20,
                left_jaw_y_min,
                (left_jaw_x_max - left_jaw_x_min) + 40,
                left_jaw_y_max - left_jaw_y_min
            ]
            bboxes.append(left_jaw_bbox)
            
            # Right jaw bbox
            right_jaw_points = landmarks[landmark_regions['jaw_contour'][-5:]]  # Last 5 points
            right_jaw_x_min = np.min(right_jaw_points[:, 0])
            right_jaw_x_max = np.max(right_jaw_points[:, 0])
            right_jaw_y_min = np.min(right_jaw_points[:, 1])
            right_jaw_y_max = np.max(right_jaw_points[:, 1])
            
            right_jaw_bbox = [
                right_jaw_x_min - 20,
                right_jaw_y_min,
                (right_jaw_x_max - right_jaw_x_min) + 40,
                right_jaw_y_max - right_jaw_y_min
            ]
            bboxes.append(right_jaw_bbox)
            
            # Jawline bbox (overall jaw contour)
            jaw_x_min = np.min(jaw_points[:, 0])
            jaw_x_max = np.max(jaw_points[:, 0])
            jaw_y_min = np.min(jaw_points[:, 1])
            jaw_y_max = np.max(jaw_points[:, 1])
            
            jawline_bbox = [
                jaw_x_min - 10,
                jaw_y_min,
                (jaw_x_max - jaw_x_min) + 20,
                jaw_y_max - jaw_y_min
            ]
            bboxes.append(jawline_bbox)
            
            # Ensure all bboxes are within image bounds
            clipped_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                clipped_bboxes.append([x, y, w, h])
            
            return clipped_bboxes
            
        except Exception as e:
            logger.error(f"Error generating bboxes from landmarks: {str(e)}")
            # Return dummy bboxes
            return [[0, 0, 50, 50]] * 5
    
    def _estimate_face_shape_from_landmarks(self, landmarks: np.ndarray) -> str:
        """
        Estimate face shape from landmarks (simplified heuristic).
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            Estimated face shape string
        """
        try:
            # Simple heuristic based on face width/height ratio
            jaw_points = landmarks[:17]  # Jaw contour
            
            face_width = np.max(jaw_points[:, 0]) - np.min(jaw_points[:, 0])
            face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
            
            ratio = face_height / face_width if face_width > 0 else 1.5
            
            if ratio > 1.5:
                return 'long'
            elif ratio > 1.3:
                return 'oval'
            elif ratio > 1.1:
                return 'square'
            else:
                return 'round'
                
        except Exception:
            return 'oval'  # Default
    
    def create_synthetic_data(
        self, 
        num_samples: int = 1000,
        base_images_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create synthetic training data using data augmentation.
        
        Args:
            num_samples: Number of synthetic samples to create
            base_images_dir: Directory containing base images (optional)
        """
        try:
            logger.info(f"Creating {num_samples} synthetic samples")
            
            # Define augmentation pipeline
            augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.5),
            ], 
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            
            # Use existing annotations as base
            base_annotations = list(self.annotations.items())
            if not base_annotations:
                logger.warning("No base annotations found for synthetic data generation")
                return
            
            synthetic_count = 0
            while synthetic_count < num_samples:
                # Randomly select a base annotation
                base_name, base_annotation = random.choice(base_annotations)
                
                try:
                    # Load base image
                    base_image_path = self.output_dir / 'images' / base_name
                    if not base_image_path.exists():
                        continue
                    
                    image = cv2.imread(str(base_image_path))
                    if image is None:
                        continue
                    
                    # Convert landmarks back to array format
                    landmarks_flat = base_annotation['landmarks']
                    landmarks = np.array(landmarks_flat).reshape(-1, 2)
                    
                    # Apply augmentation
                    augmented = augmentation_pipeline(
                        image=image,
                        keypoints=landmarks.tolist()
                    )
                    
                    augmented_image = augmented['image']
                    augmented_landmarks = np.array(augmented['keypoints'])
                    
                    # Generate new filename
                    synthetic_name = f"synthetic_{synthetic_count:06d}.jpg"
                    synthetic_path = self.output_dir / 'images' / synthetic_name
                    
                    # Save augmented image
                    cv2.imwrite(str(synthetic_path), augmented_image)
                    
                    # Generate bboxes for augmented landmarks
                    bboxes = self._generate_bboxes_from_landmarks(
                        augmented_landmarks, augmented_image.shape
                    )
                    
                    # Create synthetic annotation
                    synthetic_annotation = {
                        'landmarks': augmented_landmarks.flatten().tolist(),
                        'bboxes': bboxes,
                        'face_shape': base_annotation['face_shape'],
                        'image_width': augmented_image.shape[1],
                        'image_height': augmented_image.shape[0],
                        'source_dataset': 'synthetic',
                        'base_image': base_name
                    }
                    
                    self.annotations[synthetic_name] = synthetic_annotation
                    synthetic_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error creating synthetic sample: {str(e)}")
                    continue
            
            logger.info(f"Created {synthetic_count} synthetic samples")
            
        except Exception as e:
            logger.error(f"Error creating synthetic data: {str(e)}")
    
    def split_dataset(
        self, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ) -> None:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        try:
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("Train, val, and test ratios must sum to 1.0")
            
            # Shuffle annotations
            items = list(self.annotations.items())
            random.shuffle(items)
            
            # Calculate split indices
            total_samples = len(items)
            train_end = int(total_samples * train_ratio)
            val_end = train_end + int(total_samples * val_ratio)
            
            # Split data
            train_items = items[:train_end]
            val_items = items[train_end:val_end]
            test_items = items[val_end:]
            
            # Save split annotations
            splits = {
                'train': dict(train_items),
                'val': dict(val_items),
                'test': dict(test_items)
            }
            
            for split_name, split_annotations in splits.items():
                output_file = self.output_dir / 'annotations' / f'{split_name}_annotations.json'
                with open(output_file, 'w') as f:
                    json.dump(split_annotations, f, indent=2)
                
                logger.info(f"Saved {len(split_annotations)} {split_name} annotations to {output_file}")
            
            # Save complete annotations
            complete_file = self.output_dir / 'annotations' / 'complete_annotations.json'
            with open(complete_file, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            
            logger.info(f"Dataset split completed: Train={len(train_items)}, "
                       f"Val={len(val_items)}, Test={len(test_items)}")
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            raise
    
    def add_custom_annotation(
        self,
        image_path: Union[str, Path],
        landmarks: np.ndarray,
        face_shape: str,
        facial_regions: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        Add a custom annotation to the dataset.
        
        Args:
            image_path: Path to the image file
            landmarks: Facial landmarks array (68, 2)
            face_shape: Face shape label
            facial_regions: Optional manual facial region bboxes
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Copy image to dataset
            new_image_name = f"custom_{image_path.stem}_{len(self.annotations)}{image_path.suffix}"
            new_image_path = self.output_dir / 'images' / new_image_name
            cv2.imwrite(str(new_image_path), image)
            
            # Generate or use provided bounding boxes
            if facial_regions:
                bboxes = [
                    facial_regions.get('forehead', [0, 0, 50, 50]),
                    facial_regions.get('chin', [0, 0, 50, 50]),
                    facial_regions.get('left_jaw', [0, 0, 50, 50]),
                    facial_regions.get('right_jaw', [0, 0, 50, 50]),
                    facial_regions.get('jawline', [0, 0, 50, 50])
                ]
            else:
                bboxes = self._generate_bboxes_from_landmarks(landmarks, image.shape)
            
            # Create annotation
            annotation = {
                'landmarks': landmarks.flatten().tolist(),
                'bboxes': bboxes,
                'face_shape': face_shape,
                'image_width': image.shape[1],
                'image_height': image.shape[0],
                'source_dataset': 'custom',
                'original_path': str(image_path)
            }
            
            self.annotations[new_image_name] = annotation
            logger.info(f"Added custom annotation for {new_image_name}")
            
        except Exception as e:
            logger.error(f"Error adding custom annotation: {str(e)}")
            raise
    
    def visualize_annotations(
        self, 
        num_samples: int = 10,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Visualize random samples from the dataset.
        
        Args:
            num_samples: Number of samples to visualize
            save_path: Optional path to save visualization
        """
        try:
            if not self.annotations:
                logger.warning("No annotations to visualize")
                return
            
            # Randomly select samples
            sample_names = random.sample(list(self.annotations.keys()), 
                                       min(num_samples, len(self.annotations)))
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for i, image_name in enumerate(sample_names):
                if i >= len(axes):
                    break
                
                # Load image
                image_path = self.output_dir / 'images' / image_name
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get annotation
                annotation = self.annotations[image_name]
                landmarks = np.array(annotation['landmarks']).reshape(-1, 2)
                bboxes = annotation['bboxes']
                
                # Draw landmarks
                for point in landmarks:
                    cv2.circle(image_rgb, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
                
                # Draw bounding boxes
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                region_names = ['forehead', 'chin', 'left_jaw', 'right_jaw', 'jawline']
                
                for j, bbox in enumerate(bboxes):
                    if j < len(colors):
                        x, y, w, h = map(int, bbox)
                        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), colors[j], 2)
                
                # Display
                axes[i].imshow(image_rgb)
                axes[i].set_title(f"{annotation['face_shape']}\n{image_name}")
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(sample_names), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing annotations: {str(e)}")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        try:
            if not self.annotations:
                return {'total_samples': 0}
            
            # Count face shapes
            face_shape_counts = {}
            source_counts = {}
            
            for annotation in self.annotations.values():
                face_shape = annotation['face_shape']
                source = annotation.get('source_dataset', 'unknown')
                
                face_shape_counts[face_shape] = face_shape_counts.get(face_shape, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
            
            stats = {
                'total_samples': len(self.annotations),
                'face_shape_distribution': face_shape_counts,
                'source_distribution': source_counts,
                'output_directory': str(self.output_dir)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {str(e)}")
            return {'error': str(e)}


def download_and_prepare_300w(output_dir: Union[str, Path]) -> None:
    """
    Download and prepare the 300W dataset for training.
    
    Args:
        output_dir: Directory to save the prepared dataset
    """
    try:
        logger.info("Note: 300W dataset must be manually downloaded from:")
        logger.info("https://ibug.doc.ic.ac.uk/resources/300-W/")
        logger.info("Please download and extract the dataset, then run the conversion manually.")
        
        # Example usage:
        example_code = """
        # After downloading 300W dataset:
        from src.models.data_processing.data_preparation import DatasetCreator
        
        creator = DatasetCreator(output_dir='datasets/facial_landmarks')
        creator.convert_300w_dataset('path/to/300W/dataset')
        creator.create_synthetic_data(num_samples=5000)
        creator.split_dataset()
        creator.visualize_annotations()
        stats = creator.get_dataset_statistics()
        print(stats)
        """
        
        logger.info("Example usage:")
        logger.info(example_code)
        
    except Exception as e:
        logger.error(f"Error in download preparation: {str(e)}")


if __name__ == "__main__":
    # Example usage
    output_dir = Path("datasets/facial_landmarks")
    creator = DatasetCreator(output_dir)
    
    # Create some dummy data for demonstration
    logger.info("Creating example dataset structure...")
    
    # This would normally be called with real data
    # creator.convert_300w_dataset('path/to/300W')
    # creator.create_synthetic_data(1000)
    # creator.split_dataset()
    
    stats = creator.get_dataset_statistics()
    logger.info(f"Dataset statistics: {stats}")