"""
CNN-based Face Detection Module for Virtual Try-On System

This module replaces MediaPipe with a custom CNN model for face detection,
landmark extraction, and bounding box generation around specific facial regions
(forehead, chin, left jaw, right jaw, jawline).

Author: Stran Connect Team
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from ..models.facial_detection_model import FacialLandmarkCNN, ModelConfig, load_pretrained_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNFaceDetector:
    """
    CNN-based face detector for facial landmark detection and bounding box generation.
    
    This class provides methods to detect faces, extract facial landmarks,
    generate bounding boxes for specific facial regions, and calculate key 
    measurements needed for facial shape analysis.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[ModelConfig] = None,
        device: str = 'auto'
    ) -> None:
        """
        Initialize the CNN-based face detector.
        
        Args:
            model_path: Path to the pretrained model (optional)
            config: Model configuration
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.config = config if config else ModelConfig()
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load or create model
        if model_path and Path(model_path).exists():
            try:
                self.model = load_pretrained_model(model_path, self.config, self.device)
                logger.info(f"Loaded pretrained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained model: {str(e)}")
                logger.info("Creating new model with random weights")
                self.model = self._create_new_model()
        else:
            logger.info("Creating new model with random weights")
            self.model = self._create_new_model()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define facial region mappings
        self.region_mappings = {
            'forehead': 0,
            'chin': 1,
            'left_jaw': 2,
            'right_jaw': 3,
            'jawline': 4
        }
        
        # Landmark indices for key facial features (68-point model)
        self.landmark_indices = {
            'jaw_contour': list(range(0, 17)),  # Jawline points
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_mouth': list(range(48, 60)),
            'inner_mouth': list(range(60, 68))
        }
        
        logger.info("CNN Face Detector initialized successfully")
    
    def _create_new_model(self) -> FacialLandmarkCNN:
        """Create a new model instance."""
        from ..models.facial_detection_model import create_model
        model = create_model(self.config)
        model.to(self.device)
        return model
    
    def detect_faces(self, image: np.ndarray) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detect faces and extract landmarks and bounding boxes from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (success_flag, detection_results)
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model.predict_facial_regions(processed_image)
            
            # Check if face is detected (simple heuristic based on bbox confidence)
            bboxes = predictions['forehead'].cpu().numpy()
            if np.all(bboxes == 0) or np.any(np.isnan(bboxes)):
                logger.warning("No valid face detected in the image")
                return False, None
            
            # Convert predictions to numpy arrays
            results = self._process_predictions(predictions, image.shape)
            
            logger.info("Face detection successful")
            return True, results
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return False, None
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized_image = cv2.resize(rgb_image, self.config.input_size)
        
        # Normalize to [0, 1]
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor_image = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        tensor_image = tensor_image.to(self.device)
        
        return tensor_image
    
    def _process_predictions(
        self, 
        predictions: Dict[str, torch.Tensor], 
        original_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """
        Process model predictions and scale to original image coordinates.
        
        Args:
            predictions: Model predictions dictionary
            original_shape: Original image shape (height, width, channels)
            
        Returns:
            Processed predictions with scaled coordinates
        """
        original_height, original_width = original_shape[:2]
        
        # Scale factors from model input size to original image size
        scale_x = original_width / self.config.input_size[0]
        scale_y = original_height / self.config.input_size[1]
        
        results = {}
        
        # Process landmarks
        landmarks = predictions['landmarks'].cpu().numpy().squeeze()  # Remove batch dimension
        landmarks[:, 0] *= scale_x  # Scale x coordinates
        landmarks[:, 1] *= scale_y  # Scale y coordinates
        results['landmarks'] = landmarks
        
        # Process bounding boxes for each facial region
        bbox_results = {}
        for region_name, region_idx in self.region_mappings.items():
            if region_name in predictions:
                bbox = predictions[region_name].cpu().numpy().squeeze()
                
                # Scale bounding box coordinates
                bbox[0] *= scale_x  # x
                bbox[1] *= scale_y  # y
                bbox[2] *= scale_x  # width
                bbox[3] *= scale_y  # height
                
                bbox_results[region_name] = bbox
        
        results['bboxes'] = bbox_results
        
        # Process face shape prediction
        if 'face_shape' in predictions:
            face_shape_probs = torch.softmax(predictions['face_shape'], dim=1)
            face_shape_idx = torch.argmax(face_shape_probs, dim=1).item()
            confidence = face_shape_probs[0, face_shape_idx].item()
            
            shape_names = ['oval', 'round', 'square', 'heart', 'diamond', 'long']
            results['predicted_face_shape'] = shape_names[face_shape_idx]
            results['shape_confidence'] = confidence
        
        return results
    
    def extract_facial_measurements(
        self, 
        detection_results: Dict[str, Any], 
        image_shape: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Extract key facial measurements from detection results.
        
        Args:
            detection_results: Results from face detection
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary containing facial measurements
        """
        try:
            landmarks = detection_results['landmarks']
            bboxes = detection_results['bboxes']
            
            measurements = {}
            
            # Calculate face dimensions from bounding boxes
            forehead_bbox = bboxes['forehead']
            chin_bbox = bboxes['chin']
            left_jaw_bbox = bboxes['left_jaw']
            right_jaw_bbox = bboxes['right_jaw']
            jawline_bbox = bboxes['jawline']
            
            # Face width (distance between left and right jaw centers)
            left_jaw_center_x = left_jaw_bbox[0] + left_jaw_bbox[2] / 2
            right_jaw_center_x = right_jaw_bbox[0] + right_jaw_bbox[2] / 2
            face_width = abs(right_jaw_center_x - left_jaw_center_x)
            measurements['face_width'] = face_width
            
            # Face height (from forehead top to chin bottom)
            forehead_top = forehead_bbox[1]
            chin_bottom = chin_bbox[1] + chin_bbox[3]
            face_height = abs(chin_bottom - forehead_top)
            measurements['face_height'] = face_height
            
            # Forehead width
            forehead_width = forehead_bbox[2]
            measurements['forehead_width'] = forehead_width
            
            # Jaw width (from jawline bounding box)
            jaw_width = jawline_bbox[2]
            measurements['jaw_width'] = jaw_width
            
            # Cheekbone width (estimate from face width)
            # This is an approximation - in practice, you'd train the model 
            # to predict cheekbone-specific landmarks
            cheekbone_width = face_width * 0.9  # Heuristic
            measurements['cheekbone_width'] = cheekbone_width
            
            # Calculate face ratios
            if face_width > 0:
                measurements['face_ratio'] = face_height / face_width
                measurements['forehead_ratio'] = forehead_width / face_width
                measurements['jaw_ratio'] = jaw_width / face_width
                measurements['cheekbone_ratio'] = cheekbone_width / face_width
            else:
                measurements['face_ratio'] = 0
                measurements['forehead_ratio'] = 0
                measurements['jaw_ratio'] = 0
                measurements['cheekbone_ratio'] = 0
            
            # Calculate additional measurements using landmarks
            measurements.update(self._calculate_landmark_measurements(landmarks))
            
            logger.info("Facial measurements extracted successfully")
            return measurements
            
        except Exception as e:
            logger.error(f"Error extracting measurements: {str(e)}")
            return {}
    
    def _calculate_landmark_measurements(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate additional measurements from facial landmarks.
        
        Args:
            landmarks: Array of facial landmarks (num_landmarks, 2)
            
        Returns:
            Dictionary with additional measurements
        """
        measurements = {}
        
        try:
            # Upper face height (eyebrow to nose tip)
            if len(landmarks) >= 68:
                eyebrow_points = landmarks[self.landmark_indices['right_eyebrow'] + 
                                         self.landmark_indices['left_eyebrow']]
                nose_tip_points = landmarks[self.landmark_indices['nose_tip']]
                
                eyebrow_y = np.mean(eyebrow_points[:, 1])
                nose_tip_y = np.mean(nose_tip_points[:, 1])
                measurements['upper_face_height'] = abs(nose_tip_y - eyebrow_y)
                
                # Lower face height (nose tip to chin)
                jaw_points = landmarks[self.landmark_indices['jaw_contour']]
                chin_y = np.max(jaw_points[:, 1])  # Bottom of jaw
                measurements['lower_face_height'] = abs(chin_y - nose_tip_y)
        
        except Exception as e:
            logger.warning(f"Could not calculate landmark measurements: {str(e)}")
        
        return measurements
    
    def draw_detections(
        self, 
        image: np.ndarray, 
        detection_results: Dict[str, Any],
        draw_landmarks: bool = True,
        draw_bboxes: bool = True
    ) -> np.ndarray:
        """
        Draw detection results on the image for visualization.
        
        Args:
            image: Input image
            detection_results: Detection results from face detection
            draw_landmarks: Whether to draw facial landmarks
            draw_bboxes: Whether to draw bounding boxes
            
        Returns:
            Image with drawn detections
        """
        try:
            annotated_image = image.copy()
            
            # Draw landmarks
            if draw_landmarks and 'landmarks' in detection_results:
                landmarks = detection_results['landmarks']
                for point in landmarks:
                    cv2.circle(annotated_image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            
            # Draw bounding boxes
            if draw_bboxes and 'bboxes' in detection_results:
                bboxes = detection_results['bboxes']
                colors = {
                    'forehead': (255, 0, 0),    # Blue
                    'chin': (0, 255, 0),        # Green
                    'left_jaw': (0, 0, 255),    # Red
                    'right_jaw': (255, 255, 0), # Cyan
                    'jawline': (255, 0, 255)    # Magenta
                }
                
                for region_name, bbox in bboxes.items():
                    if region_name in colors:
                        x, y, w, h = bbox.astype(int)
                        color = colors[region_name]
                        
                        # Draw rectangle
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                        
                        # Draw label
                        cv2.putText(annotated_image, region_name, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return annotated_image
            
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return image
    
    def process_image_file(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process an image file and extract facial information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Detect faces
            success, detection_results = self.detect_faces(image)
            
            result = {
                'success': success,
                'image_path': str(image_path),
                'image_shape': image.shape
            }
            
            if success and detection_results:
                # Extract measurements
                measurements = self.extract_facial_measurements(detection_results, image.shape)
                result['measurements'] = measurements
                result['detection_results'] = detection_results
                
                # Create annotated image
                annotated_image = self.draw_detections(image, detection_results)
                result['annotated_image'] = annotated_image
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image file: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_region_bbox(
        self, 
        detection_results: Dict[str, Any], 
        region_name: str
    ) -> Optional[np.ndarray]:
        """
        Get bounding box for a specific facial region.
        
        Args:
            detection_results: Detection results from face detection
            region_name: Name of the facial region
            
        Returns:
            Bounding box as [x, y, width, height] or None if not found
        """
        try:
            if 'bboxes' in detection_results and region_name in detection_results['bboxes']:
                return detection_results['bboxes'][region_name]
            else:
                logger.warning(f"Region '{region_name}' not found in detection results")
                return None
                
        except Exception as e:
            logger.error(f"Error getting region bbox: {str(e)}")
            return None
    
    def save_model(self, save_path: Union[str, Path]) -> bool:
        """
        Save the current model state.
        
        Args:
            save_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'device': self.device
            }, save_path)
            
            logger.info(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()