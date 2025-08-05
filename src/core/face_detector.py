"""
Face Detection Module for Virtual Try-On System

This module provides comprehensive face detection and facial landmark extraction
using MediaPipe's Face Mesh solution. It identifies 468 facial landmarks and
extracts key measurements for facial shape analysis.

Author: Stran Connect Team
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Advanced face detection and landmark extraction using MediaPipe.
    
    This class provides methods to detect faces, extract facial landmarks,
    and calculate key measurements needed for facial shape analysis.
    """
    
    def __init__(
        self, 
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> None:
        """
        Initialize the FaceDetector with MediaPipe configurations.
        
        Args:
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmark positions
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Key facial landmark indices for measurements
        self.landmark_indices = {
            'forehead_top': [10, 151],
            'forehead_width': [103, 332],
            'eye_left': [33, 133],
            'eye_right': [362, 263],
            'nose_tip': [1],
            'nose_width': [218, 438],
            'mouth_corners': [61, 291],
            'mouth_center': [13, 14],
            'chin_bottom': [18],
            'jaw_left': [172],
            'jaw_right': [397],
            'cheekbone_left': [116],
            'cheekbone_right': [345],
            'face_contour': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                           397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                           172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        logger.info("FaceDetector initialized successfully")
    
    def detect_faces(self, image: np.ndarray) -> Tuple[bool, Optional[List[Any]]]:
        """
        Detect faces and extract landmarks from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (success_flag, landmarks_list)
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                logger.info(f"Detected {len(results.multi_face_landmarks)} face(s)")
                return True, results.multi_face_landmarks
            else:
                logger.warning("No faces detected in the image")
                return False, None
                
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return False, None
    
    def extract_facial_measurements(
        self, 
        landmarks: List[Any], 
        image_shape: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Extract key facial measurements from landmarks.
        
        Args:
            landmarks: Face landmarks from MediaPipe
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary containing facial measurements
        """
        try:
            height, width = image_shape[:2]
            measurements = {}
            
            # Convert landmarks to pixel coordinates
            landmark_points = []
            for landmark in landmarks[0].landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmark_points.append((x, y))
            
            # Calculate face width (distance between temples)
            left_temple = landmark_points[103]
            right_temple = landmark_points[332]
            face_width = self._calculate_distance(left_temple, right_temple)
            measurements['face_width'] = face_width
            
            # Calculate face height (forehead to chin)
            forehead_top = landmark_points[10]
            chin_bottom = landmark_points[18]
            face_height = self._calculate_distance(forehead_top, chin_bottom)
            measurements['face_height'] = face_height
            
            # Calculate forehead width
            forehead_left = landmark_points[103]
            forehead_right = landmark_points[332]
            forehead_width = self._calculate_distance(forehead_left, forehead_right)
            measurements['forehead_width'] = forehead_width
            
            # Calculate jaw width
            jaw_left = landmark_points[172]
            jaw_right = landmark_points[397]
            jaw_width = self._calculate_distance(jaw_left, jaw_right)
            measurements['jaw_width'] = jaw_width
            
            # Calculate cheekbone width
            cheekbone_left = landmark_points[116]
            cheekbone_right = landmark_points[345]
            cheekbone_width = self._calculate_distance(cheekbone_left, cheekbone_right)
            measurements['cheekbone_width'] = cheekbone_width
            
            # Calculate face ratios
            measurements['face_ratio'] = face_height / face_width if face_width > 0 else 0
            measurements['forehead_ratio'] = forehead_width / face_width if face_width > 0 else 0
            measurements['jaw_ratio'] = jaw_width / face_width if face_width > 0 else 0
            measurements['cheekbone_ratio'] = cheekbone_width / face_width if face_width > 0 else 0
            
            # Calculate additional measurements for shape analysis
            measurements['upper_face_height'] = self._calculate_distance(
                forehead_top, landmark_points[168]  # nose bridge
            )
            measurements['lower_face_height'] = self._calculate_distance(
                landmark_points[168], chin_bottom
            )
            
            logger.info("Facial measurements extracted successfully")
            return measurements
            
        except Exception as e:
            logger.error(f"Error extracting measurements: {str(e)}")
            return {}
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance between the points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def draw_landmarks(
        self, 
        image: np.ndarray, 
        landmarks: List[Any],
        draw_contour: bool = True
    ) -> np.ndarray:
        """
        Draw facial landmarks on the image for visualization.
        
        Args:
            image: Input image
            landmarks: Face landmarks
            draw_contour: Whether to draw face contour
            
        Returns:
            Image with drawn landmarks
        """
        try:
            annotated_image = image.copy()
            
            for face_landmarks in landmarks:
                if draw_contour:
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                    )
                
                # Draw key points
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style()
                )
            
            return annotated_image
            
        except Exception as e:
            logger.error(f"Error drawing landmarks: {str(e)}")
            return image
    
    def process_image_file(self, image_path: Union[str, Path]) -> Dict:
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
            success, landmarks = self.detect_faces(image)
            
            result = {
                'success': success,
                'image_path': str(image_path),
                'image_shape': image.shape
            }
            
            if success and landmarks:
                # Extract measurements
                measurements = self.extract_facial_measurements(landmarks, image.shape)
                result['measurements'] = measurements
                result['landmarks'] = landmarks
                
                # Create annotated image
                annotated_image = self.draw_landmarks(image, landmarks)
                result['annotated_image'] = annotated_image
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image file: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()