"""
Facial Shape Analysis Module for Virtual Try-On System

This module classifies facial shapes based on geometric measurements and ratios
extracted from facial landmarks. It uses expert knowledge and mathematical
analysis to determine the most suitable facial shape category.

Author: Stran Connect Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceShape(Enum):
    """Enumeration of facial shape types."""
    OVAL = "oval"
    ROUND = "round"
    SQUARE = "square"
    HEART = "heart"
    DIAMOND = "diamond"
    LONG = "long"
    UNKNOWN = "unknown"


@dataclass
class ShapeAnalysisResult:
    """
    Data class to store facial shape analysis results.
    
    Attributes:
        primary_shape: Most likely facial shape
        confidence: Confidence score (0-1)
        secondary_shape: Second most likely shape
        measurements: Raw facial measurements
        ratios: Calculated facial ratios
        characteristics: Descriptive characteristics
    """
    primary_shape: FaceShape
    confidence: float
    secondary_shape: Optional[FaceShape] = None
    measurements: Optional[Dict[str, float]] = None
    ratios: Optional[Dict[str, float]] = None
    characteristics: Optional[List[str]] = None


class FacialShapeAnalyzer:
    """
    Advanced facial shape classification using geometric analysis.
    
    This class analyzes facial measurements and ratios to classify faces
    into one of six primary shape categories with confidence scores.
    """
    
    def __init__(self) -> None:
        """Initialize the FacialShapeAnalyzer with classification thresholds."""
        
        # Define classification thresholds based on research and expert knowledge
        self.thresholds = {
            'face_ratio': {
                'oval': (1.2, 1.4),
                'round': (0.9, 1.2),
                'square': (0.9, 1.2),
                'heart': (1.3, 1.6),
                'diamond': (1.3, 1.6),
                'long': (1.4, 2.0)
            },
            'forehead_ratio': {
                'oval': (0.25, 0.35),
                'round': (0.25, 0.35),
                'square': (0.25, 0.35),
                'heart': (0.35, 0.45),
                'diamond': (0.20, 0.30),
                'long': (0.25, 0.35)
            },
            'jaw_ratio': {
                'oval': (0.20, 0.30),
                'round': (0.25, 0.35),
                'square': (0.30, 0.40),
                'heart': (0.15, 0.25),
                'diamond': (0.15, 0.25),
                'long': (0.20, 0.30)
            },
            'cheekbone_ratio': {
                'oval': (0.30, 0.40),
                'round': (0.30, 0.40),
                'square': (0.30, 0.40),
                'heart': (0.25, 0.35),
                'diamond': (0.35, 0.45),
                'long': (0.30, 0.40)
            }
        }
        
        # Weight importance of different features
        self.feature_weights = {
            'face_ratio': 0.30,
            'forehead_ratio': 0.25,
            'jaw_ratio': 0.25,
            'cheekbone_ratio': 0.20
        }
        
        logger.info("FacialShapeAnalyzer initialized successfully")
    
    def analyze_face_shape(self, measurements: Dict[str, float]) -> ShapeAnalysisResult:
        """
        Analyze facial measurements to determine face shape.
        
        Args:
            measurements: Dictionary containing facial measurements
            
        Returns:
            ShapeAnalysisResult with classification details
        """
        try:
            if not self._validate_measurements(measurements):
                logger.warning("Invalid measurements provided")
                return ShapeAnalysisResult(
                    primary_shape=FaceShape.UNKNOWN,
                    confidence=0.0,
                    measurements=measurements
                )
            
            # Calculate ratios if not present
            ratios = self._calculate_ratios(measurements)
            
            # Score each face shape
            shape_scores = self._calculate_shape_scores(ratios)
            
            # Get primary and secondary shapes
            sorted_shapes = sorted(shape_scores.items(), key=lambda x: x[1], reverse=True)
            primary_shape = FaceShape(sorted_shapes[0][0])
            primary_confidence = sorted_shapes[0][1]
            secondary_shape = FaceShape(sorted_shapes[1][0]) if len(sorted_shapes) > 1 else None
            
            # Generate characteristics
            characteristics = self._generate_characteristics(primary_shape, ratios)
            
            result = ShapeAnalysisResult(
                primary_shape=primary_shape,
                confidence=primary_confidence,
                secondary_shape=secondary_shape,
                measurements=measurements,
                ratios=ratios,
                characteristics=characteristics
            )
            
            logger.info(f"Face shape analyzed: {primary_shape.value} (confidence: {primary_confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in face shape analysis: {str(e)}")
            return ShapeAnalysisResult(
                primary_shape=FaceShape.UNKNOWN,
                confidence=0.0,
                measurements=measurements
            )
    
    def _validate_measurements(self, measurements: Dict[str, float]) -> bool:
        """
        Validate that required measurements are present and valid.
        
        Args:
            measurements: Facial measurements dictionary
            
        Returns:
            True if measurements are valid, False otherwise
        """
        required_keys = ['face_width', 'face_height', 'forehead_width', 'jaw_width', 'cheekbone_width']
        
        for key in required_keys:
            if key not in measurements or measurements[key] <= 0:
                logger.warning(f"Invalid or missing measurement: {key}")
                return False
        
        return True
    
    def _calculate_ratios(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate facial ratios from measurements.
        
        Args:
            measurements: Raw facial measurements
            
        Returns:
            Dictionary of calculated ratios
        """
        ratios = {}
        
        face_width = measurements['face_width']
        
        # Calculate primary ratios
        ratios['face_ratio'] = measurements['face_height'] / face_width
        ratios['forehead_ratio'] = measurements['forehead_width'] / face_width
        ratios['jaw_ratio'] = measurements['jaw_width'] / face_width
        ratios['cheekbone_ratio'] = measurements['cheekbone_width'] / face_width
        
        # Calculate additional ratios for fine-tuning
        ratios['forehead_to_jaw'] = measurements['forehead_width'] / measurements['jaw_width']
        ratios['cheekbone_to_jaw'] = measurements['cheekbone_width'] / measurements['jaw_width']
        ratios['cheekbone_to_forehead'] = measurements['cheekbone_width'] / measurements['forehead_width']
        
        return ratios
    
    def _calculate_shape_scores(self, ratios: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate confidence scores for each face shape.
        
        Args:
            ratios: Facial ratios dictionary
            
        Returns:
            Dictionary of shape scores
        """
        shape_scores = {}
        
        for shape in FaceShape:
            if shape == FaceShape.UNKNOWN:
                continue
                
            score = 0.0
            total_weight = 0.0
            
            for feature, weight in self.feature_weights.items():
                if feature in ratios and shape.value in self.thresholds[feature]:
                    ratio_value = ratios[feature]
                    min_threshold, max_threshold = self.thresholds[feature][shape.value]
                    
                    # Calculate how well the ratio fits within the threshold range
                    if min_threshold <= ratio_value <= max_threshold:
                        # Perfect fit within range
                        feature_score = 1.0
                    else:
                        # Calculate distance from range
                        if ratio_value < min_threshold:
                            distance = min_threshold - ratio_value
                            range_size = max_threshold - min_threshold
                        else:
                            distance = ratio_value - max_threshold
                            range_size = max_threshold - min_threshold
                        
                        # Penalize based on distance (with some tolerance)
                        feature_score = max(0.0, 1.0 - (distance / range_size))
                    
                    score += feature_score * weight
                    total_weight += weight
            
            # Normalize score
            if total_weight > 0:
                shape_scores[shape.value] = score / total_weight
            else:
                shape_scores[shape.value] = 0.0
        
        return shape_scores
    
    def _generate_characteristics(self, face_shape: FaceShape, ratios: Dict[str, float]) -> List[str]:
        """
        Generate descriptive characteristics for the face shape.
        
        Args:
            face_shape: Identified face shape
            ratios: Facial ratios
            
        Returns:
            List of characteristic descriptions
        """
        characteristics = []
        
        # Base characteristics for each shape
        base_characteristics = {
            FaceShape.OVAL: [
                "Balanced facial proportions",
                "Gently rounded jawline",
                "Forehead slightly wider than chin",
                "Versatile for most hairstyles"
            ],
            FaceShape.ROUND: [
                "Fuller cheeks and soft curves",
                "Width and height approximately equal",
                "Gentle jawline without sharp angles",
                "Benefits from volume-adding styles"
            ],
            FaceShape.SQUARE: [
                "Strong, angular jawline",
                "Broad forehead and jaw of similar width",
                "Defined facial structure",
                "Suits softening hairstyles"
            ],
            FaceShape.HEART: [
                "Wider forehead tapering to narrow chin",
                "Prominent cheekbones",
                "Inverted triangle shape",
                "Benefits from chin-balancing styles"
            ],
            FaceShape.DIAMOND: [
                "Narrow forehead and chin",
                "Widest at cheekbones",
                "Angular features",
                "Suits forehead-widening styles"
            ],
            FaceShape.LONG: [
                "Height significantly greater than width",
                "Elongated facial structure",
                "May have high forehead",
                "Benefits from width-adding styles"
            ]
        }
        
        if face_shape in base_characteristics:
            characteristics.extend(base_characteristics[face_shape])
        
        # Add specific observations based on ratios
        if ratios.get('forehead_ratio', 0) > 0.35:
            characteristics.append("Prominent forehead")
        
        if ratios.get('jaw_ratio', 0) > 0.32:
            characteristics.append("Strong jawline")
        
        if ratios.get('cheekbone_ratio', 0) > 0.38:
            characteristics.append("Prominent cheekbones")
        
        if ratios.get('face_ratio', 0) > 1.5:
            characteristics.append("Elongated face structure")
        
        return characteristics
    
    def get_shape_description(self, face_shape: FaceShape) -> Dict[str, str]:
        """
        Get detailed description and styling advice for a face shape.
        
        Args:
            face_shape: The face shape to describe
            
        Returns:
            Dictionary with description and styling advice
        """
        descriptions = {
            FaceShape.OVAL: {
                "description": "Oval faces are considered the ideal face shape with balanced proportions. The face is about 1.5 times longer than it is wide, with a gently rounded jawline and forehead that's slightly wider than the chin.",
                "styling_advice": "Lucky you! Oval faces can wear almost any hairstyle. Avoid covering too much of your face, as your balanced proportions are your best feature.",
                "recommended_styles": ["Long layers", "Side-swept bangs", "Pixie cuts", "Bob cuts", "Updos"]
            },
            FaceShape.ROUND: {
                "description": "Round faces have soft, curved lines with the width and height approximately equal. The cheeks are the widest part of the face, with a rounded chin and hairline.",
                "styling_advice": "Add height and volume to elongate your face. Avoid styles that add width at the sides or emphasize the roundness.",
                "recommended_styles": ["Long straight hair", "High ponytails", "Side parts", "Layered cuts", "Asymmetrical styles"]
            },
            FaceShape.SQUARE: {
                "description": "Square faces feature a strong, angular jawline with the forehead, cheekbones, and jaw all similar in width. The face appears geometric with defined edges.",
                "styling_advice": "Soften your angular features with curved lines and layers. Avoid blunt cuts that emphasize the square shape.",
                "recommended_styles": ["Soft waves", "Long layers", "Side-swept bangs", "Rounded cuts", "Curly styles"]
            },
            FaceShape.HEART: {
                "description": "Heart-shaped faces are wider at the forehead and temples, narrowing down to a pointed chin. The cheekbones are prominent, creating an inverted triangle shape.",
                "styling_advice": "Balance your narrow chin by adding volume to the lower half of your face. Minimize width at the forehead.",
                "recommended_styles": ["Chin-length bobs", "Flipped-out ends", "Side parts", "Volume at jawline", "Soft, wispy bangs"]
            },
            FaceShape.DIAMOND: {
                "description": "Diamond faces are narrow at the forehead and chin, with the widest point at the cheekbones. This creates an angular, diamond-like shape with defined cheekbones.",
                "styling_advice": "Add width to your forehead and chin while minimizing the width at your cheekbones. Focus on balancing the proportions.",
                "recommended_styles": ["Full bangs", "Chin-length styles", "Side parts", "Volume at temples", "Textured cuts"]
            },
            FaceShape.LONG: {
                "description": "Long faces are significantly longer than they are wide, often with a high forehead, long chin, or both. The face appears elongated and narrow.",
                "styling_advice": "Add width to your face with horizontal lines and volume at the sides. Avoid styles that add height.",
                "recommended_styles": ["Blunt bangs", "Wide waves", "Bob cuts", "Side parts", "Layers at cheekbones"]
            }
        }
        
        return descriptions.get(face_shape, {
            "description": "Face shape could not be determined.",
            "styling_advice": "Consult with a professional stylist for personalized recommendations.",
            "recommended_styles": []
        })
    
    def compare_shapes(self, shape1: FaceShape, shape2: FaceShape) -> Dict[str, any]:
        """
        Compare two face shapes and their styling differences.
        
        Args:
            shape1: First face shape
            shape2: Second face shape
            
        Returns:
            Dictionary with comparison details
        """
        desc1 = self.get_shape_description(shape1)
        desc2 = self.get_shape_description(shape2)
        
        return {
            'shape1': {
                'name': shape1.value,
                'description': desc1.get('description', ''),
                'styles': desc1.get('recommended_styles', [])
            },
            'shape2': {
                'name': shape2.value,
                'description': desc2.get('description', ''),
                'styles': desc2.get('recommended_styles', [])
            },
            'common_styles': list(set(desc1.get('recommended_styles', [])) & 
                                set(desc2.get('recommended_styles', []))),
            'different_approaches': {
                'shape1_unique': list(set(desc1.get('recommended_styles', [])) - 
                                    set(desc2.get('recommended_styles', []))),
                'shape2_unique': list(set(desc2.get('recommended_styles', [])) - 
                                    set(desc1.get('recommended_styles', [])))
            }
        }