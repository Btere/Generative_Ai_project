"""
Core modules for the Virtual Try-On System.

This package contains the main analysis components including face detection,
shape analysis, and hairstyle recommendation engines.
"""

from .face_detector import FaceDetector
from .shape_analyzer import FacialShapeAnalyzer, ShapeAnalysisResult, FaceShape
from .hairstyle_engine import (
    HairstyleRecommendationEngine, 
    HairstyleRecommendation, 
    RecommendationResult,
    HairLength,
    HairTexture,
    StyleCategory
)

__all__ = [
    'FaceDetector',
    'FacialShapeAnalyzer',
    'ShapeAnalysisResult', 
    'FaceShape',
    'HairstyleRecommendationEngine',
    'HairstyleRecommendation',
    'RecommendationResult',
    'HairLength',
    'HairTexture',
    'StyleCategory'
]