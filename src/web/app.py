"""
Flask Web Application for Virtual Try-On System

This module provides the main web interface for the virtual try-on system,
including camera integration, image upload, face analysis, and hairstyle
recommendations with comprehensive typing.

Author: Stran Connect Team
"""

import io
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, TypedDict
import json
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import cv2
import numpy as np

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.face_detector import FaceDetector
from core.shape_analyzer import FacialShapeAnalyzer, ShapeAnalysisResult, FaceShape
from core.hairstyle_engine import HairstyleRecommendationEngine, RecommendationResult
from utils.image_processor import ImageProcessor
from utils.validators import InputValidator, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisRequest(TypedDict):
    """Type definition for analysis request data."""
    image_data: str
    preferences: Optional[Dict[str, Any]]
    include_landmarks: Optional[bool]
    top_n: Optional[int]


class AnalysisResponse(TypedDict):
    """Type definition for analysis response data."""
    success: bool
    face_shape: Optional[str]
    confidence: Optional[float]
    measurements: Optional[Dict[str, float]]
    recommendations: Optional[List[Dict[str, Any]]]
    characteristics: Optional[List[str]]
    styling_principles: Optional[List[str]]
    avoid_styles: Optional[List[str]]
    personalization_notes: Optional[List[str]]
    error_message: Optional[str]
    processing_time: Optional[float]


class VirtualTryOnApp:
    """
    Main Flask application class for the virtual try-on system.
    
    This class encapsulates the web application logic, route handlers,
    and integration with the face analysis pipeline.
    """
    
    def __init__(
        self, 
        upload_folder: Union[str, Path] = "uploads",
        max_content_length: int = 16 * 1024 * 1024  # 16MB
    ) -> None:
        """
        Initialize the Flask application with configuration.
        
        Args:
            upload_folder: Directory for uploaded files
            max_content_length: Maximum file upload size in bytes
        """
        self.app = Flask(__name__)
        self.upload_folder = Path(upload_folder)
        self.upload_folder.mkdir(exist_ok=True)
        
        # Configure Flask app
        self.app.config['UPLOAD_FOLDER'] = str(self.upload_folder)
        self.app.config['MAX_CONTENT_LENGTH'] = max_content_length
        self.app.config['SECRET_KEY'] = 'stran-connect-virtual-tryon-2024'
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.shape_analyzer = FacialShapeAnalyzer()
        self.hairstyle_engine = HairstyleRecommendationEngine()
        self.image_processor = ImageProcessor()
        self.validator = InputValidator()
        
        # Register routes
        self._register_routes()
        
        # Allowed file extensions
        self.allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        
        logger.info("VirtualTryOnApp initialized successfully")
    
    def _register_routes(self) -> None:
        """Register all Flask routes with proper typing."""
        
        @self.app.route('/')
        def index() -> str:
            """Serve the main application page."""
            return render_template('index.html')
        
        @self.app.route('/api/analyze', methods=['POST'])
        def analyze_face() -> Tuple[Dict[str, Any], int]:
            """
            Analyze uploaded image for face shape and provide recommendations.
            
            Returns:
                Tuple of (response_data, status_code)
            """
            start_time = datetime.now()
            
            try:
                # Validate request
                if not request.is_json:
                    return {'success': False, 'error_message': 'Content-Type must be application/json'}, 400
                
                request_data: Dict[str, Any] = request.get_json()
                
                # Validate required fields
                validation_result = self.validator.validate_api_request(
                    request_data, 
                    required_fields=['image_data']
                )
                
                if not validation_result['is_valid']:
                    return {
                        'success': False, 
                        'error_message': '; '.join(validation_result['errors'])
                    }, 400
                
                # Process the analysis request
                analysis_result = self._process_analysis_request(request_data)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                analysis_result['processing_time'] = processing_time
                
                status_code = 200 if analysis_result['success'] else 500
                return analysis_result, status_code
                
            except Exception as e:
                logger.error(f"Error in analyze_face: {str(e)}")
                processing_time = (datetime.now() - start_time).total_seconds()
                return {
                    'success': False,
                    'error_message': f'Analysis failed: {str(e)}',
                    'processing_time': processing_time
                }, 500
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_file() -> Tuple[Dict[str, Any], int]:
            """
            Handle file uploads with validation.
            
            Returns:
                Tuple of (response_data, status_code)
            """
            try:
                if 'file' not in request.files:
                    return {'success': False, 'error_message': 'No file provided'}, 400
                
                file: FileStorage = request.files['file']
                
                if file.filename == '':
                    return {'success': False, 'error_message': 'No file selected'}, 400
                
                # Validate file
                file_validation = self._validate_uploaded_file(file)
                if not file_validation['success']:
                    return file_validation, 400
                
                # Save file securely
                filename = secure_filename(file.filename)
                file_path = self.upload_folder / filename
                file.save(str(file_path))
                
                # Convert to base64 for processing
                image = self.image_processor.load_image(file_path)
                if image is None:
                    return {'success': False, 'error_message': 'Could not process uploaded image'}, 400
                
                base64_data = self.image_processor.to_base64(image)
                
                # Clean up uploaded file
                file_path.unlink(missing_ok=True)
                
                return {
                    'success': True,
                    'image_data': base64_data,
                    'filename': filename
                }, 200
                
            except Exception as e:
                logger.error(f"Error in upload_file: {str(e)}")
                return {'success': False, 'error_message': f'Upload failed: {str(e)}'}, 500
        
        @self.app.route('/api/camera/capture', methods=['POST'])
        def capture_from_camera() -> Tuple[Dict[str, Any], int]:
            """
            Capture image from camera.
            
            Returns:
                Tuple of (response_data, status_code)
            """
            try:
                request_data: Dict[str, Any] = request.get_json() or {}
                camera_index: int = request_data.get('camera_index', 0)
                
                # Capture image
                captured_images = self.image_processor.capture_from_camera(
                    camera_index=camera_index,
                    capture_count=1
                )
                
                if not captured_images:
                    return {'success': False, 'error_message': 'Failed to capture image from camera'}, 500
                
                # Convert to base64
                base64_data = self.image_processor.to_base64(captured_images[0])
                
                return {
                    'success': True,
                    'image_data': base64_data
                }, 200
                
            except Exception as e:
                logger.error(f"Error in capture_from_camera: {str(e)}")
                return {'success': False, 'error_message': f'Camera capture failed: {str(e)}'}, 500
        
        @self.app.route('/api/preferences/validate', methods=['POST'])
        def validate_preferences() -> Tuple[Dict[str, Any], int]:
            """
            Validate user preferences.
            
            Returns:
                Tuple of (response_data, status_code)
            """
            try:
                request_data: Dict[str, Any] = request.get_json() or {}
                preferences: Dict[str, Any] = request_data.get('preferences', {})
                
                validation_result = self.validator.validate_user_preferences(preferences)
                
                return {
                    'success': validation_result['is_valid'],
                    'validation_result': validation_result
                }, 200
                
            except Exception as e:
                logger.error(f"Error in validate_preferences: {str(e)}")
                return {'success': False, 'error_message': f'Validation failed: {str(e)}'}, 500
        
        @self.app.route('/api/hairstyles/<shape>')
        def get_hairstyles_for_shape(shape: str) -> Tuple[Dict[str, Any], int]:
            """
            Get hairstyle recommendations for a specific face shape.
            
            Args:
                shape: Face shape name
                
            Returns:
                Tuple of (response_data, status_code)
            """
            try:
                # Validate face shape
                try:
                    face_shape = FaceShape(shape.lower())
                except ValueError:
                    return {'success': False, 'error_message': f'Invalid face shape: {shape}'}, 400
                
                # Get shape description
                description = self.shape_analyzer.get_shape_description(face_shape)
                
                return {
                    'success': True,
                    'face_shape': face_shape.value,
                    'description': description
                }, 200
                
            except Exception as e:
                logger.error(f"Error in get_hairstyles_for_shape: {str(e)}")
                return {'success': False, 'error_message': f'Failed to get hairstyles: {str(e)}'}, 500
        
        @self.app.route('/static/<path:filename>')
        def serve_static(filename: str) -> Any:
            """Serve static files."""
            return send_from_directory(self.app.static_folder, filename)
    
    def _process_analysis_request(self, request_data: Dict[str, Any]) -> AnalysisResponse:
        """
        Process face analysis request with comprehensive typing.
        
        Args:
            request_data: Request data containing image and preferences
            
        Returns:
            Analysis response with results or error information
        """
        try:
            # Extract parameters
            image_data: str = request_data['image_data']
            preferences: Optional[Dict[str, Any]] = request_data.get('preferences')
            include_landmarks: bool = request_data.get('include_landmarks', False)
            top_n: int = request_data.get('top_n', 5)
            
            # Load image from base64
            image: Optional[np.ndarray] = self.image_processor.load_from_base64(image_data)
            if image is None:
                return AnalysisResponse(
                    success=False,
                    face_shape=None,
                    confidence=None,
                    measurements=None,
                    recommendations=None,
                    characteristics=None,
                    styling_principles=None,
                    avoid_styles=None,
                    personalization_notes=None,
                    error_message="Could not decode image data",
                    processing_time=None
                )
            
            # Validate image quality
            quality_validation = self.image_processor.validate_image_quality(image)
            if not quality_validation['is_valid']:
                return AnalysisResponse(
                    success=False,
                    face_shape=None,
                    confidence=None,
                    measurements=None,
                    recommendations=None,
                    characteristics=None,
                    styling_principles=None,
                    avoid_styles=None,
                    personalization_notes=None,
                    error_message=f"Image quality issues: {'; '.join(quality_validation['errors'])}",
                    processing_time=None
                )
            
            # Preprocess image
            processed_image: np.ndarray = self.image_processor.preprocess_image(image)
            
            # Detect faces and extract measurements
            face_detection_success, landmarks = self.face_detector.detect_faces(processed_image)
            
            if not face_detection_success or not landmarks:
                return AnalysisResponse(
                    success=False,
                    face_shape=None,
                    confidence=None,
                    measurements=None,
                    recommendations=None,
                    characteristics=None,
                    styling_principles=None,
                    avoid_styles=None,
                    personalization_notes=None,
                    error_message="No face detected in the image",
                    processing_time=None
                )
            
            # Extract facial measurements
            measurements: Dict[str, float] = self.face_detector.extract_facial_measurements(
                landmarks, processed_image.shape
            )
            
            # Validate measurements
            measurement_validation = self.validator.validate_facial_measurements(measurements)
            if not measurement_validation['is_valid']:
                return AnalysisResponse(
                    success=False,
                    face_shape=None,
                    confidence=None,
                    measurements=None,
                    recommendations=None,
                    characteristics=None,
                    styling_principles=None,
                    avoid_styles=None,
                    personalization_notes=None,
                    error_message=f"Invalid measurements: {'; '.join(measurement_validation['errors'])}",
                    processing_time=None
                )
            
            # Analyze face shape
            shape_analysis: ShapeAnalysisResult = self.shape_analyzer.analyze_face_shape(measurements)
            
            # Get hairstyle recommendations
            recommendations: RecommendationResult = self.hairstyle_engine.get_recommendations(
                shape_analysis, preferences, top_n
            )
            
            # Format recommendations for response
            formatted_recommendations: List[Dict[str, Any]] = [
                {
                    'name': rec.name,
                    'category': rec.category.value,
                    'description': rec.description,
                    'suitability_score': rec.suitability_score,
                    'styling_tips': rec.styling_tips,
                    'maintenance_level': rec.maintenance_level,
                    'length': rec.length.value,
                    'texture_compatibility': [t.value for t in rec.texture_compatibility],
                    'image_url': rec.image_url
                }
                for rec in recommendations.top_recommendations
            ]
            
            return AnalysisResponse(
                success=True,
                face_shape=shape_analysis.primary_shape.value,
                confidence=shape_analysis.confidence,
                measurements=measurements,
                recommendations=formatted_recommendations,
                characteristics=shape_analysis.characteristics,
                styling_principles=recommendations.styling_principles,
                avoid_styles=recommendations.avoid_styles,
                personalization_notes=recommendations.personalization_notes,
                error_message=None,
                processing_time=None
            )
            
        except Exception as e:
            logger.error(f"Error processing analysis request: {str(e)}")
            return AnalysisResponse(
                success=False,
                face_shape=None,
                confidence=None,
                measurements=None,
                recommendations=None,
                characteristics=None,
                styling_principles=None,
                avoid_styles=None,
                personalization_notes=None,
                error_message=f"Analysis failed: {str(e)}",
                processing_time=None
            )
    
    def _validate_uploaded_file(self, file: FileStorage) -> Dict[str, Union[bool, str]]:
        """
        Validate uploaded file with comprehensive checks.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Check file extension
            if file.filename:
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in self.allowed_extensions:
                    return {
                        'success': False,
                        'error_message': f'Invalid file type. Allowed: {", ".join(self.allowed_extensions)}'
                    }
            
            # Create temporary file data for validation
            file_data = {
                'filename': file.filename or 'unknown',
                'content_type': file.content_type or 'unknown',
                'size': 0  # We'll estimate this
            }
            
            # Validate using our validator
            validation_result = self.validator.validate_uploaded_file(file_data)
            
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'error_message': '; '.join(validation_result['errors'])
                }
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Error validating uploaded file: {str(e)}")
            return {'success': False, 'error_message': f'Validation failed: {str(e)}'}
    
    def run(
        self, 
        host: str = '0.0.0.0', 
        port: int = 5000, 
        debug: bool = False
    ) -> None:
        """
        Run the Flask application.
        
        Args:
            host: Host address to bind to
            port: Port number to listen on
            debug: Whether to run in debug mode
        """
        logger.info(f"Starting Virtual Try-On application on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_app() -> Flask:
    """
    Factory function to create Flask application instance.
    
    Returns:
        Configured Flask application
    """
    virtual_tryon = VirtualTryOnApp()
    return virtual_tryon.app


if __name__ == '__main__':
    # Create and run the application
    app_instance = VirtualTryOnApp()
    app_instance.run(debug=True)