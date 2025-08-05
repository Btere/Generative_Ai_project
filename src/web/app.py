"""
Flask Web Application for Virtual Try-On System

This module provides the main web interface for the Stran Connect virtual try-on
system, including endpoints for image upload, camera capture, face analysis,
and hairstyle recommendations.

Author: Stran Connect Team
"""

import os
import sys
import json
import uuid
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import numpy as np
import cv2
import logging

# Import our modules
from core.face_detector import FaceDetector
from core.shape_analyzer import FacialShapeAnalyzer, FaceShape
from core.hairstyle_engine import HairstyleRecommendationEngine
from utils.image_processor import ImageProcessor
from utils.validators import InputValidator, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'stran-connect-virtual-tryon-2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for frontend integration
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

# Initialize system components
try:
    face_detector = FaceDetector()
    shape_analyzer = FacialShapeAnalyzer()
    hairstyle_engine = HairstyleRecommendationEngine()
    image_processor = ImageProcessor()
    validator = InputValidator()
    
    logger.info("All system components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize system components: {str(e)}")
    raise

# Create upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.route('/')
def index():
    """Main application page."""
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'components': {
            'face_detector': 'ready',
            'shape_analyzer': 'ready',
            'hairstyle_engine': 'ready',
            'image_processor': 'ready'
        }
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Handle image upload from file or base64 data.
    
    Expected JSON payload:
    {
        "image_data": "base64_encoded_image_data",
        "filename": "optional_filename",
        "preferences": {
            "preferred_length": "medium",
            "hair_texture": "wavy",
            "maintenance_preference": "low"
        }
    }
    """
    try:
        # Get request data
        if request.is_json:
            data = request.get_json()
        else:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        # Validate request
        validation_result = validator.validate_api_request(
            data, 
            required_fields=['image_data']
        )
        
        if not validation_result['is_valid']:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_result['errors']
            }), 400
        
        # Process image
        image_data = data['image_data']
        preferences = data.get('preferences', {})
        
        # Load image from base64
        image = image_processor.load_from_base64(image_data)
        if image is None:
            return jsonify({'error': 'Could not load image from provided data'}), 400
        
        # Validate image quality
        quality_validation = image_processor.validate_image_quality(image)
        if not quality_validation['is_valid']:
            return jsonify({
                'error': 'Image quality validation failed',
                'details': quality_validation['errors'],
                'warnings': quality_validation.get('warnings', [])
            }), 400
        
        # Preprocess image
        processed_image = image_processor.preprocess_image(image)
        
        # Analyze face
        analysis_result = analyze_face_complete(processed_image, preferences)
        
        # Save processed image if analysis was successful
        if analysis_result['success']:
            session_id = str(uuid.uuid4())
            session['current_session'] = session_id
            
            # Save original and processed images
            original_path = UPLOAD_DIR / f"{session_id}_original.jpg"
            processed_path = UPLOAD_DIR / f"{session_id}_processed.jpg"
            
            image_processor.save_image(image, original_path)
            image_processor.save_image(processed_image, processed_path)
            
            analysis_result['session_id'] = session_id
            analysis_result['image_info'] = image_processor.get_image_info(processed_image)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/api/camera/capture', methods=['POST'])
def capture_from_camera():
    """
    Capture image from camera and analyze.
    
    Expected JSON payload:
    {
        "camera_index": 0,
        "preferences": {
            "preferred_length": "medium",
            "hair_texture": "wavy"
        }
    }
    """
    try:
        data = request.get_json() or {}
        camera_index = data.get('camera_index', 0)
        preferences = data.get('preferences', {})
        
        # Capture image from camera
        captured_images = image_processor.capture_from_camera(
            camera_index=camera_index,
            capture_count=1
        )
        
        if not captured_images:
            return jsonify({'error': 'Failed to capture image from camera'}), 400
        
        image = captured_images[0]
        
        # Validate image quality
        quality_validation = image_processor.validate_image_quality(image)
        if not quality_validation['is_valid']:
            return jsonify({
                'error': 'Captured image quality is insufficient',
                'details': quality_validation['errors'],
                'warnings': quality_validation.get('warnings', [])
            }), 400
        
        # Preprocess image
        processed_image = image_processor.preprocess_image(image)
        
        # Analyze face
        analysis_result = analyze_face_complete(processed_image, preferences)
        
        # Save images if analysis was successful
        if analysis_result['success']:
            session_id = str(uuid.uuid4())
            session['current_session'] = session_id
            
            original_path = UPLOAD_DIR / f"{session_id}_camera.jpg"
            processed_path = UPLOAD_DIR / f"{session_id}_processed.jpg"
            
            image_processor.save_image(image, original_path)
            image_processor.save_image(processed_image, processed_path)
            
            analysis_result['session_id'] = session_id
            analysis_result['image_info'] = image_processor.get_image_info(processed_image)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error in capture_from_camera: {str(e)}")
        return jsonify({'error': 'Camera capture failed', 'details': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_face():
    """
    Analyze uploaded image for face shape and provide recommendations.
    
    This is a simplified endpoint that expects image data and returns analysis.
    """
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Load and process image
        image = image_processor.load_from_base64(data['image_data'])
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Get analysis
        result = analyze_face_complete(image, data.get('preferences', {}))
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze_face: {str(e)}")
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500


@app.route('/api/recommendations/<face_shape>')
def get_recommendations_for_shape(face_shape):
    """Get hairstyle recommendations for a specific face shape."""
    try:
        # Validate face shape
        try:
            shape_enum = FaceShape(face_shape.lower())
        except ValueError:
            return jsonify({'error': f'Invalid face shape: {face_shape}'}), 400
        
        # Get preferences from query parameters
        preferences = {}
        if request.args.get('preferred_length'):
            preferences['preferred_length'] = request.args.get('preferred_length')
        if request.args.get('hair_texture'):
            preferences['hair_texture'] = request.args.get('hair_texture')
        if request.args.get('maintenance_preference'):
            preferences['maintenance_preference'] = request.args.get('maintenance_preference')
        
        # Create mock analysis result for the shape
        from core.shape_analyzer import ShapeAnalysisResult
        mock_analysis = ShapeAnalysisResult(
            primary_shape=shape_enum,
            confidence=0.85
        )
        
        # Get recommendations
        recommendations = hairstyle_engine.get_recommendations(
            mock_analysis,
            preferences,
            top_n=int(request.args.get('top_n', 10))
        )
        
        return jsonify(format_recommendations_response(recommendations))
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({'error': 'Failed to get recommendations', 'details': str(e)}), 500


@app.route('/api/face-shapes')
def get_face_shapes():
    """Get list of all supported face shapes with descriptions."""
    try:
        shapes_info = {}
        
        for shape in FaceShape:
            if shape != FaceShape.UNKNOWN:
                description = shape_analyzer.get_shape_description(shape)
                shapes_info[shape.value] = {
                    'name': shape.value.title(),
                    'description': description.get('description', ''),
                    'styling_advice': description.get('styling_advice', ''),
                    'recommended_styles': description.get('recommended_styles', [])
                }
        
        return jsonify(shapes_info)
        
    except Exception as e:
        logger.error(f"Error getting face shapes: {str(e)}")
        return jsonify({'error': 'Failed to get face shapes', 'details': str(e)}), 500


@app.route('/api/session/<session_id>/image/<image_type>')
def get_session_image(session_id, image_type):
    """Get processed images from a session."""
    try:
        # Validate session
        if session.get('current_session') != session_id:
            return jsonify({'error': 'Invalid session'}), 403
        
        # Validate image type
        if image_type not in ['original', 'processed', 'camera']:
            return jsonify({'error': 'Invalid image type'}), 400
        
        # Find image file
        image_path = UPLOAD_DIR / f"{session_id}_{image_type}.jpg"
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(str(image_path), mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error getting session image: {str(e)}")
        return jsonify({'error': 'Failed to retrieve image', 'details': str(e)}), 500


@app.route('/api/preferences/validate', methods=['POST'])
def validate_preferences():
    """Validate user preferences."""
    try:
        preferences = request.get_json()
        if not preferences:
            return jsonify({'error': 'No preferences provided'}), 400
        
        validation_result = validator.validate_user_preferences(preferences)
        
        return jsonify({
            'valid': validation_result['is_valid'],
            'errors': validation_result.get('errors', []),
            'warnings': validation_result.get('warnings', [])
        })
        
    except Exception as e:
        logger.error(f"Error validating preferences: {str(e)}")
        return jsonify({'error': 'Validation failed', 'details': str(e)}), 500


def analyze_face_complete(image: np.ndarray, preferences: Dict = None) -> Dict:
    """
    Complete face analysis pipeline.
    
    Args:
        image: Input image as numpy array
        preferences: User preferences dictionary
        
    Returns:
        Dictionary with complete analysis results
    """
    try:
        result = {
            'success': False,
            'face_detected': False,
            'face_shape': None,
            'confidence': 0.0,
            'measurements': {},
            'recommendations': [],
            'styling_principles': [],
            'avoid_styles': [],
            'personalization_notes': [],
            'processing_time': 0.0
        }
        
        start_time = datetime.now()
        
        # Step 1: Detect face and extract landmarks
        face_success, landmarks = face_detector.detect_faces(image)
        
        if not face_success or not landmarks:
            result['error'] = 'No face detected in the image'
            return result
        
        result['face_detected'] = True
        
        # Step 2: Extract facial measurements
        measurements = face_detector.extract_facial_measurements(landmarks, image.shape)
        
        if not measurements:
            result['error'] = 'Could not extract facial measurements'
            return result
        
        result['measurements'] = measurements
        
        # Step 3: Analyze face shape
        shape_analysis = shape_analyzer.analyze_face_shape(measurements)
        
        if shape_analysis.primary_shape == FaceShape.UNKNOWN:
            result['error'] = 'Could not determine face shape'
            return result
        
        result['face_shape'] = shape_analysis.primary_shape.value
        result['confidence'] = shape_analysis.confidence
        result['secondary_shape'] = shape_analysis.secondary_shape.value if shape_analysis.secondary_shape else None
        result['characteristics'] = shape_analysis.characteristics or []
        
        # Step 4: Get hairstyle recommendations
        recommendations = hairstyle_engine.get_recommendations(
            shape_analysis,
            preferences,
            top_n=10
        )
        
        # Format recommendations for response
        result['recommendations'] = [
            {
                'name': rec.name,
                'category': rec.category.value,
                'description': rec.description,
                'suitability_score': round(rec.suitability_score, 2),
                'styling_tips': rec.styling_tips,
                'maintenance_level': rec.maintenance_level,
                'length': rec.length.value,
                'texture_compatibility': [t.value for t in rec.texture_compatibility]
            }
            for rec in recommendations.top_recommendations
        ]
        
        result['styling_principles'] = recommendations.styling_principles
        result['avoid_styles'] = recommendations.avoid_styles
        result['personalization_notes'] = recommendations.personalization_notes
        
        # Calculate processing time
        end_time = datetime.now()
        result['processing_time'] = (end_time - start_time).total_seconds()
        
        result['success'] = True
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_face_complete: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': 'Analysis failed',
            'details': str(e)
        }


def format_recommendations_response(recommendations) -> Dict:
    """Format recommendations for API response."""
    return {
        'face_shape': recommendations.face_shape.value,
        'recommendations': [
            {
                'name': rec.name,
                'category': rec.category.value,
                'description': rec.description,
                'suitability_score': round(rec.suitability_score, 2),
                'styling_tips': rec.styling_tips,
                'maintenance_level': rec.maintenance_level,
                'length': rec.length.value,
                'texture_compatibility': [t.value for t in rec.texture_compatibility]
            }
            for rec in recommendations.top_recommendations
        ],
        'styling_principles': recommendations.styling_principles,
        'avoid_styles': recommendations.avoid_styles,
        'personalization_notes': recommendations.personalization_notes,
        'total_recommendations': len(recommendations.all_recommendations)
    }


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development server configuration
    import argparse
    
    parser = argparse.ArgumentParser(description='Stran Connect Virtual Try-On Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Stran Connect Virtual Try-On Server...")
    logger.info(f"Server will be available at: http://{args.host}:{args.port}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )