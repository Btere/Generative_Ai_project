"""
Validation Utilities for Virtual Try-On System

This module provides comprehensive validation functions for user inputs,
image data, and system parameters to ensure data integrity and security.

Author: Stran Connect Team
"""

import re
import mimetypes
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """
    Comprehensive input validation for virtual try-on system.
    
    This class provides validation methods for various types of inputs
    including files, user preferences, and system parameters.
    """
    
    def __init__(self) -> None:
        """Initialize the InputValidator with configuration parameters."""
        
        # File validation parameters
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_image_types = {
            'image/jpeg', 'image/jpg', 'image/png', 
            'image/bmp', 'image/tiff', 'image/webp'
        }
        self.allowed_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
        }
        
        # Image dimension limits
        self.min_image_dimensions = (100, 100)
        self.max_image_dimensions = (4096, 4096)
        
        # User preference validation patterns
        self.valid_face_shapes = {
            'oval', 'round', 'square', 'heart', 'diamond', 'long', 'unknown'
        }
        self.valid_hair_lengths = {'short', 'medium', 'long', 'extra_long'}
        self.valid_hair_textures = {'straight', 'wavy', 'curly', 'coily'}
        self.valid_maintenance_levels = {'low', 'medium', 'high'}
        
        logger.info("InputValidator initialized successfully")
    
    def validate_uploaded_file(
        self, 
        file_data: Dict[str, Any]
    ) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate uploaded file data.
        
        Args:
            file_data: Dictionary containing file information
                      Expected keys: 'filename', 'content_type', 'size', 'content'
            
        Returns:
            Dictionary with validation results
        """
        try:
            result = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Check required fields
            required_fields = ['filename', 'content_type', 'size']
            missing_fields = [field for field in required_fields if field not in file_data]
            
            if missing_fields:
                result['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
                result['is_valid'] = False
                return result
            
            # Validate filename
            filename_validation = self._validate_filename(file_data['filename'])
            if not filename_validation['is_valid']:
                result['errors'].extend(filename_validation['errors'])
                result['is_valid'] = False
            
            # Validate file size
            size_validation = self._validate_file_size(file_data['size'])
            if not size_validation['is_valid']:
                result['errors'].extend(size_validation['errors'])
                result['is_valid'] = False
            
            # Validate content type
            content_type_validation = self._validate_content_type(file_data['content_type'])
            if not content_type_validation['is_valid']:
                result['errors'].extend(content_type_validation['errors'])
                result['is_valid'] = False
            
            # Additional security checks
            security_validation = self._validate_file_security(file_data)
            if not security_validation['is_valid']:
                result['errors'].extend(security_validation['errors'])
                result['warnings'].extend(security_validation['warnings'])
                result['is_valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating uploaded file: {str(e)}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': []
            }
    
    def _validate_filename(self, filename: str) -> Dict[str, Union[bool, List[str]]]:
        """Validate filename format and security."""
        result = {'is_valid': True, 'errors': []}
        
        if not filename:
            result['errors'].append("Filename cannot be empty")
            result['is_valid'] = False
            return result
        
        # Check for dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in filename for char in dangerous_chars):
            result['errors'].append("Filename contains invalid characters")
            result['is_valid'] = False
        
        # Check file extension
        file_path = Path(filename)
        if file_path.suffix.lower() not in self.allowed_extensions:
            result['errors'].append(f"Invalid file extension. Allowed: {', '.join(self.allowed_extensions)}")
            result['is_valid'] = False
        
        # Check filename length
        if len(filename) > 255:
            result['errors'].append("Filename too long (max 255 characters)")
            result['is_valid'] = False
        
        return result
    
    def _validate_file_size(self, file_size: int) -> Dict[str, Union[bool, List[str]]]:
        """Validate file size limits."""
        result = {'is_valid': True, 'errors': []}
        
        if file_size <= 0:
            result['errors'].append("File size must be greater than 0")
            result['is_valid'] = False
        elif file_size > self.max_file_size:
            max_mb = self.max_file_size / (1024 * 1024)
            result['errors'].append(f"File too large (max {max_mb:.1f}MB)")
            result['is_valid'] = False
        
        return result
    
    def _validate_content_type(self, content_type: str) -> Dict[str, Union[bool, List[str]]]:
        """Validate MIME type."""
        result = {'is_valid': True, 'errors': []}
        
        if not content_type:
            result['errors'].append("Content type cannot be empty")
            result['is_valid'] = False
        elif content_type not in self.allowed_image_types:
            result['errors'].append(f"Invalid content type. Allowed: {', '.join(self.allowed_image_types)}")
            result['is_valid'] = False
        
        return result
    
    def _validate_file_security(self, file_data: Dict[str, Any]) -> Dict[str, Union[bool, List[str]]]:
        """Perform security validation on file data."""
        result = {'is_valid': True, 'errors': [], 'warnings': []}
        
        # Check for executable extensions (additional security)
        filename = file_data.get('filename', '')
        executable_extensions = {'.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.js', '.vbs'}
        
        if any(filename.lower().endswith(ext) for ext in executable_extensions):
            result['errors'].append("Executable files are not allowed")
            result['is_valid'] = False
        
        # Validate content type matches file extension
        if 'filename' in file_data and 'content_type' in file_data:
            guessed_type, _ = mimetypes.guess_type(file_data['filename'])
            if guessed_type and guessed_type != file_data['content_type']:
                result['warnings'].append("Content type doesn't match file extension")
        
        return result
    
    def validate_user_preferences(
        self, 
        preferences: Dict[str, Any]
    ) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate user preference data.
        
        Args:
            preferences: Dictionary containing user preferences
            
        Returns:
            Dictionary with validation results
        """
        try:
            result = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Validate hair length preference
            if 'preferred_length' in preferences:
                if preferences['preferred_length'] not in self.valid_hair_lengths:
                    result['errors'].append(
                        f"Invalid hair length. Valid options: {', '.join(self.valid_hair_lengths)}"
                    )
                    result['is_valid'] = False
            
            # Validate hair texture
            if 'hair_texture' in preferences:
                if preferences['hair_texture'] not in self.valid_hair_textures:
                    result['errors'].append(
                        f"Invalid hair texture. Valid options: {', '.join(self.valid_hair_textures)}"
                    )
                    result['is_valid'] = False
            
            # Validate maintenance preference
            if 'maintenance_preference' in preferences:
                if preferences['maintenance_preference'] not in self.valid_maintenance_levels:
                    result['errors'].append(
                        f"Invalid maintenance level. Valid options: {', '.join(self.valid_maintenance_levels)}"
                    )
                    result['is_valid'] = False
            
            # Validate age (if provided)
            if 'age' in preferences:
                age = preferences['age']
                if not isinstance(age, int) or age < 13 or age > 120:
                    result['errors'].append("Age must be between 13 and 120")
                    result['is_valid'] = False
            
            # Validate lifestyle (if provided)
            if 'lifestyle' in preferences:
                valid_lifestyles = {'active', 'busy', 'relaxed', 'professional', 'creative'}
                if preferences['lifestyle'] not in valid_lifestyles:
                    result['warnings'].append(f"Unrecognized lifestyle. Common options: {', '.join(valid_lifestyles)}")
            
            # Validate hair thickness (if provided)
            if 'hair_thickness' in preferences:
                valid_thickness = {'fine', 'medium', 'thick'}
                if preferences['hair_thickness'] not in valid_thickness:
                    result['errors'].append(f"Invalid hair thickness. Valid options: {', '.join(valid_thickness)}")
                    result['is_valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating user preferences: {str(e)}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': []
            }
    
    def validate_facial_measurements(
        self, 
        measurements: Dict[str, float]
    ) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate facial measurement data.
        
        Args:
            measurements: Dictionary containing facial measurements
            
        Returns:
            Dictionary with validation results
        """
        try:
            result = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Required measurements
            required_measurements = [
                'face_width', 'face_height', 'forehead_width', 
                'jaw_width', 'cheekbone_width'
            ]
            
            # Check for required measurements
            missing_measurements = [
                key for key in required_measurements 
                if key not in measurements
            ]
            
            if missing_measurements:
                result['errors'].append(
                    f"Missing required measurements: {', '.join(missing_measurements)}"
                )
                result['is_valid'] = False
                return result
            
            # Validate measurement values
            for key, value in measurements.items():
                if not isinstance(value, (int, float)):
                    result['errors'].append(f"Measurement '{key}' must be a number")
                    result['is_valid'] = False
                    continue
                
                if value <= 0:
                    result['errors'].append(f"Measurement '{key}' must be positive")
                    result['is_valid'] = False
                
                if value > 10000:  # Reasonable upper limit for pixel measurements
                    result['warnings'].append(f"Measurement '{key}' seems unusually large")
            
            # Validate measurement ratios for reasonableness
            if result['is_valid']:
                ratio_validation = self._validate_measurement_ratios(measurements)
                if not ratio_validation['is_valid']:
                    result['warnings'].extend(ratio_validation['warnings'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating facial measurements: {str(e)}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': []
            }
    
    def _validate_measurement_ratios(
        self, 
        measurements: Dict[str, float]
    ) -> Dict[str, Union[bool, List[str]]]:
        """Validate facial measurement ratios for reasonableness."""
        result = {'is_valid': True, 'warnings': []}
        
        try:
            # Calculate key ratios
            face_ratio = measurements['face_height'] / measurements['face_width']
            forehead_ratio = measurements['forehead_width'] / measurements['face_width']
            jaw_ratio = measurements['jaw_width'] / measurements['face_width']
            
            # Check face ratio (should be between 0.8 and 2.0 for reasonable faces)
            if face_ratio < 0.8 or face_ratio > 2.0:
                result['warnings'].append(
                    f"Unusual face height-to-width ratio: {face_ratio:.2f}"
                )
            
            # Check if forehead is wider than face (shouldn't happen)
            if forehead_ratio > 1.1:
                result['warnings'].append("Forehead appears wider than face width")
            
            # Check if jaw is wider than face (shouldn't happen)
            if jaw_ratio > 1.1:
                result['warnings'].append("Jaw appears wider than face width")
            
            # Check if cheekbones are much narrower than forehead
            if 'cheekbone_width' in measurements:
                cheekbone_ratio = measurements['cheekbone_width'] / measurements['forehead_width']
                if cheekbone_ratio < 0.5:
                    result['warnings'].append("Unusual cheekbone-to-forehead ratio")
            
        except (KeyError, ZeroDivisionError) as e:
            result['warnings'].append(f"Could not validate measurement ratios: {str(e)}")
        
        return result
    
    def validate_api_request(
        self, 
        request_data: Dict[str, Any],
        required_fields: List[str] = None
    ) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate API request data.
        
        Args:
            request_data: Dictionary containing request data
            required_fields: List of required field names
            
        Returns:
            Dictionary with validation results
        """
        try:
            result = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Check for required fields
            if required_fields:
                missing_fields = [
                    field for field in required_fields 
                    if field not in request_data or request_data[field] is None
                ]
                
                if missing_fields:
                    result['errors'].append(
                        f"Missing required fields: {', '.join(missing_fields)}"
                    )
                    result['is_valid'] = False
            
            # Validate data types and formats
            if 'image_data' in request_data:
                image_validation = self._validate_base64_image(request_data['image_data'])
                if not image_validation['is_valid']:
                    result['errors'].extend(image_validation['errors'])
                    result['is_valid'] = False
            
            # Validate confidence thresholds
            if 'min_confidence' in request_data:
                confidence = request_data['min_confidence']
                if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    result['errors'].append("min_confidence must be a number between 0 and 1")
                    result['is_valid'] = False
            
            # Validate top_n parameter
            if 'top_n' in request_data:
                top_n = request_data['top_n']
                if not isinstance(top_n, int) or not 1 <= top_n <= 20:
                    result['errors'].append("top_n must be an integer between 1 and 20")
                    result['is_valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating API request: {str(e)}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': []
            }
    
    def _validate_base64_image(self, base64_data: str) -> Dict[str, Union[bool, List[str]]]:
        """Validate base64 encoded image data."""
        result = {'is_valid': True, 'errors': []}
        
        if not base64_data:
            result['errors'].append("Image data cannot be empty")
            result['is_valid'] = False
            return result
        
        # Check for data URL prefix
        if base64_data.startswith('data:'):
            if ',' not in base64_data:
                result['errors'].append("Invalid data URL format")
                result['is_valid'] = False
                return result
            
            # Extract content type and data
            header, data = base64_data.split(',', 1)
            
            # Validate content type in header
            if 'image/' not in header:
                result['errors'].append("Data URL must contain image content type")
                result['is_valid'] = False
            
            base64_data = data
        
        # Basic base64 format validation
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', base64_data):
            result['errors'].append("Invalid base64 format")
            result['is_valid'] = False
        
        # Check reasonable length (rough estimate)
        if len(base64_data) > self.max_file_size * 1.4:  # Base64 adds ~33% overhead
            result['errors'].append("Base64 image data too large")
            result['is_valid'] = False
        
        return result
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        try:
            # Remove dangerous characters
            safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
            
            # Limit length
            if len(safe_filename) > 100:
                name, ext = Path(safe_filename).stem, Path(safe_filename).suffix
                safe_filename = name[:90] + ext
            
            # Ensure not empty
            if not safe_filename or safe_filename == '.':
                safe_filename = 'image.jpg'
            
            return safe_filename
            
        except Exception as e:
            logger.error(f"Error sanitizing filename: {str(e)}")
            return 'image.jpg'
    
    def validate_system_config(
        self, 
        config: Dict[str, Any]
    ) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate system configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            Dictionary with validation results
        """
        try:
            result = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Validate camera settings
            if 'camera_index' in config:
                if not isinstance(config['camera_index'], int) or config['camera_index'] < 0:
                    result['errors'].append("camera_index must be a non-negative integer")
                    result['is_valid'] = False
            
            # Validate image processing settings
            if 'max_image_size' in config:
                max_size = config['max_image_size']
                if not isinstance(max_size, (list, tuple)) or len(max_size) != 2:
                    result['errors'].append("max_image_size must be a tuple/list of 2 integers")
                    result['is_valid'] = False
                elif not all(isinstance(x, int) and x > 0 for x in max_size):
                    result['errors'].append("max_image_size values must be positive integers")
                    result['is_valid'] = False
            
            # Validate detection confidence
            if 'min_detection_confidence' in config:
                confidence = config['min_detection_confidence']
                if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    result['errors'].append("min_detection_confidence must be between 0 and 1")
                    result['is_valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating system config: {str(e)}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': []
            }