"""
Image Processing Utilities for Virtual Try-On System

This module provides comprehensive image processing capabilities including
image capture, preprocessing, optimization, and validation for the virtual
try-on pipeline.

Author: Stran Connect Team
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Comprehensive image processing utilities for virtual try-on system.
    
    This class handles image capture, preprocessing, enhancement, and
    optimization for facial analysis and recommendation systems.
    """
    
    def __init__(
        self,
        max_image_size: Tuple[int, int] = (1024, 1024),
        quality: int = 85,
        supported_formats: List[str] = None
    ) -> None:
        """
        Initialize the ImageProcessor with configuration parameters.
        
        Args:
            max_image_size: Maximum dimensions for processed images
            quality: JPEG quality for compressed images (1-100)
            supported_formats: List of supported image formats
        """
        self.max_image_size = max_image_size
        self.quality = quality
        self.supported_formats = supported_formats or [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
        ]
        
        # Image enhancement parameters
        self.enhancement_params = {
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.9, 1.1),
            'sharpness_range': (0.8, 1.2),
            'color_range': (0.9, 1.1)
        }
        
        logger.info("ImageProcessor initialized successfully")
    
    def capture_from_camera(
        self,
        camera_index: int = 0,
        capture_count: int = 1,
        delay: float = 0.5
    ) -> List[np.ndarray]:
        """
        Capture images from camera device.
        
        Args:
            camera_index: Camera device index (usually 0 for default camera)
            capture_count: Number of images to capture
            delay: Delay between captures in seconds
            
        Returns:
            List of captured images as numpy arrays
        """
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera with index {camera_index}")
            
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            captured_images = []
            
            # Warm up camera
            for _ in range(5):
                cap.read()
            
            # Capture images
            for i in range(capture_count):
                ret, frame = cap.read()
                
                if ret:
                    # Flip image horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    captured_images.append(frame)
                    logger.info(f"Captured image {i + 1}/{capture_count}")
                    
                    if i < capture_count - 1:
                        cv2.waitKey(int(delay * 1000))
                else:
                    logger.warning(f"Failed to capture image {i + 1}")
            
            cap.release()
            logger.info(f"Successfully captured {len(captured_images)} images")
            return captured_images
            
        except Exception as e:
            logger.error(f"Error capturing from camera: {str(e)}")
            return []
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file path with validation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            image_path = Path(image_path)
            
            # Validate file exists and format
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if image_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            # Load image using OpenCV
            image = cv2.imread(str(image_path))
            
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            logger.info(f"Successfully loaded image: {image_path}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    def load_from_base64(self, base64_string: str) -> Optional[np.ndarray]:
        """
        Load image from base64 encoded string.
        
        Args:
            base64_string: Base64 encoded image data
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            logger.info("Successfully loaded image from base64")
            return image_bgr
            
        except Exception as e:
            logger.error(f"Error loading image from base64: {str(e)}")
            return None
    
    def preprocess_image(
        self,
        image: np.ndarray,
        resize: bool = True,
        enhance: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for face detection and analysis.
        
        Args:
            image: Input image as numpy array
            resize: Whether to resize image to max dimensions
            enhance: Whether to apply image enhancement
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image
        """
        try:
            processed_image = image.copy()
            
            # Resize image if needed
            if resize:
                processed_image = self._resize_image(processed_image)
            
            # Enhance image quality
            if enhance:
                processed_image = self._enhance_image(processed_image)
            
            # Normalize pixel values
            if normalize:
                processed_image = self._normalize_image(processed_image)
            
            logger.info("Image preprocessing completed")
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        max_width, max_height = self.max_image_size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        # Only resize if image is larger than max size
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_image = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return resized_image
        
        return image
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality using various filters and adjustments.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Apply automatic brightness adjustment
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(self._calculate_brightness_factor(image))
            
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # Apply sharpening
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # Convert back to OpenCV format
            enhanced_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Apply noise reduction
            enhanced_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    def _calculate_brightness_factor(self, image: np.ndarray) -> float:
        """
        Calculate optimal brightness adjustment factor.
        
        Args:
            image: Input image
            
        Returns:
            Brightness factor
        """
        # Convert to grayscale for brightness calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Target brightness (around 127 for 8-bit images)
        target_brightness = 127
        
        # Calculate adjustment factor
        if mean_brightness < target_brightness:
            factor = min(1.3, target_brightness / mean_brightness)
        else:
            factor = max(0.7, target_brightness / mean_brightness)
        
        return factor
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Ensure values are in valid range
        normalized = np.clip(image, 0, 255)
        return normalized.astype(np.uint8)
    
    def save_image(
        self,
        image: np.ndarray,
        output_path: Union[str, Path],
        quality: Optional[int] = None
    ) -> bool:
        """
        Save image to file with specified quality.
        
        Args:
            image: Image to save
            output_path: Output file path
            quality: JPEG quality (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use specified quality or default
            save_quality = quality or self.quality
            
            # Save based on file extension
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(
                    str(output_path),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, save_quality]
                )
            elif output_path.suffix.lower() == '.png':
                cv2.imwrite(
                    str(output_path),
                    image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6]
                )
            else:
                cv2.imwrite(str(output_path), image)
            
            logger.info(f"Successfully saved image to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False
    
    def to_base64(self, image: np.ndarray, format: str = 'JPEG') -> str:
        """
        Convert image to base64 string.
        
        Args:
            image: Input image
            format: Output format (JPEG, PNG, etc.)
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=self.quality)
            
            # Encode to base64
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Add data URL prefix
            mime_type = f"image/{format.lower()}"
            data_url = f"data:{mime_type};base64,{base64_string}"
            
            return data_url
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            return ""
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Union[int, float, str]]:
        """
        Get comprehensive information about an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image information
        """
        try:
            height, width, channels = image.shape
            
            # Calculate file size estimate
            estimated_size = height * width * channels
            
            # Calculate brightness and contrast metrics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Calculate color distribution
            b_mean = np.mean(image[:, :, 0])
            g_mean = np.mean(image[:, :, 1])
            r_mean = np.mean(image[:, :, 2])
            
            info = {
                'width': width,
                'height': height,
                'channels': channels,
                'estimated_size_bytes': estimated_size,
                'aspect_ratio': width / height,
                'brightness': float(brightness),
                'contrast': float(contrast),
                'color_means': {
                    'blue': float(b_mean),
                    'green': float(g_mean),
                    'red': float(r_mean)
                },
                'is_portrait': height > width,
                'is_landscape': width > height,
                'is_square': abs(width - height) < min(width, height) * 0.1
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting image info: {str(e)}")
            return {}
    
    def create_thumbnail(
        self,
        image: np.ndarray,
        size: Tuple[int, int] = (150, 150)
    ) -> np.ndarray:
        """
        Create a thumbnail of the image.
        
        Args:
            image: Input image
            size: Thumbnail dimensions
            
        Returns:
            Thumbnail image
        """
        try:
            # Calculate aspect ratio preserving dimensions
            height, width = image.shape[:2]
            target_width, target_height = size
            
            aspect_ratio = width / height
            target_aspect = target_width / target_height
            
            if aspect_ratio > target_aspect:
                # Image is wider, fit by width
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Image is taller, fit by height
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # Resize image
            thumbnail = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            
            # Create canvas and center thumbnail
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            canvas.fill(255)  # White background
            
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = thumbnail
            
            return canvas
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    def validate_image_quality(self, image: np.ndarray) -> Dict[str, Union[bool, List[str], float]]:
        """
        Validate image quality for face detection.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_result: Dict[str, Union[bool, List[str], float]] = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'quality_score': 1.0
            }
            
            height, width = image.shape[:2]
            
            # Check minimum dimensions
            if width < 200 or height < 200:
                validation_result['errors'].append("Image too small (minimum 200x200)")
                validation_result['is_valid'] = False
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                validation_result['warnings'].append("Unusual aspect ratio may affect detection")
                validation_result['quality_score'] *= 0.8
            
            # Check brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness < 50:
                validation_result['warnings'].append("Image is very dark")
                validation_result['quality_score'] *= 0.7
            elif brightness > 200:
                validation_result['warnings'].append("Image is very bright")
                validation_result['quality_score'] *= 0.8
            
            # Check contrast
            contrast = np.std(gray)
            if contrast < 20:
                validation_result['warnings'].append("Low contrast image")
                validation_result['quality_score'] *= 0.8
            
            # Check for blur (using Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100:
                validation_result['warnings'].append("Image appears to be blurry")
                validation_result['quality_score'] *= 0.6
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating image quality: {str(e)}")
            return {
                'is_valid': False,
                'warnings': [],
                'errors': [f"Validation failed: {str(e)}"],
                'quality_score': 0.0
            }