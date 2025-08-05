"""
Hairstyle Recommendation Engine for Virtual Try-On System

This module provides intelligent hairstyle recommendations based on facial shape
analysis, personal preferences, and expert styling knowledge. It combines
rule-based logic with machine learning insights to deliver personalized suggestions.

Author: Stran Connect Team
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .shape_analyzer import FaceShape, ShapeAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HairLength(Enum):
    """Enumeration of hair length categories."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    EXTRA_LONG = "extra_long"


class HairTexture(Enum):
    """Enumeration of hair texture types."""
    STRAIGHT = "straight"
    WAVY = "wavy"
    CURLY = "curly"
    COILY = "coily"


class StyleCategory(Enum):
    """Enumeration of hairstyle categories."""
    PIXIE = "pixie"
    BOB = "bob"
    LOBS = "lobs"  # Long bob
    LAYERS = "layers"
    BANGS = "bangs"
    UPDOS = "updos"
    BRAIDS = "braids"
    WAVES = "waves"
    STRAIGHT_STYLES = "straight_styles"


@dataclass
class HairstyleRecommendation:
    """
    Data class for individual hairstyle recommendations.
    
    Attributes:
        name: Name of the hairstyle
        category: Style category
        description: Detailed description
        suitability_score: How well it suits the face shape (0-1)
        styling_tips: List of styling advice
        maintenance_level: Required maintenance (low/medium/high)
        length: Recommended hair length
        texture_compatibility: Compatible hair textures
        image_url: URL to reference image (optional)
    """
    name: str
    category: StyleCategory
    description: str
    suitability_score: float
    styling_tips: List[str]
    maintenance_level: str
    length: HairLength
    texture_compatibility: List[HairTexture]
    image_url: Optional[str] = None


@dataclass
class RecommendationResult:
    """
    Data class for complete recommendation results.
    
    Attributes:
        face_shape: Analyzed face shape
        top_recommendations: List of top recommended styles
        all_recommendations: Complete list of all recommendations
        styling_principles: General styling principles for the face shape
        avoid_styles: Styles to avoid
        personalization_notes: Personalized notes and considerations
    """
    face_shape: FaceShape
    top_recommendations: List[HairstyleRecommendation]
    all_recommendations: List[HairstyleRecommendation]
    styling_principles: List[str]
    avoid_styles: List[str]
    personalization_notes: List[str]


class HairstyleRecommendationEngine:
    """
    Advanced hairstyle recommendation system using facial shape analysis.
    
    This class provides personalized hairstyle suggestions based on facial
    geometry, hair characteristics, and styling preferences.
    """
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the recommendation engine.
        
        Args:
            data_path: Path to hairstyle database (optional)
        """
        self.data_path = Path(data_path) if data_path else Path(__file__).parent.parent.parent / "data" / "hairstyles"
        
        # Initialize hairstyle database
        self._load_hairstyle_database()
        
        # Define compatibility matrices
        self._initialize_compatibility_rules()
        
        logger.info("HairstyleRecommendationEngine initialized successfully")
    
    def _load_hairstyle_database(self) -> None:
        """Load hairstyle database from JSON files or create default database."""
        try:
            database_file = self.data_path / "hairstyles_db.json"
            
            if database_file.exists():
                with open(database_file, 'r') as f:
                    self.hairstyle_db = json.load(f)
                logger.info(f"Loaded hairstyle database from {database_file}")
            else:
                # Create default database
                self.hairstyle_db = self._create_default_database()
                self._save_database()
                logger.info("Created default hairstyle database")
                
        except Exception as e:
            logger.warning(f"Error loading database: {str(e)}. Using default database.")
            self.hairstyle_db = self._create_default_database()
    
    def _create_default_database(self) -> Dict:
        """Create a comprehensive default hairstyle database."""
        return {
            "styles": {
                # Pixie cuts
                "classic_pixie": {
                    "name": "Classic Pixie",
                    "category": "pixie",
                    "description": "A timeless short cut with layers and texture that frames the face beautifully.",
                    "styling_tips": [
                        "Use texturizing products for volume",
                        "Style with fingers for a tousled look",
                        "Regular trims every 4-6 weeks"
                    ],
                    "maintenance_level": "medium",
                    "length": "short",
                    "texture_compatibility": ["straight", "wavy", "curly"],
                    "face_shape_scores": {
                        "oval": 0.9,
                        "round": 0.3,
                        "square": 0.7,
                        "heart": 0.8,
                        "diamond": 0.8,
                        "long": 0.4
                    }
                },
                "textured_pixie": {
                    "name": "Textured Pixie",
                    "category": "pixie",
                    "description": "A modern pixie with choppy layers and edgy texture for a contemporary look.",
                    "styling_tips": [
                        "Use sea salt spray for texture",
                        "Scrunch while drying",
                        "Add pomade for definition"
                    ],
                    "maintenance_level": "medium",
                    "length": "short",
                    "texture_compatibility": ["wavy", "curly"],
                    "face_shape_scores": {
                        "oval": 0.8,
                        "round": 0.4,
                        "square": 0.8,
                        "heart": 0.7,
                        "diamond": 0.9,
                        "long": 0.5
                    }
                },
                
                # Bob cuts
                "classic_bob": {
                    "name": "Classic Bob",
                    "category": "bob",
                    "description": "A chin-length bob with clean lines that suits many face shapes.",
                    "styling_tips": [
                        "Blow dry with round brush for smoothness",
                        "Use heat protectant",
                        "Trim every 6-8 weeks"
                    ],
                    "maintenance_level": "medium",
                    "length": "short",
                    "texture_compatibility": ["straight", "wavy"],
                    "face_shape_scores": {
                        "oval": 0.9,
                        "round": 0.6,
                        "square": 0.5,
                        "heart": 0.8,
                        "diamond": 0.7,
                        "long": 0.8
                    }
                },
                "angled_bob": {
                    "name": "Angled Bob",
                    "category": "bob",
                    "description": "A bob that's longer in front and shorter in back, creating dynamic movement.",
                    "styling_tips": [
                        "Style with flat iron for sleek finish",
                        "Use volumizing products at roots",
                        "Maintain angle with regular trims"
                    ],
                    "maintenance_level": "high",
                    "length": "short",
                    "texture_compatibility": ["straight", "wavy"],
                    "face_shape_scores": {
                        "oval": 0.8,
                        "round": 0.7,
                        "square": 0.6,
                        "heart": 0.9,
                        "diamond": 0.8,
                        "long": 0.7
                    }
                },
                
                # Long bobs (Lobs)
                "straight_lob": {
                    "name": "Straight Lob",
                    "category": "lobs",
                    "description": "A shoulder-length bob with straight, sleek styling.",
                    "styling_tips": [
                        "Use smoothing serum",
                        "Blow dry with paddle brush",
                        "Finish with flat iron for sleekness"
                    ],
                    "maintenance_level": "medium",
                    "length": "medium",
                    "texture_compatibility": ["straight", "wavy"],
                    "face_shape_scores": {
                        "oval": 0.8,
                        "round": 0.9,
                        "square": 0.7,
                        "heart": 0.7,
                        "diamond": 0.8,
                        "long": 0.9
                    }
                },
                "wavy_lob": {
                    "name": "Wavy Lob",
                    "category": "lobs",
                    "description": "A shoulder-length bob with soft, beachy waves.",
                    "styling_tips": [
                        "Use wave spray on damp hair",
                        "Scrunch while air drying",
                        "Diffuse for extra volume"
                    ],
                    "maintenance_level": "low",
                    "length": "medium",
                    "texture_compatibility": ["wavy", "curly"],
                    "face_shape_scores": {
                        "oval": 0.9,
                        "round": 0.8,
                        "square": 0.8,
                        "heart": 0.8,
                        "diamond": 0.7,
                        "long": 0.8
                    }
                },
                
                # Layered styles
                "long_layers": {
                    "name": "Long Layers",
                    "category": "layers",
                    "description": "Long hair with graduated layers for movement and volume.",
                    "styling_tips": [
                        "Use volumizing mousse",
                        "Blow dry with large round brush",
                        "Layer-specific cutting every 8-10 weeks"
                    ],
                    "maintenance_level": "low",
                    "length": "long",
                    "texture_compatibility": ["straight", "wavy", "curly"],
                    "face_shape_scores": {
                        "oval": 0.9,
                        "round": 0.9,
                        "square": 0.8,
                        "heart": 0.6,
                        "diamond": 0.7,
                        "long": 0.7
                    }
                },
                "face_framing_layers": {
                    "name": "Face-Framing Layers",
                    "category": "layers",
                    "description": "Strategic layers that highlight facial features and add softness.",
                    "styling_tips": [
                        "Curl front pieces away from face",
                        "Use light-hold styling cream",
                        "Focus on layered pieces during styling"
                    ],
                    "maintenance_level": "medium",
                    "length": "medium",
                    "texture_compatibility": ["straight", "wavy"],
                    "face_shape_scores": {
                        "oval": 0.8,
                        "round": 0.8,
                        "square": 0.9,
                        "heart": 0.7,
                        "diamond": 0.8,
                        "long": 0.8
                    }
                },
                
                # Bangs styles
                "side_swept_bangs": {
                    "name": "Side-Swept Bangs",
                    "category": "bangs",
                    "description": "Soft bangs swept to one side for a flattering, versatile look.",
                    "styling_tips": [
                        "Blow dry bangs with small round brush",
                        "Use light-hold hairspray",
                        "Trim every 3-4 weeks"
                    ],
                    "maintenance_level": "high",
                    "length": "medium",
                    "texture_compatibility": ["straight", "wavy"],
                    "face_shape_scores": {
                        "oval": 0.9,
                        "round": 0.7,
                        "square": 0.8,
                        "heart": 0.9,
                        "diamond": 0.8,
                        "long": 0.6
                    }
                },
                "blunt_bangs": {
                    "name": "Blunt Bangs",
                    "category": "bangs",
                    "description": "Straight-across bangs that create a bold, dramatic look.",
                    "styling_tips": [
                        "Blow dry straight down",
                        "Use flat iron for precision",
                        "Regular trimming essential"
                    ],
                    "maintenance_level": "high",
                    "length": "any",
                    "texture_compatibility": ["straight"],
                    "face_shape_scores": {
                        "oval": 0.7,
                        "round": 0.4,
                        "square": 0.5,
                        "heart": 0.6,
                        "diamond": 0.8,
                        "long": 0.9
                    }
                }
            }
        }
    
    def _save_database(self) -> None:
        """Save the hairstyle database to JSON file."""
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            database_file = self.data_path / "hairstyles_db.json"
            
            with open(database_file, 'w') as f:
                json.dump(self.hairstyle_db, f, indent=2)
            
            logger.info(f"Saved hairstyle database to {database_file}")
            
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")
    
    def _initialize_compatibility_rules(self) -> None:
        """Initialize face shape compatibility rules and styling principles."""
        
        self.styling_principles = {
            FaceShape.OVAL: [
                "Maintain natural proportions",
                "Avoid covering too much of the face",
                "Most styles are flattering",
                "Focus on enhancing natural beauty"
            ],
            FaceShape.ROUND: [
                "Add height and volume on top",
                "Create vertical lines",
                "Avoid width at ear level",
                "Use asymmetrical styles"
            ],
            FaceShape.SQUARE: [
                "Soften angular jawline",
                "Add curves and layers",
                "Avoid blunt cuts",
                "Use side parts"
            ],
            FaceShape.HEART: [
                "Balance narrow chin",
                "Add volume at jawline",
                "Minimize forehead width",
                "Use chin-length styles"
            ],
            FaceShape.DIAMOND: [
                "Widen forehead and chin",
                "Minimize cheekbone width",
                "Use full bangs",
                "Add volume at temples"
            ],
            FaceShape.LONG: [
                "Add width to face",
                "Use horizontal lines",
                "Avoid height on top",
                "Focus on cheekbone level"
            ]
        }
        
        self.avoid_styles = {
            FaceShape.OVAL: [
                "Styles that completely hide the face"
            ],
            FaceShape.ROUND: [
                "Blunt bobs at chin level",
                "Center parts",
                "Styles that add width at ears",
                "Very short pixie cuts"
            ],
            FaceShape.SQUARE: [
                "Blunt cuts",
                "Center parts",
                "Straight-across bangs",
                "Severe geometric styles"
            ],
            FaceShape.HEART: [
                "Short styles that emphasize forehead",
                "Styles with volume on top",
                "Very long straight hair",
                "Full bangs"
            ],
            FaceShape.DIAMOND: [
                "Styles that add width at cheekbones",
                "Slicked-back looks",
                "Very short cuts",
                "Center parts without bangs"
            ],
            FaceShape.LONG: [
                "Very long straight hair",
                "High ponytails",
                "Tall updos",
                "Styles that add height"
            ]
        }
    
    def get_recommendations(
        self,
        shape_analysis: ShapeAnalysisResult,
        preferences: Optional[Dict] = None,
        top_n: int = 5
    ) -> RecommendationResult:
        """
        Get personalized hairstyle recommendations based on face shape analysis.
        
        Args:
            shape_analysis: Result from facial shape analysis
            preferences: User preferences (hair length, texture, etc.)
            top_n: Number of top recommendations to return
            
        Returns:
            RecommendationResult with personalized recommendations
        """
        try:
            if shape_analysis.primary_shape == FaceShape.UNKNOWN:
                logger.warning("Cannot provide recommendations for unknown face shape")
                return self._create_empty_result()
            
            # Generate all recommendations
            all_recommendations = self._generate_all_recommendations(
                shape_analysis.primary_shape, preferences
            )
            
            # Sort by suitability score
            all_recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
            
            # Get top recommendations
            top_recommendations = all_recommendations[:top_n]
            
            # Generate personalization notes
            personalization_notes = self._generate_personalization_notes(
                shape_analysis, preferences
            )
            
            result = RecommendationResult(
                face_shape=shape_analysis.primary_shape,
                top_recommendations=top_recommendations,
                all_recommendations=all_recommendations,
                styling_principles=self.styling_principles.get(shape_analysis.primary_shape, []),
                avoid_styles=self.avoid_styles.get(shape_analysis.primary_shape, []),
                personalization_notes=personalization_notes
            )
            
            logger.info(f"Generated {len(all_recommendations)} recommendations for {shape_analysis.primary_shape.value} face shape")
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._create_empty_result()
    
    def _generate_all_recommendations(
        self,
        face_shape: FaceShape,
        preferences: Optional[Dict] = None
    ) -> List[HairstyleRecommendation]:
        """Generate all hairstyle recommendations for a face shape."""
        recommendations = []
        
        for style_id, style_data in self.hairstyle_db["styles"].items():
            # Get base suitability score
            base_score = style_data["face_shape_scores"].get(face_shape.value, 0.5)
            
            # Apply preference adjustments
            adjusted_score = self._apply_preference_adjustments(
                base_score, style_data, preferences
            )
            
            # Create recommendation object
            recommendation = HairstyleRecommendation(
                name=style_data["name"],
                category=StyleCategory(style_data["category"]),
                description=style_data["description"],
                suitability_score=adjusted_score,
                styling_tips=style_data["styling_tips"],
                maintenance_level=style_data["maintenance_level"],
                length=HairLength(style_data["length"]),
                texture_compatibility=[HairTexture(t) for t in style_data["texture_compatibility"]],
                image_url=style_data.get("image_url")
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _apply_preference_adjustments(
        self,
        base_score: float,
        style_data: Dict,
        preferences: Optional[Dict] = None
    ) -> float:
        """Apply user preferences to adjust recommendation scores."""
        if not preferences:
            return base_score
        
        adjusted_score = base_score
        
        # Length preference
        if "preferred_length" in preferences:
            preferred_length = preferences["preferred_length"]
            style_length = style_data["length"]
            
            if preferred_length == style_length:
                adjusted_score *= 1.2  # Boost matching length
            elif abs(["short", "medium", "long", "extra_long"].index(preferred_length) - 
                    ["short", "medium", "long", "extra_long"].index(style_length)) > 1:
                adjusted_score *= 0.7  # Penalize very different lengths
        
        # Texture compatibility
        if "hair_texture" in preferences:
            hair_texture = preferences["hair_texture"]
            if hair_texture in style_data["texture_compatibility"]:
                adjusted_score *= 1.1  # Boost compatible textures
            else:
                adjusted_score *= 0.6  # Penalize incompatible textures
        
        # Maintenance preference
        if "maintenance_preference" in preferences:
            maintenance_pref = preferences["maintenance_preference"]
            style_maintenance = style_data["maintenance_level"]
            
            if maintenance_pref == style_maintenance:
                adjusted_score *= 1.15  # Boost matching maintenance level
        
        # Ensure score stays within bounds
        return min(1.0, max(0.0, adjusted_score))
    
    def _generate_personalization_notes(
        self,
        shape_analysis: ShapeAnalysisResult,
        preferences: Optional[Dict] = None
    ) -> List[str]:
        """Generate personalized notes and considerations."""
        notes = []
        
        # Face shape specific notes
        if shape_analysis.confidence < 0.7:
            notes.append(
                f"Your face shape analysis shows moderate confidence ({shape_analysis.confidence:.1%}). "
                "Consider consulting with a professional stylist for personalized advice."
            )
        
        if shape_analysis.secondary_shape:
            notes.append(
                f"Your face has characteristics of both {shape_analysis.primary_shape.value} and "
                f"{shape_analysis.secondary_shape.value} shapes. Look for styles that work well for both."
            )
        
        # Characteristic-based notes
        if shape_analysis.characteristics:
            if "Strong jawline" in shape_analysis.characteristics:
                notes.append("Your strong jawline is a beautiful feature that can be enhanced with softening styles.")
            
            if "Prominent forehead" in shape_analysis.characteristics:
                notes.append("Consider bangs or side-swept styles to balance your forehead proportions.")
            
            if "Prominent cheekbones" in shape_analysis.characteristics:
                notes.append("Your cheekbones are a striking feature - choose styles that showcase them.")
        
        # Preference-based notes
        if preferences:
            if preferences.get("lifestyle") == "busy":
                notes.append("Based on your busy lifestyle, consider low-maintenance styles that look great with minimal styling.")
            
            if preferences.get("hair_thickness") == "fine":
                notes.append("For fine hair, look for styles that add volume and texture.")
            elif preferences.get("hair_thickness") == "thick":
                notes.append("Your thick hair gives you many styling options - don't be afraid to try bold cuts.")
        
        return notes
    
    def _create_empty_result(self) -> RecommendationResult:
        """Create an empty recommendation result for error cases."""
        return RecommendationResult(
            face_shape=FaceShape.UNKNOWN,
            top_recommendations=[],
            all_recommendations=[],
            styling_principles=[],
            avoid_styles=[],
            personalization_notes=["Unable to generate recommendations. Please try again or consult a professional stylist."]
        )
    
    def add_custom_style(self, style_data: Dict) -> bool:
        """
        Add a custom hairstyle to the database.
        
        Args:
            style_data: Dictionary containing style information
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Validate required fields
            required_fields = ["name", "category", "description", "face_shape_scores"]
            if not all(field in style_data for field in required_fields):
                logger.error("Missing required fields for custom style")
                return False
            
            # Generate unique ID
            style_id = style_data["name"].lower().replace(" ", "_")
            
            # Add to database
            self.hairstyle_db["styles"][style_id] = style_data
            
            # Save to file
            self._save_database()
            
            logger.info(f"Added custom style: {style_data['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom style: {str(e)}")
            return False
    
    def get_style_details(self, style_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific hairstyle.
        
        Args:
            style_name: Name of the hairstyle
            
        Returns:
            Dictionary with style details or None if not found
        """
        for style_data in self.hairstyle_db["styles"].values():
            if style_data["name"].lower() == style_name.lower():
                return style_data
        
        logger.warning(f"Style not found: {style_name}")
        return None