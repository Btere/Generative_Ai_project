# Stran Connect Virtual Try-On System

A sophisticated virtual try-on pipeline that analyzes facial shapes and recommends hairstyles based on individual facial features.

## Features

- **Face Detection & Analysis**: Advanced facial landmark detection using MediaPipe
- **Facial Shape Classification**: Geometric analysis to determine facial shape types (oval, round, square, heart, diamond, long)
- **Hairstyle Recommendations**: AI-powered suggestions based on facial shape compatibility
- **Multi-Input Support**: Camera capture (web/mobile) and photo upload functionality
- **Modern Web Interface**: Responsive design with real-time processing

## System Architecture

```
src/
├── core/
│   ├── face_detector.py      # Face detection and landmark extraction
│   ├── shape_analyzer.py     # Facial shape classification
│   └── hairstyle_engine.py   # Recommendation algorithm
├── utils/
│   ├── image_processor.py    # Image preprocessing utilities
│   └── validators.py         # Input validation
├── web/
│   ├── app.py               # Flask web application
│   ├── static/              # CSS, JS, images
│   └── templates/           # HTML templates
└── data/
    └── hairstyles/          # Hairstyle database and rules
```

## Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python src/web/app.py
```

## Usage

1. Navigate to `http://localhost:5000`
2. Upload a photo or use camera capture
3. Get facial shape analysis and hairstyle recommendations
4. View detailed explanations and styling tips

## Technical Details

- **Face Detection**: MediaPipe Face Mesh for 468 facial landmarks
- **Shape Analysis**: Geometric ratios and proportions calculation
- **Classification**: Machine learning-based facial shape determination
- **Recommendations**: Rule-based engine with expert styling knowledge

## Supported Facial Shapes

- **Oval**: Balanced proportions, versatile styling options
- **Round**: Fuller cheeks, suits volume-adding styles
- **Square**: Strong jawline, softening styles recommended
- **Heart**: Wider forehead, balancing lower face volume
- **Diamond**: Wider cheekbones, forehead and chin emphasis
- **Long**: Elongated face, width-adding styles preferred
