# Custom CNN Model for Facial Landmark Detection

This guide explains how to use the custom CNN model implementation that replaces MediaPipe for better facial landmark detection and bounding box generation.

## 🎯 **Features**

- **Custom CNN Architecture**: ResNet-based backbone with attention mechanisms
- **Multi-Task Learning**: Simultaneous landmark detection, bounding box regression, and face shape classification
- **Precise Bounding Boxes**: Targeted detection of forehead, chin, left jaw, right jaw, and jawline regions
- **Comprehensive Training Pipeline**: Complete training infrastructure with data augmentation
- **Production Ready**: Integrated with the existing virtual try-on system

## 📁 **Project Structure**

```
src/
├── models/
│   ├── facial_detection_model.py      # Custom CNN architecture
│   ├── training/
│   │   ├── trainer.py                 # Training pipeline
│   │   └── train_model.py             # Training script
│   ├── data_processing/
│   │   └── data_preparation.py        # Dataset creation utilities
│   └── inference/
└── core/
    └── cnn_face_detector.py           # CNN-based face detector (replaces MediaPipe)
```

## 🚀 **Quick Start**

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Training Data

#### Option A: Use 300W Dataset
```python
from src.models.data_processing.data_preparation import DatasetCreator

# Download 300W dataset from: https://ibug.doc.ic.ac.uk/resources/300-W/
creator = DatasetCreator(output_dir='datasets/facial_landmarks')
creator.convert_300w_dataset('path/to/300W/dataset')
creator.create_synthetic_data(num_samples=5000)
creator.split_dataset()
creator.visualize_annotations()
```

#### Option B: Create Custom Dataset
```python
# Add your own annotated images
creator = DatasetCreator(output_dir='datasets/facial_landmarks')

# For each image, provide landmarks and face shape
import numpy as np
landmarks = np.array([[x1, y1], [x2, y2], ...])  # 68 landmarks
creator.add_custom_annotation(
    image_path='path/to/image.jpg',
    landmarks=landmarks,
    face_shape='oval'  # oval, round, square, heart, diamond, long
)

creator.split_dataset()
```

### 3. Train the Model

```bash
# Basic training
python src/models/training/train_model.py \
    --data_dir datasets/facial_landmarks \
    --epochs 100 \
    --batch_size 32

# Advanced training with custom config
python src/models/training/train_model.py \
    --data_dir datasets/facial_landmarks \
    --config custom_config.json \
    --epochs 200 \
    --learning_rate 0.0005 \
    --bbox_weight 2.0
```

### 4. Use Trained Model

```python
from src.core.cnn_face_detector import CNNFaceDetector

# Initialize with trained model
detector = CNNFaceDetector(
    model_path='models/checkpoints/facial_landmark_cnn_best.pth',
    device='auto'
)

# Process image
import cv2
image = cv2.imread('test_image.jpg')
success, results = detector.detect_faces(image)

if success:
    # Get measurements
    measurements = detector.extract_facial_measurements(results, image.shape)
    
    # Visualize results
    annotated_image = detector.draw_detections(image, results)
    cv2.imshow('Results', annotated_image)
    cv2.waitKey(0)
```

## 🏗️ **Model Architecture**

### Core Components

1. **ResNet Backbone**: Feature extraction with residual connections
2. **Attention Modules**: Spatial attention for important facial regions
3. **Multi-Head Outputs**:
   - Landmark detection (68 points × 2 coordinates)
   - Bounding box regression (5 regions × 4 coordinates)
   - Face shape classification (6 classes)

### Bounding Box Regions

The model generates bounding boxes for:

1. **Forehead**: Upper facial region above eyebrows
2. **Chin**: Lower central facial region
3. **Left Jaw**: Left side jawline and cheek
4. **Right Jaw**: Right side jawline and cheek
5. **Jawline**: Complete jaw contour

### Loss Function

Multi-component loss with configurable weights:

```python
total_loss = (
    landmark_weight * landmark_loss +      # MSE for landmark coordinates
    bbox_weight * bbox_loss +              # Smooth L1 for bounding boxes
    shape_weight * shape_loss +            # CrossEntropy for face shape
    consistency_weight * consistency_loss   # Geometric consistency
)
```

## 📊 **Training Configuration**

### Model Parameters

```python
class ModelConfig:
    # Architecture
    num_landmarks: int = 68
    num_bboxes: int = 5
    input_size: Tuple[int, int] = (224, 224)
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-4
    
    # Loss weights
    landmark_weight: float = 1.0
    bbox_weight: float = 1.0
    shape_weight: float = 0.5
    consistency_weight: float = 0.3
```

### Data Augmentation

- Horizontal flip (50% probability)
- Rotation (±15 degrees)
- Brightness/contrast adjustment
- Gaussian noise
- Blur effects
- Random scaling

## 🎓 **Training From Scratch**

### Step 1: Collect Data

You need training data with:
- Facial images
- 68-point facial landmarks
- Face shape labels
- Optional: Manual bounding box annotations

### Step 2: Popular Datasets

**Recommended datasets for training:**

1. **300W Dataset** (recommended)
   - 68-point landmarks
   - ~7,000 images
   - Indoor/outdoor variations
   - Download: https://ibug.doc.ic.ac.uk/resources/300-W/

2. **WFLW Dataset**
   - 98-point landmarks
   - Large-scale dataset
   - More challenging scenarios

3. **CelebA**
   - Large celebrity dataset
   - Can be annotated with landmark detection tools

### Step 3: Data Annotation Format

```json
{
  "image001.jpg": {
    "landmarks": [x1, y1, x2, y2, ...],  // 136 values (68 points × 2)
    "bboxes": [
      [x1, y1, w1, h1],  // forehead
      [x2, y2, w2, h2],  // chin
      [x3, y3, w3, h3],  // left_jaw
      [x4, y4, w4, h4],  // right_jaw
      [x5, y5, w5, h5]   // jawline
    ],
    "face_shape": "oval",  // oval, round, square, heart, diamond, long
    "image_width": 640,
    "image_height": 480
  }
}
```

### Step 4: Training Tips

1. **Start Small**: Begin with a subset of data to verify the pipeline
2. **Monitor Losses**: Watch individual loss components (landmark, bbox, shape)
3. **Learning Rate Scheduling**: Use ReduceLROnPlateau for better convergence
4. **Early Stopping**: Monitor validation loss to prevent overfitting
5. **Data Balance**: Ensure balanced representation of different face shapes

## 🔧 **Model Integration**

The custom CNN model integrates seamlessly with the existing system:

```python
# In src/web/app.py
from core.cnn_face_detector import CNNFaceDetector

# Replace MediaPipe detector
self.face_detector = CNNFaceDetector(
    model_path='models/checkpoints/best_model.pth',
    config=model_config,
    device='auto'
)
```

## 📈 **Performance Optimization**

### Hardware Requirements

- **Minimum**: CPU-only training (slow)
- **Recommended**: GPU with 8GB+ VRAM
- **Optimal**: Multiple GPUs for faster training

### Training Time Estimates

- **Small dataset** (1K images): 2-4 hours on GPU
- **Medium dataset** (10K images): 8-12 hours on GPU
- **Large dataset** (50K+ images): 1-3 days on GPU

### Inference Speed

- **CPU**: ~100-200ms per image
- **GPU**: ~10-20ms per image

## 🐛 **Troubleshooting**

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train_model.py --batch_size 16
   ```

2. **Poor Convergence**
   ```bash
   # Adjust learning rate and loss weights
   python train_model.py --learning_rate 0.0001 --bbox_weight 2.0
   ```

3. **Data Loading Errors**
   - Verify annotation file format
   - Check image file paths
   - Ensure consistent coordinate systems

### Debug Mode

```bash
python train_model.py --debug --data_dir datasets/facial_landmarks
```

## 🔬 **Model Evaluation**

### Metrics to Monitor

1. **Landmark Accuracy**: Mean pixel error for landmark positions
2. **Bounding Box IoU**: Intersection over Union for facial regions
3. **Face Shape Accuracy**: Classification accuracy for face shapes
4. **Consistency Score**: Geometric relationship preservation

### Evaluation Script

```python
from src.models.training.trainer import ModelTrainer

# Evaluate trained model
trainer = ModelTrainer(model, config, train_loader, val_loader)
metrics = trainer.validate_epoch()
print(f"Validation Loss: {metrics['total_loss']}")
print(f"Shape Accuracy: {metrics['shape_accuracy']}")
```

## 🎯 **Advantages Over MediaPipe**

1. **Custom Training**: Tailored to your specific use case
2. **Better Bounding Boxes**: Precise facial region detection
3. **Integrated Face Shape**: Direct classification without separate analysis
4. **Controllable Accuracy**: Train longer for better performance
5. **Production Flexibility**: Deploy on any hardware configuration

## 📝 **Example Training Session**

```bash
# 1. Prepare data
python -c "
from src.models.data_processing.data_preparation import DatasetCreator
creator = DatasetCreator('datasets/facial_landmarks')
creator.convert_300w_dataset('path/to/300W')
creator.create_synthetic_data(2000)
creator.split_dataset()
"

# 2. Train model
python src/models/training/train_model.py \
    --data_dir datasets/facial_landmarks \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --bbox_weight 1.5 \
    --output_dir models/checkpoints \
    --model_name custom_facial_cnn

# 3. Test integration
python -c "
from src.core.cnn_face_detector import CNNFaceDetector
detector = CNNFaceDetector('models/checkpoints/custom_facial_cnn_best.pth')
result = detector.process_image_file('test_image.jpg')
print('Success:', result['success'])
"
```

## 🤝 **Contributing**

To improve the model:

1. **Add More Data**: Expand training dataset with diverse images
2. **Architecture Improvements**: Experiment with different backbones
3. **Loss Function Tuning**: Optimize loss weights for your use case
4. **Augmentation Strategies**: Add domain-specific augmentations

## 📞 **Support**

For issues and questions:
- Check the troubleshooting section above
- Review training logs for error messages
- Verify data format and file paths
- Monitor GPU memory usage during training

The custom CNN model provides superior control and performance compared to MediaPipe, especially when trained on domain-specific data for virtual try-on applications.