<<<<<<< HEAD
# road-damage-detection
=======
# Road Damage Detection Model

An advanced deep learning model for detecting and classifying road damage using EfficientNetV2 architecture. This project implements a comprehensive pipeline for training and evaluating road damage detection models with various optimizations for performance and accuracy.

## Features

- **Advanced Model Architecture**
  - Based on EfficientNetV2-S with custom modifications
  - Feature pyramid fusion with attention mechanisms
  - Mixed precision training support
  - Optimized for A100 GPU performance

- **Data Processing**
  - Automated preprocessing pipeline
  - CLAHE enhancement for better feature detection
  - Advanced data augmentation techniques
  - Efficient data loading and batching

- **Training Optimizations**
  - Custom training loop with performance optimizations
  - Learning rate scheduling with warmup
  - Mixed precision training
  - Early stopping and model checkpointing
  - Class weight balancing

## Requirements

```
tensorflow
keras
numpy
opencv-python
scikit-learn
seaborn
matplotlib
tqdm
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/seungdori/road-damage-detection.git
cd road-damage-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from road_damage_classifier import RoadDamageClassifier

classifier = RoadDamageClassifier(
    train_dir='data/train',
    test_dir='data/test',
    valid_dir='data/valid',
    img_size=224,
    batch_size=32,
    learning_rate=1e-3,
    epochs=50
)

# Train the model
history = classifier.train()

# Evaluate the model
results = classifier.evaluate()
```

### Command Line Interface

```bash
python main.py --preprocess \
               --batch-size 128 \
               --epochs 300 \
               --img-size 224 \
               --learning-rate 0.001 \
               --output-dir ./models
```

### Arguments

- `--preprocess`: Enable data preprocessing
- `--use-pretrained`: Use a pre-trained model
- `--model-path`: Path to pre-trained model file
- `--batch-size`: Batch size (default: 128)
- `--epochs`: Number of training epochs (default: 300)
- `--img-size`: Input image size (default: 224)
- `--learning-rate`: Learning rate (default: 0.001)
- `--output-dir`: Directory to save models and results (default: ./models)

## Model Architecture

The model uses a modified EfficientNetV2-S backbone with the following enhancements:

1. Multi-scale feature extraction
2. Feature pyramid fusion
3. Attention mechanisms
4. Custom classification head
5. Regularization techniques

## Training Pipeline

The training pipeline includes:

1. Data preprocessing and augmentation
2. Mixed precision training
3. Learning rate scheduling
4. Early stopping
5. Model checkpointing
6. Performance monitoring and visualization

## Performance Optimization

- GPU memory optimization
- XLA compilation
- Mixed precision training
- Efficient data loading
- Custom training loop

## Results Visualization

The training process generates:
- Training history plots
- Confusion matrices
- Classification reports
- Performance metrics

## Directory Structure

```
road-damage-detection/
├── data/
│   ├── train/
│   ├── test/
│   └── valid/
├── models/
├── road_damage_classifier.py
├── main.py
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- EfficientNetV2 paper and implementation
- TensorFlow team for mixed precision training support
- Road damage dataset creators and contributors
>>>>>>> b4da64f (Initial commit)
