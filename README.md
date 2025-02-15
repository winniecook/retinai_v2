# retinai_v2
Deep learning system for retinal disease classification using EfficientNetV2. Achieves 86% accuracy in distinguishing between normal retinas, cataracts, and glaucoma using a two-stage training approach.

An advanced deep learning pipeline for retinal disease classification using EfficientNetV2 with Feature Pyramid Network.

## Features

- EfficientNetV2 backbone with FPN for multi-scale feature extraction
- Advanced preprocessing pipeline with vessel segmentation
- Two-stage training strategy with progressive learning rates
- Advanced augmentation techniques (RandAugment, MixUp, CutMix)
- Experiment tracking and visualization
- Multi-GPU training support

## Project Structure

```
retinal_efficientnet/
├── configs/           # Configuration files
├── data/             # Data directory
├── outputs/          # Model outputs and logs
└── src/             # Source code
    ├── models/      # Model architectures
    └── utils/       # Utility functions
```

## Setup

1. Create conda environment:
```bash
conda create -n retinal_env python=3.8
conda activate retinal_env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Pipeline

1. Preprocess data:
```bash
python src/preprocess_advanced.py
```

2. Train model:
```bash
python src/train_two_stage.py
```

## Model Architecture

- Backbone: EfficientNetV2-S
- Feature Pyramid Network for multi-scale feature extraction
- Custom attention mechanisms
- Advanced regularization techniques

## Installation
1. Clone the repository:


git clone https://github.com/winniecook/retinai_v2.git
cd retinai_v2

2. Install dependencies:
bash

pip install -r requirements.txt
Run the complete pipeline:
bash
3. Run the Complete Pipeline

./run_all.sh

📁 Project Structure
CopyInsert
retinal_efficientnet/
├── src/
│   ├── models/
│   │   └── efficientnet_fpn.py    # Model architecture
│   ├── utils/
│   │   ├── augmentations.py       # Data augmentation
│   │   └── dataset.py             # Dataset handling
│   ├── train_two_stage.py         # Training script
│   └── analyze_results.py         # Analysis tools
├── run_all.sh                     # Pipeline script
└── requirements.txt               # Dependencies

🔧 Detailed Usage
Training
The model employs a two-stage training approach:

Stage 1: Train only the classifier layers

Frozen EfficientNetV2 backbone
Higher learning rate (1e-3)
Optimized for feature extraction

Stage 2: Fine-tune the entire network

Unfrozen backbone
Lower learning rate (1e-4)
End-to-end optimization
bash

python src/train_two_stage.py
Analysis
Generate comprehensive performance metrics:

python src/analyze_results.py

## Outputs include:

ROC curves
Confusion matrices
Class-wise metrics
Training/validation curves
📈 Results
The model demonstrates robust performance:

Training Accuracy: 92.22%
Validation Accuracy: 77.36%
Test Accuracy: 86.00%
Key strengths:

Excellent normal case detection (97% recall)
Strong cataract identification (94% precision)
High-precision glaucoma detection (92% precision)

🔍 Model Architecture

EfficientNetV2-Small
├── Backbone: EfficientNetV2
├── Custom Classifier
│   ├── Dropout (0.3)
│   └── Linear Layer
└── Training Features
    ├── Early Stopping
    ├── LR Scheduling
    └── Class Weighting

📚 Citation
If you use this code in your research, please cite:


@software{cook2025retinal,
  author = {Cook, Winnie},
  title = {RetinAI: Retinal Disease Classification using EfficientNetV2},
  year = {2025},
  url = {https://github.com/winniecook/retinai_v2}
}

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
EfficientNetV2 implementation from the timm library
PyTorch team for the deep learning framework
Albumentations for image augmentation tools
📧 Contact
Winnie Cook - GitHub

Project Link: https://github.com/winniecook/retinai_v2 EOL

