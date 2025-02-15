"""
Model Performance Analysis
------------------------
Comprehensive analysis of the retinal disease classification model.
Generates performance metrics, visualizations, and detailed reports.

Outputs:
- ROC curves
- Confusion matrices
- Class-wise metrics
- Training/validation curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score
)
import torch
import json
from pathlib import Path
import logging

from utils.dataset import RetinalDataset
from utils.augmentations import get_valid_transforms
from train_two_stage import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

def load_model_and_data(model_path, test_data_path, device):
    """
    Load trained model and test dataset.
    
    Args:
        model_path: Path to model checkpoint
        test_data_path: Path to test data directory
        device: Computation device (cuda/cpu)
        
    Returns:
        model: Loaded model
        test_loader: DataLoader for test set
    """
    # Load model
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = RetinalDataset(test_data_path, transform=get_valid_transforms())
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    return model, test_loader

def calculate_metrics(y_true, y_pred, y_score):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction probabilities
        
    Returns:
        DataFrame: Performance metrics for each class
    """
    classes = ['Normal', 'Cataract', 'Glaucoma']
    metrics = []
    
    for i, class_name in enumerate(classes):
        # Binary metrics for each class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        metrics.append({
            'Class': class_name,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1-Score': f1,
            'AUC-ROC': roc_auc
        })
    
    return pd.DataFrame(metrics)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Generate and save confusion matrix plot.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    classes = ['Normal', 'Cataract', 'Glaucoma']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, y_score, save_path):
    """
    Generate and save ROC curves for each class.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction probabilities
        save_path: Path to save the plot
    """
    classes = ['Normal', 'Cataract', 'Glaucoma']
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(classes):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr,
            label=f'{class_name} (AUC = {roc_auc:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    """
    Main analysis pipeline.
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Load model and data
    model, test_loader = load_model_and_data(
        'outputs/best_model_stage2.pth',
        'processed_data/test',
        device
    )
    
    # Collect predictions
    all_targets = []
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_score = np.array(all_scores)
    
    # Generate metrics and plots
    metrics_df = calculate_metrics(y_true, y_pred, y_score)
    plot_confusion_matrix(y_true, y_pred, output_dir / 'confusion_matrix.png')
    plot_roc_curves(y_true, y_score, output_dir / 'roc_curves.png')
    
    # Save metrics
    metrics_df.to_csv(output_dir / 'performance_metrics.csv', index=False)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(metrics_df.to_string(index=False))
    
    print("\nResults saved in outputs directory:")
    print("1. performance_metrics.csv")
    print("2. confusion_matrix.png")
    print("3. roc_curves.png")

if __name__ == '__main__':
    main()
