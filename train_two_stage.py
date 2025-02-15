import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from models.efficientnet_fpn import create_model
from utils.dataset import RetinalDataset
from utils.augmentations import get_train_transforms, get_valid_transforms, MixUpCutMix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, loader, criterion, optimizer, device, mixup_cutmix=None, scheduler=None):
    model.train()
    losses = AverageMeter()
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup or cutmix
        if mixup_cutmix is not None:
            inputs, targets_a, targets_b, lam = mixup_cutmix(inputs, targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1 - lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update statistics
        losses.update(loss.item(), inputs.size(0))
        
        if batch_idx % 10 == 0:
            logger.info(f'Train Batch: [{batch_idx}/{len(loader)}] Loss: {losses.avg:.4f}')
    
    return losses.avg

def validate(model, loader, criterion, device, classes):
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            losses.update(loss.item(), inputs.size(0))
            
            # Store predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=classes)
    
    return losses.avg, cm, report

def plot_confusion_matrix(cm, classes, output_dir, epoch):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_epoch_{epoch}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='~/retinal_efficientnet/processed_data')
    parser.add_argument('--output_dir', type=str, default='~/retinal_efficientnet/outputs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs_stage1', type=int, default=100)
    parser.add_argument('--num_epochs_stage2', type=int, default=100)
    parser.add_argument('--lr_stage1', type=float, default=1e-3)
    parser.add_argument('--lr_stage2', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    # Setup
    args.data_dir = str(Path(args.data_dir).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data loading
    train_dataset = RetinalDataset(
        args.data_dir,
        split='train',
        transform=get_train_transforms()
    )
    val_dataset = RetinalDataset(
        args.data_dir,
        split='val',
        transform=get_valid_transforms()
    )
    test_dataset = RetinalDataset(
        args.data_dir,
        split='test',
        transform=get_valid_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_dataset.get_sampler(),
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(num_classes=3, pretrained=True).to(device)
    
    # Stage 1: Train only the new layers
    logger.info("Stage 1: Training only new layers")
    best_val_acc = 0
    patience_counter = 0
    best_epoch = 0
    
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_stage1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr_stage1,
        epochs=args.num_epochs_stage1,
        steps_per_epoch=len(train_loader)
    )
    
    mixup_cutmix = MixUpCutMix(mixup_alpha=1.0, cutmix_alpha=1.0)
    
    for epoch in range(args.num_epochs_stage1):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs_stage1}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, mixup_cutmix, scheduler)
        val_loss, cm, report = validate(model, val_loader, criterion, device, train_dataset.classes)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, train_dataset.classes, output_dir, epoch)
        
        # Save best model and check early stopping
        val_acc = np.diag(cm).sum() / cm.sum()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / 'best_model_stage1.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Only stop if we've trained for minimum epochs and patience is exceeded
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Stage 2: Fine-tune all layers
    logger.info("\nStage 2: Fine-tuning all layers")
    model.load_state_dict(torch.load(output_dir / 'best_model_stage1.pth'))
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_stage2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr_stage2,
        epochs=args.num_epochs_stage2,
        steps_per_epoch=len(train_loader)
    )
    
    best_val_acc = 0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(args.num_epochs_stage2):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs_stage2}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, mixup_cutmix, scheduler)
        val_loss, cm, report = validate(model, val_loader, criterion, device, train_dataset.classes)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, train_dataset.classes, output_dir, epoch + args.num_epochs_stage1)
        
        # Save best model and check early stopping
        val_acc = np.diag(cm).sum() / cm.sum()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / 'best_model_stage2.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Only stop if we've trained for minimum epochs and patience is exceeded
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation with ensemble
    logger.info("\nEvaluating on test set...")
    # Load both stage models for ensemble
    model_stage1 = create_model(num_classes=3, pretrained=False).to(device)
    model_stage2 = create_model(num_classes=3, pretrained=False).to(device)
    model_stage1.load_state_dict(torch.load(output_dir / 'best_model_stage1.pth'))
    model_stage2.load_state_dict(torch.load(output_dir / 'best_model_stage2.pth'))
    model_stage1.eval()
    model_stage2.eval()
    
    # Ensemble prediction
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = model_stage1(inputs)
            outputs2 = model_stage2(inputs)
            outputs = (outputs1 + outputs2) / 2  # Average predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate final metrics
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=train_dataset.classes)
    
    logger.info("\nFinal Test Results:")
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save final confusion matrix
    plot_confusion_matrix(cm, train_dataset.classes, output_dir, 'final_ensemble')
    
    # Save detailed results
    with open(output_dir / 'final_results.txt', 'w') as f:
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
        f.write("\n\nPer-class Accuracy:\n")
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        for cls, acc in zip(train_dataset.classes, per_class_acc):
            f.write(f"{cls}: {acc:.4f}\n")

if __name__ == '__main__':
    main()
