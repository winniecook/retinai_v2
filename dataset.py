import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

class RetinalDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.classes = ['normal', 'cataract', 'glaucoma']
        
        # Load all samples
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob('*.png'):
                self.samples.append((str(img_path), class_idx))
        
        # Calculate class weights for balanced sampling
        labels = [s[1] for s in self.samples]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        self.class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image (9 channels: RGB + Vessel + HSV)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label

    def get_sampler(self):
        """Create a weighted sampler for balanced training."""
        sample_weights = [self.class_weights[label] for _, label in self.samples]
        return torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
