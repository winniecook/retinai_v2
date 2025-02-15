import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from skimage import filters, feature, morphology
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetinalPreprocessor:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        
    def extract_vessels(self, image):
        """Extract blood vessels using multi-scale line detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Multi-scale line detection
        scales = [1, 2, 3]
        vessel_response = np.zeros_like(enhanced, dtype=float)
        
        for scale in scales:
            sigma = scale
            hessian = feature.hessian_matrix(enhanced, sigma=sigma)
            eigenvals = feature.hessian_matrix_eigvals(hessian)
            vessel_response += filters.frangi(enhanced, sigmas=range(1, scale + 1))
        
        # Normalize and threshold
        vessel_response = (vessel_response - vessel_response.min()) / (vessel_response.max() - vessel_response.min())
        vessels = (vessel_response > filters.threshold_otsu(vessel_response)).astype(np.uint8) * 255
        
        # Clean up
        vessels = morphology.remove_small_objects(vessels.astype(bool), min_size=50)
        vessels = morphology.remove_small_holes(vessels, area_threshold=50)
        
        return vessels.astype(np.uint8)

    def enhance_image(self, image):
        """Apply multiple enhancement techniques."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge LAB channels
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Extract vessels
        vessels = self.extract_vessels(image)
        
        # Create multi-channel image
        hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
        
        # Stack different representations
        channels = [
            enhanced_rgb,  # Original enhanced RGB
            cv2.cvtColor(vessels, cv2.COLOR_GRAY2RGB),  # Vessel map
            hsv,  # HSV representation
        ]
        
        return np.concatenate(channels, axis=2)  # Stack along channel dimension

    def process_image(self, img_path):
        """Process a single image with advanced techniques."""
        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Basic preprocessing
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply enhancements
        enhanced = self.enhance_image(image)
        
        # Normalize
        enhanced = enhanced.astype(np.float32) / 255.0
        
        # Standardize each channel
        for i in range(enhanced.shape[2]):
            mean = enhanced[..., i].mean()
            std = enhanced[..., i].std()
            enhanced[..., i] = (enhanced[..., i] - mean) / (std + 1e-7)
        
        return enhanced

def process_and_split_data(
    src_dir='~/retinal_project2/data',
    dest_dir='~/retinal_efficientnet/processed_data',
    img_size=(224, 224),
    train_size=0.8,  
    val_size=0.1,    
    test_size=0.1,   
    num_workers=4,
    min_samples_per_class=100  
):
    """Process the dataset with parallel execution."""
    src_dir = Path(src_dir).expanduser()
    dest_dir = Path(dest_dir).expanduser()
    
    logger.info(f"Processing data from {src_dir} to {dest_dir}")
    
    # Create destination directories
    for split in ['train', 'val', 'test']:
        for cls in ['normal', 'cataract', 'glaucoma']:
            os.makedirs(dest_dir / split / cls, exist_ok=True)
    
    preprocessor = RetinalPreprocessor(output_size=img_size)
    
    def process_class(class_name):
        logger.info(f"Processing {class_name} images...")
        src_class_dir = src_dir / class_name
        images = list(src_class_dir.glob('*.png'))
        num_images = len(images)
        logger.info(f"Found {num_images} images in {class_name}")
        
        if num_images < min_samples_per_class:
            logger.warning(f"Warning: {class_name} has fewer than {min_samples_per_class} images")
        
        # Split data with fixed random state for reproducibility
        train_imgs, temp_imgs = train_test_split(
            images, train_size=train_size, random_state=42, shuffle=True
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.5, random_state=42, shuffle=True
        )
        
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        def process_and_save(img_path, split):
            try:
                processed = preprocessor.process_image(img_path)
                
                # Convert to PIL and save
                processed = (processed * 255).astype(np.uint8)
                processed = Image.fromarray(processed)
                
                dest_path = dest_dir / split / class_name / img_path.name
                processed.save(dest_path, 'PNG', optimize=True)
                return True
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                return False
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for split, imgs in splits.items():
                logger.info(f"Processing {split} split for {class_name}")
                futures = [
                    executor.submit(process_and_save, img_path, split)
                    for img_path in imgs
                ]
                
                # Track progress
                successful = sum(future.result() for future in tqdm(futures, desc=f"{split}-{class_name}"))
                logger.info(f"Successfully processed {successful}/{len(imgs)} images for {split}-{class_name}")
    
    # Process each class
    for cls in ['normal', 'cataract', 'glaucoma']:
        process_class(cls)
    
    # Print dataset statistics
    logger.info("\nDataset Statistics:")
    for split in ['train', 'val', 'test']:
        logger.info(f"\n{split.upper()}:")
        for cls in ['normal', 'cataract', 'glaucoma']:
            count = len(list((dest_dir / split / cls).glob('*.png')))
            logger.info(f"{cls}: {count} images")

if __name__ == '__main__':
    process_and_split_data()
