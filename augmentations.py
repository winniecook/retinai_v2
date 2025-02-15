import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MixUpCutMix:
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        
    def mixup(self, x, y):
        """Performs mixup on the input sample and accompanying label."""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix(self, x, y):
        """Performs cutmix on the input sample and accompanying label."""
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y_a, y_b, lam

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, x, y):
        if np.random.rand() > self.prob:
            return x, y, y, 1.

        if np.random.rand() > self.switch_prob:
            return self.mixup(x, y)
        else:
            return self.cutmix(x, y)

def get_train_transforms(height=224, width=224):
    return A.Compose([
        A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.ElasticTransform(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.MultiplicativeNoise(p=0.5),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406] * 3,  # Repeat for each channel group
            std=[0.229, 0.224, 0.225] * 3,
        ),
        ToTensorV2(),
    ])

def get_valid_transforms(height=224, width=224):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406] * 3,  # Repeat for each channel group
            std=[0.229, 0.224, 0.225] * 3,
        ),
        ToTensorV2(),
    ])
