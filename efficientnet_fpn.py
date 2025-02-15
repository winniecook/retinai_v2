import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        self.attention = nn.ModuleList([
            CBAM(out_channels) for _ in in_channels_list
        ])
        
    def forward(self, features):
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]
        
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
        
        outs = [
            attention(conv(lateral))
            for lateral, conv, attention in zip(laterals, self.fpn_convs, self.attention)
        ]
        
        return outs

class EfficientNetFPN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNetV2
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=pretrained, features_only=True)
        
        # Get feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        features = self.backbone(dummy_input)
        in_channels_list = [feat.shape[1] for feat in features]
        
        # FPN
        self.fpn = FPN(in_channels_list, out_channels=256)
        
        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Multiple classification heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            ) for _ in range(len(in_channels_list))
        ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # FPN
        fpn_features = self.fpn(features)
        
        # Multiple predictions
        predictions = []
        for feat, classifier in zip(fpn_features, self.classifiers):
            feat = self.pool(feat).view(feat.size(0), -1)
            pred = classifier(feat)
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(0)

def create_model(num_classes=3, pretrained=True):
    return EfficientNetFPN(num_classes=num_classes, pretrained=pretrained)
