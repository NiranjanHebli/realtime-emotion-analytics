import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from logger_config import setup_logger

logger = setup_logger(__name__)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # Spatial attention considers max and average pooling across the channel dimension
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(attn))

class EmotionCNN_Attention(nn.Module):
    def __init__(self, num_classes=4):
        super(EmotionCNN_Attention, self).__init__()
        # Input size: 1x48x48
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Unique feature: Attention mechanism
        self.attention = SpatialAttention()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After two MaxPool2d (48 -> 24 -> 12), the spatial dimension is 12x12
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 32x24x24
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 64x12x12
        x = self.attention(x) # Apply attention on feature maps
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test model instantiation and tensor shapes
    model = EmotionCNN_Attention(num_classes=4)
    dummy_input = torch.randn(8, 1, 48, 48) # Batch size 8, 1 channel, 48x48
    output = model(dummy_input)
    logger.info(f"Output shape: {output.shape}") # Expected: [8, 4]
