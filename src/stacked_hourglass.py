import torch
import torch.nn as nn
from hourglass import HourglassBlock

class HeatmapHead(nn.Module):
    """Head for predicting heatmaps"""
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, num_keypoints, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

class StackedHourglass(nn.Module):
    def __init__(self, in_channels=3, num_keypoints=5, num_stacks=2, num_channels=256, depth=4):
        """
        Stacked Hourglass Network for keypoint detection
        
        Args:
            in_channels (int): Number of input channels (default: 3 for RGB)
            num_keypoints (int): Number of keypoints to detect (default: 5 for face landmarks)
            num_stacks (int): Number of stacked hourglass blocks (default: 2)
            num_channels (int): Number of channels in the network (default: 256)
            depth (int): Depth of each hourglass block (default: 4)
        """
        super().__init__()
        
        self.num_stacks = num_stacks
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks for initial feature extraction
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        
        # Create stacked hourglass blocks
        self.hourglass_blocks = nn.ModuleList([
            HourglassBlock(num_channels, num_channels, depth)
            for _ in range(num_stacks)
        ])
        
        # Create heatmap heads for each stack
        self.heatmap_heads = nn.ModuleList([
            HeatmapHead(num_channels, num_keypoints)
            for _ in range(num_stacks)
        ])
        
        # Create intermediate feature processing
        self.inter_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, 1),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_stacks - 1)
        ])
        
        # Create heatmap processing
        self.heatmap_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_keypoints, num_channels, 1),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_stacks - 1)
        ])

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            list: List of heatmap predictions from each stack
        """
        # Initial feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        
        # Store predictions from each stack
        predictions = []
        
        # Process through each stack
        for i in range(self.num_stacks):
            # Hourglass block
            hg_out = self.hourglass_blocks[i](x)
            
            # Generate heatmap prediction
            heatmap = self.heatmap_heads[i](hg_out)
            predictions.append(heatmap)
            
            # If not the last stack, process features for next stack
            if i < self.num_stacks - 1:
                # Process hourglass output
                hg_features = self.inter_features[i](hg_out)
                
                # Process heatmap features
                heatmap_features = self.heatmap_features[i](heatmap)
                
                # Combine features for next stack
                x = hg_features + heatmap_features + x
        
        return predictions

    def get_loss(self, predictions, targets):
        """
        Calculate MSE loss for all stacks
        
        Args:
            predictions (list): List of heatmap predictions from each stack
            targets (torch.Tensor): Target heatmaps
            
        Returns:
            torch.Tensor: Total loss
        """
        total_loss = 0
        for pred in predictions:
            total_loss += nn.MSELoss()(pred, targets)
        return total_loss 