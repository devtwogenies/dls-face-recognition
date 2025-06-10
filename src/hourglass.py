import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)

class HourglassBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4):
        super().__init__()
        self.depth = depth
        
        # Downsampling path
        self.down_path = nn.ModuleList()
        # Upsampling path
        self.up_path = nn.ModuleList()
        # Residual connections
        self.residual = nn.ModuleList()
        
        # Create the hourglass structure
        for i in range(depth):
            # Downsampling
            self.down_path.append(
                nn.Sequential(
                    ResidualBlock(in_channels if i == 0 else out_channels, out_channels),
                    nn.MaxPool2d(2)
                )
            )
            
            # Upsampling
            self.up_path.append(
                nn.Sequential(
                    ResidualBlock(out_channels, out_channels),
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
            )
            
            # Residual connections
            self.residual.append(
                ResidualBlock(in_channels if i == 0 else out_channels, out_channels)
            )
        
        # Bottom of the hourglass
        self.bottom = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        # Store intermediate features for skip connections
        features = []
        
        # Downsampling path
        for i in range(self.depth):
            x = self.down_path[i](x)
            features.append(x)
        
        # Bottom of the hourglass
        x = self.bottom(x)
        
        # Upsampling path with skip connections
        for i in range(self.depth-1, -1, -1):
            x = self.up_path[i](x)
            # Add residual connection
            x = x + self.residual[i](features[i])
        
        return x 