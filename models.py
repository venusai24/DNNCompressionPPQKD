"""
models.py

ResNet-20 implementation for CIFAR-10 dataset.
This follows the original ResNet paper for CIFAR-10 (He et al., 2016).

ResNet-20 uses the following structure:
- Initial 3x3 conv layer (16 filters)
- 3 stages with [16, 32, 64] filters
- Each stage has 3 residual blocks (n=3 for ResNet-20: 6n+2 = 20 layers)
- Global average pooling
- Fully connected layer for 10 classes

Architecture: 1 + 3*3*2 + 1 = 20 layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-20.
    Uses two 3x3 convolutions with batch normalization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for the first convolution (used for downsampling)
    """
    expansion = 1  # BasicBlock doesn't expand channels
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        
        # If dimensions change, use 1x1 convolution to match dimensions
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through the basic block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after residual connection
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet implementation for CIFAR-10.
    
    Args:
        block (nn.Module): The residual block type (BasicBlock for ResNet-20)
        num_blocks (list): Number of blocks in each stage
        num_classes (int): Number of output classes (10 for CIFAR-10)
    """
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        
        self.in_channels = 16
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(
            3,  # RGB input
            16, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        
        # Three stages with different feature map sizes
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Create a stage with multiple residual blocks.
        
        Args:
            block (nn.Module): Block type to use
            out_channels (int): Number of output channels for this stage
            num_blocks (int): Number of blocks in this stage
            stride (int): Stride for the first block (for downsampling)
            
        Returns:
            nn.Sequential: A sequential container of blocks
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        This follows the initialization scheme from the original ResNet paper.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Initial convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Three stages of residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global average pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # Classifier
        out = self.fc(out)
        
        return out


def resnet20(num_classes=10):
    """
    Constructs a ResNet-20 model for CIFAR-10.
    
    ResNet-20 has 3 blocks per stage (n=3), giving:
    1 + 3*2*3 + 1 = 20 convolutional layers
    
    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        
    Returns:
        ResNet: A ResNet-20 model
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    """
    Constructs a ResNet-32 model for CIFAR-10.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        
    Returns:
        ResNet: A ResNet-32 model
    """
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet56(num_classes=10):
    """
    Constructs a ResNet-56 model for CIFAR-10.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        
    Returns:
        ResNet: A ResNet-56 model
    """
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


if __name__ == "__main__":
    """
    Test the ResNet-20 model to verify architecture.
    """
    # Create model
    model = resnet20(num_classes=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"ResNet-20 for CIFAR-10")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 10)")
    
    assert output.shape == (batch_size, 10), "Output shape mismatch!"
    print("\n✓ Model architecture verified successfully!")