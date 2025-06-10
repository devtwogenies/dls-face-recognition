import torch
import torch.nn as nn
import pytest
from Hourglass import ResidualBlock, HourglassBlock

def test_residual_block_same_channels():
    """Test ResidualBlock with same input and output channels"""
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    
    # Create input tensor
    x = torch.randn(batch_size, channels, height, width)
    
    # Create residual block
    block = ResidualBlock(channels, channels)
    
    # Forward pass
    output = block(x)
    
    # Check output shape
    assert output.shape == (batch_size, channels, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_residual_block_different_channels():
    """Test ResidualBlock with different input and output channels"""
    batch_size = 2
    in_channels = 32
    out_channels = 64
    height = 32
    width = 32
    
    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create residual block
    block = ResidualBlock(in_channels, out_channels)
    
    # Forward pass
    output = block(x)
    
    # Check output shape
    assert output.shape == (batch_size, out_channels, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_residual_block_gradients():
    """Test if gradients flow through ResidualBlock"""
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    
    # Create input tensor with requires_grad=True
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    
    # Create residual block
    block = ResidualBlock(channels, channels)
    
    # Forward pass
    output = block(x)
    
    # Compute loss and backward pass
    loss = output.mean()
    loss.backward()
    
    # Check if gradients were computed
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_hourglass_block_output_shape():
    """Test if HourglassBlock maintains input shape"""
    batch_size = 2
    channels = 64
    height = 64
    width = 64
    depth = 4
    
    # Create input tensor
    x = torch.randn(batch_size, channels, height, width)
    
    # Create hourglass block
    block = HourglassBlock(channels, channels, depth)
    
    # Forward pass
    output = block(x)
    
    # Check output shape matches input shape
    assert output.shape == (batch_size, channels, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_hourglass_block_different_channels():
    """Test HourglassBlock with different input and output channels"""
    batch_size = 2
    in_channels = 32
    out_channels = 64
    height = 64
    width = 64
    depth = 4
    
    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create hourglass block
    block = HourglassBlock(in_channels, out_channels, depth)
    
    # Forward pass
    output = block(x)
    
    # Check output shape
    assert output.shape == (batch_size, out_channels, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_hourglass_block_gradients():
    """Test if gradients flow through HourglassBlock"""
    batch_size = 2
    channels = 64
    height = 64
    width = 64
    depth = 4
    
    # Create input tensor with requires_grad=True
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)
    
    # Create hourglass block
    block = HourglassBlock(channels, channels, depth)
    
    # Forward pass
    output = block(x)
    
    # Compute loss and backward pass
    loss = output.mean()
    loss.backward()
    
    # Check if gradients were computed
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_hourglass_block_different_depths():
    """Test HourglassBlock with different depths"""
    batch_size = 2
    channels = 64
    height = 64
    width = 64
    
    # Test different depths
    for depth in [2, 3, 4]:
        # Create input tensor
        x = torch.randn(batch_size, channels, height, width)
        
        # Create hourglass block
        block = HourglassBlock(channels, channels, depth)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        assert output.shape == (batch_size, channels, height, width)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

def test_hourglass_block_skip_connections():
    """Test if skip connections are working in HourglassBlock"""
    batch_size = 2
    channels = 64
    height = 64
    width = 64
    depth = 4
    
    # Create input tensor
    x = torch.randn(batch_size, channels, height, width)
    
    # Create hourglass block
    block = HourglassBlock(channels, channels, depth)
    
    # Forward pass
    output = block(x)
    
    # Check if output is different from input (skip connections should modify the features)
    assert not torch.allclose(output, x)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

if __name__ == '__main__':
    pytest.main([__file__]) 