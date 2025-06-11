import torch
import torch.nn as nn
import pytest
from stacked_hourglass import StackedHourglass

def test_stacked_hourglass_initialization():
    """Test StackedHourglass initialization with different parameters"""
    # Test default parameters
    model = StackedHourglass()
    assert model.num_stacks == 2
    assert isinstance(model.hourglass_blocks, nn.ModuleList)
    assert len(model.hourglass_blocks) == 2
    assert len(model.heatmap_heads) == 2
    
    # Test custom parameters
    model = StackedHourglass(
        in_channels=1,
        num_keypoints=10,
        num_stacks=3,
        num_channels=128,
        depth=3
    )
    assert model.num_stacks == 3
    assert len(model.hourglass_blocks) == 3
    assert len(model.heatmap_heads) == 3

def test_stacked_hourglass_forward_shape():
    """Test if StackedHourglass maintains correct output shapes"""
    batch_size = 2
    in_channels = 3
    num_keypoints = 5
    height = 256
    width = 256
    
    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create model
    model = StackedHourglass(
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        num_stacks=2
    )
    
    # Forward pass
    predictions = model(x)
    
    # Check predictions
    assert isinstance(predictions, list)
    assert len(predictions) == 2  # num_stacks
    
    # Check each prediction shape
    # Note: height and width are halved due to initial stride=2 convolution
    expected_height = height // 2
    expected_width = width // 2
    for pred in predictions:
        assert pred.shape == (batch_size, num_keypoints, expected_height, expected_width)
        assert not torch.isnan(pred).any()
        assert not torch.isinf(pred).any()

def test_stacked_hourglass_gradients():
    """Test if gradients flow through StackedHourglass"""
    batch_size = 2
    in_channels = 3
    num_keypoints = 5
    height = 256
    width = 256
    
    # Create input tensor with requires_grad=True
    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    
    # Create model
    model = StackedHourglass(
        in_channels=in_channels,
        num_keypoints=num_keypoints
    )
    
    # Forward pass
    predictions = model(x)
    
    # Compute loss and backward pass
    target = torch.randn_like(predictions[0])
    loss = model.get_loss(predictions, target)
    loss.backward()
    
    # Check if gradients were computed
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_stacked_hourglass_loss():
    """Test StackedHourglass loss calculation"""
    batch_size = 2
    in_channels = 3
    num_keypoints = 5
    height = 256
    width = 256
    
    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create model
    model = StackedHourglass(
        in_channels=in_channels,
        num_keypoints=num_keypoints
    )
    
    # Forward pass
    predictions = model(x)
    
    # Create target tensor
    target = torch.randn_like(predictions[0])
    
    # Calculate loss
    loss = model.get_loss(predictions, target)
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss >= 0

def test_stacked_hourglass_different_stacks():
    """Test StackedHourglass with different number of stacks"""
    batch_size = 2
    in_channels = 3
    num_keypoints = 5
    height = 256
    width = 256
    
    # Test different number of stacks
    for num_stacks in [1, 2, 3]:
        # Create input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Create model
        model = StackedHourglass(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            num_stacks=num_stacks
        )
        
        # Forward pass
        predictions = model(x)
        
        # Check predictions
        assert len(predictions) == num_stacks
        for pred in predictions:
            assert pred.shape == (batch_size, num_keypoints, height//2, width//2)
            assert not torch.isnan(pred).any()
            assert not torch.isinf(pred).any()

def test_stacked_hourglass_feature_processing():
    """Test if feature processing between stacks works correctly"""
    batch_size = 2
    in_channels = 3
    num_keypoints = 5
    height = 256
    width = 256
    num_stacks = 2
    
    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create model
    model = StackedHourglass(
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        num_stacks=num_stacks
    )
    
    # Forward pass
    predictions = model(x)
    
    # Check that predictions from different stacks are different
    # (indicating that feature processing is working)
    assert not torch.allclose(predictions[0], predictions[1])

def test_stacked_hourglass_intermediate_supervision():
    """Test if intermediate supervision is working correctly"""
    batch_size = 2
    in_channels = 3
    num_keypoints = 5
    height = 256
    width = 256
    num_stacks = 2
    
    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create model
    model = StackedHourglass(
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        num_stacks=num_stacks
    )
    
    # Forward pass
    predictions = model(x)
    
    # Create target tensor
    target = torch.randn_like(predictions[0])
    
    # Calculate loss
    loss = model.get_loss(predictions, target)
    
    # Check that loss is computed for all stacks
    assert isinstance(loss, torch.Tensor)
    assert loss > 0

if __name__ == '__main__':
    pytest.main([__file__]) 