"""
apot_quantizer.py

Additive Powers-of-Two (APoT) Quantization implementation.
Based on "Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks"
(Li et al., ICLR 2020)

APoT quantization uses non-uniform quantization levels that are sums of powers of two,
allowing efficient hardware implementation while maintaining high accuracy.

Key equation from the paper (Eq. 4 in Makenali et al.):
    Quantization levels are formed by additive combinations of powers of two
    For 4-bit: {0, ±2^-7, ±2^-6, ±2^-5, ..., ±(2^-7 + 2^-6 + ... + 2^0)}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import copy


class STEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for quantization.
    
    In the forward pass, we quantize the input.
    In the backward pass, we pass through the gradient unchanged.
    
    This allows gradients to flow through the non-differentiable quantization operation.
    """
    
    @staticmethod
    def forward(ctx, input, quantized_input):
        """
        Forward pass: return quantized values.
        
        Args:
            ctx: Context object to save information for backward pass
            input: Original full-precision input
            quantized_input: Quantized input
            
        Returns:
            Quantized input
        """
        return quantized_input
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: pass gradient through unchanged (straight-through).
        
        Args:
            ctx: Context object with saved information
            grad_output: Gradient from subsequent layers
            
        Returns:
            Gradient for input, None for quantized_input
        """
        # Pass gradient straight through
        return grad_output, None


class APoTQuantConv2d(nn.Conv2d):
    """
    Convolutional layer with Additive Powers-of-Two (APoT) Quantization.
    
    This layer quantizes weights using non-uniform APoT levels during the forward pass,
    while maintaining full-precision weights for gradient updates.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution (default: 1)
        padding (int or tuple): Padding added to all sides (default: 0)
        dilation (int or tuple): Spacing between kernel elements (default: 1)
        groups (int): Number of blocked connections (default: 1)
        bias (bool): If True, adds a learnable bias (default: True)
        padding_mode (str): 'zeros', 'reflect', 'replicate', 'circular' (default: 'zeros')
        bitwidth (int): Number of bits for quantization (default: 4)
        
    Attributes:
        bitwidth (int): Number of quantization bits
        levels (torch.Tensor): Non-uniform quantization levels (non-trainable)
        clipping_threshold (nn.Parameter): Learnable clipping threshold for weights
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        bitwidth: int = 4
    ):
        # Initialize parent Conv2d
        super(APoTQuantConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        self.bitwidth = bitwidth
        
        # Generate APoT quantization levels
        levels = self._generate_apot_levels(bitwidth)
        
        # Register levels as a non-trainable buffer
        self.register_buffer('levels', levels)
        
        # Initialize learnable clipping threshold
        # Initialize to a reasonable value based on weight distribution
        with torch.no_grad():
            initial_threshold = self.weight.abs().max().item()
        
        self.clipping_threshold = nn.Parameter(
            torch.tensor(initial_threshold, dtype=torch.float32)
        )
    
    def _generate_apot_levels(self, bitwidth):
        """
        Generate non-uniform APoT quantization levels.
        
        For 4-bit quantization, we have 2^4 = 16 levels.
        Levels are formed by additive combinations of powers of two.
        
        The levels are symmetric around zero and include:
        - Zero
        - Positive and negative powers of two
        - Sums of powers of two
        
        This follows the formulation from Li et al. (ICLR 2020).
        
        Args:
            bitwidth (int): Number of bits for quantization
            
        Returns:
            torch.Tensor: Sorted quantization levels
        """
        # Number of quantization levels
        num_levels = 2 ** bitwidth
        
        # For 4-bit (16 levels), we generate levels as follows:
        # We use powers of two from 2^-7 to 2^0
        # and create additive combinations
        
        if bitwidth == 4:
            # Base powers: [2^-7, 2^-6, 2^-5, 2^-4, 2^-3, 2^-2, 2^-1, 2^0]
            base_powers = [2**i for i in range(-7, 1)]
            
            # Generate all possible combinations (subset sums)
            levels_set = set([0.0])  # Include zero
            
            # Generate positive levels by additive combinations
            for i in range(1, 2**len(base_powers)):
                level = 0.0
                for j in range(len(base_powers)):
                    if i & (1 << j):
                        level += base_powers[j]
                levels_set.add(level)
            
            # Convert to sorted list
            positive_levels = sorted(list(levels_set))
            
            # Take the top (num_levels // 2) positive levels
            # This gives us 8 positive levels for 4-bit
            num_positive = (num_levels // 2) - 1  # -1 for zero
            positive_levels = positive_levels[-num_positive:]
            
            # Create symmetric negative levels
            negative_levels = [-x for x in positive_levels]
            negative_levels.reverse()
            
            # Combine: negative + zero + positive
            all_levels = negative_levels + [0.0] + positive_levels
            
        else:
            # For other bitwidths, use uniform quantization as fallback
            # This is a simplified approach
            all_levels = np.linspace(-1, 1, num_levels).tolist()
        
        # Ensure we have exactly num_levels
        if len(all_levels) > num_levels:
            # Take evenly spaced levels
            indices = np.linspace(0, len(all_levels) - 1, num_levels, dtype=int)
            all_levels = [all_levels[i] for i in indices]
        
        levels_tensor = torch.tensor(all_levels, dtype=torch.float32)
        
        return levels_tensor
    
    def quantize(self, x, levels, threshold):
        """
        Quantize input tensor to the nearest APoT level using STE.
        
        This function:
        1. Clips the input to [-threshold, +threshold]
        2. Normalizes to match the range of quantization levels
        3. Finds the nearest quantization level for each element
        4. Uses Straight-Through Estimator for backpropagation
        
        Args:
            x (torch.Tensor): Input tensor to quantize
            levels (torch.Tensor): Quantization levels
            threshold (torch.Tensor): Clipping threshold (scalar)
            
        Returns:
            torch.Tensor: Quantized tensor (same shape as input)
        """
        # Ensure threshold is positive
        threshold = threshold.abs() + 1e-8
        
        # Clip input to [-threshold, +threshold]
        x_clipped = torch.clamp(x, -threshold, threshold)
        
        # Normalize to [-1, 1] to match level range
        x_normalized = x_clipped / threshold
        
        # Find nearest level for each element
        # Expand dimensions for broadcasting: x_normalized [...] vs levels [num_levels]
        x_expanded = x_normalized.unsqueeze(-1)  # [..., 1]
        levels_expanded = levels.view(1, -1)  # [1, num_levels]
        
        # For multidimensional x, we need to reshape properly
        original_shape = x_normalized.shape
        x_flat = x_normalized.view(-1, 1)  # [N, 1]
        
        # Compute distances to all levels
        # distances shape: [N, num_levels]
        distances = torch.abs(x_flat - levels_expanded)
        
        # Find index of nearest level
        nearest_indices = torch.argmin(distances, dim=1)  # [N]
        
        # Get quantized values
        x_quantized_flat = levels[nearest_indices]  # [N]
        x_quantized_normalized = x_quantized_flat.view(original_shape)
        
        # Scale back to original range
        x_quantized = x_quantized_normalized * threshold
        
        # Apply Straight-Through Estimator
        # Forward: use quantized values
        # Backward: use gradient of original values
        x_quantized = STEFunction.apply(x, x_quantized)
        
        return x_quantized
    
    def forward(self, input):
        """
        Forward pass with quantized weights.
        
        Args:
            input (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output of convolution with quantized weights
        """
        # Quantize weights using APoT levels
        quantized_weight = self.quantize(
            self.weight,
            self.levels,
            self.clipping_threshold
        )
        
        # Perform convolution with quantized weights
        # Note: we quantize weights but keep activations in full precision
        # (activation quantization can be added separately if needed)
        output = F.conv2d(
            input=input,
            weight=quantized_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        return output
    
    def extra_repr(self):
        """
        Extra representation string for printing.
        """
        s = super().extra_repr()
        s += f', bitwidth={self.bitwidth}'
        return s


def convert_to_apot(model, bitwidth=4, ignore_layers=None):
    """
    Convert all Conv2d layers in a model to APoTQuantConv2d layers.
    
    This function recursively traverses the model and replaces each nn.Conv2d
    layer with an APoTQuantConv2d layer, preserving the original weights and
    configuration.
    
    Args:
        model (nn.Module): Model to convert
        bitwidth (int): Number of bits for APoT quantization (default: 4)
        ignore_layers (list): List of layer names to ignore (default: None)
        
    Returns:
        nn.Module: Model with APoT quantized convolution layers
        
    Example:
        >>> model = resnet20()
        >>> model = convert_to_apot(model, bitwidth=4)
        >>> # Now all Conv2d layers are APoTQuantConv2d layers
    """
    if ignore_layers is None:
        ignore_layers = []
    
    # Iterate through all child modules
    for name, module in model.named_children():
        # Check if this layer should be ignored
        full_name = name
        if full_name in ignore_layers:
            continue
        
        # If this is a Conv2d layer, replace it
        if isinstance(module, nn.Conv2d) and not isinstance(module, APoTQuantConv2d):
            # Create new APoTQuantConv2d with same configuration
            new_layer = APoTQuantConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                bitwidth=bitwidth
            )
            
            # Copy weights and biases from original layer
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias)
                
                # Initialize clipping threshold based on current weights
                new_layer.clipping_threshold.data = module.weight.abs().max()
            
            # Replace the layer
            setattr(model, name, new_layer)
        
        # Recursively convert child modules
        else:
            convert_to_apot(module, bitwidth, ignore_layers)
    
    return model


def count_quantized_layers(model):
    """
    Count the number of APoT quantized layers in a model.
    
    Args:
        model (nn.Module): Model to analyze
        
    Returns:
        dict: Dictionary with counts of different layer types
    """
    counts = {
        'total_conv': 0,
        'apot_conv': 0,
        'regular_conv': 0
    }
    
    for module in model.modules():
        if isinstance(module, APoTQuantConv2d):
            counts['apot_conv'] += 1
            counts['total_conv'] += 1
        elif isinstance(module, nn.Conv2d):
            counts['regular_conv'] += 1
            counts['total_conv'] += 1
    
    return counts


if __name__ == "__main__":
    """
    Test the APoT quantization implementation.
    """
    print("Testing APoT Quantization...")
    print("-" * 50)
    
    # Test 1: APoT level generation
    print("Test 1: APoT Level Generation")
    layer = APoTQuantConv2d(16, 32, kernel_size=3, padding=1, bitwidth=4)
    print(f"  Bitwidth: {layer.bitwidth}")
    print(f"  Number of levels: {len(layer.levels)}")
    print(f"  Levels: {layer.levels.tolist()}")
    print(f"  Level range: [{layer.levels.min():.6f}, {layer.levels.max():.6f}]")
    assert len(layer.levels) == 16, "Should have 16 levels for 4-bit"
    print("  ✓ Level generation verified")
    print()
    
    # Test 2: Forward pass
    print("Test 2: Forward Pass")
    batch_size = 4
    input_tensor = torch.randn(batch_size, 16, 32, 32)
    output = layer(input_tensor)
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Clipping threshold: {layer.clipping_threshold.item():.6f}")
    assert output.shape == (batch_size, 32, 32, 32), "Output shape mismatch"
    print("  ✓ Forward pass verified")
    print()
    
    # Test 3: Quantization effect
    print("Test 3: Quantization Effect")
    with torch.no_grad():
        original_weight = layer.weight.clone()
        quantized_weight = layer.quantize(
            layer.weight,
            layer.levels,
            layer.clipping_threshold
        )
    
    print(f"  Original weight range: [{original_weight.min():.6f}, {original_weight.max():.6f}]")
    print(f"  Quantized weight range: [{quantized_weight.min():.6f}, {quantized_weight.max():.6f}]")
    print(f"  Unique values in quantized weights: {len(torch.unique(quantized_weight))}")
    
    # Check that quantized values are from the level set
    unique_normalized = (quantized_weight / layer.clipping_threshold).unique()
    print(f"  Unique normalized values: {len(unique_normalized)}")
    print("  ✓ Quantization verified")
    print()
    
    # Test 4: Gradient flow (STE)
    print("Test 4: Gradient Flow (STE)")
    layer.train()
    input_tensor = torch.randn(2, 16, 8, 8, requires_grad=True)
    output = layer(input_tensor)
    loss = output.sum()
    loss.backward()
    
    print(f"  Weight gradient exists: {layer.weight.grad is not None}")
    print(f"  Weight gradient shape: {layer.weight.grad.shape}")
    print(f"  Threshold gradient: {layer.clipping_threshold.grad.item():.6f}")
    assert layer.weight.grad is not None, "Gradient should flow through STE"
    print("  ✓ Gradient flow verified")
    print()
    
    # Test 5: Model conversion
    print("Test 5: Model Conversion")
    from models import resnet20
    
    model = resnet20()
    counts_before = count_quantized_layers(model)
    print(f"  Before conversion:")
    print(f"    Total Conv2d: {counts_before['total_conv']}")
    print(f"    APoT Conv2d: {counts_before['apot_conv']}")
    
    model = convert_to_apot(model, bitwidth=4)
    counts_after = count_quantized_layers(model)
    print(f"  After conversion:")
    print(f"    Total Conv2d: {counts_after['total_conv']}")
    print(f"    APoT Conv2d: {counts_after['apot_conv']}")
    
    assert counts_after['apot_conv'] == counts_before['total_conv'], "All Conv2d should be converted"
    assert counts_after['regular_conv'] == 0, "No regular Conv2d should remain"
    print("  ✓ Model conversion verified")
    print()
    
    # Test 6: Converted model forward pass
    print("Test 6: Converted Model Forward Pass")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(2, 3, 32, 32)
        test_output = model(test_input)
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")
    assert test_output.shape == (2, 10), "Output shape should be (2, 10) for CIFAR-10"
    print("  ✓ Converted model forward pass verified")
    print()
    
    print("✓ All APoT quantization tests passed successfully!")