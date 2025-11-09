"""
fpgm_pruner.py

Filter Pruning via Geometric Median (FPGM) implementation.
Based on "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration"
(He et al., CVPR 2019)

The FPGM method prunes filters that are closest to the geometric median of all filters,
as these filters are considered redundant. The geometric median is the point that minimizes
the sum of Euclidean distances to all other points.

Key equation from the paper (Eq. 3 in Makenali et al.):
    GM = argmin_{x ∈ F} Σ ||x - f_i||_2
    where F is the set of all filters
"""

import torch
import torch.nn as nn
import numpy as np


class FPGMPruner:
    """
    FPGM (Filter Pruning via Geometric Median) pruner.
    
    This class implements the FPGM pruning strategy which identifies and prunes
    filters that are closest to the geometric median of all filters in a layer.
    
    The intuition is that filters near the geometric median are redundant and
    contribute less unique information to the network.
    
    Attributes:
        device (torch.device): Device to perform computations on
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize the FPGM pruner.
        
        Args:
            device (str or torch.device): Device for computations (default: 'cpu')
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
    
    def calculate_geometric_median(self, filter_tensor, max_iter=100, tol=1e-5):
        """
        Calculate the geometric median of a set of filters.
        
        The geometric median is the point that minimizes the sum of Euclidean distances
        to all other points. We use the practical approach from the FPGM paper:
        find the filter that minimizes the sum of distances to all other filters.
        
        This is based on Equation (3) from the Makenali paper:
            GM = argmin_{x ∈ F} Σ ||x - f_i||_2
        
        Args:
            filter_tensor (torch.Tensor): Flattened filters of shape [num_filters, filter_dim]
            max_iter (int): Maximum iterations for refinement (default: 100)
            tol (float): Tolerance for convergence (default: 1e-5)
        
        Returns:
            torch.Tensor: The geometric median (shape: [filter_dim])
        
        Algorithm:
            1. For each filter, calculate sum of distances to all other filters
            2. The filter with minimum sum is the approximate geometric median
            3. This is the discrete geometric median (exact for discrete sets)
        """
        num_filters = filter_tensor.shape[0]
        
        if num_filters == 0:
            raise ValueError("Cannot calculate geometric median of empty tensor")
        
        if num_filters == 1:
            return filter_tensor[0]
        
        # Calculate pairwise distances between all filters
        # For efficiency, we compute: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        # Shape: [num_filters, num_filters]
        distances = torch.cdist(filter_tensor, filter_tensor, p=2)
        
        # Sum of distances from each filter to all other filters
        # Shape: [num_filters]
        sum_distances = distances.sum(dim=1)
        
        # Find the filter with minimum sum of distances
        # This filter is the geometric median (discrete case)
        min_idx = torch.argmin(sum_distances)
        geometric_median = filter_tensor[min_idx]
        
        return geometric_median
    
    def get_pruning_mask(self, layer, prune_rate):
        """
        Generate a binary pruning mask for a convolutional layer using FPGM.
        
        The mask indicates which filters to keep (1) and which to prune (0).
        Filters closest to the geometric median are pruned as they are considered
        redundant.
        
        Args:
            layer (nn.Conv2d): Convolutional layer to prune
            prune_rate (float): Fraction of filters to prune (0.0 to 1.0)
        
        Returns:
            torch.Tensor: Binary mask of shape [num_filters] where:
                         1 = keep filter, 0 = prune filter
        
        Raises:
            ValueError: If layer is not nn.Conv2d or prune_rate is invalid
        
        Example:
            >>> pruner = FPGMPruner()
            >>> layer = nn.Conv2d(64, 128, 3)
            >>> mask = pruner.get_pruning_mask(layer, prune_rate=0.3)
            >>> # mask has shape [128], with 30% of values being 0
        """
        # Validate inputs
        if not isinstance(layer, nn.Conv2d):
            raise ValueError(f"Expected nn.Conv2d layer, got {type(layer)}")
        
        if not 0.0 <= prune_rate <= 1.0:
            raise ValueError(f"prune_rate must be in [0.0, 1.0], got {prune_rate}")
        
        # Get weight tensor from the layer
        # Shape: [out_channels, in_channels, kernel_h, kernel_w]
        weight = layer.weight.data
        num_filters = weight.shape[0]
        
        # Handle edge cases
        if num_filters == 0:
            return torch.ones(0, dtype=torch.bool, device=weight.device)
        
        if prune_rate == 0.0:
            return torch.ones(num_filters, dtype=torch.bool, device=weight.device)
        
        if prune_rate == 1.0:
            # Keep at least one filter
            mask = torch.zeros(num_filters, dtype=torch.bool, device=weight.device)
            mask[0] = 1
            return mask
        
        # Flatten each filter to a 1D vector
        # Shape: [num_filters, in_channels * kernel_h * kernel_w]
        flattened_filters = weight.view(num_filters, -1)
        
        # Move to computation device
        flattened_filters = flattened_filters.to(self.device)
        
        # Calculate the geometric median of all filters
        geometric_median = self.calculate_geometric_median(flattened_filters)
        
        # Calculate Euclidean distance of each filter from the geometric median
        # Shape: [num_filters]
        distances = torch.norm(flattened_filters - geometric_median, p=2, dim=1)
        
        # Calculate number of filters to prune
        num_to_prune = int(np.ceil(num_filters * prune_rate))
        
        # Ensure at least one filter remains
        num_to_prune = min(num_to_prune, num_filters - 1)
        
        # Get indices of filters to prune (those closest to geometric median)
        # These are the most redundant filters
        _, sorted_indices = torch.sort(distances)
        prune_indices = sorted_indices[:num_to_prune]
        
        # Create binary mask: 1 = keep, 0 = prune
        mask = torch.ones(num_filters, dtype=torch.bool, device=weight.device)
        mask[prune_indices] = 0
        
        return mask
    
    def apply_mask(self, layer, mask):
        """
        Apply a pruning mask to a convolutional layer by zeroing out pruned filters.
        
        This is a utility method to actually apply the pruning mask to the layer weights.
        Pruned filters (mask=0) will have all their weights set to zero.
        
        Args:
            layer (nn.Conv2d): Convolutional layer to prune
            mask (torch.Tensor): Binary mask of shape [num_filters]
        
        Example:
            >>> pruner = FPGMPruner()
            >>> mask = pruner.get_pruning_mask(layer, prune_rate=0.3)
            >>> pruner.apply_mask(layer, mask)
            >>> # Now 30% of filters in layer have all weights = 0
        """
        if not isinstance(layer, nn.Conv2d):
            raise ValueError(f"Expected nn.Conv2d layer, got {type(layer)}")
        
        # Expand mask to match weight dimensions
        # Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        # Mask shape: [out_channels] -> [out_channels, 1, 1, 1]
        expanded_mask = mask.view(-1, 1, 1, 1).float()
        
        # Apply mask by element-wise multiplication
        layer.weight.data *= expanded_mask.to(layer.weight.device)
        
        # Also zero out corresponding biases if they exist
        if layer.bias is not None:
            layer.bias.data *= mask.float().to(layer.bias.device)
    
    def get_pruning_info(self, layer, mask):
        """
        Get information about the pruning operation.
        
        Args:
            layer (nn.Conv2d): The pruned layer
            mask (torch.Tensor): The pruning mask
        
        Returns:
            dict: Dictionary containing pruning statistics
        """
        num_filters = mask.shape[0]
        num_pruned = (mask == 0).sum().item()
        num_kept = (mask == 1).sum().item()
        actual_prune_rate = num_pruned / num_filters if num_filters > 0 else 0.0
        
        # Calculate parameter reduction
        weight_shape = layer.weight.shape
        params_per_filter = np.prod(weight_shape[1:])  # in_channels * kernel_h * kernel_w
        
        if layer.bias is not None:
            params_per_filter += 1
        
        total_params = params_per_filter * num_filters
        pruned_params = params_per_filter * num_pruned
        remaining_params = params_per_filter * num_kept
        
        info = {
            'num_filters': num_filters,
            'num_pruned': num_pruned,
            'num_kept': num_kept,
            'prune_rate': actual_prune_rate,
            'total_params': total_params,
            'pruned_params': pruned_params,
            'remaining_params': remaining_params,
            'param_reduction': pruned_params / total_params if total_params > 0 else 0.0
        }
        
        return info


if __name__ == "__main__":
    """
    Test the FPGM pruner to verify functionality.
    """
    print("Testing FPGM Pruner...")
    print("-" * 50)
    
    # Create a sample convolutional layer
    layer = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    print(f"Test layer: Conv2d(32, 64, kernel_size=3)")
    print(f"Weight shape: {layer.weight.shape}")
    print()
    
    # Initialize pruner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pruner = FPGMPruner(device=device)
    print(f"Pruner device: {device}")
    print()
    
    # Test different pruning rates
    prune_rates = [0.0, 0.3, 0.5, 0.7]
    
    for prune_rate in prune_rates:
        print(f"Testing prune_rate = {prune_rate}")
        
        # Get pruning mask
        mask = pruner.get_pruning_mask(layer, prune_rate)
        
        # Get pruning info
        info = pruner.get_pruning_info(layer, mask)
        
        print(f"  Filters: {info['num_filters']}")
        print(f"  Pruned: {info['num_pruned']}")
        print(f"  Kept: {info['num_kept']}")
        print(f"  Actual prune rate: {info['prune_rate']:.3f}")
        print(f"  Parameter reduction: {info['param_reduction']:.3f}")
        
        # Verify mask properties
        assert mask.shape[0] == 64, "Mask shape mismatch"
        assert mask.dtype == torch.bool, "Mask should be boolean"
        assert info['num_pruned'] + info['num_kept'] == 64, "Mask count mismatch"
        print("  ✓ Mask verified")
        print()
    
    # Test mask application
    print("Testing mask application...")
    test_layer = nn.Conv2d(16, 32, kernel_size=3)
    original_weight = test_layer.weight.data.clone()
    
    mask = pruner.get_pruning_mask(test_layer, prune_rate=0.5)
    pruner.apply_mask(test_layer, mask)
    
    # Verify that pruned filters are zeroed
    for i, keep in enumerate(mask):
        if not keep:
            # This filter should be all zeros
            assert torch.all(test_layer.weight.data[i] == 0), f"Filter {i} not zeroed"
        else:
            # This filter should not be all zeros (assuming random init)
            assert not torch.all(test_layer.weight.data[i] == 0), f"Filter {i} incorrectly zeroed"
    
    print("  ✓ Mask application verified")
    print()
    
    # Test geometric median calculation
    print("Testing geometric median calculation...")
    # Create a simple 2D test case
    points = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]  # This should be close to the geometric median
    ])
    
    gm = pruner.calculate_geometric_median(points)
    print(f"  Test points shape: {points.shape}")
    print(f"  Geometric median: {gm.tolist()}")
    
    # The geometric median should be one of the input points (discrete case)
    distances = torch.cdist(gm.unsqueeze(0), points, p=2)
    min_dist = distances.min().item()
    assert min_dist < 1e-5, "Geometric median should be one of the input points"
    print("  ✓ Geometric median verified")
    print()
    
    print("✓ All FPGM pruner tests passed successfully!")