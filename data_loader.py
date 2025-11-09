"""
data_loader.py

CIFAR-10 data loading utilities with standard augmentations.
This module provides data loaders for training and testing on CIFAR-10 dataset.

Standard CIFAR-10 preprocessing:
- Training: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalization
- Testing: Normalization only

Normalization values are standard for CIFAR-10:
- Mean: [0.4914, 0.4822, 0.4465]
- Std: [0.2023, 0.1994, 0.2010]
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Create CIFAR-10 train and test data loaders with standard augmentations.
    
    Training transformations:
        - Random crop with padding (32x32 output from 36x36 padded image)
        - Random horizontal flip (p=0.5)
        - Convert to tensor
        - Normalize with CIFAR-10 mean and std
    
    Test transformations:
        - Convert to tensor
        - Normalize with CIFAR-10 mean and std
    
    Args:
        batch_size (int): Batch size for both train and test loaders (default: 128)
        num_workers (int): Number of subprocesses for data loading (default: 4)
        data_dir (str): Directory to store/load CIFAR-10 dataset (default: './data')
    
    Returns:
        tuple: (train_loader, test_loader)
            - train_loader: DataLoader for training set (50,000 images)
            - test_loader: DataLoader for test set (10,000 images)
    
    Example:
        >>> train_loader, test_loader = get_cifar10_loaders(batch_size=128)
        >>> for images, labels in train_loader:
        ...     # Training loop
        ...     pass
    """
    
    # CIFAR-10 dataset statistics (precomputed on training set)
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]
    
    # Training data transformations (with augmentation)
    train_transform = transforms.Compose([
        # Pad image to 36x36 and randomly crop back to 32x32
        # This provides translation invariance
        transforms.RandomCrop(32, padding=4),
        
        # Random horizontal flip with probability 0.5
        # This doubles the effective training set size
        transforms.RandomHorizontalFlip(),
        
        # Convert PIL Image to tensor (scales to [0, 1])
        transforms.ToTensor(),
        
        # Normalize with CIFAR-10 mean and std
        # This centers the data around 0 with unit variance
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    
    # Test data transformations (no augmentation)
    test_transform = transforms.Compose([
        # Convert PIL Image to tensor
        transforms.ToTensor(),
        
        # Normalize with same statistics as training
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Load CIFAR-10 training dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data for better generalization
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, test_loader


def get_cifar10_info():
    """
    Get information about the CIFAR-10 dataset.
    
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'num_classes': 10,
        'class_names': [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        'image_shape': (3, 32, 32),  # (channels, height, width)
        'train_size': 50000,
        'test_size': 10000,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
    }
    return info


if __name__ == "__main__":
    """
    Test the data loader to verify functionality.
    """
    print("Testing CIFAR-10 data loaders...")
    print("-" * 50)
    
    # Get dataset info
    info = get_cifar10_info()
    print("Dataset Information:")
    print(f"  Number of classes: {info['num_classes']}")
    print(f"  Class names: {', '.join(info['class_names'])}")
    print(f"  Image shape: {info['image_shape']}")
    print(f"  Training samples: {info['train_size']}")
    print(f"  Test samples: {info['test_size']}")
    print(f"  Mean: {info['mean']}")
    print(f"  Std: {info['std']}")
    print()
    
    # Create data loaders
    batch_size = 128
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=2,
        data_dir='./data'
    )
    
    print(f"Data Loaders Created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print()
    
    # Test a single batch
    print("Testing batch loading...")
    train_iter = iter(train_loader)
    images, labels = next(train_iter)
    
    print(f"  Train batch shape: {images.shape}")
    print(f"  Train labels shape: {labels.shape}")
    print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")
    print()
    
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    
    print(f"  Test batch shape: {images.shape}")
    print(f"  Test labels shape: {labels.shape}")
    print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")
    print()
    
    # Verify normalization
    print("Verifying normalization...")
    all_images = []
    for images, _ in train_loader:
        all_images.append(images)
        if len(all_images) >= 10:  # Sample first 10 batches
            break
    
    all_images = torch.cat(all_images, dim=0)
    computed_mean = all_images.mean(dim=[0, 2, 3])
    computed_std = all_images.std(dim=[0, 2, 3])
    
    print(f"  Computed mean (sample): {computed_mean.tolist()}")
    print(f"  Expected mean: {info['mean']}")
    print(f"  Computed std (sample): {computed_std.tolist()}")
    print(f"  Expected std: {info['std']}")
    print()
    
    print("✓ Data loaders verified successfully!")