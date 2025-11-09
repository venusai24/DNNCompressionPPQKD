"""
train_spq.py

Simultaneous Pruning and Quantization (SPQ) training script.
This implements Algorithm 1 from "Integrating Pruning with Quantization for 
Efficient Deep Neural Networks Compression" (Makenali et al., 2025).

SPQ Pipeline:
1. Convert model to APoT quantization immediately
2. Train with quantization-aware training
3. Apply FPGM pruning at the end of each epoch (on full-precision latent weights)
4. Key difference from PPQ: Pruning and quantization happen simultaneously

This follows the order: Quantize → (Train + Prune iteratively)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import os
import argparse
from tqdm import tqdm
import time

# Import custom modules
from models import resnet20
from data_loader import get_cifar10_loaders
from fpgm_pruner import FPGMPruner
from apot_quantizer import convert_to_apot, count_quantized_layers, APoTQuantConv2d


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def get_apot_conv_layers(model):
    """
    Get all APoTQuantConv2d layers from the model.
    
    Args:
        model (nn.Module): Model to extract layers from
        
    Returns:
        list: List of (name, layer) tuples for all APoTQuantConv2d layers
    """
    apot_layers = []
    for name, module in model.named_modules():
        if isinstance(module, APoTQuantConv2d):
            apot_layers.append((name, module))
    return apot_layers


def apply_pruning_masks_spq(model, masks):
    """
    Apply pruning masks to APoT quantized layers for SPQ.
    
    In SPQ, masks are applied to the forward pass but gradients flow to all weights.
    This is different from PPQ where pruned weights have zero gradients.
    
    The mask is applied by zeroing out the full-precision latent weights,
    which will then be quantized in the forward pass.
    
    Args:
        model (nn.Module): Model with APoT layers
        masks (dict): Dictionary mapping layer names to masks
    """
    for name, module in model.named_modules():
        if name in masks and isinstance(module, APoTQuantConv2d):
            mask = masks[name]
            # Expand mask to match weight dimensions
            expanded_mask = mask.view(-1, 1, 1, 1).float().to(module.weight.device)
            
            # Apply mask to latent full-precision weights
            # This zeros out pruned filters for the forward pass
            module.weight.data *= expanded_mask
            
            # Also zero out bias for pruned filters
            if module.bias is not None:
                module.bias.data *= mask.float().to(module.bias.device)
            
            # NOTE: In SPQ, we do NOT register gradient hooks
            # All weights (including pruned ones) receive gradient updates
            # The mask is re-applied after each epoch based on new weight values


def compute_pruning_masks_spq(model, pruner, prune_rate, epoch):
    """
    Compute FPGM pruning masks for all APoT layers based on latent weights.
    
    This implements Algorithm 1, Lines 11-17.
    Critically, the geometric median is calculated on the full-precision
    latent weights (not the quantized weights).
    
    Args:
        model (nn.Module): Model with APoT layers
        pruner (FPGMPruner): FPGM pruner instance
        prune_rate (float): Fraction of filters to prune
        epoch (int): Current epoch number (for logging)
        
    Returns:
        dict: Dictionary mapping layer names to pruning masks
    """
    masks = {}
    
    print(f"\nComputing FPGM masks for epoch {epoch + 1} (prune_rate={prune_rate:.2%})...")
    
    # Get all APoT quantized layers
    apot_layers = get_apot_conv_layers(model)
    
    for name, layer in apot_layers:
        # Algorithm 1, Line 13: Calculate GM on full-precision latent weights
        # The layer.weight contains the full-precision latent weights
        # (not the quantized weights used in forward pass)
        mask = pruner.get_pruning_mask(layer, prune_rate)
        masks[name] = mask
        
        # Get pruning info
        info = pruner.get_pruning_info(layer, mask)
        print(f"  {name}: {info['num_kept']}/{info['num_filters']} filters kept "
              f"({info['prune_rate']:.2%} pruned)")
    
    return masks


def count_parameters(model):
    """
    Count total and non-zero parameters in the model.
    
    Args:
        model (nn.Module): Model to count parameters
        
    Returns:
        tuple: (total_params, non_zero_params)
    """
    total_params = 0
    non_zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        non_zero_params += torch.count_nonzero(param).item()
    
    return total_params, non_zero_params


def train_spq_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                    pruner, prune_rate, apply_pruning=True):
    """
    Train one epoch with SPQ (Simultaneous Pruning and Quantization).
    
    This implements Algorithm 1, Lines 2-17.
    
    Args:
        model (nn.Module): Model with APoT quantization
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch (int): Current epoch number
        pruner (FPGMPruner): FPGM pruner instance
        prune_rate (float): Fraction of filters to prune
        apply_pruning (bool): Whether to apply pruning at end of epoch
        
    Returns:
        tuple: (average_loss, accuracy, masks)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training iteration loop (Algorithm 1, Lines 2-10)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [Train]')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass (Algorithm 1, Lines 4-5)
        # APoTQuantConv2d layers automatically quantize weights during forward
        # The quantized weights are used for convolution
        # But the full-precision latent weights are maintained for updates
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass (Algorithm 1, Lines 6-8)
        # Gradients flow through STE to full-precision latent weights
        loss.backward()
        
        # Update latent weights (Algorithm 1, Line 8)
        # ALL weights are updated (including those that will be pruned)
        # This is the key difference from PPQ
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # End-of-epoch pruning (Algorithm 1, Lines 11-17)
    masks = None
    if apply_pruning:
        # Compute pruning masks based on current latent weights
        # Algorithm 1, Line 13: GM is calculated on full-precision weights
        masks = compute_pruning_masks_spq(model, pruner, prune_rate, epoch)
        
        # Apply masks to latent weights for next epoch (Algorithm 1, Line 15)
        apply_pruning_masks_spq(model, masks)
        
        # Calculate sparsity
        total_params, non_zero_params = count_parameters(model)
        sparsity = 1.0 - (non_zero_params / total_params)
        print(f"Sparsity after epoch {epoch + 1}: {sparsity:.2%} "
              f"({non_zero_params:,}/{total_params:,} params remaining)")
    
    return avg_loss, accuracy, masks


def train_spq(args):
    """
    Main SPQ training function.
    
    This implements the complete Algorithm 1 from the Makenali paper.
    
    Args:
        args: Command-line arguments
        
    Returns:
        float: Final test accuracy
    """
    print("\n" + "="*80)
    print("SPQ (SIMULTANEOUS PRUNING AND QUANTIZATION) TRAINING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create baseline model
    print("\nCreating ResNet-20 model...")
    model = resnet20(num_classes=10)
    
    total_params_before, _ = count_parameters(model)
    print(f"Total parameters before quantization: {total_params_before:,}")
    
    # Convert to APoT quantization immediately (before training)
    # This is the key difference from PPQ
    print(f"\nConverting to APoT quantization (bitwidth={args.bitwidth})...")
    model = convert_to_apot(model, bitwidth=args.bitwidth)
    model = model.to(device)
    
    # Verify conversion
    counts = count_quantized_layers(model)
    print(f"Converted {counts['apot_conv']} Conv2d layers to APoT quantization")
    
    # Count parameters after quantization
    total_params_after, _ = count_parameters(model)
    print(f"Total parameters after quantization: {total_params_after:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    print(f"\nLR schedule: milestones={milestones}, gamma=0.1")
    
    # Initialize FPGM pruner
    pruner = FPGMPruner(device=device)
    
    # Get all APoT layers
    apot_layers = get_apot_conv_layers(model)
    print(f"Found {len(apot_layers)} APoT quantized layers to prune")
    
    print(f"\nTraining configuration:")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Prune rate per epoch: {args.prune_rate:.2%}")
    print(f"  Bitwidth: {args.bitwidth}")
    print(f"  Initial LR: {args.lr}")
    
    # Training loop (Algorithm 1, Line 1: for epoch = 1 to n)
    best_acc = 0.0
    start_time = time.time()
    final_masks = None
    
    print(f"\nStarting SPQ training for {args.epochs} epochs...")
    print("="*80)
    
    for epoch in range(args.epochs):
        # Train one epoch with simultaneous pruning and quantization
        # Algorithm 1, Lines 2-17
        train_loss, train_acc, masks = train_spq_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            pruner=pruner,
            prune_rate=args.prune_rate,
            apply_pruning=True  # Apply pruning at end of each epoch
        )
        
        # Store masks from this epoch
        if masks is not None:
            final_masks = masks
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'masks': final_masks,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'bitwidth': args.bitwidth,
                'prune_rate': args.prune_rate
            }, os.path.join(args.save_dir, 'spq_compressed_model.pth'))
        
        # Print results
        total_params, non_zero_params = count_parameters(model)
        sparsity = 1.0 - (non_zero_params / total_params)
        
        print(f'\nEpoch {epoch + 1}/{args.epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | '
              f'Best: {best_acc:.2f}%')
        print(f'Sparsity: {sparsity:.2%} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print("-"*80)
    
    elapsed_time = time.time() - start_time
    
    # Load best model for final evaluation
    print(f"\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'spq_compressed_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    _, final_acc = evaluate(model, test_loader, criterion, device)
    
    # Final statistics
    total_params, non_zero_params = count_parameters(model)
    final_sparsity = 1.0 - (non_zero_params / total_params)
    
    print("\n" + "="*80)
    print("SPQ TRAINING COMPLETED")
    print("="*80)
    print(f"Training time: {elapsed_time/60:.2f} minutes")
    print(f"Final test accuracy: {final_acc:.2f}%")
    print(f"Best test accuracy: {checkpoint['best_acc']:.2f}%")
    print(f"Final sparsity: {final_sparsity:.2%}")
    print(f"Parameters: {non_zero_params:,}/{total_params:,} remaining")
    print(f"Model saved to: {os.path.join(args.save_dir, 'spq_compressed_model.pth')}")
    print("="*80)
    
    return final_acc


def main():
    parser = argparse.ArgumentParser(description='SPQ: Simultaneous Pruning and Quantization Training')
    
    # General settings
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save models')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    
    # Pruning settings
    parser.add_argument('--prune-rate', type=float, default=0.05,
                        help='Fraction of filters to prune per epoch')
    
    # Quantization settings
    parser.add_argument('--bitwidth', type=int, default=4,
                        help='Bitwidth for APoT quantization')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print configuration
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Run SPQ training
    final_acc = train_spq(args)
    
    print(f"\nSPQ training finished with final accuracy: {final_acc:.2f}%")


if __name__ == '__main__':
    main()