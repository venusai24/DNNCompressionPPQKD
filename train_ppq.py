"""
train_ppq.py

Post-Pruning Quantization (PPQ) training script.
This implements Algorithm 2 from "Integrating Pruning with Quantization for 
Efficient Deep Neural Networks Compression" (Makenali et al., 2025).

PPQ Pipeline:
1. Stage 1: Train baseline full-precision model
2. Stage 2: Apply incremental FPGM pruning with retraining
3. Stage 3: Apply APoT quantization with quantization-aware training (QAT)

This follows the order: Train → Prune → Quantize
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
from apot_quantizer import convert_to_apot, count_quantized_layers


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch (int): Current epoch number
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
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
    
    return avg_loss, accuracy


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


def get_conv_layers(model):
    """
    Get all Conv2d layers from the model.
    
    Args:
        model (nn.Module): Model to extract layers from
        
    Returns:
        list: List of (name, layer) tuples for all Conv2d layers
    """
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers


def apply_pruning_masks(model, masks):
    """
    Apply pruning masks to model and zero out gradients for pruned filters.
    
    Args:
        model (nn.Module): Model to apply masks to
        masks (dict): Dictionary mapping layer names to masks
    """
    for name, module in model.named_modules():
        if name in masks and isinstance(module, nn.Conv2d):
            mask = masks[name]
            # Expand mask to match weight dimensions
            expanded_mask = mask.view(-1, 1, 1, 1).float().to(module.weight.device)
            
            # Apply mask to weights
            module.weight.data *= expanded_mask
            
            # Zero out bias for pruned filters
            if module.bias is not None:
                module.bias.data *= mask.float().to(module.bias.device)
            
            # Register hook to zero gradients for pruned filters
            def hook_factory(mask):
                def hook(grad):
                    # Zero out gradients for pruned filters
                    mask_expanded = mask.view(-1, 1, 1, 1).float().to(grad.device)
                    return grad * mask_expanded
                return hook
            
            # Remove existing hooks if any
            if hasattr(module.weight, '_backward_hooks'):
                module.weight._backward_hooks.clear()
            
            # Register new hook
            module.weight.register_hook(hook_factory(mask))


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


# ============================================================================
# STAGE 1: Full-Precision Training
# ============================================================================

def stage1_baseline_training(args):
    """
    Stage 1: Train baseline full-precision model.
    
    This corresponds to training a standard ResNet-20 model on CIFAR-10
    without any compression techniques.
    
    Args:
        args: Command-line arguments
        
    Returns:
        float: Best test accuracy achieved
    """
    print("\n" + "="*80)
    print("STAGE 1: FULL-PRECISION BASELINE TRAINING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating ResNet-20 model...")
    model = resnet20(num_classes=10)
    model = model.to(device)
    
    total_params, _ = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    # Decay at 50% and 75% of total epochs
    milestones = [int(args.baseline_epochs * 0.5), int(args.baseline_epochs * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Training loop
    print(f"\nTraining for {args.baseline_epochs} epochs...")
    print(f"LR schedule: milestones={milestones}, gamma=0.1")
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.baseline_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'baseline_model.pth'))
        
        # Print results
        print(f'Epoch {epoch}/{args.baseline_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | '
              f'Best: {best_acc:.2f}% | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    elapsed_time = time.time() - start_time
    print(f"\nStage 1 completed in {elapsed_time/60:.2f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.save_dir, 'baseline_model.pth')}")
    
    return best_acc


# ============================================================================
# STAGE 2: Incremental Pruning
# ============================================================================

def stage2_incremental_pruning(args):
    """
    Stage 2: Apply incremental FPGM pruning.
    
    This implements the incremental pruning loop from Algorithm 2 (Lines 1-20).
    The pruning is done in stages, with retraining between each stage.
    
    Args:
        args: Command-line arguments
        
    Returns:
        float: Best test accuracy after pruning
    """
    print("\n" + "="*80)
    print("STAGE 2: INCREMENTAL FPGM PRUNING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load baseline model
    print("\nLoading baseline model...")
    model = resnet20(num_classes=10)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'baseline_model.pth')))
    model = model.to(device)
    
    # Evaluate baseline
    criterion = nn.CrossEntropyLoss()
    _, baseline_acc = evaluate(model, test_loader, criterion, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Initialize pruner
    pruner = FPGMPruner(device=device)
    
    # Get all conv layers
    conv_layers = get_conv_layers(model)
    print(f"\nFound {len(conv_layers)} Conv2d layers to prune")
    
    # Calculate pruning schedule
    num_stages = args.pruning_stages
    epochs_per_stage = args.pruning_epochs // num_stages
    prune_rate_per_stage = args.total_prune_rate / num_stages
    
    print(f"\nPruning configuration:")
    print(f"  Total pruning epochs: {args.pruning_epochs}")
    print(f"  Number of stages: {num_stages}")
    print(f"  Epochs per stage: {epochs_per_stage}")
    print(f"  Total prune rate: {args.total_prune_rate:.2%}")
    print(f"  Prune rate per stage: {prune_rate_per_stage:.2%}")
    
    # Optimizer for pruning stage
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.prune_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler for entire pruning phase
    milestones = [int(args.pruning_epochs * 0.5), int(args.pruning_epochs * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Track masks across all stages
    all_masks = {}
    best_acc = 0.0
    global_epoch = 0
    
    # Incremental pruning loop (Algorithm 2, Lines 1-20)
    for stage in range(1, num_stages + 1):
        print(f"\n{'-'*80}")
        print(f"PRUNING STAGE {stage}/{num_stages}")
        print(f"{'-'*80}")
        
        # Train for epochs_per_stage (Algorithm 2, Lines 3-7)
        for epoch_in_stage in range(1, epochs_per_stage + 1):
            global_epoch += 1
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, global_epoch
            )
            
            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # Update learning rate
            scheduler.step()
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
            
            print(f'Stage {stage}, Epoch {epoch_in_stage}/{epochs_per_stage} '
                  f'(Global {global_epoch}/{args.pruning_epochs}): '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Apply pruning at end of stage (Algorithm 2, Lines 8-15)
        print(f"\nApplying FPGM pruning (stage {stage})...")
        
        stage_masks = {}
        for name, layer in conv_layers:
            # Generate mask for this layer (Algorithm 2, Lines 10-14)
            mask = pruner.get_pruning_mask(layer, prune_rate_per_stage)
            
            # Combine with previous masks (cumulative pruning)
            if name in all_masks:
                # Keep a filter only if it was kept in all previous stages
                mask = mask & all_masks[name]
            
            stage_masks[name] = mask
            
            # Get pruning info
            info = pruner.get_pruning_info(layer, mask)
            print(f"  {name}: {info['num_kept']}/{info['num_filters']} filters kept "
                  f"({info['prune_rate']:.2%} pruned)")
        
        # Update global masks
        all_masks.update(stage_masks)
        
        # Apply masks to model (Algorithm 2, Line 15)
        # This zeros out weights and gradients for pruned filters
        apply_pruning_masks(model, all_masks)
        
        # Count remaining parameters
        total_params, non_zero_params = count_parameters(model)
        sparsity = 1.0 - (non_zero_params / total_params)
        print(f"\nCumulative sparsity: {sparsity:.2%} "
              f"({non_zero_params:,}/{total_params:,} params remaining)")
        
        # Evaluate after pruning
        _, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Accuracy after stage {stage} pruning: {test_acc:.2f}%")
    
    # Save final pruned model
    print(f"\nSaving pruned model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'masks': all_masks,
        'sparsity': sparsity
    }, os.path.join(args.save_dir, 'pruned_model.pth'))
    
    print(f"\nStage 2 completed")
    print(f"Final accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.save_dir, 'pruned_model.pth')}")
    
    return test_acc


# ============================================================================
# STAGE 3: Quantization-Aware Training
# ============================================================================

def stage3_quantization_aware_training(args):
    """
    Stage 3: Apply APoT quantization and perform quantization-aware training.
    
    This implements the QAT phase from Algorithm 2 (Lines 21-33).
    
    Args:
        args: Command-line arguments
        
    Returns:
        float: Final test accuracy after quantization
    """
    print("\n" + "="*80)
    print("STAGE 3: QUANTIZATION-AWARE TRAINING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load pruned model (Algorithm 2, Line 21)
    print("\nLoading pruned model...")
    model = resnet20(num_classes=10)
    checkpoint = torch.load(os.path.join(args.save_dir, 'pruned_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    masks = checkpoint['masks']
    sparsity = checkpoint['sparsity']
    
    print(f"Loaded model with sparsity: {sparsity:.2%}")
    
    # Evaluate pruned model before quantization
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    _, pruned_acc = evaluate(model, test_loader, criterion, device)
    print(f"Pruned model accuracy: {pruned_acc:.2f}%")
    
    # Convert to APoT quantized model (Algorithm 2, Line 22)
    print(f"\nConverting to APoT quantization (bitwidth={args.bitwidth})...")
    model = model.cpu()  # Move to CPU for conversion
    model = convert_to_apot(model, bitwidth=args.bitwidth)
    model = model.to(device)
    
    # Verify conversion
    counts = count_quantized_layers(model)
    print(f"Converted {counts['apot_conv']} Conv2d layers to APoT quantization")
    
    # Reapply pruning masks to quantized model
    # This ensures pruned filters remain zero during QAT
    print("\nReapplying pruning masks to quantized model...")
    apply_pruning_masks(model, masks)
    
    # Evaluate immediately after quantization
    _, quant_acc = evaluate(model, test_loader, criterion, device)
    print(f"Accuracy after quantization (before QAT): {quant_acc:.2f}%")
    
    # Quantization-Aware Training (Algorithm 2, Lines 23-31)
    print(f"\nStarting Quantization-Aware Training for {args.qat_epochs} epochs...")
    
    # Optimizer for QAT - typically use lower learning rate
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.qat_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler for QAT
    milestones = [int(args.qat_epochs * 0.5), int(args.qat_epochs * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(1, args.qat_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Reapply masks after each epoch to ensure pruned filters stay zero
        apply_pruning_masks(model, masks)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'masks': masks,
                'sparsity': sparsity,
                'bitwidth': args.bitwidth
            }, os.path.join(args.save_dir, 'ppq_compressed_model.pth'))
        
        print(f'Epoch {epoch}/{args.qat_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | '
              f'Best: {best_acc:.2f}% | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Final evaluation
    checkpoint = torch.load(os.path.join(args.save_dir, 'ppq_compressed_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    _, final_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"\nStage 3 completed")
    print(f"Final PPQ compressed model accuracy: {final_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.save_dir, 'ppq_compressed_model.pth')}")
    
    return final_acc


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PPQ: Post-Pruning Quantization Training')
    
    # General settings
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save models')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Stage 1: Baseline training
    parser.add_argument('--baseline-epochs', type=int, default=200,
                        help='Number of epochs for baseline training')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate for baseline training')
    
    # Stage 2: Pruning
    parser.add_argument('--pruning-epochs', type=int, default=160,
                        help='Total number of epochs for pruning')
    parser.add_argument('--pruning-stages', type=int, default=4,
                        help='Number of incremental pruning stages')
    parser.add_argument('--total-prune-rate', type=float, default=0.5,
                        help='Total proportion of filters to prune')
    parser.add_argument('--prune-lr', type=float, default=0.01,
                        help='Learning rate for pruning stage')
    
    # Stage 3: Quantization
    parser.add_argument('--qat-epochs', type=int, default=50,
                        help='Number of epochs for quantization-aware training')
    parser.add_argument('--bitwidth', type=int, default=4,
                        help='Bitwidth for APoT quantization')
    parser.add_argument('--qat-lr', type=float, default=0.001,
                        help='Learning rate for QAT')
    
    # Optimizer settings
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    
    # Pipeline control
    parser.add_argument('--skip-stage1', action='store_true',
                        help='Skip stage 1 (baseline training)')
    parser.add_argument('--skip-stage2', action='store_true',
                        help='Skip stage 2 (pruning)')
    parser.add_argument('--skip-stage3', action='store_true',
                        help='Skip stage 3 (quantization)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*80)
    print("PPQ (POST-PRUNING QUANTIZATION) TRAINING")
    print("="*80)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Run pipeline
    results = {}
    
    if not args.skip_stage1:
        results['baseline_acc'] = stage1_baseline_training(args)
    
    if not args.skip_stage2:
        results['pruned_acc'] = stage2_incremental_pruning(args)
    
    if not args.skip_stage3:
        results['final_acc'] = stage3_quantization_aware_training(args)
    
    # Print final summary
    print("\n" + "="*80)
    print("PPQ TRAINING COMPLETED")
    print("="*80)
    if 'baseline_acc' in results:
        print(f"Stage 1 - Baseline accuracy: {results['baseline_acc']:.2f}%")
    if 'pruned_acc' in results:
        print(f"Stage 2 - Pruned accuracy: {results['pruned_acc']:.2f}%")
    if 'final_acc' in results:
        print(f"Stage 3 - Final PPQ accuracy: {results['final_acc']:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()