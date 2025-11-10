"""
train_ppq_enhanced.py

Enhanced PPQ training with comprehensive logging matching train_student_kd.py.
Post-Pruning Quantization (PPQ) - Algorithm 2 from Makenali et al.

Pipeline:
1. Stage 1: Train baseline full-precision model
2. Stage 2: Apply incremental FPGM pruning with retraining
3. Stage 3: Apply APoT quantization with quantization-aware training (QAT)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import os
import argparse
from tqdm import tqdm
import time
import json
from collections import defaultdict

from models import resnet20
from data_loader import get_cifar10_loaders
from fpgm_pruner import FPGMPruner
from apot_quantizer import convert_to_apot, count_quantized_layers


class TrainingLogger:
    """Logger for tracking all training metrics."""
    
    def __init__(self, save_dir, stage_name):
        self.save_dir = save_dir
        self.stage_name = stage_name
        self.history = defaultdict(list)
        self.best_metrics = {
            'best_acc': 0.0,
            'best_epoch': 0
        }
    
    def log_epoch(self, epoch, metrics):
        """Log metrics for an epoch."""
        for key, value in metrics.items():
            self.history[key].append(value)
    
    def save(self):
        """Save training history to JSON."""
        log_path = os.path.join(self.save_dir, f'training_log_{self.stage_name}.json')
        with open(log_path, 'w') as f:
            json.dump({
                'stage': self.stage_name,
                'history': dict(self.history),
                'best_metrics': self.best_metrics
            }, f, indent=2)
        print(f"  Training log saved to: {log_path}")


def get_model_size(model, filepath=None):
    """Get model size in MB."""
    if filepath and os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
    else:
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


def count_parameters(model):
    """Count total and non-zero parameters."""
    total_params = 0
    non_zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        non_zero_params += torch.count_nonzero(param).item()
    
    return total_params, non_zero_params


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch with detailed metrics."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track gradient norms
    grad_norms = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    
    return avg_loss, accuracy, avg_grad_norm


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
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
    """Get all Conv2d layers."""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers


def apply_pruning_masks(model, masks):
    """Apply pruning masks and zero gradients."""
    for name, module in model.named_modules():
        if name in masks and isinstance(module, nn.Conv2d):
            mask = masks[name]
            expanded_mask = mask.view(-1, 1, 1, 1).float().to(module.weight.device)
            
            module.weight.data *= expanded_mask
            
            if module.bias is not None:
                module.bias.data *= mask.float().to(module.bias.device)
            
            def hook_factory(mask):
                def hook(grad):
                    mask_expanded = mask.view(-1, 1, 1, 1).float().to(grad.device)
                    return grad * mask_expanded
                return hook
            
            # Clear existing hooks safely
            if hasattr(module.weight, '_backward_hooks') and module.weight._backward_hooks is not None:
                module.weight._backward_hooks.clear()
            
            module.weight.register_hook(hook_factory(mask))


def stage1_baseline_training(args):
    """Stage 1: Full-precision baseline training with comprehensive logging."""
    print("\n" + "="*100)
    print("STAGE 1: FULL-PRECISION BASELINE TRAINING")
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.baseline_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Weight Decay: {args.weight_decay}")
    
    # Load data
    print(f"\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Training samples: {len(train_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    
    # Create model
    print(f"\nCreating ResNet-20 model...")
    model = resnet20(num_classes=10)
    model = model.to(device)
    
    total_params, _ = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    milestones = [int(args.baseline_epochs * 0.5), int(args.baseline_epochs * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    print(f"  LR Schedule: MultiStepLR")
    print(f"    Milestones: {milestones}")
    print(f"    Gamma: 0.1")
    
    # Initialize logger
    logger = TrainingLogger(args.save_dir, 'stage1_baseline')
    
    # Training loop
    print(f"\n{'='*100}")
    print(f"TRAINING")
    print(f"{'='*100}")
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.baseline_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.log_epoch(epoch, {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'grad_norm': grad_norm,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        })
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            logger.best_metrics['best_acc'] = best_acc
            logger.best_metrics['best_epoch'] = epoch
            
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'baseline_model.pth'))
            print(f"\n  ✓ New best model saved! (Epoch {epoch}, Acc: {best_acc:.2f}%)")
        
        # Print epoch summary
        print(f'\nEpoch {epoch}/{args.baseline_epochs} ({epoch_time:.1f}s):')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Grad Norm: {grad_norm:.4f}')
        print(f'  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
        print(f'  Best  - Acc: {best_acc:.2f}% (Epoch {logger.best_metrics["best_epoch"]})')
        print(f'  LR: {current_lr:.6f}')
        print('-' * 100)
    
    elapsed_time = time.time() - start_time
    
    # Save logger
    logger.save()
    
    # Summary
    print(f"\n{'='*100}")
    print(f"STAGE 1 COMPLETED")
    print(f"{'='*100}")
    print(f"  Training time: {elapsed_time/60:.2f} minutes")
    print(f"  Best accuracy: {best_acc:.2f}%")
    print(f"  Best epoch: {logger.best_metrics['best_epoch']}")
    print(f"  Model saved to: {os.path.join(args.save_dir, 'baseline_model.pth')}")
    
    model_size = get_model_size(model, os.path.join(args.save_dir, 'baseline_model.pth'))
    print(f"  Model size: {model_size:.2f} MB")
    
    return best_acc


def stage2_incremental_pruning(args):
    """Stage 2: Incremental pruning with comprehensive logging."""
    print("\n" + "="*100)
    print("STAGE 2: INCREMENTAL FPGM PRUNING")
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Pruning epochs: {args.pruning_epochs}")
    print(f"  Pruning stages: {args.pruning_stages}")
    print(f"  Total prune rate: {args.total_prune_rate:.2%}")
    print(f"  Prune rate per stage: {args.total_prune_rate / args.pruning_stages:.2%}")
    print(f"  Learning rate: {args.prune_lr}")
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Load baseline model
    print(f"\nLoading baseline model...")
    model = resnet20(num_classes=10)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'baseline_model.pth')))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    _, baseline_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    
    # Initialize pruner
    pruner = FPGMPruner(device=device)
    conv_layers = get_conv_layers(model)
    print(f"  Found {len(conv_layers)} Conv2d layers to prune")
    
    # Pruning configuration
    num_stages = args.pruning_stages
    epochs_per_stage = args.pruning_epochs // num_stages
    prune_rate_per_stage = args.total_prune_rate / num_stages
    
    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.prune_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    milestones = [int(args.pruning_epochs * 0.5), int(args.pruning_epochs * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Initialize logger
    logger = TrainingLogger(args.save_dir, 'stage2_pruning')
    
    # Training loop
    print(f"\n{'='*100}")
    print(f"PRUNING TRAINING")
    print(f"{'='*100}")
    
    all_masks = {}
    best_acc = 0.0
    global_epoch = 0
    start_time = time.time()
    
    for stage in range(1, num_stages + 1):
        print(f"\n{'-'*100}")
        print(f"PRUNING STAGE {stage}/{num_stages}")
        print(f"{'-'*100}")
        
        for epoch_in_stage in range(1, epochs_per_stage + 1):
            global_epoch += 1
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, grad_norm = train_epoch(
                model, train_loader, criterion, optimizer, device, global_epoch
            )
            
            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # Update scheduler
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Calculate current sparsity
            total_params, non_zero_params = count_parameters(model)
            sparsity = 1.0 - (non_zero_params / total_params)
            
            # Log metrics
            logger.log_epoch(global_epoch, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'grad_norm': grad_norm,
                'learning_rate': current_lr,
                'sparsity': sparsity,
                'stage': stage,
                'epoch_time': epoch_time
            })
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                logger.best_metrics['best_acc'] = best_acc
                logger.best_metrics['best_epoch'] = global_epoch
            
            # Print epoch summary
            print(f'\nStage {stage}, Epoch {epoch_in_stage}/{epochs_per_stage} (Global {global_epoch}/{args.pruning_epochs}) ({epoch_time:.1f}s):')
            print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Grad Norm: {grad_norm:.4f}')
            print(f'  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
            print(f'  Sparsity: {sparsity:.2%}, LR: {current_lr:.6f}')
            print('-' * 100)
        
        # Apply pruning at end of stage
        print(f"\nApplying FPGM pruning (stage {stage})...")
        
        stage_masks = {}
        for name, layer in conv_layers:
            mask = pruner.get_pruning_mask(layer, prune_rate_per_stage)
            
            if name in all_masks:
                mask = mask & all_masks[name]
            
            stage_masks[name] = mask
            
            info = pruner.get_pruning_info(layer, mask)
            print(f"  {name}: {info['num_kept']}/{info['num_filters']} filters kept ({info['prune_rate']:.2%} pruned)")
        
        all_masks.update(stage_masks)
        apply_pruning_masks(model, all_masks)
        
        # Calculate cumulative sparsity
        total_params, non_zero_params = count_parameters(model)
        sparsity = 1.0 - (non_zero_params / total_params)
        print(f"\nCumulative sparsity: {sparsity:.2%} ({non_zero_params:,}/{total_params:,} params)")
        
        # Evaluate after pruning
        _, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Accuracy after stage {stage} pruning: {test_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    
    # Save final pruned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'masks': all_masks,
        'sparsity': sparsity
    }, os.path.join(args.save_dir, 'pruned_model.pth'))
    
    # Save logger
    logger.save()
    
    # Summary
    print(f"\n{'='*100}")
    print(f"STAGE 2 COMPLETED")
    print(f"{'='*100}")
    print(f"  Training time: {elapsed_time/60:.2f} minutes")
    print(f"  Final accuracy: {test_acc:.2f}%")
    print(f"  Best accuracy: {best_acc:.2f}%")
    print(f"  Final sparsity: {sparsity:.2%}")
    print(f"  Parameters: {non_zero_params:,}/{total_params:,} remaining")
    print(f"  Model saved to: {os.path.join(args.save_dir, 'pruned_model.pth')}")
    
    model_size = get_model_size(model, os.path.join(args.save_dir, 'pruned_model.pth'))
    print(f"  Model size: {model_size:.2f} MB")
    
    return test_acc


def stage3_quantization_aware_training(args):
    """Stage 3: QAT with comprehensive logging."""
    print("\n" + "="*100)
    print("STAGE 3: QUANTIZATION-AWARE TRAINING")
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  QAT epochs: {args.qat_epochs}")
    print(f"  Bitwidth: {args.bitwidth}")
    print(f"  Learning rate: {args.qat_lr}")
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Load pruned model
    print(f"\nLoading pruned model...")
    model = resnet20(num_classes=10)
    checkpoint = torch.load(os.path.join(args.save_dir, 'pruned_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    masks = checkpoint['masks']
    sparsity = checkpoint['sparsity']
    
    print(f"  Loaded model with sparsity: {sparsity:.2%}")
    
    # Evaluate before quantization
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    _, pruned_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Pruned model accuracy: {pruned_acc:.2f}%")
    
    # Convert to APoT quantization
    print(f"\nConverting to APoT quantization (bitwidth={args.bitwidth})...")
    model = model.cpu()
    model = convert_to_apot(model, bitwidth=args.bitwidth)
    model = model.to(device)
    
    counts = count_quantized_layers(model)
    print(f"  Converted {counts['apot_conv']} layers to APoT")
    
    # Reapply pruning masks
    apply_pruning_masks(model, masks)
    
    # Evaluate after quantization
    _, quant_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Accuracy after quantization (before QAT): {quant_acc:.2f}%")
    
    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.qat_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    milestones = [int(args.qat_epochs * 0.5), int(args.qat_epochs * 0.75)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Initialize logger
    logger = TrainingLogger(args.save_dir, 'stage3_qat')
    
    # Training loop
    print(f"\n{'='*100}")
    print(f"QAT TRAINING")
    print(f"{'='*100}")
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.qat_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Reapply masks
        apply_pruning_masks(model, masks)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.log_epoch(epoch, {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'grad_norm': grad_norm,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        })
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            logger.best_metrics['best_acc'] = best_acc
            logger.best_metrics['best_epoch'] = epoch
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'masks': masks,
                'sparsity': sparsity,
                'bitwidth': args.bitwidth
            }, os.path.join(args.save_dir, 'ppq_compressed_model.pth'))
            print(f"\n  ✓ New best model saved! (Epoch {epoch}, Acc: {best_acc:.2f}%)")
        
        # Print epoch summary
        print(f'\nEpoch {epoch}/{args.qat_epochs} ({epoch_time:.1f}s):')
        print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Grad Norm: {grad_norm:.4f}')
        print(f'  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
        print(f'  Best  - Acc: {best_acc:.2f}% (Epoch {logger.best_metrics["best_epoch"]})')
        print(f'  LR: {current_lr:.6f}')
        print('-' * 100)
    
    elapsed_time = time.time() - start_time
    
    # Save logger
    logger.save()
    
    # Summary
    print(f"\n{'='*100}")
    print(f"STAGE 3 COMPLETED")
    print(f"{'='*100}")
    print(f"  Training time: {elapsed_time/60:.2f} minutes")
    print(f"  Pruned accuracy (before QAT): {pruned_acc:.2f}%")
    print(f"  Quantized accuracy (before QAT): {quant_acc:.2f}%")
    print(f"  Final accuracy (after QAT): {best_acc:.2f}%")
    print(f"  Improvement from QAT: {best_acc - quant_acc:+.2f}%")
    print(f"  Model saved to: {os.path.join(args.save_dir, 'ppq_compressed_model.pth')}")
    
    model_size = get_model_size(model, os.path.join(args.save_dir, 'ppq_compressed_model.pth'))
    print(f"  Model size: {model_size:.2f} MB")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='PPQ with Comprehensive Logging')
    
    # General
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Stage 1
    parser.add_argument('--baseline-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    
    # Stage 2
    parser.add_argument('--pruning-epochs', type=int, default=160)
    parser.add_argument('--pruning-stages', type=int, default=4)
    parser.add_argument('--total-prune-rate', type=float, default=0.5)
    parser.add_argument('--prune-lr', type=float, default=0.01)
    
    # Stage 3
    parser.add_argument('--qat-epochs', type=int, default=50)
    parser.add_argument('--bitwidth', type=int, default=4)
    parser.add_argument('--qat-lr', type=float, default=0.001)
    
    # Optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    
    # Pipeline control
    parser.add_argument('--skip-stage1', action='store_true')
    parser.add_argument('--skip-stage2', action='store_true')
    parser.add_argument('--skip-stage3', action='store_true')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*100)
    print("PPQ TRAINING - COMPREHENSIVE LOGGING")
    print("="*100)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Run pipeline
    results = {}
    overall_start = time.time()
    
    if not args.skip_stage1:
        results['baseline_acc'] = stage1_baseline_training(args)
    
    if not args.skip_stage2:
        results['pruned_acc'] = stage2_incremental_pruning(args)
    
    if not args.skip_stage3:
        results['final_acc'] = stage3_quantization_aware_training(args)
    
    overall_time = time.time() - overall_start
    
    # Final summary
    print("\n" + "="*100)
    print("PPQ PIPELINE COMPLETED")
    print("="*100)
    
    print(f"\nTotal Time: {overall_time/60:.2f} minutes ({overall_time/3600:.2f} hours)")
    
    if 'baseline_acc' in results:
        print(f"\nStage 1 - Baseline Training:")
        print(f"  Accuracy: {results['baseline_acc']:.2f}%")
        baseline_size = get_model_size(None, os.path.join(args.save_dir, 'baseline_model.pth'))
        print(f"  Model size: {baseline_size:.2f} MB")
    
    if 'pruned_acc' in results:
        print(f"\nStage 2 - Incremental Pruning:")
        print(f"  Accuracy: {results['pruned_acc']:.2f}%")
        if 'baseline_acc' in results:
            print(f"  Accuracy drop: {results['baseline_acc'] - results['pruned_acc']:.2f}%")
        
        # Load pruned model info
        checkpoint = torch.load(os.path.join(args.save_dir, 'pruned_model.pth'))
        if 'sparsity' in checkpoint:
            print(f"  Sparsity: {checkpoint['sparsity']:.2%}")
        pruned_size = get_model_size(None, os.path.join(args.save_dir, 'pruned_model.pth'))
        print(f"  Model size: {pruned_size:.2f} MB")
    
    if 'final_acc' in results:
        print(f"\nStage 3 - Quantization-Aware Training:")
        print(f"  Accuracy: {results['final_acc']:.2f}%")
        if 'pruned_acc' in results:
            print(f"  QAT improvement: {results['final_acc'] - results['pruned_acc']:+.2f}%")
        
        # Load final model info
        checkpoint = torch.load(os.path.join(args.save_dir, 'ppq_compressed_model.pth'))
        if 'bitwidth' in checkpoint:
            print(f"  Bitwidth: {checkpoint['bitwidth']} bits")
        if 'sparsity' in checkpoint:
            print(f"  Sparsity: {checkpoint['sparsity']:.2%}")
        final_size = get_model_size(None, os.path.join(args.save_dir, 'ppq_compressed_model.pth'))
        print(f"  Model size: {final_size:.2f} MB")
    
    # Overall compression metrics
    if 'baseline_acc' in results and 'final_acc' in results:
        print(f"\n" + "="*100)
        print("OVERALL COMPRESSION RESULTS")
        print("="*100)
        
        accuracy_retention = (results['final_acc'] / results['baseline_acc']) * 100
        print(f"  Baseline → Compressed:")
        print(f"    Accuracy: {results['baseline_acc']:.2f}% → {results['final_acc']:.2f}%")
        print(f"    Accuracy retention: {accuracy_retention:.2f}%")
        print(f"    Accuracy drop: {results['baseline_acc'] - results['final_acc']:.2f}%")
        
        if os.path.exists(os.path.join(args.save_dir, 'baseline_model.pth')) and \
           os.path.exists(os.path.join(args.save_dir, 'ppq_compressed_model.pth')):
            baseline_size = get_model_size(None, os.path.join(args.save_dir, 'baseline_model.pth'))
            final_size = get_model_size(None, os.path.join(args.save_dir, 'ppq_compressed_model.pth'))
            size_reduction = baseline_size / final_size
            print(f"    Model size: {baseline_size:.2f} MB → {final_size:.2f} MB")
            print(f"    Size reduction: {size_reduction:.2f}×")
        
        # Calculate efficiency score
        checkpoint = torch.load(os.path.join(args.save_dir, 'ppq_compressed_model.pth'))
        if 'sparsity' in checkpoint:
            # Approximate compression from sparsity and bitwidth
            sparsity = checkpoint['sparsity']
            bitwidth = checkpoint.get('bitwidth', 4)
            param_compression = 1.0 / (1.0 - sparsity)
            bit_compression = 32 / bitwidth
            total_compression = param_compression * bit_compression
            
            print(f"\n  Compression Breakdown:")
            print(f"    Parameter compression: {param_compression:.2f}× (from {sparsity:.2%} sparsity)")
            print(f"    Bit compression: {bit_compression:.2f}× (from {bitwidth}-bit quantization)")
            print(f"    Total theoretical compression: {total_compression:.2f}×")
            
            efficiency_score = (total_compression * results['final_acc']) / 100
            print(f"\n  Efficiency Score: {efficiency_score:.2f}")
            print(f"    (compression × accuracy)")
    
    print("\n" + "="*100)
    print("✓ ALL STAGES COMPLETED")
    print("="*100)
    
    print(f"\nGenerated Files:")
    if not args.skip_stage1:
        print(f"  - checkpoints/baseline_model.pth")
        print(f"  - checkpoints/training_log_stage1_baseline.json")
    if not args.skip_stage2:
        print(f"  - checkpoints/pruned_model.pth")
        print(f"  - checkpoints/training_log_stage2_pruning.json")
    if not args.skip_stage3:
        print(f"  - checkpoints/ppq_compressed_model.pth")
        print(f"  - checkpoints/training_log_stage3_qat.json")
    
    print("\n" + "="*100)


if __name__ == '__main__':
    main()