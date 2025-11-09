"""
train_student_kd.py

Knowledge Distillation training for compressed student model.
This implements an improvement to the PPQ pipeline by combining Stage 3 (QAT) 
with Knowledge Distillation from the full-precision teacher.

Pipeline:
1. Load full-precision teacher model (from PPQ Stage 1)
2. Load pruned student model and convert to APoT quantization
3. Train student with KD from teacher + QAT
4. Save final compressed model with KD

This approach combines:
- Pruning (from Stage 2)
- Quantization-Aware Training (Stage 3)
- Knowledge Distillation (our improvement)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import time

# Import custom modules
from models import resnet20
from data_loader import get_cifar10_loaders
from apot_quantizer import convert_to_apot, count_quantized_layers
from kd_loss import distillation_loss, distillation_loss_with_components


def evaluate(model, test_loader, device, criterion=None):
    """
    Evaluate the student model on the test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device: Device to evaluate on
        criterion: Optional loss function for computing test loss
        
    Returns:
        tuple: (test_loss, test_accuracy) if criterion provided, else test_accuracy
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    
    if criterion is not None:
        avg_loss = running_loss / len(test_loader)
        return avg_loss, accuracy
    else:
        return accuracy


def train_epoch_kd(student_model, teacher_model, train_loader, optimizer, 
                   device, temperature, alpha, epoch):
    """
    Train student model for one epoch using Knowledge Distillation.
    
    Args:
        student_model (nn.Module): Student model to train
        teacher_model (nn.Module): Teacher model (frozen)
        train_loader (DataLoader): Training data loader
        optimizer: Optimizer for student model
        device: Device to train on
        temperature (float): Temperature for KD
        alpha (float): Balance between KD and hard loss
        epoch (int): Current epoch number
        
    Returns:
        tuple: (avg_loss, accuracy)
    """
    student_model.train()
    teacher_model.eval()  # Teacher always in eval mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Get teacher predictions (no gradient computation needed)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        
        # Get student predictions (with gradient computation)
        optimizer.zero_grad()
        student_logits = student_model(inputs)
        
        # Calculate Knowledge Distillation loss
        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hard_labels=targets,
            temperature=temperature,
            alpha=alpha
        )
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = student_logits.max(1)
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


def main():
    """
    Main training function for Knowledge Distillation with QAT.
    """
    # ========================================================================
    # Argument Parsing
    # ========================================================================
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation Training for Compressed Student'
    )
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for knowledge distillation (default: 4.0)')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='Weight for distillation loss (default: 0.9)')
    
    # Model settings
    parser.add_argument('--bitwidth', type=int, default=4,
                        help='Bitwidth for APoT quantization (default: 4)')
    
    # Data settings
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Optimizer settings
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    
    # Paths
    parser.add_argument('--teacher-path', type=str, 
                        default='checkpoints/baseline_model.pth',
                        help='Path to teacher model weights')
    parser.add_argument('--student-base-path', type=str,
                        default='checkpoints/pruned_model.pth',
                        help='Path to pruned student base weights')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save final model')
    
    # Training control
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate without training')
    parser.add_argument('--log-components', action='store_true',
                        help='Log individual loss components during training')
    
    args = parser.parse_args()
    
    # ========================================================================
    # Setup
    # ========================================================================
    print("\n" + "="*80)
    print("KNOWLEDGE DISTILLATION TRAINING FOR COMPRESSED STUDENT")
    print("="*80)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha (KD weight): {args.alpha}")
    print(f"  Bitwidth: {args.bitwidth}")
    print(f"  Batch Size: {args.batch_size}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ========================================================================
    # Load Data
    # ========================================================================
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # ========================================================================
    # Instantiate Teacher Model
    # ========================================================================
    print("\nLoading Teacher Model (Full-Precision)...")
    teacher_model = resnet20(num_classes=10)
    
    # Load teacher weights
    if not os.path.exists(args.teacher_path):
        print(f"ERROR: Teacher model not found at {args.teacher_path}")
        print("Please run PPQ Stage 1 first to create the teacher model:")
        print("  python train_ppq.py --skip-stage2 --skip-stage3")
        return
    
    teacher_model.load_state_dict(torch.load(args.teacher_path))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set to evaluation mode (frozen)
    
    # Evaluate teacher
    teacher_acc = evaluate(teacher_model, test_loader, device)
    print(f"  Teacher accuracy: {teacher_acc:.2f}%")
    
    # Count teacher parameters
    teacher_params, _ = count_parameters(teacher_model)
    print(f"  Teacher parameters: {teacher_params:,}")
    
    # ========================================================================
    # Instantiate Student Model
    # ========================================================================
    print("\nLoading Student Model (Pruned + Quantized)...")
    student_model = resnet20(num_classes=10)
    
    # Load pruned weights
    if not os.path.exists(args.student_base_path):
        print(f"ERROR: Pruned model not found at {args.student_base_path}")
        print("Please run PPQ Stage 2 first to create the pruned model:")
        print("  python train_ppq.py --skip-stage1 --skip-stage3")
        return
    
    checkpoint = torch.load(args.student_base_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        student_model.load_state_dict(checkpoint['model_state_dict'])
        if 'sparsity' in checkpoint:
            print(f"  Loaded pruned model with sparsity: {checkpoint['sparsity']:.2%}")
    else:
        student_model.load_state_dict(checkpoint)
    
    # Convert to APoT quantization
    print(f"  Converting to APoT quantization (bitwidth={args.bitwidth})...")
    student_model = convert_to_apot(student_model, bitwidth=args.bitwidth)
    student_model = student_model.to(device)
    
    # Verify conversion
    counts = count_quantized_layers(student_model)
    print(f"  Converted {counts['apot_conv']} Conv2d layers to APoT")
    
    # Count student parameters
    student_params_total, student_params_nonzero = count_parameters(student_model)
    student_sparsity = 1.0 - (student_params_nonzero / student_params_total)
    print(f"  Student total parameters: {student_params_total:,}")
    print(f"  Student non-zero parameters: {student_params_nonzero:,}")
    print(f"  Student sparsity: {student_sparsity:.2%}")
    
    # Calculate compression ratio
    compression_ratio = teacher_params / student_params_nonzero
    print(f"  Compression ratio: {compression_ratio:.2f}×")
    
    # Evaluate student before KD training
    student_acc_before = evaluate(student_model, test_loader, device)
    print(f"  Student accuracy (before KD): {student_acc_before:.2f}%")
    
    # ========================================================================
    # Create Optimizer
    # ========================================================================
    print("\nSetting up optimizer...")
    optimizer = optim.SGD(
        student_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (optional but recommended)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
        gamma=0.1
    )
    
    print(f"  Optimizer: SGD")
    print(f"  Initial LR: {args.lr}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  LR Schedule: MultiStepLR at 50% and 75% of epochs")
    
    # ========================================================================
    # Training Loop (QAT + KD)
    # ========================================================================
    if not args.evaluate_only:
        print("\n" + "="*80)
        print("STARTING KNOWLEDGE DISTILLATION TRAINING")
        print("="*80)
        
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, args.epochs + 1):
            # Train for one epoch
            if args.log_components:
                # Train with component logging (slower but informative)
                train_loss, train_acc = train_epoch_with_logging(
                    student_model, teacher_model, train_loader, optimizer,
                    device, args.temperature, args.alpha, epoch
                )
            else:
                # Standard training
                train_loss, train_acc = train_epoch_kd(
                    student_model, teacher_model, train_loader, optimizer,
                    device, args.temperature, args.alpha, epoch
                )
            
            # Evaluate on test set
            test_loss, test_acc = evaluate(
                student_model, test_loader, device, 
                criterion=nn.CrossEntropyLoss()
            )
            
            # Update learning rate
            scheduler.step()
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'teacher_acc': teacher_acc,
                    'compression_ratio': compression_ratio,
                    'sparsity': student_sparsity,
                    'bitwidth': args.bitwidth,
                    'temperature': args.temperature,
                    'alpha': args.alpha
                }, os.path.join(args.save_dir, 'final_student_model_kd.pth'))
            
            # Print epoch results
            print(f'\nEpoch {epoch}/{args.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  Best Test Acc: {best_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 80)
        
        elapsed_time = time.time() - start_time
        
        # ====================================================================
        # Final Results
        # ====================================================================
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        
        # Load best model
        checkpoint = torch.load(os.path.join(args.save_dir, 'final_student_model_kd.pth'))
        student_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        final_acc = evaluate(student_model, test_loader, device)
        
        print(f"\nResults:")
        print(f"  Training time: {elapsed_time/60:.2f} minutes")
        print(f"  Teacher accuracy: {teacher_acc:.2f}%")
        print(f"  Student accuracy (before KD): {student_acc_before:.2f}%")
        print(f"  Student accuracy (after KD): {final_acc:.2f}%")
        print(f"  Best accuracy: {checkpoint['best_acc']:.2f}%")
        print(f"  Improvement from KD: {final_acc - student_acc_before:.2f}%")
        print(f"  Gap to teacher: {teacher_acc - final_acc:.2f}%")
        
        print(f"\nCompression Statistics:")
        print(f"  Sparsity: {student_sparsity:.2%}")
        print(f"  Bitwidth: {args.bitwidth}")
        print(f"  Compression ratio: {compression_ratio:.2f}×")
        print(f"  Parameters: {student_params_nonzero:,} / {teacher_params:,}")
        
        print(f"\nModel saved to: {os.path.join(args.save_dir, 'final_student_model_kd.pth')}")
        print("="*80)
    
    else:
        # Evaluate only mode
        print("\n" + "="*80)
        print("EVALUATION ONLY MODE")
        print("="*80)
        
        final_acc = evaluate(student_model, test_loader, device)
        print(f"\nResults:")
        print(f"  Teacher accuracy: {teacher_acc:.2f}%")
        print(f"  Student accuracy: {final_acc:.2f}%")
        print(f"  Gap to teacher: {teacher_acc - final_acc:.2f}%")
        print(f"  Compression ratio: {compression_ratio:.2f}×")


def train_epoch_with_logging(student_model, teacher_model, train_loader, optimizer,
                             device, temperature, alpha, epoch):
    """
    Train with detailed loss component logging (for debugging/analysis).
    """
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    running_hard_loss = 0.0
    running_soft_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        
        optimizer.zero_grad()
        student_logits = student_model(inputs)
        
        # Get loss components
        loss_dict = distillation_loss_with_components(
            student_logits, teacher_logits, targets, temperature, alpha
        )
        
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        running_hard_loss += loss_dict['hard_loss'].item()
        running_soft_loss += loss_dict['soft_loss'].item()
        
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar with component info
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'hard': running_hard_loss / (batch_idx + 1),
            'soft': running_soft_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


if __name__ == '__main__':
    main()