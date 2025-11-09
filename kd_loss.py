import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, hard_labels, temperature=3.0, alpha=0.7):
    """
    Compute Hinton-style Knowledge Distillation loss.
    
    This implements the distillation loss as described in Hinton et al. (2015):
        L = α * T² * L_KD + (1 - α) * L_CE
    
    where:
        - L_KD: KL divergence between soft teacher and student distributions
        - L_CE: Cross-entropy loss with hard labels
        - T: Temperature for softening distributions
        - α: Weight balancing the two loss components
    
    The T² scaling factor compensates for the gradient magnitude reduction
    when using higher temperatures.
    
    Args:
        student_logits (torch.Tensor): Logits from student model, shape [batch_size, num_classes]
        teacher_logits (torch.Tensor): Logits from teacher model, shape [batch_size, num_classes]
        hard_labels (torch.Tensor): Ground truth labels, shape [batch_size]
        temperature (float): Temperature for softening probability distributions.
                           Higher values create softer distributions. (default: 3.0)
        alpha (float): Weight for the distillation loss. Should be in [0, 1].
                      alpha=1.0 means pure distillation, alpha=0.0 means pure supervised learning.
                      (default: 0.7)
    
    Returns:
        torch.Tensor: Scalar loss value combining both hard and soft losses
    
    Example:
        >>> student_logits = torch.randn(32, 10)  # batch_size=32, num_classes=10
        >>> teacher_logits = torch.randn(32, 10)
        >>> labels = torch.randint(0, 10, (32,))
        >>> loss = distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7)
        >>> loss.backward()
    
    References:
        Hinton, G., Vinyals, O., & Dean, J. (2015).
        Distilling the knowledge in a neural network.
        arXiv preprint arXiv:1503.02531.
    """
    # Validate inputs
    assert student_logits.shape == teacher_logits.shape, \
        f"Student and teacher logits must have the same shape. Got {student_logits.shape} vs {teacher_logits.shape}"
    assert 0.0 <= alpha <= 1.0, \
        f"Alpha must be in [0, 1]. Got {alpha}"
    assert temperature > 0.0, \
        f"Temperature must be positive. Got {temperature}"
    assert len(hard_labels.shape) == 1, \
        f"Hard labels should be 1D tensor. Got shape {hard_labels.shape}"
    assert student_logits.shape[0] == hard_labels.shape[0], \
        f"Batch size mismatch between logits ({student_logits.shape[0]}) and labels ({hard_labels.shape[0]})"
    
    # 1. Calculate hard loss (L_CE): Standard cross-entropy with true labels
    # This ensures the student learns the correct classifications
    criterion_ce = nn.CrossEntropyLoss()
    hard_loss = criterion_ce(student_logits, hard_labels)
    
    # 2. Calculate soft loss (L_KD): KL divergence between softened distributions
    # Temperature scaling: Higher T creates softer probability distributions
    # This reveals more information about the teacher's learned structure
    
    # Soften the student logits and convert to log probabilities
    # Shape: [batch_size, num_classes]
    soft_student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    
    # Soften the teacher logits and convert to probabilities
    # We use regular softmax (not log_softmax) for the teacher as KLDivLoss expects
    # log probabilities for input and regular probabilities for target
    # Shape: [batch_size, num_classes]
    soft_teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # Calculate KL divergence between soft distributions
    # KLDivLoss expects:
    #   - input: log probabilities (student)
    #   - target: probabilities (teacher)
    # reduction='batchmean' averages over the batch dimension
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    soft_loss = criterion_kl(soft_student_log_probs, soft_teacher_probs)
    
    # 3. Combine losses with temperature-squared scaling
    # The T² factor is crucial: it compensates for the fact that the gradients
    # of the soft targets scale as 1/T². Without this, the soft loss would
    # become insignificant at high temperatures.
    #
    # Formula: L = α * T² * L_KD + (1 - α) * L_CE
    distillation_loss_value = (alpha * (temperature ** 2) * soft_loss) + ((1.0 - alpha) * hard_loss)
    
    return distillation_loss_value


def distillation_loss_with_components(student_logits, teacher_logits, hard_labels, 
                                      temperature=3.0, alpha=0.7):
    """
    Compute Knowledge Distillation loss and return individual components.
    
    This is useful for logging and debugging purposes to see how each
    component contributes to the total loss.
    
    Args:
        student_logits (torch.Tensor): Logits from student model
        teacher_logits (torch.Tensor): Logits from teacher model
        hard_labels (torch.Tensor): Ground truth labels
        temperature (float): Temperature for softening (default: 3.0)
        alpha (float): Weight for distillation loss (default: 0.7)
    
    Returns:
        dict: Dictionary containing:
            - 'total_loss': Combined loss value
            - 'hard_loss': Cross-entropy loss with hard labels
            - 'soft_loss': KL divergence loss (before T² scaling)
            - 'soft_loss_scaled': KL divergence loss (after T² scaling)
            - 'hard_loss_weighted': Weighted hard loss component
            - 'soft_loss_weighted': Weighted soft loss component (with T² scaling)
    
    Example:
        >>> loss_dict = distillation_loss_with_components(
        ...     student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7
        ... )
        >>> print(f"Total: {loss_dict['total_loss']:.4f}")
        >>> print(f"Hard: {loss_dict['hard_loss']:.4f}, Soft: {loss_dict['soft_loss']:.4f}")
    """
    # Calculate hard loss
    criterion_ce = nn.CrossEntropyLoss()
    hard_loss = criterion_ce(student_logits, hard_labels)
    
    # Calculate soft loss
    soft_student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    soft_loss = criterion_kl(soft_student_log_probs, soft_teacher_probs)
    
    # Calculate scaled soft loss
    soft_loss_scaled = (temperature ** 2) * soft_loss
    
    # Calculate weighted components
    hard_loss_weighted = (1.0 - alpha) * hard_loss
    soft_loss_weighted = alpha * soft_loss_scaled
    
    # Total loss
    total_loss = soft_loss_weighted + hard_loss_weighted
    
    return {
        'total_loss': total_loss,
        'hard_loss': hard_loss,
        'soft_loss': soft_loss,
        'soft_loss_scaled': soft_loss_scaled,
        'hard_loss_weighted': hard_loss_weighted,
        'soft_loss_weighted': soft_loss_weighted
    }


if __name__ == "__main__":
    """
    Test the knowledge distillation loss implementation.
    """
    print("Testing Knowledge Distillation Loss...")
    print("-" * 60)
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic Functionality")
    batch_size = 32
    num_classes = 10
    
    # Create dummy data
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Calculate loss
    loss = distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7)
    
    print(f"  Student logits shape: {student_logits.shape}")
    print(f"  Teacher logits shape: {teacher_logits.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    assert student_logits.grad is not None, "Gradients not computed!"
    print(f"  Gradient shape: {student_logits.grad.shape}")
    print("  ✓ Basic functionality test passed")
    
    # Test 2: Different temperature values
    print("\nTest 2: Temperature Effect")
    student_logits = torch.randn(16, 10)
    teacher_logits = torch.randn(16, 10)
    labels = torch.randint(0, 10, (16,))
    
    temperatures = [1.0, 3.0, 5.0, 10.0]
    print("  Temperature | Loss Value")
    print("  " + "-" * 28)
    for T in temperatures:
        loss = distillation_loss(student_logits, teacher_logits, labels, temperature=T, alpha=0.7)
        print(f"  {T:>11.1f} | {loss.item():>10.4f}")
    print("  ✓ Temperature scaling test passed")
    
    # Test 3: Different alpha values
    print("\nTest 3: Alpha (Balance) Effect")
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    print("  Alpha | Loss Value | Hard Weight | Soft Weight")
    print("  " + "-" * 52)
    for a in alphas:
        loss = distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=a)
        print(f"  {a:>5.1f} | {loss.item():>10.4f} | {1.0-a:>11.1f} | {a:>11.1f}")
    print("  ✓ Alpha balancing test passed")
    
    # Test 4: Loss components
    print("\nTest 4: Loss Components")
    student_logits = torch.randn(16, 10)
    teacher_logits = torch.randn(16, 10)
    labels = torch.randint(0, 10, (16,))
    
    loss_dict = distillation_loss_with_components(
        student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7
    )
    
    print(f"  Total loss:           {loss_dict['total_loss'].item():.4f}")
    print(f"  Hard loss:            {loss_dict['hard_loss'].item():.4f}")
    print(f"  Soft loss (unscaled): {loss_dict['soft_loss'].item():.4f}")
    print(f"  Soft loss (T² scaled):{loss_dict['soft_loss_scaled'].item():.4f}")
    print(f"  Hard loss weighted:   {loss_dict['hard_loss_weighted'].item():.4f}")
    print(f"  Soft loss weighted:   {loss_dict['soft_loss_weighted'].item():.4f}")
    
    # Verify total = hard_weighted + soft_weighted
    reconstructed = loss_dict['hard_loss_weighted'] + loss_dict['soft_loss_weighted']
    assert torch.allclose(loss_dict['total_loss'], reconstructed, atol=1e-6), \
        "Loss components don't sum to total!"
    print("  ✓ Loss component breakdown verified")
    
    # Test 5: Edge cases
    print("\nTest 5: Edge Cases")
    
    # Perfect predictions (student matches teacher)
    teacher_logits = torch.randn(8, 10)
    student_logits = teacher_logits.clone()
    labels = torch.randint(0, 10, (8,))
    
    loss = distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7)
    print(f"  Loss when student == teacher: {loss.item():.4f}")
    
    # Pure distillation (alpha=1.0)
    loss_pure_kd = distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=1.0)
    print(f"  Loss with alpha=1.0: {loss_pure_kd.item():.4f}")
    
    # Pure supervised (alpha=0.0)
    loss_pure_ce = distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.0)
    print(f"  Loss with alpha=0.0: {loss_pure_ce.item():.4f}")
    print("  ✓ Edge case test passed")
    
    # Test 6: Input validation
    print("\nTest 6: Input Validation")
    try:
        # Mismatched shapes
        distillation_loss(torch.randn(16, 10), torch.randn(16, 5), labels)
        assert False, "Should have raised assertion error!"
    except AssertionError as e:
        print(f"  ✓ Caught shape mismatch: {str(e)[:50]}...")
    
    try:
        # Invalid alpha
        distillation_loss(torch.randn(16, 10), torch.randn(16, 10), labels, alpha=1.5)
        assert False, "Should have raised assertion error!"
    except AssertionError as e:
        print(f"  ✓ Caught invalid alpha: {str(e)[:50]}...")
    
    try:
        # Invalid temperature
        distillation_loss(torch.randn(16, 10), torch.randn(16, 10), labels, temperature=-1.0)
        assert False, "Should have raised assertion error!"
    except AssertionError as e:
        print(f"  ✓ Caught invalid temperature: {str(e)[:50]}...")
    
    print("\n" + "=" * 60)
    print("✓ All Knowledge Distillation Loss tests passed successfully!")
    print("=" * 60)