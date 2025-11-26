# DNN Compression with Pruning, Quantization, and Knowledge Distillation

This project explores and implements various techniques for compressing Deep Neural Networks (DNNs), focusing on image classification with ResNet on the CIFAR-10 dataset. The primary goal is to reduce model size and computational complexity while maintaining high accuracy.

The implemented compression techniques are:
-   **Pruning:** Filter Pruning via Geometric Median (FPGM)
-   **Quantization:** Additive Powers-of-Two (APoT) Quantization
-   **Knowledge Distillation (KD):** Transferring knowledge from a larger "teacher" model to a smaller "student" model.

The project provides different pipelines to combine these techniques, based on the paper "Integrating Pruning with Quantization for Efficient Deep Neural Networks Compression" by Makenali et al. (2025).

## Key Components

### Models

-   **Architecture:** The project uses ResNet architectures (specifically ResNet-20, ResNet-32, and ResNet-56) adapted for the CIFAR-10 dataset.
-   **Implementation:** `models.py` contains the PyTorch implementation of these models.

### Data Loader

-   **Dataset:** The CIFAR-10 dataset is used for training and evaluation.
-   **Implementation:** `data_loader.py` provides data loaders with standard augmentations (random crop, horizontal flip) for training and normalization for both training and testing.

### Pruning

-   **Method:** Filter Pruning via Geometric Median (FPGM) is used to prune entire filters from convolutional layers. FPGM identifies redundant filters by finding the filters closest to the geometric median of all filters in a layer.
-   **Implementation:** `fpgm_pruner.py` contains the implementation of the FPGM pruner.

### Quantization

-   **Method:** Additive Powers-of-Two (APoT) quantization is a non-uniform quantization scheme where the quantization levels are sums of powers of two. This allows for efficient hardware implementation.
-   **Implementation:** `apot_quantizer.py` provides the `APoTQuantConv2d` layer and a function to convert a model to use these quantized layers. It uses a Straight-Through Estimator (STE) for backpropagation.

### Knowledge Distillation

-   **Method:** The knowledge distillation loss proposed by Hinton et al. is used. It combines a "soft loss" (KL divergence between student and teacher logits) and a "hard loss" (cross-entropy with ground truth labels).
-   **Implementation:** `kd_loss.py` implements the distillation loss function.

## Training Pipelines

The project includes three main training scripts, each implementing a different compression strategy.

### 1. Post-Pruning Quantization (PPQ)

This pipeline follows the order: **Train -> Prune -> Quantize**. It is implemented in `train_ppq.py` and consists of three stages:

-   **Stage 1: Baseline Training:** Train a full-precision ResNet-20 model on CIFAR-10.
-   **Stage 2: Incremental Pruning:** Apply FPGM pruning incrementally with retraining to recover accuracy.
-   **Stage 3: Quantization-Aware Training (QAT):** Convert the pruned model to an APoT quantized model and fine-tune it.

**To run the full PPQ pipeline:**
```bash
python train_ppq.py --baseline-epochs 200 --pruning-epochs 160 --qat-epochs 50
```

You can also run individual stages using the `--skip-stage` flags. For example, to run only Stage 3, assuming Stages 1 and 2 have been completed:
```bash
python train_ppq.py --skip-stage1 --skip-stage2 --qat-epochs 50
```

### 2. Simultaneous Pruning and Quantization (SPQ)

This pipeline follows the order: **Quantize -> (Train + Prune iteratively)**. It is implemented in `train_spq.py`.

-   The model is immediately converted to use APoT quantization.
-   The model is trained with QAT, and at the end of each epoch, FPGM pruning is applied to the full-precision latent weights.

**To run the SPQ pipeline:**
```bash
python train_spq.py --epochs 200 --prune-rate 0.05
```
This will train the model for 200 epochs, pruning 5% of the remaining filters at the end of each epoch.

### 3. Knowledge Distillation for a Compressed Student

This script (`train_student_kd.py`) demonstrates how to use knowledge distillation to improve the accuracy of a student model that has already been pruned and quantized.

**Pipeline:**
1.  Load a full-precision teacher model (e.g., the baseline model from PPQ Stage 1).
2.  Load a pruned and quantized student model (e.g., the output of PPQ Stage 2, which is then quantized).
3.  Train the student model using a combination of the distillation loss and the standard cross-entropy loss.

**To run this training:**
```bash
# First, make sure you have a baseline teacher model and a pruned student model.
# You can generate them using train_ppq.py:
# python train_ppq.py --skip-stage3  # This will run stage 1 and 2

# Then, run the KD training:
python train_student_kd.py \
    --teacher-path checkpoints/baseline_model.pth \
    --student-base-path checkpoints/pruned_model.pth \
    --epochs 50 \
    --lr 0.01 \
    --temperature 4.0 \
    --alpha 0.9
```

## Model Analysis

The `test.py` script is a utility to analyze and compare the compressed models. It computes:
-   File size
-   Total and non-zero parameters (sparsity)
-   Theoretical compression ratio

**Usage:**
```bash
python test.py <path_to_model_A> <path_to_model_B>
```
For example, to compare the baseline model with the final PPQ compressed model:
```bash
python test.py checkpoints/baseline_model.pth checkpoints/ppq_compressed_model.pth
```

This will provide a detailed comparison of the two models, showing the effectiveness of the compression.
