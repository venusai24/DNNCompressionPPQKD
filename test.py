# Python — compute theoretical and empirical compression
import os
import sys
import torch

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict'):
            if k in ckpt:
                return ckpt[k]
        # assume it's a plain dict of tensors (state_dict)
        return ckpt
    return ckpt

def analyze_checkpoint(path, sample_unique_max=10000):
    info = {}
    info['path'] = path
    info['filesize_bytes'] = os.path.getsize(path)
    ckpt = torch.load(path, map_location='cpu')
    info.update({k: ckpt.get(k, None) for k in ('bitwidth','sparsity','masks') if isinstance(ckpt, dict)})
    state_dict = extract_state_dict(ckpt)
    total = 0
    nonzero = 0
    dtypes = {}
    unique_values = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        num = v.numel()
        total += num
        nonzero += int(torch.count_nonzero(v).item())
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + num
        # sample unique for large tensors to avoid heavy ops
        if num <= sample_unique_max:
            try:
                unique_values[k] = int(torch.unique(v).numel())
            except:
                unique_values[k] = None
        else:
            try:
                vals = v.reshape(-1)
                idx = torch.randperm(len(vals))[:sample_unique_max]
                unique_values[k] = int(torch.unique(vals[idx]).numel())
            except:
                unique_values[k] = None

    info['total_params'] = total
    info['nonzero_params'] = nonzero
    info['sparsity'] = 1.0 - (nonzero / total) if total>0 else None
    info['dtypes'] = dtypes
    info['per_tensor_unique_sample'] = {k: unique_values[k] for k in list(unique_values)[:10]}
    return info

def pretty_print(info):
    print(f"Path: {info['path']}")
    print(f"  File size: {info['filesize_bytes']/1e6:.3f} MB")
    print(f"  Total params (elements in state_dict tensors): {info['total_params']:,}")
    print(f"  Non-zero params: {info['nonzero_params']:,}")
    print(f"  Sparsity (1 - nonzero/total): {info['sparsity']:.3%}")
    if 'bitwidth' in info and info['bitwidth'] is not None:
        print(f"  Saved bitwidth: {info['bitwidth']}")
    if 'sparsity' in info and info.get('sparsity') is not None:
        print(f"  (Checkpoint may include 'sparsity' metadata: {info.get('sparsity')})")
    print(f"  Dtypes summary: {info['dtypes']}")
    print(f"  Sample per-tensor unique counts (first 10 tensors): {info['per_tensor_unique_sample']}")
    print()

def theoretical_size_bytes(total_params, nonzero_params, bitwidth=None, assume_baseline_fp32=True):
    # baseline naive size (all params stored in float32)
    baseline_bytes = total_params * 4 if assume_baseline_fp32 else total_params * 4
    # compressed theoretical: store only nonzero values at bitwidth (no index overhead)
    if bitwidth is None:
        # fallback: assume float32 storage for nonzero values
        compressed_bytes = nonzero_params * 4
    else:
        compressed_bytes = nonzero_params * (bitwidth / 8.0)
    return baseline_bytes, compressed_bytes

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_models.py <path_A> <path_B>")
        sys.exit(1)
    a_path, b_path = sys.argv[1], sys.argv[2]
    a = analyze_checkpoint(a_path)
    b = analyze_checkpoint(b_path)
    print("=== CHECKPOINT A ===")
    pretty_print(a)
    print("=== CHECKPOINT B ===")
    pretty_print(b)

    # Empirical file size ratio
    file_ratio = (a['filesize_bytes'] / b['filesize_bytes']) if b['filesize_bytes']>0 else None
    print(f"Empirical file size ratio A / B: {file_ratio:.3f}" if file_ratio else "N/A")

    # Theoretical size estimates (assume baseline fp32)
    # For each, baseline is its own total*4; compressed estimate uses its nonzero*bitwidth (if present) else nonzero*4
    for label, info in (('A', a), ('B', b)):
        bw = info.get('bitwidth', None)
        baseline_bytes, compressed_bytes = theoretical_size_bytes(info['total_params'], info['nonzero_params'], bitwidth=bw)
        print(f"Theoretical sizes for {label}: baseline(float32)={baseline_bytes/1e6:.3f}MB, compressed(ideal)={compressed_bytes/1e6:.3f}MB", end='')
        if baseline_bytes and compressed_bytes:
            print(f", theoretical compression = {baseline_bytes/compressed_bytes:.3f}x")
        else:
            print()
    # Cross-comparison: A baseline -> B compressed
    bw_b = b.get('bitwidth', None)
    baseline_a = a['total_params'] * 4
    compressed_b = b['nonzero_params'] * (bw_b/8.0 if bw_b else 4)
    if compressed_b>0:
        print(f"\nIf A is baseline(float32) and B is stored as nonzero*bitwidth: A/B theoretical compression = {baseline_a/compressed_b:.3f}x")
    else:
        print("\nCannot compute A/B theoretical compression (division by zero).")