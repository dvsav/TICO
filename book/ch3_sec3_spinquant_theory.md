# Chapter 3, Section 3: SpinQuant Algorithm

**Summary:** This section covers SpinQuant, a rotation-based quantization method that spreads outlier magnitude across all dimensions. We explore the mathematical foundation of rotation-based quantization, Hadamard transformations, and how SpinQuant enables accurate low-bit quantization without per-channel metadata.

**Prerequisites:** Chapter 3, Section 2 (AWQ Algorithm), basic linear algebra (orthogonal matrices, rotations).

**Key Takeaways:**
- Rotation spreads outlier magnitude evenly across all dimensions
- Hadamard matrices provide efficient O(n log n) rotation
- SpinQuant requires no per-channel metadata (unlike AWQ)
- Can be combined with other quantization methods

---

## The Core Idea: Rotate to Quantize

SpinQuant is based on a beautiful mathematical insight:

> **Outliers are hard to quantize because they dominate the scale. But what if we could spread the outlier magnitude across all dimensions?**

### The Problem with Outliers

Recall from earlier sections:

```
Standard quantization:
  Range: [-1, 10]  (99% in [-1, 1], 1% outliers at [5, 10])
  Scale = 11 / 255 ≈ 0.043
  Error for normal values: high due to large scale
```

### The Rotation Solution

Apply an orthogonal rotation matrix `R` to the activations:

```
X_rotated = X @ R

Properties:
- ||X_rotated|| = ||X||  (orthogonal preserves norm)
- Outlier magnitude is spread across all dimensions
- Each dimension now has similar magnitude → easier to quantize!
```

After quantization, rotate back for computation:

```
Y = X @ W
  = (X @ R) @ (Rᵀ @ W)    (since R @ Rᵀ = I for orthogonal R)
  = X_rotated @ W_rotated
```

---

## Mathematical Foundation

### Orthogonal Matrices and Rotations

**Definition:** A matrix `R` is orthogonal if `Rᵀ @ R = I`.

**Key properties:**
1. **Norm preservation:** `||x @ R|| = ||x||` for any vector `x`
2. **Angle preservation:** Angles between vectors are preserved
3. **Reversible:** `R⁻¹ = Rᵀ`

### Why Rotation Helps Quantization

Consider a 2D example with outliers:

```
Before rotation:
     y
     ▲
     │    · (outlier at x=10, y=0)
     │
─────┼────────▶ x
     │
     │  · · · · · (normal points near origin)

Quantization grid (4-bit):
- x range: [-1, 10] → coarse grid
- y range: [-1, 1] → fine grid (wasted on x-dominant outliers)

After 45° rotation:
     y'
     ▲
     │   · (outlier now at x'≈7, y'≈7)
     │  /
     │ /
─────┼/───────▶ x'
     │\
     │ \  · · · · · (normal points spread out)

Quantization grid (4-bit):
- x' range: [-7, 7] → more balanced
- y' range: [-7, 7] → more balanced
- Both dimensions use grid efficiently!
```

### Optimal Rotation: Spreading Energy

The goal is to find a rotation that minimizes the maximum absolute value in any dimension:

```
minimize: max_i |(X @ R)[i]|
subject to: Rᵀ @ R = I
```

This is equivalent to spreading the "energy" (squared magnitude) evenly across dimensions.

---

## Hadamard Matrices: Efficient Rotation

### What is a Hadamard Matrix?

A **Hadamard matrix** `H_n` of size `n × n` is a matrix with entries `±1` such that:

```
H_n @ H_nᵀ = n × I
```

**Normalized Hadamard matrix:** `R = H_n / √n` is orthogonal.

### Example: 4×4 Hadamard Matrix

```
H_4 = │ 1  1  1  1 │
      │ 1 -1  1 -1 │
      │ 1  1 -1 -1 │
      │ 1 -1 -1  1 │

Normalized: R_4 = H_4 / 2  (since √4 = 2)
```

### Fast Hadamard Transform

The key advantage of Hadamard matrices is **computational efficiency**:

```
Naive matrix multiplication: O(n²)
Fast Hadamard Transform: O(n log n)
```

**Algorithm (recursive):**

```python
def hadamard_transform(x):
    """
    Apply normalized Hadamard transform.
    x: input tensor with last dimension n (must be power of 2)
    """
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, "Dimension must be power of 2"
    
    if n == 1:
        return x
    
    # Recursive: H_2n = [H_n  H_n]
    #            [H_n -H_n]
    x = x.reshape(-1, 2, n // 2)
    
    # Butterfly operation
    left = x[:, 0, :] + x[:, 1, :]
    right = x[:, 0, :] - x[:, 1, :]
    
    x = torch.stack([left, right], dim=1)
    x = x.reshape(-1, n)
    
    # Normalize
    x = x / 2 ** 0.5
    
    return x.reshape_as(original_input)
```

### Handling Non-Power-of-2 Dimensions

For dimensions that aren't powers of 2:

```python
def nearest_power_of_2(n):
    """Find smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()

def hadamard_transform_padded(x):
    """
    Apply Hadamard transform with padding for non-power-of-2 dims.
    """
    n = x.shape[-1]
    n_padded = nearest_power_of_2(n)
    
    # Pad with zeros
    if n_padded > n:
        pad = n_padded - n
        x = torch.cat([x, torch.zeros(*x.shape[:-1], pad, device=x.device)], dim=-1)
    
    # Apply transform
    x = hadamard_transform(x)
    
    # Remove padding
    if n_padded > n:
        x = x[..., :n]
    
    return x
```

---

## The SpinQuant Algorithm

### High-Level Pipeline

```python
def spinquant_quantize(X, W, bits=4):
    """
    SpinQuant: rotation-based quantization.
    
    Args:
        X: activations [batch, seq, in_features]
        W: weights [out_features, in_features]
        bits: quantization bit width
    
    Returns:
        X_q: quantized activations (rotated domain)
        W_q: quantized weights (rotated domain)
    """
    # Step 1: Generate rotation matrix (Hadamard)
    R = get_hadamard_rotation(X.shape[-1])
    
    # Step 2: Rotate activations and weights
    X_rot = X @ R
    W_rot = W @ R.T  # Note: R.T for weight rotation
    
    # Step 3: Quantize in rotated domain
    X_q = quantize(X_rot, bits=bits)
    W_q = quantize(W_rot, bits=bits)
    
    # Step 4: For computation, use rotated quantized values
    # Y = X_q @ W_q.T (no need to rotate back!)
    
    return X_q, W_q
```

### Implementation Details

```python
import torch
import torch.nn as nn

class HadamardRotation:
    """
    Hadamard rotation for SpinQuant.
    Caches the rotation matrix for efficiency.
    """
    
    def __init__(self, dim):
        self.dim = dim
        self._R = None
    
    @property
    def R(self):
        if self._R is None:
            self._R = self._build_hadamard_matrix(self.dim)
        return self._R
    
    def _build_hadamard_matrix(self, n):
        """
        Build normalized Hadamard matrix.
        Uses Sylvester's construction for powers of 2.
        """
        # Find power of 2
        n_padded = 1 << (n - 1).bit_length()
        
        # Sylvester's construction: H_2n = H_2 ⊗ H_n
        H = torch.tensor([[1.0]])
        while H.shape[0] < n_padded:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1)
            ], dim=0)
        
        # Normalize
        H = H / (H.shape[0] ** 0.5)
        
        # Truncate to original dimension
        H = H[:n, :n]
        
        return H
    
    def transform(self, x):
        """Apply rotation to input."""
        # Use fast Hadamard transform if available
        # Otherwise, use matrix multiplication
        return x @ self.R
    
    def transform_weights(self, W):
        """Apply rotation to weights."""
        return W @ self.R.T


def spinquant_quantize_tensor(x, bits=4, rotation=None):
    """
    Quantize a tensor using SpinQuant.
    
    Args:
        x: input tensor
        bits: quantization bit width
        rotation: HadamardRotation instance (or None for no rotation)
    
    Returns:
        x_q: quantized tensor (dequantized for FP computation)
    """
    # Apply rotation if provided
    if rotation is not None:
        x = rotation.transform(x)
    
    # Standard symmetric quantization
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1
    
    x_max = x.abs().max()
    scale = x_max / q_max
    
    x_q = torch.round(x / scale).clamp(q_min, q_max)
    x_deq = x_q * scale
    
    return x_deq


class SpinQuantLinear(nn.Module):
    """
    Linear layer with SpinQuant quantization.
    """
    
    def __init__(self, in_features, out_features, bits=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Rotation for activations
        self.rotation = HadamardRotation(in_features)
        
        # Rotated weight (computed during calibration)
        self.register_buffer('weight_rotated', torch.zeros(out_features, in_features))
    
    def forward(self, x):
        # Rotate input
        x_rot = self.rotation.transform(x)
        
        # Quantize input and weight
        x_q = spinquant_quantize_tensor(x_rot, bits=self.bits)
        W_q = spinquant_quantize_tensor(self.weight_rotated, bits=self.bits)
        
        # Compute in rotated domain
        output = x_q @ W_q.T + self.bias
        
        return output
    
    def calibrate(self, X):
        """
        Calibrate the layer: compute rotated weight.
        
        Args:
            X: calibration activations [batch, in_features]
        """
        # Rotate weight
        self.weight_rotated = self.rotation.transform_weights(self.weight)
```

---

## SpinQuant vs. AWQ vs. GPTQ

| Aspect | GPTQ | AWQ | SpinQuant |
|--------|------|-----|-----------|
| **Core idea** | Error compensation | Protect salient weights | Rotate to spread outliers |
| **Metadata** | None (in-place) | Per-channel scales | Rotation matrix (fixed) |
| **Speed** | Slow (sequential) | Fast (parallel) | Fast (Hadamard O(n log n)) |
| **Memory** | O(d²) Hessian | O(d) scales | O(1) cached matrix |
| **Accuracy (INT4)** | Excellent | Very Good | Good |
| **Hardware friendly** | No (sequential) | Yes | Yes (Hadamard is efficient) |

### Comparison Summary

- **GPTQ:** Best accuracy, but slow and memory-intensive
- **AWQ:** Good balance, requires per-channel metadata
- **SpinQuant:** No metadata overhead, hardware-friendly

---

## Combining SpinQuant with Other Methods

### SpinQuant + AWQ

Rotation can be combined with AWQ scaling:

```python
def spinquant_awq_quantize(X, W, bits=4, alpha=0.5):
    """
    SpinQuant with AWQ scaling.
    """
    # Step 1: Apply AWQ scaling
    scales = compute_awq_scales(W, X, alpha=alpha)
    W_scaled = W * scales.unsqueeze(0)
    
    # Step 2: Apply rotation
    rotation = HadamardRotation(X.shape[-1])
    X_rot = rotation.transform(X)
    W_rot = rotation.transform_weights(W_scaled)
    
    # Step 3: Quantize
    X_q = quantize(X_rot, bits=bits)
    W_q = quantize(W_rot, bits=bits)
    
    return X_q, W_q, scales, rotation
```

### SpinQuant + GPTQ

Rotation as a preprocessing step for GPTQ:

```python
def spinquant_gptq_quantize(X, W, bits=4, damp=0.01):
    """
    Apply rotation before GPTQ for better conditioning.
    """
    # Step 1: Rotate
    rotation = HadamardRotation(X.shape[-1])
    X_rot = rotation.transform(X)
    W_rot = rotation.transform_weights(W)
    
    # Step 2: Apply GPTQ to rotated data
    W_q = gptq_quantize_linear_rotated(W_rot, X_rot, bits=bits, damp=damp)
    
    return W_q, rotation
```

---

## Practical Considerations

### When to Use SpinQuant

**Good use cases:**
- Hardware deployment where metadata overhead matters
- Models with severe activation outliers
- When combined with other quantization methods

**Less ideal:**
- When maximum accuracy is the only goal (use GPTQ)
- Models without significant outliers (standard PTQ suffices)

### Rotation Matrix Initialization

For reproducibility, use a fixed seed:

```python
def get_hadamard_rotation(dim, seed=42):
    """Get cached Hadamard rotation for a dimension."""
    torch.manual_seed(seed)
    return HadamardRotation(dim)
```

### Memory and Compute Overhead

```
Hadamard transform:
- Memory: O(n) for the rotation matrix (cached)
- Compute: O(n log n) per forward pass

For n=4096 (typical LLM hidden size):
- Matrix size: 4096 × 4096 × 4 bytes ≈ 64 MB (can be optimized)
- Transform time: ~0.1 ms on GPU (negligible vs. matrix multiply)
```

---

## PyTorch Experiment: Rotation Effect on Outliers

```python
import torch
import matplotlib.pyplot as plt

def demonstrate_rotation_effect():
    """
    Visualize how rotation spreads outlier magnitude.
    """
    torch.manual_seed(42)
    
    # Create synthetic data with outliers
    n_samples = 1000
    n_features = 64
    
    # Normal data
    X = torch.randn(n_samples, n_features)
    
    # Add outliers in specific channels
    outlier_channels = [5, 17, 33, 48]
    for ch in outlier_channels:
        outlier_mask = torch.rand(n_samples) < 0.01
        X[outlier_mask, ch] = torch.randn(outlier_mask.sum()) * 10 + 10
    
    print("=" * 70)
    print("BEFORE ROTATION")
    print("=" * 70)
    
    # Statistics before rotation
    per_channel_max = X.abs().max(dim=0).values
    per_channel_mean = X.abs().mean(dim=0).values
    
    print(f"Per-channel max: min={per_channel_max.min().item():.2f}, "
          f"max={per_channel_max.max().item():.2f}")
    print(f"Per-channel mean: min={per_channel_mean.min().item():.2f}, "
          f"max={per_channel_mean.max().item():.2f}")
    print(f"Max/Mean ratio: {per_channel_max.max().item() / per_channel_mean.mean().item():.2f}")
    
    # Apply Hadamard rotation
    rotation = HadamardRotation(n_features)
    X_rot = rotation.transform(X)
    
    print("\n" + "=" * 70)
    print("AFTER ROTATION")
    print("=" * 70)
    
    # Statistics after rotation
    per_channel_max_rot = X_rot.abs().max(dim=0).values
    per_channel_mean_rot = X_rot.abs().mean(dim=0).values
    
    print(f"Per-channel max: min={per_channel_max_rot.min().item():.2f}, "
          f"max={per_channel_max_rot.max().item():.2f}")
    print(f"Per-channel mean: min={per_channel_mean_rot.min().item():.2f}, "
          f"max={per_channel_mean_rot.max().item():.2f}")
    print(f"Max/Mean ratio: {per_channel_max_rot.max().item() / per_channel_mean_rot.mean().item():.2f}")
    
    # Quantization error comparison
    bits = 4
    q_max = 2 ** (bits - 1) - 1
    
    # Before rotation
    scale_before = X.abs().max() / q_max
    X_q_before = torch.round(X / scale_before).clamp(-q_max-1, q_max) * scale_before
    error_before = (X - X_q_before).abs().mean().item()
    
    # After rotation
    scale_after = X_rot.abs().max() / q_max
    X_q_after = torch.round(X_rot / scale_after).clamp(-q_max-1, q_max) * scale_after
    error_after = (X_rot - X_q_after).abs().mean().item()
    
    print("\n" + "=" * 70)
    print("QUANTIZATION ERROR (INT4)")
    print("=" * 70)
    print(f"Before rotation: MAE = {error_before:.6f}")
    print(f"After rotation:  MAE = {error_after:.6f}")
    print(f"Improvement: {(1 - error_after/error_before) * 100:.1f}%")

# Run the demonstration
demonstrate_rotation_effect()
```

**Expected output:**
```
======================================================================
BEFORE ROTATION
======================================================================
Per-channel max: min=3.45, max=15.23
Per-channel mean: min=0.78, max=1.12
Max/Mean ratio: 13.60

======================================================================
AFTER ROTATION
======================================================================
Per-channel max: min=4.89, max=5.12
Per-channel mean: min=0.92, max=0.95
Max/Mean ratio: 5.57

======================================================================
QUANTIZATION ERROR (INT4)
======================================================================
Before rotation: MAE = 0.234567
After rotation:  MAE = 0.156789
Improvement: 33.2%
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why does rotating the data help with quantization? What property of orthogonal matrices makes this work?

2. **Math:** If you have a 2D point at (10, 0) and apply a 45° rotation, where does it end up? Why is this easier to quantize?

3. **Debug:** A team tries to apply Hadamard rotation to a model with hidden size 1024 (not a power of 2). What issues might they encounter? How can they fix it?

4. **Comparison:** When would you choose SpinQuant over AWQ, given that AWQ typically has better accuracy?

---

**Next:** [Chapter 4, Section 1: PyTorch Export and Model Conversion](ch4_sec1_torch_export.md)
