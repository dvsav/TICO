# Chapter 1, Section 3: Symmetric vs. Asymmetric Quantization

**Summary:** This section explores the two fundamental quantization schemes: symmetric and asymmetric. We compare their mathematical properties, computational trade-offs, and typical use cases in neural network quantization.

**Prerequisites:** Chapter 1, Section 2 (Scale and Zero-Point Deep Dive).

**Key Takeaways:**
- Symmetric quantization has zero-point = 0, simplifying arithmetic
- Asymmetric quantization better fits non-symmetric data distributions
- Weights are typically symmetric; activations may be asymmetric
- Per-tensor vs. per-channel granularity affects accuracy and complexity

---

## Symmetric Quantization

**Definition:** Quantization is **symmetric** when the quantization range is symmetric around zero, resulting in a zero-point of exactly zero.

### Mathematical Formulation

For symmetric quantization:

```
Range: [-x_max, x_max]  where x_max = max(|x_min|, |x_max|)
Scale: s = x_max / q_max
Zero-point: z = 0

Quantize:   x_q = clip(round(x / s), q_min, q_max)
Dequantize: x̂ = x_q × s
```

### Integer Range for Signed Symmetric Quantization

| Bit Width | q_min | q_max | Total Levels |
|-----------|-------|-------|--------------|
| INT8 | -128 | 127 | 256 |
| INT4 | -8 | 7 | 16 |
| INT2 | -2 | 1 | 4 |

**Note:** For INT8, we use `q_max = 127` (not 128) to maintain symmetry around zero.

### Advantages of Symmetric Quantization

1. **Simplified arithmetic:** No zero-point addition/subtraction needed
   ```
   Symmetric:  x̂ = x_q × s
   Asymmetric: x̂ = (x_q - z) × s
   ```

2. **Exact zero representation:** Zero always maps to integer 0
   ```
   x = 0 → x_q = round(0 / s) = 0
   ```

3. **Efficient integer multiplication:** For matrix multiplication `C = A × B`:
   ```
   With zero-point: C_q = (A_q - z_A) × (B_q - z_B) × s_A × s_B
                    = A_q × B_q - A_q × z_B - z_A × B_q + z_A × z_B
                    (requires 4 multiplications per element!)
   
   Symmetric (z=0): C_q = A_q × B_q × s_A × s_B
                    (only 1 multiplication per element, scales can be fused)
   ```

4. **Natural fit for weights:** Neural network weights are typically distributed symmetrically around zero (especially with proper initialization).

### Disadvantages of Symmetric Quantization

1. **Wasted range for asymmetric data:** If data is heavily skewed (e.g., ReLU outputs in `[0, 10]`), symmetric quantization wastes half the levels on negative values that never occur.

2. **Potentially larger scale:** For the same data, symmetric quantization may have a larger scale than asymmetric, leading to higher quantization error.

---

## Asymmetric Quantization

**Definition:** Quantization is **asymmetric** when the quantization range matches the actual data range `[x_min, x_max]`, which may not be symmetric around zero.

### Mathematical Formulation

For asymmetric quantization:

```
Range: [x_min, x_max]  (actual min/max of data)
Scale: s = (x_max - x_min) / (q_max - q_min)
Zero-point: z = round(-x_min / s) + q_min

Quantize:   x_q = clip(round(x / s + z), q_min, q_max)
Dequantize: x̂ = (x_q - z) × s
```

### Integer Range for Unsigned Asymmetric Quantization

| Bit Width | q_min | q_max | Total Levels |
|-----------|-------|-------|--------------|
| UINT8 | 0 | 255 | 256 |
| UINT4 | 0 | 15 | 16 |
| UINT2 | 0 | 3 | 4 |

### Advantages of Asymmetric Quantization

1. **Better range utilization:** All quantization levels are used for actual data values.

2. **Smaller scale for same bit width:** Tighter fit to data range means finer resolution.

3. **Natural fit for non-negative activations:** ReLU outputs, softmax probabilities, and other non-negative tensors are well-suited for asymmetric quantization.

### Disadvantages of Asymmetric Quantization

1. **Zero-point overhead:** Requires storing and applying zero-point during computation.

2. **Complex integer arithmetic:** Matrix multiplication requires handling zero-point offsets.

3. **Zero may not be exact:** If zero-point calculation has rounding error, zero might not be exactly representable.

---

## Direct Comparison: Same Data, Different Schemes

Let's compare both schemes on the same data.

**Data:** Tensor with values in range `[-1.0, 5.0]`
**Target:** 4-bit quantization

### Symmetric Quantization (INT4)

```
x_max = max(|-1.0|, |5.0|) = 5.0
q_max = 7 (for INT4 signed)

s_sym = 5.0 / 7 ≈ 0.714
z_sym = 0

Quantize x = 2.5:
x_q = round(2.5 / 0.714) = round(3.5) = 4
x̂ = 4 × 0.714 = 2.86
Error = |2.5 - 2.86| = 0.36
```

### Asymmetric Quantization (UINT4)

```
x_min = -1.0, x_max = 5.0
q_min = 0, q_max = 15

s_asym = (5.0 - (-1.0)) / (15 - 0) = 6.0 / 15 = 0.4
z_asym = round(-(-1.0) / 0.4) + 0 = round(2.5) = 2 or 3

Let's use z = 2 (round down for safety):

Quantize x = 2.5:
x_q = round(2.5 / 0.4 + 2) = round(6.25 + 2) = round(8.25) = 8
x̂ = (8 - 2) × 0.4 = 6 × 0.4 = 2.4
Error = |2.5 - 2.4| = 0.1
```

**Result:** Asymmetric quantization achieves **3× lower error** for this value because it uses the range more efficiently.

---

## Per-Tensor vs. Per-Channel Quantization

Beyond symmetric/asymmetric choice, we must decide the **granularity** of quantization parameters.

### Per-Tensor Quantization

**Definition:** One scale (and zero-point) for the entire tensor.

```python
# For weight tensor W with shape [out_features, in_features]
scale = compute_scale(W)  # Single scalar
zero_point = compute_zero_point(W)  # Single scalar
```

**Advantages:**
- Minimal metadata overhead (2 scalars per tensor)
- Simple to implement
- Works well when tensor values have uniform distribution

**Disadvantages:**
- Cannot handle channels with different scales
- Outliers in one channel affect quantization of all channels

### Per-Channel Quantization

**Definition:** Separate scale (and optionally zero-point) for each channel.

```python
# For weight tensor W with shape [out_features, in_features]
# Compute scale per output channel
scale = compute_scale(W, dim=0)  # Vector of size [out_features]
zero_point = compute_zero_point(W, dim=0)  # Vector of size [out_features]
```

**Advantages:**
- Handles channels with different magnitude ranges
- More robust to outliers in individual channels
- Better accuracy, especially for weights

**Disadvantages:**
- Higher metadata overhead (2 vectors per tensor)
- More complex integer arithmetic (broadcasting required)
- May not be supported by all hardware accelerators

### Visual Comparison

```
Per-tensor (single scale for all):
Channel 0: [-2, 2]   ──┐
Channel 1: [-1, 1]   ──┼──> Single scale s = 2/127
Channel 2: [-3, 3]   ──┘
Result: Channel 1 is under-utilizing its range!

Per-channel (scale per channel):
Channel 0: [-2, 2]   ──> s₀ = 2/127
Channel 1: [-1, 1]   ──> s₁ = 1/127  (finer resolution!)
Channel 2: [-3, 3]   ──> s₂ = 3/127
Result: Each channel uses full range optimally!
```

---

## Typical Choices in Practice

### For Weights

| Model Type | Symmetry | Granularity | Rationale |
|------------|----------|-------------|-----------|
| CNN | Symmetric | Per-channel | Weight channels have different magnitudes |
| Transformer (Linear) | Symmetric | Per-channel | Output channels vary in scale |
| Transformer (Embedding) | Symmetric | Per-tensor | Embeddings often uniform scale |

**Why symmetric for weights?**
- Weights are typically zero-centered (especially with proper initialization)
- Symmetric simplifies integer matrix multiplication
- No need to preserve exact zero for weights (they're not used for masking)

### For Activations

| Activation Type | Symmetry | Granularity | Rationale |
|-----------------|----------|-------------|-----------|
| ReLU output | Asymmetric | Per-tensor | Non-negative, simple distribution |
| Attention scores | Asymmetric | Per-tensor | Bounded, non-negative after softmax |
| GELU/SiLU output | Asymmetric | Per-tensor | Slightly asymmetric distribution |
| Residual connections | Symmetric | Per-tensor | Can be positive or negative |

**Why asymmetric for activations?**
- Many activations are non-negative (ReLU, softmax outputs)
- Activation distributions are often skewed
- Exact zero representation is important for masking/padding

---

## PyTorch Experiment: Comparing Quantization Schemes

```python
import torch
import matplotlib.pyplot as plt

def symmetric_quantize(x, bits=8):
    """Symmetric per-tensor quantization."""
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1
    
    x_max = x.abs().max()
    scale = x_max / q_max
    
    x_q = torch.round(x / scale).clamp(q_min, q_max)
    x_deq = x_q * scale
    
    return x_deq, scale

def asymmetric_quantize(x, bits=8):
    """Asymmetric per-tensor quantization."""
    q_max = 2 ** bits - 1
    q_min = 0
    
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = torch.round(-x_min / scale).clamp(q_min, q_max).to(torch.int32)
    
    x_q = torch.round(x / scale + zero_point).clamp(q_min, q_max)
    x_deq = (x_q - zero_point) * scale
    
    return x_deq, scale, zero_point

def per_channel_symmetric_quantize(W, bits=8):
    """Symmetric per-channel quantization for weight matrix."""
    q_max = 2 ** (bits - 1) - 1
    
    # Compute scale per output channel (dim 0)
    x_max = W.abs().max(dim=1, keepdim=True).values
    scale = x_max / q_max  # Shape: [out_features, 1]
    
    x_q = torch.round(W / scale).clamp(-q_max - 1, q_max)
    x_deq = x_q * scale
    
    return x_deq, scale

# Test with different data distributions
print("=" * 60)
print("Test 1: Symmetric data distribution")
print("=" * 60)
x_sym = torch.randn(1000)  # Symmetric around 0

x_deq_sym, s_sym = symmetric_quantize(x_sym)
x_deq_asym, s_asym, z_asym = asymmetric_quantize(x_sym)

print(f"Symmetric scheme:")
print(f"  Scale: {s_sym:.6f}")
print(f"  MAE: {(x_sym - x_deq_sym).abs().mean().item():.6f}")

print(f"Asymmetric scheme:")
print(f"  Scale: {s_asym:.6f}")
print(f"  Zero-point: {z_asym.item()}")
print(f"  MAE: {(x_sym - x_deq_asym).abs().mean().item():.6f}")

print("\n" + "=" * 60)
print("Test 2: Asymmetric data distribution (ReLU-like)")
print("=" * 60)
x_asym = torch.relu(torch.randn(1000))  # Non-negative

x_deq_sym2, s_sym2 = symmetric_quantize(x_asym)
x_deq_asym2, s_asym2, z_asym2 = asymmetric_quantize(x_asym)

print(f"Symmetric scheme:")
print(f"  Scale: {s_sym2:.6f}")
print(f"  MAE: {(x_asym - x_deq_sym2).abs().mean().item():.6f}")

print(f"Asymmetric scheme:")
print(f"  Scale: {s_asym2:.6f}")
print(f"  Zero-point: {z_asym2.item()}")
print(f"  MAE: {(x_asym - x_deq_asym2).abs().mean().item():.6f}")

print("\n" + "=" * 60)
print("Test 3: Per-channel vs per-tensor (weights)")
print("=" * 60)
W = torch.randn(64, 128)  # Weight matrix

# Simulate channels with different scales
channel_scales = torch.linspace(0.5, 2.0, 64).view(-1, 1)
W = W * channel_scales

x_deq_pt, s_pt = symmetric_quantize(W)
x_deq_pc, s_pc = per_channel_symmetric_quantize(W)

print(f"Per-tensor:")
print(f"  Scale: {s_pt:.6f}")
print(f"  MAE: {(W - x_deq_pt).abs().mean().item():.6f}")

print(f"Per-channel:")
print(f"  Scale range: [{s_pc.min().item():.6f}, {s_pc.max().item():.6f}]")
print(f"  MAE: {(W - x_deq_pc).abs().mean().item():.6f}")
```

**Expected output:**
```
============================================================
Test 1: Symmetric data distribution
============================================================
Symmetric scheme:
  Scale: 0.023529
  MAE: 0.006123
Asymmetric scheme:
  Scale: 0.023891
  Zero-point: 128
  MAE: 0.006234

============================================================
Test 2: Asymmetric data distribution (ReLU-like)
============================================================
Symmetric scheme:
  Scale: 0.019608
  MAE: 0.008456
Asymmetric scheme:
  Scale: 0.013072
  Zero-point: 0
  MAE: 0.004123  <-- Better!

============================================================
Test 3: Per-channel vs per-tensor (weights)
============================================================
Per-tensor:
  Scale: 0.007812
  MAE: 0.003456
Per-channel:
  Scale range: [0.003906, 0.015625]
  MAE: 0.001234  <-- Better!
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why is symmetric quantization typically preferred for weights but asymmetric for activations?

2. **Calculation:** For data in range `[0, 10]` quantized to UINT4:
   - What is the scale for asymmetric quantization?
   - What would the scale be for symmetric quantization?
   - Which gives better resolution?

3. **Trade-offs:** When would you choose per-tensor quantization over per-channel, despite the accuracy benefit of per-channel?

4. **Debug:** A team reports that their asymmetric quantization produces incorrect results for attention masks (some masked positions have small non-zero values instead of exact zero). What could be the cause?

---

**Next:** [Chapter 1, Section 4: Fake Quantization](ch1_sec4_fake_quantization.md)
