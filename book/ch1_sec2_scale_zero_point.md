# Chapter 1, Section 2: Scale and Zero-Point Deep Dive

**Summary:** This section provides a comprehensive understanding of scale and zero-point—the two fundamental quantization parameters. We explore how they're computed, their mathematical properties, and why zero-point matters for preserving certain values exactly.

**Prerequisites:** Chapter 1, Section 1 (Quantization Fundamentals).

**Key Takeaways:**
- Scale determines the resolution of quantization levels
- Zero-point enables exact representation of specific values (especially zero)
- Scale computation differs for symmetric vs asymmetric quantization
- Zero-point is critical for padding and certain neural network operations

---

## Scale: The Resolution Factor

The **scale** `s` is a positive floating-point number that determines how "fine" or "coarse" the quantization grid is.

### Mathematical Definition

Given a floating-point range `[x_min, x_max]` and integer range `[q_min, q_max]`:

```
s = (x_max - x_min) / (q_max - q_min)
```

This formula ensures that the full integer range maps to the full floating-point range.

### Intuition: Scale as "Step Size"

Think of scale as the **distance between adjacent quantization levels** on the floating-point axis:

```
FP32:  0    s   2s   3s   4s   5s   6s   7s
       |----|----|----|----|----|----|----|
INT8:  0    1    2    3    4    5    6    7
```

- Smaller scale → finer grid → more precise representation
- Larger scale → coarser grid → more quantization error

### Scale and Bit Width Relationship

For a fixed range `[x_min, x_max]`, the scale depends on bit width `b`:

```
s = (x_max - x_min) / (2^b - 1)    [for unsigned]
s = (x_max - x_min) / (2^b - 2)    [for signed, accounting for asymmetry]
```

**Key insight:** Halving the bit width roughly **doubles** the scale, which doubles the maximum rounding error.

---

## Zero-Point: The Offset

The **zero-point** `z` is an integer value that corresponds to the floating-point value `0`.

### Why Zero-Point Matters

**Problem:** In asymmetric quantization, zero might not align with any quantization level.

**Example:** Quantize range `[2.0, 10.0]` to 3-bit unsigned integers `[0, 7]`:

```
Scale: s = (10 - 2) / 7 = 8/7 ≈ 1.14

Without zero-point:
  x = 0 → x_q = round(0 / 1.14) = 0 → x̂ = 0 × 1.14 = 0  ✓

But wait—0 is OUTSIDE our range [2.0, 10.0]!
If we clip: x_q = clip(0, 0, 7) = 0 → x̂ = 0  (but this is wrong)
```

**Solution:** Introduce zero-point to shift the quantization grid:

```
With zero-point:
  z = round(0 / s) - q_min = round(0 / 1.14) - 0 = 0
  
  But we need: x̂ = (x_q - z) × s = 0 when x = 0
  
  So: z = -q_min + round(-x_min / s)
  z = 0 + round(-2.0 / 1.14) = round(-1.75) = -2
  
  Verify: x = 2.0 → x_q = round(2.0 / 1.14) + (-2) = round(1.75) - 2 = 0
          x̂ = (0 - (-2)) × 1.14 = 2.28 ≈ 2.0  ✓
```

### Zero-Point Formula

For asymmetric quantization:

```
z = clip(round(-x_min / s) + q_min, q_min, q_max)
```

This ensures that `x = 0` maps to an integer within `[q_min, q_max]`.

### Symmetric Quantization: Zero-Point = 0

When the quantization range is symmetric around zero (e.g., `[-5.0, 5.0]`), the zero-point is exactly `0`:

```
x_min = -5.0, x_max = 5.0
s = (5 - (-5)) / (q_max - q_min) = 10 / (q_max - q_min)

z = round(-(-5.0) / s) + q_min = round(5.0 / s) + q_min

For symmetric range around 0: z = 0  (by design)
```

**Advantage:** Symmetric quantization simplifies the dequantization formula:

```
With zero-point: x̂ = (x_q - z) × s
Symmetric (z=0): x̂ = x_q × s
```

---

## Complete Quantization/Dequantization Pipeline

Here's the full pipeline with both scale and zero-point:

### Forward (Quantization)

```python
def quantize(x, s, z, q_min, q_max):
    """Quantize floating-point to integer."""
    x_q = torch.round(x / s) - z
    x_q = torch.clamp(x_q, q_min, q_max)
    return x_q
```

### Inverse (Dequantization)

```python
def dequantize(x_q, s, z):
    """Dequantize integer back to floating-point."""
    x̂ = (x_q + z) * s  # Note: +z because we subtracted during quantize
    return x̂
```

**Note:** Different frameworks use different conventions. PyTorch often uses:

```
x_q = round(x / s + z)  # zero-point added before rounding
x̂ = (x_q - z) × s       # zero-point subtracted after dequantizing
```

Both conventions are mathematically equivalent—just be consistent!

---

## Practical Example: INT8 Asymmetric Quantization

Let's work through a complete example with realistic values.

**Given:**
- Floating-point range: `[x_min, x_max] = [-2.3, 7.8]`
- Target: INT8 unsigned (`q_min = 0`, `q_max = 255`)

**Step 1: Compute scale**

```
s = (x_max - x_min) / (q_max - q_min)
s = (7.8 - (-2.3)) / (255 - 0)
s = 10.1 / 255
s ≈ 0.0396
```

**Step 2: Compute zero-point**

```
z = round(-x_min / s) + q_min
z = round(-(-2.3) / 0.0396) + 0
z = round(2.3 / 0.0396)
z = round(58.08)
z = 58
```

**Step 3: Verify zero-point maps correctly**

```
x = 0 → x_q = round(x / s + z)
x = 0 → x_q = round(0 / 0.0396 + 58) = round(58) = 58
x̂ = (58 - 58) × 0.0396 = 0  ✓
```

**Step 4: Quantize sample values**

| x (FP32) | x_q (INT8) | x̂ (dequantized) | Error |
|----------|------------|-----------------|-------|
| -2.3 | 0 | (0 - 58) × 0.0396 = -2.30 | 0.00 |
| 0.0 | 58 | (58 - 58) × 0.0396 = 0.00 | 0.00 |
| 3.5 | 146 | (146 - 58) × 0.0396 = 3.48 | 0.02 |
| 7.8 | 255 | (255 - 58) × 0.0396 = 7.80 | 0.00 |

---

## Why Exact Zero Representation Matters

You might wonder: "Why go through the trouble of computing zero-point? Can't we just use symmetric quantization?"

There are several reasons why exact zero representation is critical:

### 1. Padding and Masking

Neural networks frequently use zero for padding:

```python
# Padding in attention masks
mask = torch.tensor([[1, 1, 0, 0]])  # Last two positions are padding

# If zero isn't exactly representable:
# quantize(0) = 1 (off by one)
# dequantize(1) = 0.04 (not zero!)

# This corrupts the masking logic!
```

### 2. Sparse Activations

Many activations in neural networks are exactly zero:
- ReLU outputs zero for negative inputs
- Dropout sets values to zero
- Embedding padding indices are zero

If zero isn't exactly representable, these become small non-zero values, which can accumulate and cause numerical issues.

### 3. Integer Arithmetic

When performing integer matrix multiplication on hardware accelerators, having zero exactly representable enables:
- Efficient skipping of zero values (sparse computation)
- Correct handling of padding without special cases
- Simpler hardware design

---

## PyTorch Experiment: Scale and Zero-Point

Let's verify scale and zero-point computation in PyTorch:

```python
import torch

def compute_qparams(x, bits=8, symmetric=False):
    """Compute scale and zero-point for a tensor."""
    q_min = 0 if not symmetric else -(2 ** (bits - 1))
    q_max = (2 ** bits - 1) if not symmetric else (2 ** (bits - 1) - 1)
    
    if symmetric:
        # Symmetric: use max absolute value
        x_max = x.abs().max()
        scale = x_max / q_max
        zero_point = torch.tensor(0, dtype=torch.int32)
    else:
        # Asymmetric: use min and max
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / (q_max - q_min)
        zero_point = torch.round(-x_min / scale).to(torch.int32)
        zero_point = torch.clamp(zero_point, q_min, q_max)
    
    return scale, zero_point

def quantize_dequantize(x, scale, zero_point, bits=8, symmetric=False):
    """Full quantize-dequantize cycle."""
    q_min = 0 if not symmetric else -(2 ** (bits - 1))
    q_max = (2 ** bits - 1) if not symmetric else (2 ** (bits - 1) - 1)
    
    # Quantize
    x_q = torch.round(x / scale + zero_point)
    x_q = torch.clamp(x_q, q_min, q_max).to(torch.int32)
    
    # Dequantize
    x_deq = (x_q - zero_point) * scale
    
    return x_deq, x_q

# Test with a tensor that has non-symmetric range
x = torch.tensor([-2.3, 0.0, 3.5, 7.8])

# Asymmetric quantization
scale, zp = compute_qparams(x, bits=8, symmetric=False)
print(f"Asymmetric (INT8):")
print(f"  Scale: {scale:.6f}")
print(f"  Zero-point: {zp.item()}")

x_deq, x_q = quantize_dequantize(x, scale, zp, bits=8, symmetric=False)
print(f"  Quantized values: {x_q.tolist()}")
print(f"  Dequantized: {x_deq.tolist()}")
print(f"  Errors: {(x - x_deq).abs().tolist()}")

# Symmetric quantization
scale_sym, zp_sym = compute_qparams(x, bits=8, symmetric=True)
print(f"\nSymmetric (INT8):")
print(f"  Scale: {scale_sym:.6f}")
print(f"  Zero-point: {zp_sym.item()}")

x_deq_sym, x_q_sym = quantize_dequantize(x, scale_sym, zp_sym, bits=8, symmetric=True)
print(f"  Quantized values: {x_q_sym.tolist()}")
print(f"  Dequantized: {x_deq_sym.tolist()}")
```

**Expected output:**
```
Asymmetric (INT8):
  Scale: 0.039608
  Zero-point: 58
  Quantized values: [0, 58, 146, 255]
  Dequantized: [-2.297, 0.0, 3.485, 7.8]
  Errors: [0.003, 0.0, 0.015, 0.0]

Symmetric (INT8):
  Scale: 0.030588
  Zero-point: 0
  Quantized values: [-75, 0, 114, 255]
  Dequantized: [-2.294, 0.0, 3.487, 7.8]
```

---

## Knowledge Checkpoint

1. **Formula recall:** Write down the formulas for computing scale and zero-point in asymmetric quantization.

2. **Calculation:** Given range `[-1.5, 4.5]` and 4-bit unsigned integers, compute the scale and zero-point. Verify that zero maps correctly.

3. **Conceptual:** Why is symmetric quantization often preferred for weights but not always for activations?

4. **Debug:** In the PyTorch experiment, what would happen if we forgot to clamp the zero-point to `[q_min, q_max]`?

---

**Next:** [Chapter 1, Section 3: Symmetric vs. Asymmetric Quantization](ch1_sec3_symmetric_vs_asymmetric.md)
