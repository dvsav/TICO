# Chapter 1, Section 1: Quantization Fundamentals

**Summary:** This section introduces the mathematical foundation of quantization: the mapping from continuous floating-point values to discrete integer levels. We derive the core quantization and dequantization formulas and understand the sources of quantization error.

**Prerequisites:** Chapter 0 (Introduction), basic understanding of floating-point representation.

**Key Takeaways:**
- Quantization is a mapping from high-precision to low-precision representation
- The dequantization formula: `x_float ≈ (x_int - z) × s`
- Quantization error is unavoidable but can be minimized
- Bit width determines the number of available quantization levels

---

## The Mathematical Definition of Quantization

At its core, **quantization** is a function that maps a continuous (or high-precision) value to a discrete set of levels. For neural network quantization, we typically map floating-point numbers to integers.

### The Quantization Formula

Given a floating-point value `x ∈ ℝ`, the quantization operation produces an integer `x_q ∈ ℤ`:

```
x_q = clip(round(x / s + z), q_min, q_max)
```

Where:
- `s` is the **scale** (a positive floating-point number)
- `z` is the **zero-point** (an integer that maps to floating-point value 0)
- `round(·)` rounds to the nearest integer
- `clip(v, q_min, q_max)` constrains values to `[q_min, q_max]`
- `q_min` and `q_max` are determined by the bit width

**Note:** For symmetric quantization, `z = 0`, simplifying the formula to `x_q = clip(round(x / s), q_min, q_max)`. The zero-point becomes essential for asymmetric quantization, where the range is not centered around zero.

### The Dequantization Formula

To recover an approximation of the original value, we **dequantize**:

```
x̂ = (x_q - z) × s
```

Where:
- `x̂` is the dequantized (reconstructed) value
- `z` is the **zero-point** (an integer, often 0 for symmetric quantization)
- `s` is the scale

**Key insight:** The dequantized value `x̂` is generally *not equal* to the original `x`. The difference `|x - x̂|` is the **quantization error**.

---

## Understanding Scale Through an Example

Let's work through a concrete example to build intuition.

**Problem:** Quantize values in range `[-3.0, 5.0]` to 4-bit signed integers.

**Step 1: Determine integer range**

For 4-bit signed integers:
```
q_min = -2^(4-1) = -8
q_max = 2^(4-1) - 1 = 7
```

**Step 2: Compute the scale**

The scale maps the integer range to the floating-point range:

```
s = (x_max - x_min) / (q_max - q_min)
s = (5.0 - (-3.0)) / (7 - (-8))
s = 8.0 / 15
s ≈ 0.533
```

**Step 3: Quantize a sample value**

This example uses **symmetric quantization** (zero-point z = 0).

Let's quantize `x = 2.5`:

```
x_q = round(2.5 / 0.533 + 0)  # z = 0 for symmetric
x_q = round(4.69)
x_q = clip(5, -8, 7)
x_q = 5
```

**Step 4: Dequantize to verify**

```
x̂ = (5 - 0) × 0.533  # z = 0
x̂ ≈ 2.67
```

**Quantization error:** `|2.5 - 2.67| = 0.17`

**Note:** This example demonstrates symmetric quantization where the zero-point is 0. For asymmetric quantization (covered in Section 3), the zero-point would be non-zero to account for the asymmetric range.

---

## Visualizing Quantization Levels

Consider a simpler case: quantizing to 3-bit integers (8 levels) over range `[0, 7]`.

```
Floating-point axis:
0    1    2    3    4    5    6    7
|----|----|----|----|----|----|----|

Quantization levels (with s=1, z=0):
INT8:  0    1    2    3    4    5    6    7
       ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
FP32:  0    1    2    3    4    5    6    7
```

Any floating-point value between levels gets rounded. For example:
- `x = 2.3` → `x_q = 2` → `x̂ = 2.0` (error = 0.3)
- `x = 2.7` → `x_q = 3` → `x̂ = 3.0` (error = 0.3)
- `x = 2.5` → `x_q = 2` or `3` → `x̂ = 2.0` or `3.0` (error = 0.5)

**Maximum quantization error** (without clipping) is `s/2` — half the distance between adjacent levels.

---

## Bit Width and Quantization Levels

The number of available quantization levels is determined by the bit width `b`:

| Bit Width | Integer Type | Levels | q_min | q_max |
|-----------|--------------|--------|-------|-------|
| 8-bit | INT8 | 256 | -128 | 127 |
| 8-bit | UINT8 | 256 | 0 | 255 |
| 4-bit | INT4 | 16 | -8 | 7 |
| 4-bit | UINT4 | 16 | 0 | 15 |
| 2-bit | INT2 | 4 | -2 | 1 |

**Trade-off:** Higher bit width → more levels → lower quantization error → larger model size.

For neural networks:
- **FP32 (32-bit):** Baseline, ~4 bytes per parameter
- **INT8 (8-bit):** 4× compression, ~1 byte per parameter
- **INT4 (4-bit):** 8× compression, ~0.5 bytes per parameter

---

## Sources of Quantization Error

There are two primary sources of error:

### 1. Rounding Error

Even within the representable range, most floating-point values fall between quantization levels:

```
x = 2.5, s = 1.0 → x_q = round(2.5) = 2 or 3 → x̂ = 2.0 or 3.0
Error = |2.5 - 2.0| = 0.5  or  |2.5 - 3.0| = 0.5
```

This error is **bounded** by `s/2` (assuming no clipping).

### 2. Clipping Error

Values outside `[x_min, x_max]` get clipped to the nearest representable extreme:

```
x = 10.0, x_max = 7.0 → x_q = 7 → x̂ = 7.0
Error = |10.0 - 7.0| = 3.0
```

This error can be **unbounded** and is typically much larger than rounding error.

**Key design decision:** Choosing the quantization range involves a trade-off:
- **Narrow range:** Smaller scale → lower rounding error, but more clipping
- **Wide range:** Less clipping, but larger scale → higher rounding error

Optimal quantization finds the sweet spot that minimizes *total* error (rounding + clipping).

---

## PyTorch Experiment: Basic Quantization

Let's verify these concepts with a simple PyTorch example:

```python
import torch

def quantize_tensor(x, bits=8, symmetric=True):
    """
    Quantization demo with proper zero-point handling.
    """
    # Determine quantization range
    if symmetric:
        q_max = 2 ** (bits - 1) - 1
        q_min = -q_max - 1
        zero_point = 0
    else:
        q_max = 2 ** bits - 1
        q_min = 0
        # Compute zero-point for asymmetric quantization
        scale = (x.max() - x.min()) / (q_max - q_min)
        zero_point = torch.round(-x.min() / scale).to(torch.int32)
    
    # Compute scale
    if symmetric:
        scale = x.abs().max() / q_max
    else:
        scale = (x.max() - x.min()) / (q_max - q_min)
    
    # Quantize: x_q = round(x / s + z)
    x_q = torch.round(x / scale + zero_point).clamp(q_min, q_max)
    
    # Dequantize: x̂ = (x_q - z) × s
    x_deq = (x_q - zero_point) * scale
    
    # Compute error
    error = (x - x_deq).abs().mean().item()
    
    return x_deq, scale, zero_point

# Test with random tensor (symmetric quantization, z=0)
x = torch.randn(1000)
x_deq, scale, zero_point = quantize_tensor(x, bits=8)

# Compute error separately
error = (x - x_deq).abs().mean().item()

print(f"Scale: {scale:.6f}")
print(f"Zero-point: {zero_point}")
print(f"Mean absolute error: {error:.6f}")
print(f"Max absolute error: {(x - x_deq).abs().max().item():.6f}")
```

**Expected output:**
```
Scale: ~0.15 (depends on random seed)
Zero-point: 0  (symmetric quantization)
Mean absolute error: ~0.04
Max absolute error: ~0.08 (clipping may increase this)
```

---

## Knowledge Checkpoint

Before proceeding to Section 2, verify your understanding:

1. **Formula recall:** Write down the dequantization formula from memory. What does each term represent?

2. **Calculation:** You have values in range `[-5.0, 10.0]` and want to quantize to 4-bit signed integers. What is the scale? What integer does `x = 7.5` map to?

3. **Conceptual:** Why does clipping error tend to be more harmful than rounding error?

4. **PyTorch check:** If you run the experiment above with `bits=4` instead of `bits=8`, what happens to the error? Why?

---

**Next:** [Chapter 1, Section 2: Scale and Zero-Point Deep Dive](ch1_sec2_scale_zero_point.md)
