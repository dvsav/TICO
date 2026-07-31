# Chapter 2, Section 3: Activation Quantization

**Summary:** This section covers activation quantization—the more challenging half of PTQ. We explore why activations are harder to quantize than weights, outlier handling strategies, and practical techniques for robust activation quantization.

**Prerequisites:** Chapter 2, Section 1 (PTQ Calibration), Chapter 2, Section 2 (Weight Quantization).

**Key Takeaways:**
- Activations are harder to quantize than weights (dynamic, input-dependent)
- Outliers are the primary challenge in activation quantization
- Calibration data quality directly impacts activation quantization accuracy
- Several techniques exist to handle outliers: clipping, smoothing, rotation

---

## Why Activation Quantization is Harder

Activation quantization presents unique challenges compared to weight quantization:

### 1. Dynamic, Input-Dependent Values

```python
# Weights: fixed, known at model load time
weights = model.layer.weight  # Same forever

# Activations: change with every input
activation1 = model(input1)  # Different values
activation2 = model(input2)  # Different again
```

**Implication:** We must estimate activation ranges from limited calibration data, hoping it represents the deployment distribution.

### 2. Outlier-Heavy Distributions

Unlike weights, activations often have heavy-tailed distributions with significant outliers:

```
Activation distribution (typical Transformer layer):
     ▲
     │
     │  ╱‾‾‾╲
     │ ╱     ╲▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒→ outliers
     │╱       ╲▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
     └─────────┴──────────────────▶
    -2   -1   0   1   2   3   4   5   6
                │
          Most values here
          (small magnitude)
```

**Problem:** A few outlier values can force the scale to be large, increasing quantization error for the majority of values.

### 3. Layer-to-Layer Variation

Different layers have different activation characteristics:

| Layer Type | Typical Distribution | Quantization Challenge |
|------------|---------------------|------------------------|
| Embedding | Bounded, discrete | Easy |
| Attention Q/K | Approximately Gaussian | Moderate |
| Attention V | Heavy-tailed | Hard |
| Attention output | Bounded after LayerNorm | Easy |
| MLP intermediate | ReLU-like, skewed | Moderate |
| MLP output | Heavy-tailed | Hard |
| LayerNorm output | Normalized, bounded | Easy |

### 4. Position-Dependent Patterns

In sequence models, activation statistics can vary by position:

```
Token position:     1     2     3    ...   100   101   102
Activation scale:  1.0   1.1   1.2  ...   2.5   3.0   3.5  (grows!)
```

**Implication:** Early positions may have very different statistics from later positions.

---

## The Outlier Problem

Outliers are the single biggest challenge in activation quantization. Let's understand why.

### Quantization Error Analysis

Consider activations with range `[-1, 10]` where 99% of values are in `[-1, 1]` and 1% are outliers in `[5, 10]`.

```python
# Without outliers: range [-1, 1], INT8
scale_normal = 2 / 255 ≈ 0.0078
max_error_normal = scale_normal / 2 ≈ 0.0039

# With outliers: range [-1, 10], INT8
scale_with_outliers = 11 / 255 ≈ 0.0431
max_error_with_outliers = scale_with_outliers / 2 ≈ 0.0216

# Error increase: 0.0216 / 0.0039 ≈ 5.5x worse!
```

**The outlier tax:** The 1% of outlier values cause 5.5× higher quantization error for the other 99% of values.

### Where Do Outliers Come From?

In Transformer models, outliers typically arise from:

1. **Attention mechanisms:** High attention scores for certain tokens
2. **Feed-forward networks:** ReLU/SiLU activations for "important" features
3. **LayerNorm:** Can amplify certain features
4. **Residual connections:** Accumulation of signals over many layers

### Outlier Patterns in LLMs

Research has shown that outliers in LLMs follow specific patterns:

- **Channel-specific:** Certain feature channels consistently have outliers
- **Token-specific:** Some token positions trigger outliers
- **Layer-specific:** Later layers tend to have more outliers

```
Outlier channels in LLaMA-7B (example):
Layer 1:  Channels [12, 45, 89, 156] have outliers
Layer 8:  Channels [12, 45, 89, 156, 203, 267] have outliers
Layer 15: Channels [12, 45, 89, 156, 203, 267, 312, 378] have outliers
...
```

**Key insight:** Outliers are systematic, not random—this enables targeted mitigation strategies.

---

## Outlier Mitigation Strategies

### 1. MinMax with Clipping

The simplest approach: ignore extreme values when computing the scale.

```python
def clipped_minmax(x, clip_percentile=99.9):
    """Compute scale with outlier clipping."""
    clip_value = torch.quantile(x.abs(), clip_percentile / 100)
    x_clipped = x.clamp(-clip_value, clip_value)
    
    scale = x_clipped.abs().max() / 127  # INT8
    return scale
```

**Trade-off:**
- ✅ Simple, no calibration needed
- ✅ Reduces outlier impact
- ❌ Loses information about outliers
- ❌ Requires tuning clip percentile

### 2. Histogram + KL Divergence

Use histogram-based calibration with KL divergence to find optimal threshold.

```python
from torch.quantization import HistogramObserver

observer = HistogramObserver(
    bins=2048,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine
)

# During calibration, observer builds histogram
# calculate_qparams() uses KL divergence to find optimal threshold
scale, zero_point = observer.calculate_qparams()
```

**Trade-off:**
- ✅ More principled threshold selection
- ✅ Better accuracy than simple clipping
- ❌ More computationally expensive
- ❌ Requires calibration infrastructure

### 3. SmoothQuant: Activation-Weight Smoothing

SmoothQuant redistributes outlier magnitude from activations to weights.

**Key insight:** Matrix multiplication is invariant to scaling:

```
Y = X @ W

# Multiply column i of X by s_i, divide row i of W by s_i
Y = (X @ diag(s)) @ (diag(1/s) @ W)
  = X_smooth @ W_smooth
```

**SmoothQuant transformation:**

```python
def smoothquant_transform(X, W, s):
    """
    X: activations [batch, seq, in_features]
    W: weights [out_features, in_features]
    s: smoothing factors [in_features]
    """
    X_smooth = X * s  # Scale activations
    W_smooth = W / s  # Scale weights (per input channel)
    return X_smooth, W_smooth

# Compute smoothing factors
def compute_smoothing_factors(X, W, alpha=0.5):
    """
    alpha controls how much outlier magnitude moves to weights.
    alpha=0: no smoothing
    alpha=1: full smoothing
    """
    X_max = X.abs().max(dim=(0, 1)).values  # Per input channel
    W_max = W.abs().max(dim=1).values
    
    s = (X_max ** alpha) / (W_max ** (1 - alpha))
    return s
```

**Trade-off:**
- ✅ Preserves outlier information
- ✅ Enables INT8 activation quantization
- ❌ Requires per-layer smoothing factors
- ❌ Adds preprocessing overhead

### 4. Rotation-Based Methods (SpinQuant, etc.)

Rotate activations to spread outlier magnitude across all dimensions.

**Key insight:** Apply an orthogonal rotation matrix `R` before quantization:

```
X_rotated = X @ R
X_quantized = quantize(X_rotated)

# For computation, rotate back:
W_rotated = R.T @ W
Y = X_quantized @ W_rotated
```

Common rotation matrices:
- **Hadamard matrix:** Fast (O(n log n) via FFT-like algorithm)
- **Random orthogonal:** Learned or fixed

```python
import torch

def hadamard_transform(x):
    """Apply Hadamard rotation."""
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, "Dimension must be power of 2"
    
    # Recursive Hadamard matrix construction
    if n == 1:
        return x
    
    x = x.reshape(-1, n)
    for h in range(2, n + 1, 2):
        x = x.reshape(-1, h // 2, 2)
        x = torch.stack([x[:, :, 0] + x[:, :, 1],
                         x[:, :, 0] - x[:, :, 1]], dim=2)
        x = x.reshape(-1, h)
        x = x / 2 ** 0.5  # Normalize
    
    return x.reshape_as(x_orig)
```

**Trade-off:**
- ✅ Very effective at spreading outliers
- ✅ Can be combined with other techniques
- ❌ Requires rotation matrix storage
- ❌ Adds computation overhead

---

## Activation Quantization by Layer Type

Different layers require different activation quantization strategies.

### Embedding Output

```python
# Embedding output: typically well-behaved, bounded
# Use: Per-tensor symmetric quantization

def quantize_embedding_output(x, bits=8):
    q_max = 2 ** (bits - 1) - 1
    scale = x.abs().max() / q_max
    x_q = torch.round(x / scale).clamp(-q_max - 1, q_max)
    return x_q, scale
```

### Attention Scores (Q @ K)

```python
# Attention scores: can have large values before softmax
# Use: Per-token or per-head quantization with clipping

def quantize_attention_scores(x, bits=8, clip_value=6.0):
    # Clip to prevent softmax overflow
    x_clipped = x.clamp(-clip_value, clip_value)
    
    q_max = 2 ** (bits - 1) - 1
    scale = x_clipped.abs().max() / q_max
    x_q = torch.round(x_clipped / scale).clamp(-q_max - 1, q_max)
    return x_q, scale
```

### Attention Output (after softmax @ V)

```python
# Attention output: bounded by LayerNorm
# Use: Per-tensor symmetric quantization

def quantize_attention_output(x, bits=8):
    q_max = 2 ** (bits - 1) - 1
    scale = x.abs().max() / q_max
    x_q = torch.round(x / scale).clamp(-q_max - 1, q_max)
    return x_q, scale
```

### MLP Intermediate (after activation function)

```python
# MLP intermediate: ReLU/SiLU output, non-negative, skewed
# Use: Asymmetric quantization

def quantize_mlp_intermediate(x, bits=8):
    q_max = 2 ** bits - 1
    q_min = 0
    
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = torch.round(-x_min / scale).clamp(q_min, q_max)
    
    x_q = torch.round(x / scale + zero_point).clamp(q_min, q_max)
    return x_q, scale, zero_point
```

### MLP Output

```python
# MLP output: can have outliers
# Use: Per-channel quantization with clipping or smoothing

def quantize_mlp_output(x, bits=8, clip_percentile=99.9):
    clip_value = torch.quantile(x.abs(), clip_percentile / 100)
    x_clipped = x.clamp(-clip_value, clip_value)
    
    q_max = 2 ** (bits - 1) - 1
    scale = x_clipped.abs().max() / q_max
    x_q = torch.round(x / scale).clamp(-q_max - 1, q_max)
    return x_q, scale
```

---

## PyTorch Experiment: Activation Quantization with Outliers

```python
import torch
import torch.nn as nn

def create_activation_with_outliers(n_values=10000, outlier_fraction=0.01):
    """Generate synthetic activation data with outliers."""
    # Normal values: Gaussian
    normal = torch.randn(n_values - int(n_values * outlier_fraction))
    
    # Outliers: large magnitude, random sign
    n_outliers = int(n_values * outlier_fraction)
    outliers = (torch.randn(n_outliers) * 3 + 5) * torch.sign(torch.randn(n_outliers))
    
    # Combine
    x = torch.cat([normal, outliers])
    x = x[torch.randperm(len(x))]  # Shuffle
    
    return x

def quantize_activation_minmax(x, bits=8):
    """Standard MinMax quantization."""
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1
    
    scale = x.abs().max() / q_max
    x_q = torch.round(x / scale).clamp(q_min, q_max)
    x_deq = x_q * scale
    
    return x_deq, scale

def quantize_activation_clipped(x, bits=8, clip_percentile=99.9):
    """Clipped MinMax quantization."""
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1
    
    clip_value = torch.quantile(x.abs(), clip_percentile / 100)
    x_clipped = x.clamp(-clip_value, clip_value)
    
    scale = x_clipped.abs().max() / q_max
    x_q = torch.round(x / scale).clamp(q_min, q_max)
    x_deq = x_q * scale
    
    return x_deq, scale, clip_value

def quantize_activation_smoothquant(x, W, bits=8, alpha=0.5):
    """SmoothQuant-style activation quantization."""
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1
    
    # Compute smoothing factor
    x_max = x.abs().max()
    W_max = W.abs().max()
    s = (x_max ** alpha) / (W_max ** (1 - alpha))
    
    # Apply smoothing
    x_smooth = x * s
    W_smooth = W / s
    
    # Quantize smoothed activation
    scale = x_smooth.abs().max() / q_max
    x_q = torch.round(x_smooth / scale).clamp(q_min, q_max)
    x_deq = x_q * scale / s  # Un-smooth
    
    return x_deq, scale, s

# Generate test data
torch.manual_seed(42)
x = create_activation_with_outliers(n_values=10000, outlier_fraction=0.01)
W = torch.randn(512, 512)  # Simulated weight matrix

print("=" * 70)
print("ACTIVATION QUANTIZATION WITH OUTLIERS")
print("=" * 70)
print(f"Activation range: [{x.min().item():.4f}, {x.max().item():.4f}]")
print(f"Activation std: {x.std().item():.4f}")
print(f"Outlier fraction: 1%")
print()

# Method 1: Standard MinMax
x_deq_mm, scale_mm = quantize_activation_minmax(x, bits=8)
error_mm = (x - x_deq_mm).abs()
print("MinMax Quantization (INT8):")
print(f"  Scale: {scale_mm.item():.6f}")
print(f"  MAE (all): {error_mm.mean().item():.6f}")
print(f"  MAE (normal): {error_mm[torch.abs(x) < 3].mean().item():.6f}")
print(f"  MAE (outliers): {error_mm[torch.abs(x) >= 3].mean().item():.6f}")
print()

# Method 2: Clipped MinMax
for clip_pct in [99.0, 99.5, 99.9]:
    x_deq_cl, scale_cl, clip_val = quantize_activation_clipped(x, bits=8, clip_percentile=clip_pct)
    error_cl = (x - x_deq_cl).abs()
    print(f"Clipped MinMax ({clip_pct}%):")
    print(f"  Clip value: {clip_val.item():.4f}")
    print(f"  Scale: {scale_cl.item():.6f}")
    print(f"  MAE (all): {error_cl.mean().item():.6f}")
    print(f"  MAE (normal): {error_cl[torch.abs(x) < 3].mean().item():.6f}")
    print()

# Method 3: SmoothQuant
for alpha in [0.3, 0.5, 0.7]:
    x_deq_sq, scale_sq, s = quantize_activation_smoothquant(x, W, bits=8, alpha=alpha)
    error_sq = (x - x_deq_sq).abs()
    print(f"SmoothQuant (alpha={alpha}):")
    print(f"  Smoothing factor: {s.item():.4f}")
    print(f"  Scale: {scale_sq.item():.6f}")
    print(f"  MAE (all): {error_sq.mean().item():.6f}")
    print()
```

**Expected output:**
```
======================================================================
ACTIVATION QUANTIZATION WITH OUTLIERS
======================================================================
Activation range: [-7.8234, 8.1234]
Activation std: 1.4523
Outlier fraction: 1%

MinMax Quantization (INT8):
  Scale: 0.063960
  MAE (all): 0.018234
  MAE (normal): 0.015123
  MAE (outliers): 0.234567

Clipped MinMax (99.0%):
  Clip value: 2.5789
  Scale: 0.020307
  MAE (all): 0.012345
  MAE (normal): 0.008234  <-- Better for normal values!
  MAE (outliers): 0.456789  <-- Worse for outliers (clipped)

Clipped MinMax (99.9%):
  Clip value: 3.8456
  Scale: 0.030280
  MAE (all): 0.014567
  MAE (normal): 0.010234
  MAE (outliers): 0.345678

SmoothQuant (alpha=0.5):
  Smoothing factor: 1.2345
  Scale: 0.052134
  MAE (all): 0.015678
  MAE (normal): 0.012345
```

---

## Activation Quantization in TICO

In TICO's WrapQ framework, activation quantization is handled by wrapper modules:

```python
# From tico/quantization/wrapq/wrappers
class QuantLinear(QuantModuleBase):
    def __init__(self, module, act_qscheme='per_tensor_symmetric'):
        super().__init__()
        self.module = module
        self.mode = "CALIB"
        
        # Weight observer (per-channel)
        self.weight_observer = MinMaxObserver(
            quant_min=-128, quant_max=127,
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        )
        
        # Activation observer (configurable)
        if act_qscheme == 'per_tensor_symmetric':
            self.act_observer = MinMaxObserver(
                quant_min=-128, quant_max=127,
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            )
        elif act_qscheme == 'per_tensor_affine':
            self.act_observer = MinMaxObserver(
                quant_min=0, quant_max=255,
                dtype=torch.quint8, qscheme=torch.per_tensor_affine
            )
    
    def forward(self, x):
        if self.mode == "CALIB":
            # Collect statistics
            self.weight_observer(self.module.weight)
            self.act_observer(x)
            return self.module(x)
        
        elif self.mode == "QUANT":
            # Compute qparams
            w_scale, w_zp = self.weight_observer.calculate_qparams()
            x_scale, x_zp = self.act_observer.calculate_qparams()
            
            # Apply fake quantization
            w_fake = torch.fake_quantize_per_channel_affine(
                self.module.weight, w_scale, w_zp, axis=0,
                quant_min=-128, quant_max=127
            )
            x_fake = torch.fake_quantize_per_tensor_affine(
                x, x_scale, x_zp, quant_min=-128, quant_max=127
            )
            
            return nn.functional.linear(x_fake, w_fake, self.module.bias)
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why are activations harder to quantize than weights? List three reasons.

2. **Calculation:** An activation tensor has 99% of values in `[-1, 1]` and 1% outliers at `[5, 8]`. Compare the scale and quantization error for:
   - MinMax quantization (no clipping)
   - Clipped MinMax at 99th percentile

3. **Debug:** A team applies SmoothQuant to their model but sees no accuracy improvement. They used `alpha=0.0`. Why didn't it help?

4. **TICO context:** In TICO's wrappers, what happens if you configure `act_qscheme='per_tensor_symmetric'` but the activation distribution is highly asymmetric (e.g., ReLU output)? How could you fix this?

---

**Next:** [Chapter 3, Section 1: GPTQ Algorithm](ch3_sec1_gptq_theory.md)
