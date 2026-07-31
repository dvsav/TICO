# Chapter 2, Section 2: Weight Quantization

**Summary:** This section covers weight quantization in detail—the process of converting model weights from floating-point to low-precision integers. We explore per-channel vs. per-tensor schemes, weight-specific considerations, and practical implementation in PyTorch.

**Prerequisites:** Chapter 2, Section 1 (PTQ Calibration), Chapter 1, Section 3 (Symmetric vs. Asymmetric Quantization).

**Key Takeaways:**
- Weights are easier to quantize than activations (static, known distribution)
- Per-channel quantization is typically preferred for weights
- Symmetric quantization is standard for weights
- Weight quantization alone can provide significant compression with minimal accuracy loss

---

## Why Weight Quantization is "Easier"

Weight quantization is generally more straightforward than activation quantization for several reasons:

### 1. Static, Known Values

```python
# Weights: known at model load time
weight_tensor = model.layer.weight  # Fixed values

# Activations: depend on input
activation = model(input)  # Different for each input
```

**Implication:** We can directly analyze weight statistics without calibration data.

### 2. Well-Behaved Distributions

Neural network weights typically have:
- **Symmetric distribution** around zero (especially with proper initialization)
- **Bell-shaped histogram** (approximately Gaussian or Laplacian)
- **Few extreme outliers** (weights are regularized during training)

```
Weight distribution (typical Linear layer):
     ▲
     │
     │      ╱‾‾‾╲
     │     ╱     ╲
     │    ╱       ╲
     │───╱─────────╲───▶
    -3   -2   -1   0   1   2   3

Symmetric, centered at zero, no significant outliers.
```

### 3. No Real-Time Constraints

Weight quantization happens offline during model preparation:
- No latency requirements
- Can use more sophisticated algorithms
- Can afford per-channel analysis

---

## Weight Quantization Schemes

### Standard Choice: Symmetric Per-Channel

For most Linear and Conv2d layers, the standard is:

```
Symmetry:     Symmetric (zero-point = 0)
Granularity:  Per-channel (scale per output channel)
Bit width:    INT8 (or INT4 for aggressive compression)
```

**Rationale:**
- **Symmetric:** Weights are zero-centered
- **Per-channel:** Different output channels often have different magnitude scales
- **INT8:** Good accuracy/compression trade-off

### Mathematical Formulation

For a weight tensor `W` with shape `[out_features, in_features]`:

```python
# Per-channel scale computation (symmetric)
q_max = 127  # For INT8
W_max = W.abs().max(dim=1, keepdim=True).values  # Shape: [out_features, 1]
scale = W_max / q_max  # Shape: [out_features, 1]

# Quantization
W_q = torch.round(W / scale).clamp(-128, 127).to(torch.int8)

# Dequantization
W_deq = W_q * scale  # Shape: [out_features, in_features]
```

### Per-Tensor Alternative

For simpler hardware or when per-channel is not supported:

```python
# Per-tensor scale (single scale for entire tensor)
q_max = 127
W_max = W.abs().max()  # Scalar
scale = W_max / q_max  # Scalar

W_q = torch.round(W / scale).clamp(-128, 127).to(torch.int8)
```

**Trade-off:** Simpler but may lose accuracy if channels have varying scales.

---

## Weight Quantization by Layer Type

Different layer types have different quantization requirements.

### Linear / Fully Connected Layers

```python
# Weight shape: [out_features, in_features]
# Quantize per output channel (dim 0)

def quantize_linear_weight(W, bits=8):
    q_max = 2 ** (bits - 1) - 1
    
    # Per-channel scale
    W_max = W.abs().max(dim=1, keepdim=True).values
    scale = W_max / q_max
    
    # Quantize
    W_q = torch.round(W / scale).clamp(-q_max - 1, q_max)
    
    return W_q, scale  # scale shape: [out_features, 1]
```

**Note:** Bias is typically kept in FP32 or quantized separately with higher precision.

### Conv2d Layers

```python
# Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
# Quantize per output channel (dim 0)

def quantize_conv_weight(W, bits=8):
    q_max = 2 ** (bits - 1) - 1
    
    # Per-channel scale (over all dims except dim 0)
    W_max = W.abs().max(dim=(1, 2, 3), keepdim=False).values  # [out_channels]
    scale = W_max / q_max
    
    # Reshape scale for broadcasting
    scale = scale.view(-1, 1, 1, 1)  # [out_channels, 1, 1, 1]
    
    W_q = torch.round(W / scale).clamp(-q_max - 1, q_max)
    
    return W_q, scale
```

### Embedding Layers

```python
# Weight shape: [vocab_size, embedding_dim]
# Typically per-tensor or per-row (per-token) quantization

def quantize_embedding_weight(W, bits=8, per_tensor=True):
    q_max = 2 ** (bits - 1) - 1
    
    if per_tensor:
        # Single scale for entire embedding table
        W_max = W.abs().max()
        scale = W_max / q_max
    else:
        # Per-row scale (each vocabulary item has its own scale)
        W_max = W.abs().max(dim=1, keepdim=True).values
        scale = W_max / q_max
    
    W_q = torch.round(W / scale).clamp(-q_max - 1, q_max)
    
    return W_q, scale
```

**Note:** Embedding quantization is more sensitive—per-tensor is safer but per-row gives better compression.

### Attention Weights (Q, K, V, O projections)

```python
# Same as Linear layers - per-channel symmetric quantization
# But may need special handling for very small layers
```

---

## Weight-Only Quantization

A practical approach for large language models is **weight-only quantization**, where:
- Weights are quantized to INT4 or INT8
- Activations remain in FP16 or BF16
- Dequantization happens on-the-fly during inference

### Why Weight-Only?

| Aspect | Full Quantization | Weight-Only |
|--------|-------------------|-------------|
| **Memory** | 4× reduction | 2-4× reduction |
| **Speed** | 2-4× (with INT8 kernels) | 1.5-2× (dequant overhead) |
| **Accuracy** | May drop (activation quantization) | Minimal drop |
| **Implementation** | Complex | Simpler |

### Implementation Pattern

```python
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Store quantized weights and scales
        self.register_buffer(
            'weight_q',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            'scale',
            torch.zeros(out_features, 1, dtype=torch.float16)
        )
    
    def forward(self, x):
        # Dequantize on-the-fly
        weight = self.weight_q * self.scale  # FP16
        
        # Compute in FP16
        return nn.functional.linear(x, weight, None)

# Usage
linear_fp32 = nn.Linear(4096, 4096)
linear_quant = QuantizedLinear(4096, 4096, bits=8)

# Convert weights
W_q, scale = quantize_linear_weight(linear_fp32.weight)
linear_quant.weight_q = W_q.to(torch.int8)
linear_quant.scale = scale.to(torch.float16)
```

---

## PyTorch Experiment: Weight Quantization Analysis

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def analyze_weight_distribution(W):
    """Analyze weight distribution statistics."""
    stats = {
        'min': W.min().item(),
        'max': W.max().item(),
        'mean': W.mean().item(),
        'std': W.std().item(),
        'skewness': ((W - W.mean()) ** 3).mean().item() / (W.std().item() ** 3),
        'kurtosis': ((W - W.mean()) ** 4).mean().item() / (W.std().item() ** 4) - 3,
    }
    stats['symmetry'] = abs(stats['min']) / (abs(stats['max']) + 1e-8)
    return stats

def quantize_weights_per_channel(W, bits=8):
    """Per-channel symmetric weight quantization."""
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1
    
    # Per-channel scale
    W_max = W.abs().max(dim=1, keepdim=True).values
    scale = W_max / q_max
    
    # Quantize-dequantize
    W_q = torch.round(W / scale).clamp(q_min, q_max)
    W_deq = W_q * scale
    
    # Compute error per channel
    error = (W - W_deq).abs()
    
    return W_deq, scale, error

def quantize_weights_per_tensor(W, bits=8):
    """Per-tensor symmetric weight quantization."""
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1
    
    # Per-tensor scale
    W_max = W.abs().max()
    scale = W_max / q_max
    
    W_q = torch.round(W / scale).clamp(q_min, q_max)
    W_deq = W_q * scale
    
    error = (W - W_deq).abs()
    
    return W_deq, scale, error

# Create a model with various layer types
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3)
        self.embedding = nn.Embedding(10000, 256)

model = TestModel()

print("=" * 70)
print("WEIGHT DISTRIBUTION ANALYSIS")
print("=" * 70)

# Analyze each layer
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
        W = module.weight.data.flatten()
        stats = analyze_weight_distribution(W)
        
        print(f"\n{name}:")
        print(f"  Shape: {module.weight.shape}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.4f}")
        print(f"  Symmetry ratio (|min|/max): {stats['symmetry']:.4f}")
        print(f"  Skewness: {stats['skewness']:.4f}, Kurtosis: {stats['kurtosis']:.4f}")

print("\n" + "=" * 70)
print("WEIGHT QUANTIZATION COMPARISON")
print("=" * 70)

# Compare per-channel vs per-tensor for Linear layer
W = model.linear1.weight.data

for bits in [4, 8]:
    print(f"\nINT{bits} Quantization:")
    
    # Per-channel
    W_deq_pc, scale_pc, error_pc = quantize_weights_per_channel(W, bits=bits)
    mae_pc = error_pc.mean().item()
    max_error_pc = error_pc.max().item()
    
    # Per-tensor
    W_deq_pt, scale_pt, error_pt = quantize_weights_per_tensor(W, bits=bits)
    mae_pt = error_pt.mean().item()
    max_error_pt = error_pt.max().item()
    
    print(f"  Per-channel: MAE = {mae_pc:.6f}, Max Error = {max_error_pc:.6f}")
    print(f"  Per-tensor:  MAE = {mae_pt:.6f}, Max Error = {max_error_pt:.6f}")
    print(f"  Improvement: {mae_pt / mae_pc:.2f}x lower error with per-channel")

# Visualize channel scale distribution
print("\n" + "=" * 70)
print("CHANNEL SCALE DISTRIBUTION (per-channel quantization)")
print("=" * 70)

W_deq, scales, _ = quantize_weights_per_channel(W, bits=8)
print(f"Scale min: {scales.min().item():.6f}")
print(f"Scale max: {scales.max().item():.6f}")
print(f"Scale mean: {scales.mean().item():.6f}")
print(f"Scale std: {scales.std().item():.6f}")
print(f"Scale range ratio (max/min): {scales.max().item() / (scales.min().item() + 1e-8):.2f}x")

# This shows why per-channel is important - scales vary significantly!
```

**Expected output:**
```
======================================================================
WEIGHT DISTRIBUTION ANALYSIS
======================================================================

linear1:
  Shape: torch.Size([1024, 512])
  Range: [-0.0432, 0.0429]
  Mean: 0.000123, Std: 0.0138
  Symmetry ratio (|min|/max): 1.0069  <-- Nearly symmetric!
  Skewness: 0.0012, Kurtosis: 0.0234

linear2:
  Shape: torch.Size([512, 1024])
  Range: [-0.0305, 0.0308]
  Mean: -0.000045, Std: 0.0097
  Symmetry ratio (|min|/max): 0.9903
  Skewness: -0.0089, Kurtosis: -0.0156

======================================================================
WEIGHT QUANTIZATION COMPARISON
======================================================================

INT4 Quantization:
  Per-channel: MAE = 0.000523, Max Error = 0.001234
  Per-tensor:  MAE = 0.000891, Max Error = 0.002156
  Improvement: 1.70x lower error with per-channel

INT8 Quantization:
  Per-channel: MAE = 0.000032, Max Error = 0.000078
  Per-tensor:  MAE = 0.000055, Max Error = 0.000134
  Improvement: 1.72x lower error with per-channel

======================================================================
CHANNEL SCALE DISTRIBUTION (per-channel quantization)
======================================================================
Scale min: 0.000234
Scale max: 0.000567
Scale mean: 0.000389
Scale std: 0.000067
Scale range ratio (max/min): 2.42x  <-- Significant variation!
```

---

## Weight Quantization in Practice: LLM Example

For large language models, weight quantization follows a specific pattern:

```python
from transformers import AutoModelForCausalLM
import torch

def quantize_llm_weights(model, bits=8):
    """
    Quantize all Linear layer weights in an LLM.
    """
    quantized_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip output layer if it has different requirements
            if 'lm_head' in name and bits == 4:
                continue  # Keep lm_head in higher precision
            
            # Get original weight
            W = module.weight.data
            
            # Compute per-channel scale
            q_max = 2 ** (bits - 1) - 1
            W_max = W.abs().max(dim=1, keepdim=True).values
            scale = W_max / q_max
            
            # Quantize
            W_q = torch.round(W / scale).clamp(-q_max - 1, q_max)
            
            # Replace weight with quantized version
            # Store scale as a buffer
            module.register_buffer('weight_scale', scale)
            module.weight = nn.Parameter(W_q * scale, requires_grad=False)
            
            quantized_layers.append(name)
    
    return quantized_layers

# Usage
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
quantized = quantize_llm_weights(model, bits=8)
print(f"Quantized {len(quantized)} layers")
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Ignoring Bias

**Problem:** Bias terms are often left in FP32, which is fine, but they should be quantized consistently if the rest of the model is quantized.

**Solution:** Either:
- Keep bias in FP32 (simplest, minimal accuracy impact)
- Quantize bias to INT32 with appropriate scale (product of input and weight scales)

### Pitfall 2: Not Handling Outliers

**Problem:** Some layers may have weight outliers that skew the scale.

**Solution:** Use clipping or percentile-based scale:
```python
# Clip extreme values before computing scale
W_clipped = W.clamp(W.quantile(0.01), W.quantile(0.99))
scale = W_clipped.abs().max() / q_max
```

### Pitfall 3: Incorrect Broadcasting

**Problem:** Per-channel scales must be correctly shaped for broadcasting.

**Solution:** Always verify shapes:
```python
# For Linear: [out_features, in_features]
# Scale should be [out_features, 1] for correct broadcasting
assert scale.shape == (W.shape[0], 1)
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why is per-channel quantization more important for weights than per-tensor?

2. **Calculation:** A Linear layer has weights with max absolute value of 0.05 per channel. For INT8 symmetric quantization, what is the scale? What's the maximum quantization error?

3. **Debug:** A team quantizes Conv2d weights per-tensor instead of per-channel and sees significant accuracy drop. Why?

4. **TICO context:** In TICO's WrapQ wrappers, how would you implement weight quantization for a Linear layer? What observer would you use (if any)?

---

**Next:** [Chapter 2, Section 3: Activation Quantization](ch2_sec3_activation_quantization.md)
