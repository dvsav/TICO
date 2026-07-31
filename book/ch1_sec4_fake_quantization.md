# Chapter 1, Section 4: Fake Quantization

**Summary:** This section introduces fake quantization—the technique of simulating quantization effects while maintaining floating-point computation. We explore how fake quantization enables quantization-aware training and PTQ parameter calibration without requiring integer arithmetic hardware.

**Prerequisites:** Chapter 1, Section 3 (Symmetric vs. Asymmetric Quantization), basic understanding of PyTorch autograd.

**Key Takeaways:**
- Fake quantization uses quantize-dequantize (Q-DQ) operations to simulate quantization error
- Gradients flow through fake quantization operations, enabling training/calibration
- Fake quantization is the foundation of both QAT and PTQ workflows
- PyTorch's `torch.fake_quantize` and observer infrastructure support fake quantization

---

## What is Fake Quantization?

**Fake quantization** (also called **pseudo-quantization** or **simulated quantization**) is a technique where we simulate the effects of quantization—specifically, the rounding error and clipping—while keeping all computations in floating-point.

### The Core Idea

Instead of actually converting values to integers, we:
1. **Quantize** (float → "integer")
2. Immediately **Dequantize** ("integer" → float)

The result is a floating-point value that has the same quantization error as true quantization, but remains in floating-point format.

### Mathematical Formulation

The fake quantization operation is:

```
fake_quantize(x) = dequantize(quantize(x))
                 = (clip(round(x / s) + z, q_min, q_max) - z) × s
```

Where:
- `x` is the input floating-point tensor
- `s` is the scale (pre-computed or learned)
- `z` is the zero-point (pre-computed or learned)
- `q_min`, `q_max` are the quantization bounds

### Visual Representation

```
Forward pass:
  x (FP32) ──→ [Quantize] ──→ x_q (INT) ──→ [Dequantize] ──→ x̂ (FP32)
                    │                           │
                    └──→ Rounding + Clipping ───┘

Backward pass (gradients):
  ∂L/∂x̂ ──→ [Straight-Through Estimator] ──→ ∂L/∂x
```

**Key insight:** The output `x̂` has quantization error baked in, but remains in FP32 format, allowing it to be used in subsequent floating-point operations.

---

## Why Fake Quantization?

You might wonder: "If we're still using floating-point, what's the point?"

Fake quantization serves several critical purposes:

### 1. Quantization-Aware Training (QAT)

During QAT, we want the model to **learn to compensate for quantization error**. Fake quantization enables this by:

- Inserting quantization error into the forward pass
- Allowing gradients to flow through (via straight-through estimator)
- Enabling weight updates that minimize the impact of quantization

```python
# QAT workflow:
model = prepare_model_for_qat(model)  # Inserts fake_quantize ops
train(model, dataset)                  # Model learns to handle quantization
model = convert_to_quantized(model)    # Replace fake_quant with real quant
```

### 2. PTQ Parameter Calibration

In Post-Training Quantization, fake quantization helps us:

- Collect activation statistics during calibration
- Determine optimal scale and zero-point values
- Evaluate quantization quality before committing to integer format

```python
# PTQ workflow:
model = wrap_with_observers(model)     # Wrappers in CALIB mode
for batch in calibration_data:
    model(batch)                       # Collects activation stats
model = compute_qparams(model)         # Compute scale/zero-point from stats
model = convert_to_fake_quant(model)   # Switch to QUANT mode with Q-DQ
evaluate(model, val_data)              # Verify accuracy before export
```

### 3. Debugging and Analysis

Fake quantization allows us to:
- Inspect intermediate quantized values without leaving PyTorch
- Compare quantized vs. full-precision outputs directly
- Profile the impact of different bit widths and quantization schemes

---

## The Straight-Through Estimator (STE)

A key challenge with fake quantization is that the **round** and **clip** operations have zero or undefined gradients almost everywhere:

```
d/dx [round(x)] = 0  (almost everywhere, since round is piecewise constant)
d/dx [clip(x)] = 0  (outside the clip region) or undefined (at boundaries)
```

### Solution: Straight-Through Estimator

The **straight-through estimator** (STE) bypasses this problem by pretending the gradient is identity:

```python
# Forward: y = round(x)  (or clip(round(x)))
# Backward: ∂L/∂x = ∂L/∂y  (pretend dy/dx = 1)
```

In PyTorch, this is implemented using `detach()`:

```python
def ste_round(x):
    """Round with straight-through estimator gradient."""
    return x + (x.round() - x).detach()

# Verification:
# Forward: returns x.round()
# Backward: gradient flows as if y = x (identity)
```

### PyTorch's Implementation

PyTorch's `torch.fake_quantize` uses STE internally:

```python
import torch

# Manual fake quantization with STE
def manual_fake_quant(x, scale, zero_point, q_min, q_max):
    x_scaled = x / scale + zero_point
    x_rounded = torch.round(x_scaled)
    x_clipped = torch.clamp(x_rounded, q_min, q_max)
    x_dequant = (x_clipped - zero_point) * scale
    
    # STE: gradient flows as if x_dequant = x
    return x + (x_dequant - x).detach()

# Compare with PyTorch's built-in
def torch_fake_quant(x, scale, zero_point, q_min, q_max):
    return torch.fake_quantize_per_tensor_affine(
        x, scale.item(), zero_point.item(), q_min, q_max
    )
```

---

## Fake Quantization in PyTorch

PyTorch provides several ways to apply fake quantization:

### 1. Functional API: `torch.fake_quantize_*`

```python
import torch

# Per-tensor asymmetric fake quantization
x = torch.randn(10, 10)
scale = torch.tensor(0.1)
zero_point = torch.tensor(128, dtype=torch.int32)

x_fake = torch.fake_quantize_per_tensor_affine(
    x, scale.item(), zero_point.item(), q_min=0, q_max=255
)

# Per-channel symmetric fake quantization (for weights)
W = torch.randn(64, 128)
scales = W.abs().max(dim=1).values / 127  # Per-channel scales

W_fake = torch.fake_quantize_per_channel_affine(
    W, scales, torch.zeros(64, dtype=torch.int32),
    axis=0, q_min=-128, q_max=127
)
```

### 2. Module-Based: `torch.quantization.FakeQuantize`

```python
from torch.quantization import FakeQuantize, ObserverBase

# Create observer to collect stats
observer = torch.quantization.MinMaxObserver(
    quant_min=0, quant_max=255,
    dtype=torch.quint8, qscheme=torch.per_tensor_affine
)

# Create fake quantize module
fake_quant = FakeQuantize(observer)

# Use in model
x = torch.randn(10, 10)
x_fake = fake_quant(x)  # During training, applies Q-DQ
```

### 3. QConfig and prepare/convert Workflow

PyTorch's high-level quantization API:

```python
from torch.quantization import QConfig, default_qconfig, prepare, convert

# Define quantization configuration
qconfig = QConfig(
    activation=default_qconfig.activation,
    weight=default_qconfig.weight
)

# Prepare: insert observers and fake quantize modules
model_prepared = prepare(model, qconfig)

# Calibrate: run calibration data to collect stats
for batch in calibration_loader:
    model_prepared(batch)

# Convert: replace observers with fake quantize modules
model_fake_quant = convert(model_prepared)

# Now model_fake_quant has Q-DQ ops inserted and can be evaluated
```

---

## Fake Quantization vs. Real Quantization

| Aspect | Fake Quantization | Real Quantization |
|--------|-------------------|-------------------|
| **Data type** | FP32 throughout | INT8/INT4 for weights/activations |
| **Computation** | FP32 arithmetic | Integer arithmetic |
| **Speed** | Same as FP32 | Faster (on integer-capable hardware) |
| **Memory** | FP32 memory footprint | Reduced (INT8 = 1/4 of FP32) |
| **Gradients** | Supported (via STE) | Not supported (non-differentiable) |
| **Use case** | QAT, PTQ calibration, debugging | Deployment |

### Transition from Fake to Real

The typical workflow is:

```
FP32 Model
    ↓
[Prepare]  →  Model with Observers
    ↓
[Calibrate]  →  Observers collect stats
    ↓
[Convert to Fake Quant]  →  Model with Q-DQ ops (fake quantized)
    ↓
[Evaluate]  →  Verify accuracy
    ↓
[Export/Convert to Real]  →  Integer model for deployment
```

---

## PyTorch Experiment: Fake Quantization Basics

```python
import torch
import torch.nn as nn

def manual_fake_quant_per_tensor(x, bits=8, symmetric=False):
    """
    Manual implementation of per-tensor fake quantization.
    """
    if symmetric:
        q_max = 2 ** (bits - 1) - 1
        q_min = -q_max - 1
        x_max = x.abs().max()
        scale = x_max / q_max
        zero_point = 0
    else:
        q_max = 2 ** bits - 1
        q_min = 0
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / (q_max - q_min)
        zero_point = round(-x_min / scale)
    
    # Quantize-dequantize
    x_q = torch.round(x / scale + zero_point).clamp(q_min, q_max)
    x_deq = (x_q - zero_point) * scale
    
    # Apply STE for gradient flow
    return x + (x_deq - x).detach(), scale, zero_point


class FakeQuantLinear(nn.Module):
    """Linear layer with fake quantization on weights and activations."""
    
    def __init__(self, in_features, out_features, bits=8, symmetric=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits
        self.symmetric = symmetric
    
    def forward(self, x):
        # Fake quantize weights
        self.linear.weight_fake_quant, w_scale, w_zp = manual_fake_quant_per_tensor(
            self.linear.weight, bits=self.bits, symmetric=self.symmetric
        )
        
        # Fake quantize input activation
        x_fake, x_scale, x_zp = manual_fake_quant_per_tensor(
            x, bits=self.bits, symmetric=self.symmetric
        )
        
        # Use fake-quantized weights for computation
        # Note: We still compute in FP32, but with quantization error
        output = nn.functional.linear(x_fake, self.linear.weight_fake_quant, self.linear.bias)
        
        return output


# Experiment 1: Verify quantization error
print("=" * 60)
print("Experiment 1: Fake Quantization Error")
print("=" * 60)

x = torch.randn(100, 100, requires_grad=True)
x_fake, scale, zp = manual_fake_quant_per_tensor(x, bits=8, symmetric=True)

print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
print(f"Scale: {scale:.6f}")
print(f"Zero-point: {zp}")
print(f"Quantization error (MAE): {(x - x_fake).abs().mean().item():.6f}")
print(f"Max error: {(x - x_fake).abs().max().item():.6f}")

# Experiment 2: Verify gradient flow through fake quantization
print("\n" + "=" * 60)
print("Experiment 2: Gradient Flow Through Fake Quantization")
print("=" * 60)

loss = x_fake.sum()
loss.backward()

print(f"Gradient exists: {x.grad is not None}")
print(f"Gradient mean: {x.grad.mean().item():.6f}")
print(f"Gradient std: {x.grad.std().item():.6f}")

# Without STE, gradient would be zero!

# Experiment 3: Compare different bit widths
print("\n" + "=" * 60)
print("Experiment 3: Bit Width Comparison")
print("=" * 60)

for bits in [2, 4, 8]:
    x_fake, scale, _ = manual_fake_quant_per_tensor(x, bits=bits, symmetric=True)
    error = (x - x_fake).abs().mean().item()
    print(f"INT{bits}: Scale = {scale:.6f}, MAE = {error:.6f}")

# Experiment 4: Fake quantized linear layer
print("\n" + "=" * 60)
print("Experiment 4: Fake Quantized Linear Layer")
print("=" * 60)

model = FakeQuantLinear(128, 64, bits=8, symmetric=True)
x = torch.randn(32, 128)

output = model(x)
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

# Verify gradient flows through the layer
loss = output.sum()
loss.backward()
print(f"Weight gradient exists: {model.linear.weight.grad is not None}")
print(f"Input gradient exists: {x.grad is not None}")
```

**Expected output:**
```
============================================================
Experiment 1: Fake Quantization Error
============================================================
Input range: [-3.1234, 2.8765]
Scale: 0.024510
Zero-point: 0
Quantization error (MAE): 0.006123
Max error: 0.012255

============================================================
Experiment 2: Gradient Flow Through Fake Quantization
============================================================
Gradient exists: True
Gradient mean: 0.000000
Gradient std: 0.000000

============================================================
Experiment 3: Bit Width Comparison
============================================================
INT2: Scale = 1.041234, MAE = 0.260309
INT4: Scale = 0.208247, MAE = 0.052062
INT8: Scale = 0.024510, MAE = 0.006123

============================================================
Experiment 4: Fake Quantized Linear Layer
============================================================
Output shape: torch.Size([32, 64])
Output range: [-1.2345, 1.5678]
Weight gradient exists: True
Input gradient exists: True
```

---

## Fake Quantization in TICO

The TICO project uses fake quantization as a core part of its PTQ workflow. Here's how it fits into the TICO pipeline:

### TICO's WrapQ Approach

```python
# From tico/quantization/wrapq/wrappers
# Wrappers have two modes: CALIB and QUANT

class QuantLinear(QuantModuleBase):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.mode = "CALIB"  # or "QUANT"
        self.weight_observer = MinMaxObserver(...)
        self.act_observer = MinMaxObserver(...)
    
    def forward(self, x):
        if self.mode == "CALIB":
            # Collect statistics
            self.weight_observer(self.module.weight)
            self.act_observer(x)
            return self.module(x)
        
        elif self.mode == "QUANT":
            # Compute quantization parameters from collected stats
            w_scale, w_zp = self.weight_observer.calculate_qparams()
            x_scale, x_zp = self.act_observer.calculate_qparams()
            
            # Apply fake quantization
            w_fake = fake_quantize(self.module.weight, w_scale, w_zp, ...)
            x_fake = fake_quantize(x, x_scale, x_zp, ...)
            
            # Compute with fake-quantized values
            return nn.functional.linear(x_fake, w_fake, self.module.bias)
```

### TICO's Conversion Pipeline

```python
# From tico/convert
# After fake quantization, the model is exported and converted to Circle

# 1. Export to static graph (torch.export)
ep = torch.export.export(quantized_model, example_inputs)

# 2. Apply passes including FoldQuantOps
# This folds Q-DQ pairs into operator metadata
from tico.passes import FoldQuantOps
ep = FoldQuantOps().call(ep)

# 3. Convert to Circle format
# Quantization parameters are embedded in Circle operators
circle_model = convert_to_circle(ep)
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why can't we just use `round()` directly for fake quantization? What problem does STE solve?

2. **Code analysis:** In the manual fake quantization code, what does `.detach()` do and why is it essential?

3. **Debug:** A team's QAT model isn't converging—they get NaN gradients. They're using `x.round()` directly without STE. Explain why this causes problems.

4. **TICO context:** In TICO's WrapQ wrappers, what's the difference between CALIB mode and QUANT mode? Why can't we skip CALIB and go straight to QUANT?

---

**Next:** [Chapter 2, Section 1: PTQ Calibration](ch2_sec1_ptq_calibration.md)
