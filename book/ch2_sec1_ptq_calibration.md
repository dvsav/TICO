# Chapter 2, Section 1: PTQ Calibration

**Summary:** This section dives into the calibration phase of Post-Training Quantization (PTQ). We explore how activation statistics are collected, the role of observers, and different strategies for determining optimal quantization parameters without retraining.

**Prerequisites:** Chapter 1, Section 4 (Fake Quantization), understanding of scale and zero-point computation.

**Key Takeaways:**
- Calibration collects activation statistics to determine quantization parameters
- Observers are the primary mechanism for statistics collection in PyTorch
- MinMax and EMA observers provide different trade-offs
- KL divergence calibration offers better accuracy for certain distributions
- Calibration data quality directly impacts quantization accuracy

---

## What is Calibration?

**Calibration** is the process of collecting activation statistics from a pre-trained model by running real data through it. These statistics are then used to compute optimal quantization parameters (scale and zero-point) for each tensor that needs to be quantized.

### Why Calibration is Necessary

For **weights**, quantization is straightforward:
- Weights are known constants (model parameters)
- We can directly compute min/max from the weight tensor
- No calibration data needed

For **activations**, it's more complex:
- Activations are dynamic—they depend on the input data
- We don't know the activation range until we run data through the model
- Different inputs produce different activation distributions
- We need representative data to estimate the "true" activation range

### The Calibration Workflow

```
FP32 Model
    ↓
[Wrap with Observers]  →  Model that collects stats during forward pass
    ↓
[Run Calibration Data]  →  Observers accumulate min/max/histograms
    ↓
[Compute QParams]  →  Convert stats to scale and zero-point
    ↓
[Insert Fake Quantization]  →  Model ready for evaluation/export
```

---

## Observers: The Statistics Collectors

An **observer** is a PyTorch module that wraps a tensor and collects statistics about its values during the forward pass.

### Observer Interface

```python
class ObserverBase(nn.Module):
    def __init__(self, dtype, qscheme):
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
    
    def forward(self, x):
        # Collect statistics about x
        self._collect_stats(x)
        return x  # Return input unchanged
    
    def calculate_qparams(self):
        # Compute scale and zero-point from collected stats
        return scale, zero_point
```

### Common Observer Types

#### 1. MinMax Observer

The simplest observer—tracks the global min and max values seen during calibration.

```python
from torch.quantization import MinMaxObserver

observer = MinMaxObserver(
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine
)

# During calibration:
observer(x1)  # x1: min=-2.3, max=5.1
observer(x2)  # x2: min=-1.8, max=6.2
observer(x3)  # x3: min=-2.5, max=4.8

# After calibration:
# accumulated_min = -2.5
# accumulated_max = 6.2
scale, zero_point = observer.calculate_qparams()
```

**Advantages:**
- Simple and intuitive
- Captures the full dynamic range
- Works well for bounded distributions

**Disadvantages:**
- Sensitive to outliers (a single extreme value affects the entire range)
- May overestimate the range for long-tailed distributions

#### 2. EMA (Exponential Moving Average) Observer

Tracks a running average of min/max, giving more weight to recent batches.

```python
from torch.quantization import EMAQuantileObserver

observer = EMAQuantileObserver(
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    ema_alpha=0.1  # Lower alpha = more smoothing
)
```

**Advantages:**
- More robust to outliers than MinMax
- Adapts to changing distributions during calibration

**Disadvantages:**
- Requires tuning the EMA alpha parameter
- May under-estimate range if calibration data is not representative

#### 3. Histogram Observer

Builds a histogram of values and uses it to compute optimal thresholds.

```python
from torch.quantization import HistogramObserver

observer = HistogramObserver(
    bins=2048,  # Number of histogram bins
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine
)
```

**Advantages:**
- Enables KL divergence calibration (see below)
- More robust to outliers
- Can find optimal threshold that minimizes information loss

**Disadvantages:**
- Higher memory overhead (stores histogram)
- More computationally expensive

#### 4. Percentile Observer

Uses percentile values instead of absolute min/max.

```python
from torch.quantization import PercentileObserver

observer = PercentileObserver(
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    lower_percentile=0.5,   # Ignore bottom 0.5%
    upper_percentile=99.5   # Ignore top 0.5%
)
```

**Advantages:**
- Automatically ignores outliers
- Simple to configure

**Disadvantages:**
- May clip meaningful values if percentiles are not chosen carefully

---

## KL Divergence Calibration

**KL (Kullback-Leibler) divergence** calibration is a more sophisticated approach that finds the optimal quantization threshold by minimizing the information loss between the original and quantized distributions.

### The Idea

Instead of using min/max directly, KL divergence calibration:
1. Builds a histogram of the activation values
2. Tries different threshold values
3. For each threshold, computes the quantized distribution
4. Selects the threshold that minimizes KL divergence between original and quantized distributions

### KL Divergence Formula

```
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

Where:
- `P` is the original distribution (histogram)
- `Q` is the quantized distribution

### Algorithm

```python
def kl_divergence_calibration(histogram, num_bins=256):
    """
    Find optimal threshold using KL divergence.
    """
    min_divergence = float('inf')
    best_threshold = None
    
    # Try different thresholds
    for threshold_bin in range(num_bins // 2, num_bins):
        # Quantize histogram to threshold_bin levels
        quantized_histogram = quantize_histogram(histogram, threshold_bin)
        
        # Compute KL divergence
        divergence = compute_kl_divergence(histogram, quantized_histogram)
        
        if divergence < min_divergence:
            min_divergence = divergence
            best_threshold = threshold_bin
    
    return best_threshold
```

### When to Use KL Divergence

**Good for:**
- Activations with long-tailed distributions
- When outliers would dominate MinMax calibration
- When maximum accuracy is needed

**Not necessary for:**
- Weights (usually well-behaved distributions)
- Activations with bounded, symmetric distributions
- When calibration speed is more important than accuracy

---

## Calibration Data: Quality Matters

The choice of calibration data significantly impacts quantization quality.

### How Much Data?

| Dataset Size | Use Case | Recommendation |
|--------------|----------|----------------|
| 10-50 samples | Quick sanity check | Not recommended for production |
| 100-500 samples | Standard PTQ | Good balance of speed/accuracy |
| 1000+ samples | High-accuracy PTQ | Diminishing returns after ~500 |
| Full validation set | Maximum accuracy | Only if time permits |

### What Kind of Data?

**Ideal calibration data:**
- Representative of the target deployment distribution
- Covers the full range of expected inputs
- Diverse enough to activate different parts of the model

**For language models:**
- Use diverse text from multiple domains
- Include various sequence lengths
- Cover different topics and styles

**For vision models:**
- Include varied scenes, objects, lighting conditions
- Cover different resolutions if applicable
- Balance class distribution if relevant

### PyTorch Example: Calibration Loop

```python
import torch
from torch.utils.data import DataLoader

def calibrate_model(model, calibration_loader, device='cuda'):
    """
    Calibrate a model with observers.
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calibration_loader):
            images = images.to(device)
            
            # Forward pass - observers collect stats
            _ = model(images)
            
            # Optional: print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Calibrated {batch_idx + 1} batches")
    
    return model

# Usage
from torch.quantization import get_default_qconfig, prepare

# Prepare model with observers
qconfig = get_default_qconfig('fbgemm')  # or 'qnnpack' for ARM
model_prepared = prepare(model, qconfig)

# Calibrate
model_calibrated = calibrate_model(model_prepared, calibration_loader)

# Convert to quantized model
model_quantized = convert(model_calibrated)
```

---

## Per-Channel vs. Per-Tensor Calibration

Just like quantization, calibration can be done at different granularities.

### Per-Tensor Calibration

```python
# Single observer for entire tensor
observer = MinMaxObserver(...)

# During calibration
observer activation_tensor)  # Collects global min/max

# QParams computation
scale, zero_point = observer.calculate_qparams()  # Single values
```

### Per-Channel Calibration

```python
# Observer per channel (for Conv2d/Linear weight quantization)
from torch.quantization import default_per_channel_qconfig

qconfig = default_per_channel_qconfig

# For weight tensor [out_channels, in_channels, ...]
# Each output channel gets its own observer
# During calibration:
for i in range(out_channels):
    channel_observer[i](weight[i])  # Collect per-channel stats

# QParams computation
scales, zero_points = observer.calculate_qparams()  # Vectors
```

### Which to Choose?

| Aspect | Per-Tensor | Per-Channel |
|--------|------------|-------------|
| **Accuracy** | Good | Better (handles channel variance) |
| **Metadata overhead** | Low (2 scalars) | Higher (2 vectors) |
| **Hardware support** | Universal | May not be supported on all NPUs |
| **Recommended for** | Activations | Weights |

---

## PyTorch Experiment: Calibration Comparison

```python
import torch
import torch.nn as nn
from torch.quantization import (
    MinMaxObserver,
    HistogramObserver,
    EMAQuantileObserver,
    default_qconfig
)

def compare_observers(data, observer_classes, bits=8):
    """
    Compare different observer types on the same data.
    """
    q_min = 0
    q_max = 2 ** bits - 1
    
    results = {}
    
    for ObserverClass in observer_classes:
        # Create observer
        if ObserverClass == HistogramObserver:
            observer = ObserverClass(
                bins=2048,
                quant_min=q_min,
                quant_max=q_max,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            )
        else:
            observer = ObserverClass(
                quant_min=q_min,
                quant_max=q_max,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            )
        
        # Simulate calibration: feed data batches
        for batch in data:
            observer(batch)
        
        # Get quantization parameters
        scale, zero_point = observer.calculate_qparams()
        
        # Compute quantization error on held-out data
        test_data = torch.cat(data)
        x_q = torch.round(test_data / scale + zero_point).clamp(q_min, q_max)
        x_deq = (x_q - zero_point) * scale
        mae = (test_data - x_deq).abs().mean().item()
        
        results[ObserverClass.__name__] = {
            'scale': scale.item(),
            'zero_point': zero_point.item(),
            'mae': mae
        }
    
    return results


# Generate synthetic calibration data with outliers
torch.manual_seed(42)
n_batches = 50
batch_size = 100

# Normal data with occasional outliers
data = []
for _ in range(n_batches):
    batch = torch.randn(batch_size)
    # Add 1% outliers
    outlier_mask = torch.rand(batch_size) < 0.01
    batch[outlier_mask] = torch.randn(outlier_mask.sum()) * 5
    data.append(batch)

# Compare observers
observer_classes = [
    MinMaxObserver,
    HistogramObserver,
    EMAQuantileObserver
]

results = compare_observers(data, observer_classes)

print("Observer Comparison (with outliers):")
print("=" * 60)
for name, stats in results.items():
    print(f"{name}:")
    print(f"  Scale: {stats['scale']:.6f}")
    print(f"  Zero-point: {stats['zero_point']}")
    print(f"  MAE: {stats['mae']:.6f}")
    print()
```

**Expected output:**
```
Observer Comparison (with outliers):
============================================================
MinMaxObserver:
  Scale: 0.058824
  Zero-point: 85
  MAE: 0.015234  <-- Higher due to outlier sensitivity

HistogramObserver:
  Scale: 0.039216
  Zero-point: 128
  MAE: 0.009876  <-- Better, ignores outliers

EMAQuantileObserver:
  Scale: 0.041176
  Zero-point: 125
  MAE: 0.010543  <-- Good balance
```

---

## Calibration in TICO

In TICO's WrapQ framework, calibration is integrated into the wrapper's forward pass:

```python
# From tico/quantization/wrapq/wrappers
class QuantLinear(QuantModuleBase):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.mode = "CALIB"  # or "QUANT"
        
        # Observers for statistics collection
        self.weight_observer = MinMaxObserver(...)
        self.act_observer = MinMaxObserver(...)
    
    def forward(self, x):
        if self.mode == "CALIB":
            # Collect statistics without modifying computation
            self.weight_observer(self.module.weight)
            self.act_observer(x)
            return self.module(x)  # FP32 forward
        
        elif self.mode == "QUANT":
            # Compute qparams from collected stats
            w_scale, w_zp = self.weight_observer.calculate_qparams()
            x_scale, x_zp = self.act_observer.calculate_qparams()
            
            # Apply fake quantization
            w_fake = fake_quantize(self.module.weight, w_scale, w_zp, ...)
            x_fake = fake_quantize(x, x_scale, x_zp, ...)
            
            return nn.functional.linear(x_fake, w_fake, self.module.bias)

# Calibration loop
def calibrate(model, dataloader):
    for wrapper in model.modules():
        if isinstance(wrapper, QuantModuleBase):
            wrapper.set_mode("CALIB")
    
    for batch in dataloader:
        model(batch)  # Collects stats
    
    # Switch to QUANT mode
    for wrapper in model.modules():
        if isinstance(wrapper, QuantModuleBase):
            wrapper.set_mode("QUANT")
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why don't weights need calibration, but activations do?

2. **Debug:** A team calibrates their vision model using only images of cats. At deployment, the model processes diverse images. What problems might arise?

3. **Trade-offs:** When would you choose MinMax observer over Histogram observer, despite Histogram's better accuracy?

4. **TICO context:** In TICO's wrappers, what happens if you switch to QUANT mode before running any calibration data? What would the scale and zero-point be?

---

**Next:** [Chapter 2, Section 2: Weight Quantization](ch2_sec2_weight_quantization.md)
