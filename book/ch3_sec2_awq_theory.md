# Chapter 3, Section 2: AWQ Algorithm

**Summary:** This section covers AWQ (Activation-aware Weight Quantization), a method that protects salient weights based on activation magnitude. We explore the key insight that not all weights are equally important and how AWQ leverages activation statistics to identify and preserve critical weights.

**Prerequisites:** Chapter 3, Section 1 (GPTQ Algorithm), Chapter 2, Section 3 (Activation Quantization).

**Key Takeaways:**
- Salient weights (those connected to high-magnitude activations) are critical for accuracy
- AWQ protects salient weights with higher precision quantization
- Channel-wise scaling redistributes quantization difficulty
- AWQ enables INT4 weight quantization with minimal accuracy loss

---

## The Key Insight: Not All Weights Are Equal

AWQ is built on a fundamental observation about neural network weights:

> **Weights associated with high-magnitude activations are more important for model accuracy.**

### Salient Weights

**Definition:** A weight `W[i,j]` is **salient** if the corresponding activation `X[:,j]` has high magnitude.

```
Intuition:
  Output y = X @ W
  
  If activation X[:,j] is large, then weight column W[:,j] has big impact on output.
  If activation X[:,j] is small (near zero), weight column W[:,j] barely matters.
  
  Therefore: Quantize salient weights more carefully!
```

### Empirical Observation

Research analyzing LLM activations revealed:

1. **Channel-specific outliers:** Certain feature channels consistently have high-magnitude activations across different inputs.

2. **Persistent salience:** Salient channels remain salient across layers and inputs.

3. **Small fraction:** Only ~1% of channels are truly salient, but they contribute disproportionately to accuracy.

```
Activation magnitude per channel (LLaMA-7B, example):

Channel:    0    50   100  150  200  250  300  350  400
Magnitude:  █    █    █    ▓    █    █    ▓    █    █
            │    │    │    │    │    │    │    │    │
           low  low  low  HIGH low  low  HIGH low  low

▓ = Salient channel (top 1%)
█ = Normal channel
```

---

## The AWQ Approach

AWQ protects salient weights by applying channel-wise scaling before quantization.

### High-Level Algorithm

```python
# AWQ pseudocode
def awq_quantize(W, X, bits=4, n_salient=0.01):
    """
    W: weight matrix [out_features, in_features]
    X: activation statistics [batch, in_features]
    bits: quantization bit width
    n_salient: fraction of salient channels to protect
    """
    # Step 1: Identify salient channels
    channel_importance = compute_channel_importance(X)
    salient_channels = top_k(channel_importance, k=n_salient * in_features)
    
    # Step 2: Compute per-channel scaling factors
    # Scale down salient channels, scale up non-salient
    scales = compute_scaling_factors(W, salient_channels)
    
    # Step 3: Apply scaling
    W_scaled = W * scales  # Broadcasting: [out, 1] * [1, in]
    
    # Step 4: Quantize scaled weights
    W_q = quantize(W_scaled, bits=bits)
    
    # Step 5: Un-scale for dequantization
    W_deq = W_q * scales
    
    return W_deq, scales
```

### Channel Importance Metric

AWQ uses activation magnitude as the importance metric:

```python
def compute_channel_importance(X):
    """
    X: activations [batch, seq, in_features]
    Returns: importance score per channel [in_features]
    """
    # Mean absolute activation per channel
    importance = X.abs().mean(dim=(0, 1))
    
    # Alternative: max activation (more sensitive to outliers)
    # importance = X.abs().max(dim=(0, 1)).values
    
    return importance
```

### Scaling Factor Computation

The key innovation in AWQ is the scaling strategy:

```python
def compute_scaling_factors(W, salient_channels, alpha=0.5):
    """
    W: weight matrix [out_features, in_features]
    salient_channels: indices of salient channels
    alpha: controls scaling strength (0 = no scaling, 1 = full scaling)
    """
    in_features = W.shape[1]
    
    # Initialize scales to 1
    scales = torch.ones(in_features, device=W.device)
    
    # For salient channels: scale DOWN (makes weights smaller, easier to quantize)
    # For non-salient channels: scale UP (absorbs the scaling)
    
    salient_scale = 0.5  # Example: reduce salient by 2x
    non_salient_scale = 2.0  # Compensate
    
    for j in range(in_features):
        if j in salient_channels:
            scales[j] = salient_scale ** alpha
        else:
            scales[j] = non_salient_scale ** alpha
    
    return scales
```

**Why this works:**
- Salient weights become smaller → less quantization error relative to their importance
- Non-salient weights become larger → more quantization error, but they don't matter as much
- The scaling is designed to preserve the overall output magnitude

---

## AWQ vs. GPTQ: Different Approaches, Same Goal

Both AWQ and GPTQ aim to improve low-bit quantization, but with different strategies:

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| **Core idea** | Error compensation via Hessian | Protect salient weights |
| **Information used** | Input correlations (Hessian) | Activation magnitude |
| **Processing order** | Sequential (column-by-column) | Parallel (all at once) |
| **Memory** | O(d²) for Hessian | O(d) for scales |
| **Speed** | Slower (sequential) | Faster (parallel) |
| **Accuracy** | Excellent for INT4 | Very good for INT4 |

### Complementary Techniques

AWQ and GPTQ can be combined:

```python
# AWQ + GPTQ pipeline
def awq_gptq_quantize(W, X, bits=4):
    # Step 1: Apply AWQ scaling
    scales = compute_awq_scales(W, X)
    W_scaled = W * scales
    
    # Step 2: Apply GPTQ to scaled weights
    W_q = gptq_quantize(W_scaled, X, bits=bits)
    
    # Step 3: Un-scale
    W_deq = W_q / scales
    
    return W_deq
```

---

## The AWQ Quantization Pipeline

### Full Implementation

```python
import torch
import torch.nn as nn

def find_salient_channels(X, salient_ratio=0.01):
    """
    Identify salient channels based on activation magnitude.
    
    Args:
        X: activations [batch, seq, in_features]
        salient_ratio: fraction of channels to mark as salient
    
    Returns:
        salient_mask: boolean mask [in_features]
    """
    # Compute channel importance (mean absolute activation)
    importance = X.abs().mean(dim=(0, 1))
    
    # Find threshold for top salient_ratio channels
    k = int(X.shape[-1] * salient_ratio)
    threshold = importance.topk(k).values[-1]
    
    # Create mask
    salient_mask = importance >= threshold
    
    return salient_mask

def compute_awq_scales(W, X, salient_ratio=0.01, alpha=0.5):
    """
    Compute AWQ per-channel scaling factors.
    
    Args:
        W: weight matrix [out_features, in_features]
        X: activations [batch, seq, in_features]
        salient_ratio: fraction of salient channels
        alpha: scaling strength
    
    Returns:
        scales: per-channel scaling factors [in_features]
    """
    in_features = W.shape[1]
    
    # Find salient channels
    salient_mask = find_salient_channels(X, salient_ratio)
    
    # Compute weight magnitude per channel
    weight_mag = W.abs().mean(dim=0)  # [in_features]
    
    # Compute activation magnitude per channel
    act_mag = X.abs().mean(dim=(0, 1))  # [in_features]
    
    # Combined importance: weight × activation
    combined_importance = weight_mag * act_mag
    
    # Normalize importance
    combined_importance = combined_importance / combined_importance.max()
    
    # Compute scales: inverse relationship with importance
    # High importance → small scale (protect)
    # Low importance → large scale (sacrifice)
    scales = combined_importance ** (-alpha)
    
    # Normalize scales to have mean 1 (preserve overall magnitude)
    scales = scales / scales.mean()
    
    return scales

def awq_quantize_weights(W, X, bits=4, salient_ratio=0.01, alpha=0.5):
    """
    Full AWQ quantization pipeline.
    
    Args:
        W: weight matrix [out_features, in_features]
        X: activations [batch, seq, in_features]
        bits: quantization bit width
        salient_ratio: fraction of salient channels
        alpha: scaling strength
    
    Returns:
        W_q: quantized weights (dequantized for FP computation)
        scales: AWQ scaling factors
    """
    # Step 1: Compute AWQ scales
    scales = compute_awq_scales(W, X, salient_ratio, alpha)
    
    # Step 2: Apply scaling to weights
    W_scaled = W * scales.unsqueeze(0)  # [out, in] * [1, in]
    
    # Step 3: Quantize scaled weights
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max - 1
    
    W_max = W_scaled.abs().max()
    scale = W_max / q_max
    
    W_q = torch.round(W_scaled / scale).clamp(q_min, q_max)
    W_deq = W_q * scale
    
    # Step 4: Un-scale
    W_deq = W_deq / scales.unsqueeze(0)
    
    return W_deq, scales

# Usage: Quantize an LLM with AWQ
def quantize_llm_with_awq(model, calibration_loader, bits=4):
    """
    Quantize all Linear layers in an LLM using AWQ.
    """
    model.eval()
    
    # Collect activations for each layer
    layer_activations = {}
    
    def create_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            layer_activations[layer_name] = input.detach()
        return hook_fn
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(create_hook(name)))
    
    # Run calibration data
    with torch.no_grad():
        for batch in calibration_loader:
            model(batch)
            if len(layer_activations) >= 100:  # Collect enough samples
                break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Quantize each Linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in layer_activations:
            W = module.weight.data
            X = layer_activations[name]
            
            print(f"AWQ quantizing {name}...")
            W_q, scales = awq_quantize_weights(W, X, bits=bits)
            
            # Replace weight
            module.weight = nn.Parameter(W_q, requires_grad=False)
            
            # Store scales for inference
            module.register_buffer('awq_scales', scales)
    
    return model
```

---

## Understanding AWQ Scaling: A Visual Example

```
Before AWQ:
Weights:     [1.0] [0.5] [0.1] [0.1] [0.1] [0.1] [0.1] [0.1]
             │     │     │     │     │     │     │     │
Activations: [10]  [1]   [1]   [1]   [1]   [1]   [1]   [1]
             ↑
         Salient channel!

After AWQ scaling (alpha=0.5):
Scales:      [0.5] [1.1] [1.1] [1.1] [1.1] [1.1] [1.1] [1.1]
             ↓     ↑     ↑     ↑     ↑     ↑     ↑     ↑
Weights:     [0.5] [0.55][0.11][0.11][0.11][0.11][0.11][0.11]
             │
         Smaller → easier to quantize!

Quantization error impact:
Before: Salient weight error = 0.1 × 10 (activation) = 1.0 output error
After:  Salient weight error = 0.05 × 10 (activation) = 0.5 output error
        (50% reduction in output error!)
```

---

## AWQ Hyperparameters

### Salient Ratio

The fraction of channels marked as salient:

| Value | Effect | Recommended For |
|-------|--------|-----------------|
| 0.001 (0.1%) | Very few salient channels | Models with extreme outliers |
| 0.01 (1%) | Default | Most LLMs |
| 0.05 (5%) | More channels protected | Vision models, VLMs |

### Alpha (Scaling Strength)

Controls how aggressively salient channels are protected:

| Value | Effect | Recommended For |
|-------|--------|-----------------|
| 0.0 | No scaling (standard quantization) | Baseline |
| 0.3 | Mild scaling | INT8 quantization |
| 0.5 | Default | INT4 quantization |
| 0.7 | Aggressive scaling | INT3 or lower |

### Choosing Hyperparameters

```python
def tune_awq_hyperparameters(model, val_data, bits=4):
    """
    Grid search for optimal AWQ hyperparameters.
    """
    best_accuracy = 0
    best_hparams = {'salient_ratio': 0.01, 'alpha': 0.5}
    
    for salient_ratio in [0.005, 0.01, 0.02]:
        for alpha in [0.3, 0.5, 0.7]:
            # Quantize model
            quantized_model = quantize_llm_with_awq(
                model, calib_loader, bits=bits,
                salient_ratio=salient_ratio, alpha=alpha
            )
            
            # Evaluate
            accuracy = evaluate(quantized_model, val_data)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hparams = {'salient_ratio': salient_ratio, 'alpha': alpha}
    
    return best_hparams, best_accuracy
```

---

## AWQ Extensions and Variants

### 1. Group-wise AWQ

Instead of per-channel scaling, use group-wise scaling:

```python
def group_awq_scales(W, X, group_size=128):
    """
    AWQ with group-wise scaling (reduces metadata overhead).
    """
    in_features = W.shape[1]
    n_groups = in_features // group_size
    
    scales = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        
        # Compute group importance
        group_importance = X[:, :, start:end].abs().mean()
        
        # Compute group scale
        scale = group_importance ** (-0.5)
        scales.append(scale)
    
    return torch.tensor(scales)
```

### 2. AutoAWQ: Automated Hyperparameter Selection

Automatically tune AWQ parameters based on model characteristics:

```python
def auto_awq_scales(W, X):
    """
    Automatically determine AWQ parameters.
    """
    # Analyze activation distribution
    act_mag = X.abs().mean(dim=(0, 1))
    
    # Compute coefficient of variation (CV)
    cv = act_mag.std() / act_mag.mean()
    
    # Higher CV → more outliers → need more aggressive protection
    salient_ratio = min(0.02, max(0.005, cv * 0.01))
    alpha = min(0.7, max(0.3, cv * 0.5))
    
    return compute_awq_scales(W, X, salient_ratio, alpha)
```

### 3. AWQ for VLMs (Vision-Language Models)

Extend AWQ to handle multimodal inputs:

```python
def awq_vlm_scales(W, text_X, image_X):
    """
    AWQ scaling for VLMs with separate text and image activations.
    """
    # Compute importance for each modality
    text_importance = text_X.abs().mean(dim=(0, 1))
    image_importance = image_X.abs().mean(dim=(0, 1))
    
    # Combine (weighted by modality importance)
    combined_importance = 0.7 * text_importance + 0.3 * image_importance
    
    # Compute scales
    scales = combined_importance ** (-0.5)
    scales = scales / scales.mean()
    
    return scales
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why does AWQ scale down salient channels instead of scaling them up?

2. **Calculation:** A salient channel has activation magnitude 10× higher than average. With alpha=0.5, what is its scaling factor relative to average?

3. **Debug:** A team applies AWQ but sees no accuracy improvement. They used `salient_ratio=0.5` (50% salient channels). What went wrong?

4. **Comparison:** When would you choose AWQ over GPTQ for a production deployment?

---

**Next:** [Chapter 3, Section 3: SpinQuant Algorithm](ch3_sec3_spinquant_theory.md)
