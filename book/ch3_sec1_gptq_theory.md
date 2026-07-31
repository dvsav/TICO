# Chapter 3, Section 1: GPTQ Algorithm

**Summary:** This section introduces GPTQ (Gradient-based Post-Training Quantization), a layer-by-layer quantization method that minimizes reconstruction error using second-order information. We explore the mathematical foundation, the algorithm's key innovations, and practical implementation considerations.

**Prerequisites:** Chapter 2 (PTQ fundamentals), basic understanding of matrix operations and optimization.

**Key Takeaways:**
- GPTQ minimizes layer-wise reconstruction error, not just quantization error
- Uses approximate second-order (Hessian) information for optimal weight updates
- Processes weights column-by-column for computational efficiency
- Achieves near-lossless INT4 quantization for LLMs

---

## Motivation: Beyond Standard PTQ

Standard Post-Training Quantization (PTQ) has a fundamental limitation:

```
Standard PTQ:
  For each weight w: w_q = round(w / s) × s
  
  Problem: Each weight is quantized independently!
  Ignores how weights interact during matrix multiplication.
```

**Key insight:** The output of a Linear layer is `y = X @ W`. If we quantize weights to `W_q`, the output error is:

```
Output error = X @ W - X @ W_q = X @ (W - W_q)

This depends on BOTH the weight error (W - W_q) AND the input X!
```

GPTQ addresses this by asking: **"What if we could adjust remaining unquantized weights to compensate for already-quantized weights?"**

---

## The GPTQ Objective

GPTQ formulates quantization as an optimization problem.

### Layer-wise Reconstruction Error

For a Linear layer with input `X` and weights `W`, GPTQ seeks quantized weights `Ŵ` that minimize:

```
minimize: E[||X @ W - X @ Ŵ||²]
subject to: Ŵ[i,j] ∈ {quantization levels}
```

Where the expectation is over the calibration data distribution.

### Reformulation with Hessian

Using a Taylor expansion, this can be approximated as:

```
minimize: (W - Ŵ)ᵀ @ H @ (W - Ŵ)
subject to: Ŵ[i,j] ∈ {quantization levels}
```

Where `H = Xᵀ @ X` is the Hessian (second derivative) of the loss with respect to weights.

**Intuition:** The Hessian tells us how sensitive the output is to changes in each weight. Weights with high Hessian values are "important" and should be quantized more carefully.

---

## The GPTQ Algorithm

GPTQ solves this optimization problem efficiently using a greedy column-by-column approach.

### High-Level Algorithm

```python
# Pseudocode for GPTQ
def gptq_quantize(W, X, bits=4):
    """
    W: weight matrix [out_features, in_features]
    X: input activations [batch, in_features] (calibration data)
    bits: quantization bit width
    """
    d = W.shape[1]  # Number of input features
    Ŵ = W.clone()   # Quantized weights (initially copy of W)
    
    # Compute Hessian inverse (preconditioner)
    H = X.T @ X  # [in_features, in_features]
    H_inv = inverse(H + λI)  # Add damping for stability
    
    # Process columns one at a time
    for j in range(d):
        # Get j-th column of weights
        w_j = Ŵ[:, j]
        
        # Quantize this column
        w_j_q = quantize(w_j, bits=bits)
        
        # Compute quantization error
        error_j = w_j - w_j_q
        
        # Update remaining columns to compensate
        for k in range(j + 1, d):
            Ŵ[:, k] -= error_j @ H_inv[j, k]
        
        # Store quantized column
        Ŵ[:, j] = w_j_q
    
    return Ŵ
```

### Key Innovations

#### 1. Column-by-Column Processing

Instead of quantizing all weights independently, GPTQ processes columns sequentially:

```
Iteration 0:  [Q][ ][ ][ ]  → Quantize column 0
              ↓
Iteration 1:  [Q][Q][ ][ ]  → Quantize column 1, compensate for column 0
              ↓
Iteration 2:  [Q][Q][Q][ ]  → Quantize column 2, compensate for columns 0,1
              ↓
Iteration 3:  [Q][Q][Q][Q]  → Done!
```

#### 2. Error Compensation

After quantizing column `j`, GPTQ updates remaining columns to "undo" the quantization error:

```
Ŵ[:, k] ← Ŵ[:, k] - (W[:, j] - Ŵ[:, j]) × H_inv[j, k]
         └──────┘   └──────┘            └────┘
         updated    old value   error    compensation
```

**Intuition:** If quantizing column `j` causes output error, we can partially cancel this error by adjusting column `k` in the opposite direction.

#### 3. Hessian-Based Compensation

The compensation is weighted by the inverse Hessian `H_inv[j, k]`, which captures:
- How correlated columns `j` and `k` are (via `X[:, j]ᵀ @ X[:, k]`)
- How much adjusting column `k` affects the output

---

## Mathematical Derivation (Optional Deep Dive)

### From Reconstruction Error to Quadratic Form

Starting with the reconstruction error:

```
L(Ŵ) = E[||X @ W - X @ Ŵ||²]
     = E[tr((X @ W - X @ Ŵ)ᵀ @ (X @ W - X @ Ŵ))]
     = E[tr((W - Ŵ)ᵀ @ Xᵀ @ X @ (W - Ŵ))]
     = (W - Ŵ)ᵀ @ E[Xᵀ @ X] @ (W - Ŵ)
     = (W - Ŵ)ᵀ @ H @ (W - Ŵ)
```

Where `H = E[Xᵀ @ X]` is the expected Hessian.

### Greedy Column Selection

The optimal solution to this quadratic problem with quantization constraints is NP-hard. GPTQ uses a greedy approximation:

For each column `j`:
1. Quantize `w_j` to nearest quantization level
2. Compute the "damage" (output error) caused by this quantization
3. Update remaining columns to minimize this damage

The optimal update for remaining columns is:

```
ΔW[:, j+1:d] = -H_inv[j+1:d, j] × (w_j - ŵ_j) / H_inv[j, j]
```

This is derived by setting the gradient of the loss with respect to remaining columns to zero.

---

## Practical Implementation Details

### Efficient Hessian Inverse Computation

Computing and inverting the full Hessian is expensive for large models. GPTQ uses several optimizations:

```python
def compute_hessian_inverse(X, damping=0.01):
    """
    Compute damped Hessian inverse efficiently.
    X: [batch, in_features] - calibration activations
    """
    d = X.shape[1]
    
    # Compute Hessian: H = Xᵀ @ X
    H = X.T @ X
    
    # Add damping for numerical stability
    # This prevents division by near-zero eigenvalues
    damp = damping * torch.diag(H).mean()
    H = H + damp * torch.eye(d, device=H.device)
    
    # Compute inverse using Cholesky decomposition (more stable)
    L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    
    return H_inv
```

### Block-wise Processing

For very large models, process columns in blocks to reduce memory:

```python
def gptq_blockwise(W, X, bits=4, block_size=128):
    """
    GPTQ with block-wise processing for memory efficiency.
    """
    d = W.shape[1]
    Ŵ = W.clone()
    
    H = X.T @ X
    damp = 0.01 * torch.diag(H).mean()
    H = H + damp * torch.eye(d, device=H.device)
    
    # Process in blocks
    for block_start in range(0, d, block_size):
        block_end = min(block_start + block_size, d)
        
        # Compute inverse for this block
        H_block = H[block_start:block_end, block_start:block_end]
        H_block_inv = torch.inverse(H_block)
        
        # Process columns within block
        for j in range(block_start, block_end):
            # Quantize column j
            w_j = Ŵ[:, j]
            w_j_q = quantize(w_j, bits=bits)
            
            # Compute error
            error = w_j - w_j_q
            
            # Update remaining columns in this block
            for k in range(j + 1, block_end):
                Ŵ[:, k] -= error * H_block_inv[j - block_start, k - block_start]
            
            Ŵ[:, j] = w_j_q
    
    return Ŵ
```

### Damping Parameter

The damping parameter `λ` (or `damping`) controls numerical stability:

```
H_damped = H + λ × mean(diag(H)) × I
```

**Typical values:**
- `λ = 0.01`: Default, good for most models
- `λ = 0.1`: More stable, but less accurate compensation
- `λ = 0.001`: Less damping, may be unstable for ill-conditioned Hessians

---

## GPTQ vs. Standard PTQ: Comparison

| Aspect | Standard PTQ | GPTQ |
|--------|--------------|------|
| **Objective** | Minimize per-weight error | Minimize output reconstruction error |
| **Weight interaction** | Independent | Accounts for correlations |
| **Bit width** | INT8 typical | INT4 achievable |
| **Calibration** | For activations only | For Hessian computation |
| **Runtime** | Fast (parallel) | Slower (sequential) |
| **Memory** | Low | Higher (Hessian storage) |
| **Accuracy** | Good for INT8 | Excellent for INT4 |

---

## PyTorch Implementation: Full GPTQ

```python
import torch
import torch.nn as nn

def quantize_to_grid(w, bits=4, symmetric=True):
    """Quantize a vector to discrete grid."""
    if symmetric:
        q_max = 2 ** (bits - 1) - 1
        q_min = -q_max - 1
        
        w_max = w.abs().max()
        scale = w_max / q_max
        
        w_q = torch.round(w / scale).clamp(q_min, q_max)
        w_deq = w_q * scale
    else:
        q_max = 2 ** bits - 1
        q_min = 0
        
        w_min, w_max = w.min(), w.max()
        scale = (w_max - w_min) / (q_max - q_min)
        zero_point = torch.round(-w_min / scale).clamp(q_min, q_max)
        
        w_q = torch.round(w / scale + zero_point).clamp(q_min, q_max)
        w_deq = (w_q - zero_point) * scale
    
    return w_deq

def gptq_quantize_linear(layer, X, bits=4, damp=0.01, per_channel=True):
    """
    Apply GPTQ to a Linear layer.
    
    Args:
        layer: nn.Linear module to quantize
        X: Calibration activations [batch, seq, in_features]
        bits: Quantization bit width
        damp: Damping parameter for Hessian inversion
        per_channel: Whether to use per-channel quantization
    
    Returns:
        Quantized weight tensor
    """
    # Flatten input for Hessian computation
    # X: [batch, seq, in_features] → [batch*seq, in_features]
    if len(X.shape) == 3:
        X = X.reshape(-1, X.shape[-1])
    
    in_features = layer.in_features
    W = layer.weight.data  # [out_features, in_features]
    
    # Compute Hessian
    H = X.T @ X / X.shape[0]  # Normalize by batch size
    
    # Add damping
    damp_param = damp * torch.diag(H).mean()
    H = H + damp_param * torch.eye(in_features, device=H.device)
    
    # Compute inverse
    H_inv = torch.inverse(H)
    
    # Initialize quantized weights
    W_q = W.clone()
    
    # Process columns
    for j in range(in_features):
        # Get j-th column
        w_j = W_q[:, j]
        
        # Quantize
        if per_channel:
            # Per-channel: quantize each row independently
            # (This is a simplification; full GPTQ uses global grid)
            w_j_q = quantize_to_grid(w_j, bits=bits)
        else:
            # Per-tensor: single grid for all weights
            w_j_q = quantize_to_grid(w_j, bits=bits)
        
        # Compute quantization error
        error = w_j - w_j_q
        
        # Update remaining columns
        if j < in_features - 1:
            W_q[:, j+1:] -= torch.ger(error, H_inv[j, j+1:])
        
        # Store quantized column
        W_q[:, j] = w_j_q
    
    return W_q

# Usage example
def quantize_model_with_gptq(model, calibration_loader, bits=4):
    """Quantize all Linear layers in a model using GPTQ."""
    model.eval()
    
    # Collect calibration data
    all_activations = []
    with torch.no_grad():
        for batch in calibration_loader:
            # This assumes a simple model; adapt for your architecture
            activations = batch  # Or extract from intermediate layers
            all_activations.append(activations)
    
    X = torch.cat(all_activations, dim=0)
    
    # Quantize each Linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Quantizing {name}...")
            W_q = gptq_quantize_linear(module, X, bits=bits)
            module.weight = nn.Parameter(W_q, requires_grad=False)
    
    return model
```

---

## GPTQ in Practice: LLM Quantization

For large language models, GPTQ is typically applied layer-by-layer:

```python
def quantize_llm_with_gptq(model, calibration_loader, bits=4):
    """
    Quantize an LLM using GPTQ, processing one transformer block at a time.
    """
    model.eval()
    
    # Process each transformer block sequentially
    for layer_idx, layer in enumerate(model.model.layers):
        print(f"Processing layer {layer_idx}...")
        
        # Collect activations for this layer
        layer_inputs = []
        layer_outputs = []
        
        def hook_fn(module, input, output):
            layer_inputs.append(input[0].detach())
            layer_outputs.append(output.detach())
        
        # Register hooks on attention and MLP
        hooks = []
        for submodule in [layer.self_attn, layer.mlp]:
            hooks.append(submodule.register_forward_hook(hook_fn))
        
        # Run calibration data
        with torch.no_grad():
            for batch in calibration_loader:
                model(batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate collected activations
        X = torch.cat(layer_inputs, dim=0).reshape(-1, layer_inputs[0].shape[-1])
        
        # Quantize Linear layers in this block
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                W_q = gptq_quantize_linear(module, X, bits=bits)
                module.weight = nn.Parameter(W_q, requires_grad=False)
    
    return model
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why does GPTQ process columns sequentially instead of all at once?

2. **Math:** If the Hessian `H` is diagonal (no correlation between input features), what does the GPTQ update reduce to?

3. **Debug:** A team runs GPTQ with `damping=0.0` and gets NaN weights. What went wrong?

4. **Trade-offs:** When would you choose standard PTQ over GPTQ, despite GPTQ's better accuracy?

---

**Next:** [Chapter 3, Section 2: AWQ Algorithm](ch3_sec2_awq_theory.md)
