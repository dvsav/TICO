# Chapter 4, Section 1: PyTorch Export and Model Conversion

**Summary:** This section covers PyTorch's export infrastructure (`torch.export`) and how it enables model conversion for quantization. We explore the ExportedProgram format, graph capture, and how quantization parameters are embedded in the exported model.

**Prerequisites:** Chapter 1, Section 4 (Fake Quantization), basic understanding of PyTorch computation graphs.

**Key Takeaways:**
- `torch.export` captures models as static computation graphs
- ExportedProgram is the intermediate representation for quantization
- Quantization parameters are embedded as graph inputs or tensor metadata
- Understanding export constraints is essential for quantization workflows

---

## Why Export Matters for Quantization

Quantization requires a **static computation graph** where:
1. All operations are explicitly defined
2. Tensor shapes are known (or properly constrained)
3. Quantization parameters (scales, zero-points) are captured
4. The graph can be transformed (e.g., Q-DQ folding)

PyTorch's eager execution model doesn't provide this by default:

```python
# Eager PyTorch: dynamic, Python-driven
def forward(self, x):
    if x.shape[0] > 100:  # Dynamic control flow
        x = self.large_batch_layer(x)
    else:
        x = self.small_batch_layer(x)
    return x

# This is problematic for quantization!
# Which layer gets quantized? What are the shapes?
```

`torch.export` solves this by capturing the model as a static graph.

---

## torch.export: The Basics

### Export API

```python
import torch
import torch.export

# Define a model
model = MyModel()
model.eval()

# Example inputs (define the shape constraints)
example_inputs = (torch.randn(1, 3, 224, 224),)

# Export the model
exported_program = torch.export.export(model, example_inputs)

# The exported program contains:
# - GraphModule: the captured computation graph
# - Graph: the underlying FX graph
# - Tensor metadata (shapes, dtypes)
# - Parameters and buffers
```

### ExportedProgram Structure

```python
# Access components of the exported program
print(exported_program.graph)        # FX Graph
print(exported_program.graph_module) # GraphModule (executable)
print(exported_program.signatures)   # Input/output signatures
print(exported_program.range_constraints)  # Shape constraints

# Run the exported model
output = exported_program.module(*example_inputs)
```

### Graph Representation

The exported graph is an FX (Functional eXchange) graph:

```python
# Example exported graph for a simple Linear layer
graph:
    %x : [batch, in_features]  # Input placeholder
    %weight : [out_features, in_features]  # Parameter
    %bias : [out_features]  # Parameter
    
    %matmul = call_function[torch.ops.aten.mm.default](%x, %weight)
    %output = call_function[torch.ops.aten.add.Tensor](%matmul, %bias)
    
    return output
```

---

## Export Constraints and Limitations

### Dynamic Shapes

By default, `torch.export` requires static shapes. For dynamic shapes, use `Dim`:

```python
from torch.export import Dim

# Define dynamic dimensions
batch_dim = Dim("batch", min=1, max=128)
seq_dim = Dim("seq", min=1, max=512)

# Export with dynamic shape constraints
example_inputs = (torch.randn(4, 512),)
dynamic_shapes = {0: batch_dim, 1: seq_dim}

exported_program = torch.export.export(
    model, 
    example_inputs,
    dynamic_shapes=dynamic_shapes
)
```

### Control Flow Restrictions

`torch.export` has restrictions on Python control flow:

```python
# NOT exportable: Python-dependent control flow
def forward(self, x):
    if len(x) > 100:  # Depends on Python len()
        ...

# Exportable: Tensor-dependent control flow
def forward(self, x):
    if x.shape[0] > 100:  # Depends on tensor shape
        ...

# Exportable: Using torch.cond
def forward(self, x):
    pred = x.shape[0] > 100
    return torch.cond(pred, self.large_fn, self.small_fn, (x,))
```

### Data-Dependent Operations

Operations that depend on tensor values (not just shapes) may fail:

```python
# Problematic: value-dependent control flow
def forward(self, x):
    if x.sum() > 0:  # Depends on values, not just shape
        ...

# OK: shape-dependent operations
def forward(self, x):
    return x.view(x.shape[0], -1)  # Only depends on shape
```

---

## Exporting Quantized Models

### Fake Quantization in Exported Graphs

When exporting a model with fake quantization, the Q-DQ operations are captured:

```python
import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Fake quantization modules
        self.weight_fake_quant = FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
        self.activation_fake_quant = FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
    
    def forward(self, x):
        x = self.activation_fake_quant(x)
        weight = self.weight_fake_quant(self.linear.weight)
        return nn.functional.linear(x, weight, self.linear.bias)

# Export the quantized model
model = QuantizedLinear(512, 512)
example_inputs = (torch.randn(1, 512),)
exported_program = torch.export.export(model, example_inputs)

# The graph now contains fake_quantize operations
for node in exported_program.graph.nodes:
    if "fake_quantize" in str(node.target):
        print(f"Found fake_quantize node: {node}")
```

### Quantization Parameters in Exported Graph

Quantization parameters (scales, zero-points) are captured as graph inputs or buffers:

```python
# Access quantization parameters
for name, buffer in exported_program.state_dict().items():
    if "scale" in name or "zero_point" in name:
        print(f"{name}: {buffer.shape} = {buffer}")

# Example output:
# activation_fake_quant.scale: torch.Size([]) = 0.0235
# activation_fake_quant.zero_point: torch.Size([]) = 0
# weight_fake_quant.scale: torch.Size([512]) = tensor([...])
```

---

## Graph Transformations for Quantization

After export, the graph can be transformed to optimize quantization.

### Q-DQ Folding

Fold quantize-dequantize pairs into adjacent operators:

```python
from tico.passes import FoldQuantOps

# Before folding:
# x → quantize → dequantize → linear → ...

# After folding:
# x → linear (with embedded quantization) → ...

folded_program = FoldQuantOps().call(exported_program)
```

### Quantization Parameter Propagation

Propagate quantization parameters through the graph:

```python
from tico.passes import PropagateQParamForward, PropagateQParamBackward

# Forward propagation: push QParams toward outputs
program = PropagateQParamForward().call(exported_program)

# Backward propagation: push QParams toward inputs
program = PropagateQParamBackward().call(program)
```

### Bias Quantization

Quantize bias terms based on input and weight scales:

```python
from tico.passes import QuantizeBias

# Bias scale = input_scale × weight_scale
program = QuantizeBias().call(exported_program)
```

---

## TICO's Export Pipeline

TICO uses `torch.export` as the first step in its conversion pipeline:

```python
# Simplified TICO conversion flow
from tico.convert import convert

# 1. Prepare quantized model (WrapQ wrappers in QUANT mode)
quantized_model = prepare_and_calibrate(model, calibration_data)

# 2. Export to static graph
example_inputs = get_example_inputs(model)
exported_program = torch.export.export(quantized_model, example_inputs)

# 3. Apply quantization passes
from tico.passes import (
    FoldQuantOps,
    PropagateQParamForward,
    QuantizeBias,
    RemoveWeightDequantOp
)

passes = [
    FoldQuantOps(),
    PropagateQParamForward(),
    QuantizeBias(),
    RemoveWeightDequantOp(),
]

for p in passes:
    exported_program = p.call(exported_program)

# 4. Convert to Circle format
circle_model = convert(exported_program)
```

---

## PyTorch Experiment: Export and Inspect Quantized Model

```python
import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver

class SimpleQuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 10)
        
        # Fake quantization
        self.act_fake_quant = FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=0,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine
        )
        self.weight_fake_quant = FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    
    def forward(self, x):
        x = self.act_fake_quant(x)
        
        weight1 = self.weight_fake_quant(self.linear1.weight)
        x = nn.functional.linear(x, weight1, self.linear1.bias)
        x = self.relu(x)
        x = self.act_fake_quant(x)
        
        weight2 = self.weight_fake_quant(self.linear2.weight)
        x = nn.functional.linear(x, weight2, self.linear2.bias)
        
        return x

# Create and "calibrate" model
model = SimpleQuantizedModel()
model.eval()

# Dummy calibration (in practice, use real data)
with torch.no_grad():
    for _ in range(10):
        x = torch.randn(4, 128)
        _ = model(x)

# Export
example_inputs = (torch.randn(4, 128),)
exported_program = torch.export.export(model, example_inputs)

print("=" * 70)
print("EXPORTED GRAPH")
print("=" * 70)
print(exported_program.graph)

print("\n" + "=" * 70)
print("QUANTIZATION PARAMETERS")
print("=" * 70)
for name, buffer in exported_program.state_dict().items():
    if "scale" in name or "zero_point" in name:
        print(f"{name}: shape={buffer.shape}, value={buffer.flatten()[:5]}...")

print("\n" + "=" * 70)
print("NODE TYPES IN GRAPH")
print("=" * 70)
node_types = {}
for node in exported_program.graph.nodes:
    node_type = str(node.op)
    node_types[node_type] = node_types.get(node_type, 0) + 1

for node_type, count in sorted(node_types.items()):
    print(f"{node_type}: {count}")
```

**Expected output:**
```
======================================================================
EXPORTED GRAPH
======================================================================
graph():
    %x : [batch=4, 128]  # Input placeholder
    %weight1 : [256, 128]  # Weight parameter
    %bias1 : [256]  # Bias parameter
    ...
    
    %fake_quantize_1 = call_function[torch.ops.quantized_decomposed.quantize_per_tensor_affine](%x, %scale1, %zp1)
    %dequantize_1 = call_function[torch.ops.quantized_decomposed.dequantize_per_tensor_affine](%fake_quantize_1, %scale1, %zp1)
    
    %mm = call_function[torch.ops.aten.mm.default](%dequantize_1, %weight1)
    %add = call_function[torch.ops.aten.add.Tensor](%mm, %bias1)
    
    %relu = call_function[torch.ops.aten.relu.default](%add)
    ...
    
    return %output

======================================================================
QUANTIZATION PARAMETERS
======================================================================
act_fake_quant.scale: shape=torch.Size([]), value=tensor([0.0235])...
act_fake_quant.zero_point: shape=torch.Size([]), value=tensor([0])...
weight_fake_quant.scale: shape=torch.Size([256, 128]), value=tensor([0.0039, 0.0041, ...])...

======================================================================
NODE TYPES IN GRAPH
======================================================================
placeholder: 1
get_attr: 12
call_function: 24
output: 1
```

---

## Common Export Issues and Solutions

### Issue 1: Dynamic Control Flow

**Error:** `torch.export.ExportError: Dynamic control flow not supported`

**Solution:** Use `torch.cond` instead of Python `if`:

```python
# Before (fails)
def forward(self, x, y):
    if x.shape[0] > y.shape[0]:
        return x + y
    return x - y

# After (works)
def forward(self, x, y):
    pred = x.shape[0] > y.shape[0]
    return torch.cond(pred, lambda a, b: a + b, lambda a, b: a - b, (x, y))
```

### Issue 2: Data-Dependent Shapes

**Error:** `torch.export.ExportError: Data-dependent operator detected`

**Solution:** Avoid operations that depend on tensor values:

```python
# Before (fails)
def forward(self, x):
    n = x.sum().item()  # Data-dependent
    return x[:n]

# After (works)
def forward(self, x):
    return x[:, :128]  # Fixed slicing
```

### Issue 3: Missing Quantization Parameters

**Error:** `RuntimeError: scale/zero_point not found in state_dict`

**Solution:** Ensure observers are properly calibrated before export:

```python
# Calibrate before export
model.eval()
with torch.no_grad():
    for x in calibration_data:
        model(x)  # Collects statistics in observers

# Now export
exported_program = torch.export.export(model, example_inputs)
```

---

## Knowledge Checkpoint

1. **Conceptual:** Why does `torch.export` require static shapes by default? What problem does this solve for quantization?

2. **Debug:** A team exports their model but the graph doesn't contain any `fake_quantize` nodes. What could have gone wrong?

3. **Code analysis:** In the exported graph, what's the difference between a `placeholder` node and a `get_attr` node?

4. **TICO context:** Why does TICO apply `FoldQuantOps` pass after export? What does this pass accomplish?

---

**Next:** [Chapter 4, Section 2: End-to-End PTQ Examples](ch4_sec2_ptq_examples.md)
