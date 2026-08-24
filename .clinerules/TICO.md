## Persona
You are a software engineer experienced in Machine Learning (especially LLMs and VLMs), Python, PyTorch and model quantization and compilation for embedded platforms.

## Context

### The Repository
We are currently in a repository called `TICO`.
Below is an intoductive information about this repository.

### Introduction
TICO project is intended for the quantization of PyTorch models and their conversion to Circle format. "TICO" stands for "Torch IR To Circle ONE".

Quantization is transformation of a model (computation graph rewriting) so that the model's forward-pass computations are done in integer numbers rather than in floats and producing results that are numerically close to those of the original unquantized model (with a slight accuracy degradation).

Circle is a Samsung's binary model storage format based in Flatbuffers (like TFLite).

### Fake Quantization
Quantization requires two steps:

1. Fake quantization (at this stage the model computation graph is augmented rather than rewritten):
    a. Calibration: collecting statistics about values to be quantized:
        i. Activations: to collect the statistics you'll need to forward-pass real data through the model (i.e. you'll need a calibration dataset for this).
        ii. Constants (usually model weights): you don't need to forward-pass any data through the model, you can capture the statistics for constants right away.
    b. Computation of quantization parameters (scale, zero point) for each of the entities to be quantized based on the collected statistics.
    c. Insertion of quantize-dequantize (Q-DQ for short) operations to the forward pass computation - this is the essense of fake quantization. Q-DQ operations still operate in floats but they simulate the quantization error. A value after being fake-quantized stays in the same domain (range) as before, but it gets rounded to the nearest quantization level.
2. Fake quantization operations folding (in TICO is a part of conversion to Circle):
    a. Extraction of quantization parameters from the Q-DQ operations.
    b. Elimination of Q-DQ operations from the computation graph and embedding the quantization logic into computation graph operations.

### PTQ Quantization Flow
The quantization necessarily passes through PTQ flow. The distinctive feature of the PTQ flow is that it requires the creation of a "wrapper" model architecture semantically similar to the original unquantized one, but containing fake quantization operations. The wprapper model architecture can also have some intricate optimizations related to the specific conditions under which the quantized model will be executed (e.g. some tensors can be precomputed statically in the quantized model rather than being calculated at runtime). The wrapper model uses the weights of the original unquantized model, but adds fake quantize operations. It's the wrapper model that is exported as a static computation graph and then coverted to Circle (see the details below).

#### Fake Quantization
1. Wrapping of the original model layers, i.e. replacement of model layers with specialized wrappers (named `QuantXXX` and defined in `tico/quantization/wrapq/wrappers`). The wrappers have two modes of operation: `CALIB` and `QUANT`.
They override `forward` method that acts differently depending on the mode:
- In `CALIB` mode: Activation statistics collection (aka calibration).
- In `QUANT` mode: Activation quantization parameters computation from the statistics.
- In `QUANT` mode: Weights quantization parameters computation from weight distribution.
2. Collecting activation statistics in `CALIB` mode (implemented in QuantXXX wrappers' forward methods - see above).
3. Computing quantization parameters in QUANT mode (implemented in QuantXXX wrappers' forward methods - see above).

#### Conversion to Circle
Conversion to Circle format happens in `tico.convert` function. It consists of the following stages:
1. The model is exported from python code representation to a static computation graph (fake quantize-dequantize operations are inserted to the graph at this stage). This is done via `torch.export`.
2. The exported model then undergoes a series of graph transformations specified by the so-called "passes" (objects inheriting from `PassBase` class). This happend in `convert_exported_module_to_circle` funtion defined in `tico/utils/convert.py` file. One representative example of such pass is `FoldQuantOps` pass that performs the folding of fake quantize-dequantize operations.
3. Torch operators of the computation graph are mapped to Circle operations to form a Circle graph representation.
4. The Circle graph is serialized to Circle binary format (Flatbuffers based).

Inside `tico.convert` function the quantized_model undergoes two kinds of transformations:

It gets exported to a static graph representation (`ExportedProgram`). During this phase the forward method of the model is called and that method calls `torch.fake_quantize` function - in the final graph this becomes a respective `fake_quantize` node.
The exported model ep passes through a series of graph transformations specified by the so-called "passes" (objects inheriting from PassBase class).
build_circle function iterates over all the nodes in the model's graph (for node in ep.graph.nodes) looking for nodes referencing some data (placeholders, get_attr), extracts the referenced data (tensors) from the exported program and stores them in the CircleSubgraph object (graph.add_tensor). The actual tensor values are not stored dorectly in the CircleSubgraph object, instead they are stored in a CircleModel  object that allocates buffers for each tensor (it's a pattern similar to ExportedProgram).
build_circle function iterates over all the nodes in the model's graph (for node in ep.graph.nodes) for the 2nd time looking for nodes performing some calculations (call_function), extracts the opcodes and other information from them and stores them as circle.Operator.OperatorT in the CircleSubgraph object (graph.add_operator(circle_op)).

### Example Code
To make sense of typical TICO workflow it's best to overview the example scripts that quantize different models and convert them to Circle. Those scripts are in `tico/quantization/wrapq/examples` directory and its subdirectories (`llama`, `qwen`, `nn`).

### Public Interface Abstraction
The high-level public API that abstracts away the complexity:
```python
from tico.quantization import prepare, convert
from tico.quantization.config.gptq import GPTQConfig

# Prepare
prepared_model = prepare(model, GPTQConfig())

# Calibrate
for data in dataset:
    prepared_model(data)

# Convert
quantized_model = convert(prepared_model)
```
This interface dispatches to appropriate quantizers based on config type via a registry system.

### Multi-Algorithm Support
TICO supports multiple quantization algorithms, not just PTQ:
- **GPTQ**: Post-training weight quantization
- **SmoothQuant**: Activation-weight equalization
- **PTQ**: Post-training quantization (activations + weights)
- **PT2E**: PyTorch 2 Export-based quantization
- **FPI_GPTQ**: Fourth-order polynomial interpolation for GPTQ

Yet, PTQ is essential to obtain a quantized model in a form of Circle file.
That's why for example a model quantized with GPTQ algorithm is then wrapped with PTQ wrapper - otherwise we won't be able to obtain a quantized model.

### Two-Layer Architecture
The quantization module has a clear separation:
- **Algorithm Layer** (`tico/quantization/algorithm/`): Self-contained, independent implementations (can be used standalone)
- **Infrastructure Layer** (`tico/quantization/wrapq/`): Generic wrapper-based backend shared across algorithms

### Wrapper-Based Quantization (WrapQ)

WrapQ is now the **shared quantization backend** with these key features:
- **Registry-based wrapper discovery**: Wrappers are automatically registered and discovered via a registry system
- **Model families supported**: `llama`, `qwen_vl`, `fairseq`, `nn` (standard PyTorch modules)
- **Base class hierarchy**: All wrappers inherit from `QuantModuleBase`
- **Explicit calibration**: Users decide what/when/how to calibrate
- **Composable design**: Wrappers can be mixed with different algorithms

### Supported Model Families

Each model family is supported via hand-crafted PTQ wrapper classes that wrap each submodule of a particular model architecture. Currently the following model families are supported:
- **LLaMA**: Text decoder layers, MLPs, attention (prefill/decode) - `tico/quantization/wrapq/wrappers/llama`
- **Qwen**: Text and vision components (attention, MLPs, vision model) - `tico/quantization/wrapq/wrappers/qwen_vl`
- **Standard PyTorch (`nn`)**: Linear layers, Conv3D, etc. - `tico/quantization/wrapq/wrappers/nn`

### Variant-Aware Wrapping
The PTQ system supports execution variants:
- `"prefill"`: Full-sequence execution (prompt processing)
- `"decode"`: Single-token autoregressive decoding
- `"common"`: Variant-independent implementation

### Quantization Pass System
After fake quantization, there's a comprehensive pass-based optimization system:
- `FoldQuantOps`: Folds Q-DQ pairs into operator metadata
- `PropagateQParamForward/Backward`: Propagates quantization parameters through the graph
- `QuantizeBias`: Quantizes bias terms
- `InsertQuantizeOnDtypeMismatch`: Handles dtype conversions
- `RemoveWeightDequantOp`: Removes unnecessary dequantization operations

### Comprehensive Pass Manager
The conversion pipeline uses a sophisticated Pass Manager with multiple stages:
- Pre-edge decompositions
- Legalization passes (EP-invariant preserved)
- Optimization passes (EP-invariant relaxed)
- Quantization passes (conditional)
- Validation & checks

### Configuration System
Rich configuration with hierarchical overrides:
```python
PTQConfig(
    default_dtype=DType.uint(8),
    default_qscheme=QScheme.PER_TENSOR_ASYMM,
    wrapper_variant="prefill",
    overrides={
        "act_in": {"dtype": DType.uint(4), "qscheme": QScheme.PER_CHANNEL_ASYMM}
    },
    model_args={"vision": {"grid_thw": (8, 24, 24)}}
)
```

### Export Requirement
TICO uses `torch.export.export` (PyTorch 2.1+) internally, not just any PyTorch module. The module must be "exportable" according to torch.export's constraints.
