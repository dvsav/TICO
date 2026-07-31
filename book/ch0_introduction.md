# Chapter 0: Introduction & Learning Path

**Summary:** This chapter introduces the book's scope, learning objectives, and the path we'll follow from fundamental quantization theory to advanced algorithms like GPTQ, AWQ, and SpinQuant.

**Prerequisites:** Basic understanding of neural networks, PyTorch familiarity, and linear algebra fundamentals (matrix operations, vector spaces).

**Key Takeaways:**
- Understand what quantization is and why it matters
- Learn the book structure and progression
- Set expectations for hands-on experiments

---

## Why Quantization Matters

Modern large language models (LLMs) and vision-language models (VLMs) have grown to billions of parameters. While these models achieve remarkable accuracy, their size creates practical deployment challenges:

- **Memory footprint:** A 7B parameter model in FP32 requires ~28 GB of memory
- **Bandwidth requirements:** Loading weights from memory becomes the bottleneck
- **Energy consumption:** High-precision arithmetic consumes more power
- **Latency:** Larger models take longer to process each token

Quantization addresses these challenges by reducing the numerical precision of model weights and activations. Converting from 32-bit floating-point (FP32) to 8-bit integers (INT8) can theoretically provide:

- **4× reduction** in memory footprint
- **4× reduction** in memory bandwidth requirements
- **2-4× speedup** in inference (depending on hardware support)
- **Significant energy savings**

However, quantization is not without trade-offs. Reducing precision introduces **quantization error**, which can degrade model accuracy. The art and science of quantization lies in minimizing this error while maximizing efficiency gains.

## Book Structure

This book follows a progressive learning path:

### Part I: Quantization Fundamentals (Chapter 1)
We start with the mathematical foundations:
- What is quantization at the mathematical level?
- How do scale and zero-point work?
- What's the difference between symmetric and asymmetric quantization?
- What is fake quantization and why do we need it?

### Part II: Post-Training Quantization (Chapter 2)
We dive deep into PTQ theory and practice:
- How does calibration work?
- How are quantization parameters computed for weights?
- How are quantization parameters computed for activations?
- Where and how do we insert quantize-dequantize operations?

### Part III: Advanced Algorithms (Chapter 3)
We explore state-of-the-art quantization methods:
- **GPTQ:** Gradient-based layer-by-layer reconstruction
- **AWQ:** Activation-aware weight protection
- **SpinQuant:** Rotation-based outlier suppression

### Part IV: PyTorch Implementation (Chapter 4)
We connect theory to practice:
- Using `torch.export` for model export
- Building PTQ workflows in PyTorch
- Hands-on experiments with real models

## How to Use This Book

This book is designed for **active learning**:

1. **Read one section at a time** — Each section is self-contained and builds on previous knowledge
2. **Complete knowledge checkpoints** — Verify your understanding before moving forward
3. **Run the experiments** — Theory becomes clear when you see it in code
4. **Ask questions** — If something is unclear, flag it for discussion

## Knowledge Checkpoint

Before proceeding to Chapter 1, consider these questions:

1. **Intuition check:** Why can't we simply round FP32 weights to INT8 and expect the model to work?
2. **Math check:** If you have values in range [-2.5, 3.0] and want to quantize to 4-bit signed integers, what scale would you use?
3. **Concept check:** What's the difference between quantizing weights vs. quantizing activations?

*Don't worry if you can't answer these perfectly yet — they're designed to prime your thinking. We'll address all of these in the coming chapters.*

---

## Appendix: Why Are Integer Operations Faster?

You might wonder: "Why is INT8 multiplication faster than FP32 multiplication? They're both just numbers!"

The answer lies in **hardware complexity**:

### Floating-Point Unit (FPU) Complexity

A floating-point number has three components:
```
FP32 = [sign: 1 bit] [exponent: 8 bits] [mantissa: 23 bits]
```

**FP32 multiplication requires:**
1. **Exponent handling:** Add exponents, check for overflow/underflow
2. **Mantissa multiplication:** Multiply 23-bit fractions
3. **Normalization:** Shift result to maintain canonical form
4. **Rounding:** Apply rounding mode (round-to-nearest, round-toward-zero, etc.)
5. **Special case handling:** NaN, Inf, denormals, zero

```
FP32 multiply latency: ~3-5 cycles on modern CPUs
FP32 multiply transistor count: ~1000+ transistors per multiplier
```

### Integer Unit Simplicity

An INT8 number is just bits:
```
INT8 = [8 bits representing -128 to 127]
```

**INT8 multiplication requires:**
1. **Simple multiplication:** 8-bit × 8-bit → 16-bit result
2. **No special cases** (no NaN, no Inf, no denormals)

```
INT8 multiply latency: ~1 cycle on modern CPUs
INT8 multiply transistor count: ~100 transistors per multiplier
```

### Throughput Advantage

Modern hardware can fit **many more** integer units than floating-point units:

| Hardware | FP32 Units | INT8 Units | INT8/FP32 Ratio |
|----------|------------|------------|-----------------|
| CPU (AVX2) | 32 ops/cycle | 128 ops/cycle | 4× |
| GPU (NVIDIA) | 128 ops/cycle | 512 ops/cycle | 4× |
| NPU (Samsung) | 64 ops/cycle | 1024 ops/cycle | 16× |

### Memory Bandwidth Advantage

Beyond compute speed, quantization reduces **memory bandwidth**:

```
FP32: 4 bytes per value
INT8: 1 byte per value

Memory bandwidth required: INT8 = 25% of FP32
Memory capacity required: INT8 = 25% of FP32
```

For memory-bound workloads (like LLM inference), this **4× bandwidth reduction** often matters more than the compute speedup!

### Summary

| Factor | FP32 | INT8 | Advantage |
|--------|------|------|-----------|
| Compute latency | 3-5 cycles | 1 cycle | 3-5× |
| Hardware units | Fewer | More | 4-16× throughput |
| Memory bandwidth | 4 bytes/value | 1 byte/value | 4× |
| Power consumption | Higher | Lower | 2-4× efficiency |

**Bottom line:** Integer operations are faster because the hardware is simpler, more parallel, and more memory-efficient.

## Getting Started

To follow along with the PyTorch examples, ensure you have:
- Python 3.8+
- PyTorch 2.0+ (for `torch.export` support)
- The TICO repository (for reference implementations)

Let's begin with the fundamentals in **Chapter 1, Section 1: Quantization Fundamentals**.

---

**Next:** [Chapter 1, Section 1: Quantization Fundamentals](ch1_sec1_quantization_fundamentals.md)
