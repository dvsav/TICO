# Glossary: AI Model Quantization

| Term | Definition | First Introduced |
|------|------------|------------------|
| **Quantization** | The process of mapping continuous or high-precision values to a discrete set of lower-precision values, typically converting floating-point numbers to integers. | ch1_sec1 |
| **Post-Training Quantization (PTQ)** | Quantization applied to a pre-trained model without requiring retraining. Uses calibration data to determine quantization parameters. | ch1_sec1 |
| **Quantization-Aware Training (QAT)** | Training technique where quantization effects are simulated during training, allowing the model to adapt to quantization noise. | ch1_sec1 |
| **Scale (s)** | The multiplicative factor that maps quantized integer values back to the floating-point domain. Determines the resolution of quantization levels. | ch1_sec2 |
| **Zero-Point (z)** | The quantized integer value that corresponds to the floating-point value of zero. Used in asymmetric quantization to preserve exact zero representation. | ch1_sec2 |
| **Symmetric Quantization** | Quantization scheme where the quantization range is symmetric around zero (zero-point = 0). | ch1_sec3 |
| **Asymmetric Quantization** | Quantization scheme where the range is not symmetric around zero, requiring a non-zero zero-point. | ch1_sec3 |
| **Per-Tensor Quantization** | Single scale and zero-point applied to the entire tensor. | ch1_sec3 |
| **Per-Channel Quantization** | Separate scale (and optionally zero-point) for each channel, providing finer granularity. | ch1_sec3 |
| **Fake Quantization** | Simulation of quantization effects while computations remain in floating-point. Uses quantize-dequantize (Q-DQ) operations. | ch1_sec4 |
| **Calibration** | The process of collecting activation statistics by running real data through the model to determine optimal quantization parameters. | ch2_sec1 |
| **Observer** | PyTorch module that collects statistics (min, max, histogram) of tensor values during calibration. | ch2_sec1 |
| **Quantization Parameters (QParams)** | The scale and zero-point values that define the quantization mapping for a tensor. | ch1_sec2 |
| **Bit Width** | The number of bits used to represent quantized values (e.g., INT8, INT4). Determines the number of quantization levels. | ch1_sec1 |
| **Quantization Error** | The difference between the original floating-point value and its quantized representation. | ch1_sec1 |
| **KL Divergence** | Kullback-Leibler divergence, a metric used to find optimal quantization thresholds by minimizing information loss. | ch2_sec1 |
| **Outlier Suppression** | Techniques to handle extreme values in activations that can degrade quantization quality. | ch2_sec3 |
| **GPTQ** | Gradient-based Post-Training Quantization algorithm that minimizes reconstruction error layer-by-layer. | ch3_sec1 |
| **AWQ** | Activation-aware Weight Quantization that protects salient weights based on activation magnitude. | ch3_sec2 |
| **SpinQuant** | Rotation-based quantization method that spreads outlier impact across all dimensions. | ch3_sec3 |

---

*This glossary is continuously updated as new terms are introduced.*
