## Goal

Add wrapper-based PTQ support for **Gemma4 E2B** and make it usable in a **static-shape NPU inference flow**. The runtime should follow the same high-level principle as the existing Llama static runtime: CPU owns dynamic/control-heavy orchestration, while NPU executes fixed-shape, quantized tensor-compute subgraphs.

This issue focuses on implementation work needed for:

- activation quantization and weight quantization
- static-shape prefill and decode inference
- image-text Gemma4 E2B first
- CPU/NPU partitioning
- wrapper smoke tests and evaluation flow
- later comparison against the existing Qwen3-VL PTQ path

## Scope

### In scope for v0

- Gemma4 E2B only.
- Dense text decoder path only.
- No MoE path.
- Batch size fixed to `1` for the first static runtime.
- Static `max_seq`, for example `2048`.
- Static image input shape and static visual-token count.
- One fixed visual segment layout, for example:

  ```text
  [text prefix][image soft-token slots][text suffix][padding]
  ```

- Static prefill graph and static decode graph separated.
- Qwen3-VL style evaluation compatibility.

## Key design decision

Do **not** try to export the full Hugging Face Gemma4 forward as one NPU graph.

Instead, follow the Llama static runtime pattern:

1. prepare/wrap model modules for PTQ
2. expose static export adapters for subgraphs
3. let CPU runtime manage tokenization, layout validation, RoPE, masks, cache state, sampling, and loop control
4. let NPU run only fixed-shape tensor compute with quantized weights and quantized activations

This is necessary because Gemma4 full forward contains dynamic/control-heavy behavior:

- placeholder mask construction
- image/video/audio `masked_scatter` fusion
- dynamic mask creation through Hugging Face mask utilities
- optional cache object handling
- shared KV state bookkeeping
- optional PLE path
- optional multimodal paths

These behaviors are better handled by a static runtime wrapper rather than exported as-is.

## Proposed directory structure

```text
tico/
└── quantization/
    ├── wrapq/
    │   ├── wrappers/
    │   │   ├── registry.py                         # MODIFY: register Gemma4 wrapper modules
    │   │   └── gemma4/                              # ADD
    │   │       ├── __init__.py
    │   │       ├── utils.py
    │   │       ├── export_adapters.py
    │   │       │
    │   │       ├── quant_clippable_linear.py
    │   │       ├── quant_rmsnorm.py
    │   │       │
    │   │       ├── quant_text_scaled_word_embedding.py
    │   │       ├── quant_text_mlp.py
    │   │       ├── quant_text_attention.py
    │   │       ├── quant_text_decoder_layer.py
    │   │       ├── quant_text_model.py
    │   │       ├── quant_for_causal_lm.py
    │   │       │
    │   │       ├── quant_vision_patch_embedder.py
    │   │       ├── quant_vision_pooler.py
    │   │       ├── quant_vision_mlp.py
    │   │       ├── quant_vision_attention.py
    │   │       ├── quant_vision_encoder_layer.py
    │   │       ├── quant_vision_encoder.py
    │   │       ├── quant_vision_model.py
    │   │       │
    │   │       ├── quant_multimodal_embedder.py
    │   │       ├── quant_model.py
    │   │       └── quant_for_conditional_generation.py
    │   │
    │   └── utils/
    │       └── version.py                           # MODIFY: add Gemma4 availability check
    │
    ├── config/
    │   ├── builders.py                              # MODIFY: add build_gemma4_ptq_config
    │   └── __init__.py                              # MODIFY if builder exports are needed
    │
    ├── recipes/
    │   ├── adapters/
    │   │   ├── __init__.py                          # MODIFY: register gemma4 / gemma-4 family names
    │   │   └── gemma4.py                            # ADD: load/calib/PTQ/eval adapter
    │   │
    │   ├── debug/
    │   │   ├── static_gemma4_runtime.py             # ADD: CPU-orchestrated static runtime prototype
    │   │   └── wrapper_smoke/
    │   │       └── cases/
    │   │           └── gemma4.py                    # ADD: synthetic tiny Gemma4 smoke cases
    │   │
    │   └── data/
    │       └── vlm.py                               # Reuse when possible; add Gemma4 input normalization only if needed
    │
    └── examples/
        └── configs/
            ├── gemma4_e2b_ptq_only.yaml             # ADD: trace/parity/debug config
            ├── gemma4_e2b_gptq_ptq.yaml             # ADD: quantize.py config
            ├── gemma4_e2b_static_runtime.yaml       # ADD: inspect.py static runtime config
            └── gemma4_e2b_llava_bench_judge.yaml    # ADD: evaluation config for Qwen3-VL comparison

test/
└── quantization/
    ├── wrapq/
    │   └── wrappers/
    │       └── gemma4/                              # ADD
    │           ├── test_quantize_text_attention.py
    │           ├── test_quantize_text_decoder_layer.py
    │           ├── test_quantize_text_model.py
    │           ├── test_quantize_vision_attention.py
    │           ├── test_quantize_vision_model.py
    │           ├── test_quantize_model.py
    │           └── test_quantize_for_conditional_generation.py
    │
    ├── recipes/
    │   ├── test_gemma4_adapter.py                   # ADD
    │   ├── test_static_gemma4_runtime.py            # ADD
    │   └── test_wrapper_smoke_registry.py           # MODIFY: include Gemma4 smoke case registration
    │
    └── config/
        └── test_builders.py                         # MODIFY: validate build_gemma4_ptq_config
```

## CPU/NPU partition

| Component | Location | Notes |
|---|---:|---|
| Tokenizer / processor | CPU | Dynamic text/image preprocessing. |
| Image resize/pad | CPU | Must produce fixed static image profile. |
| Static layout validation | CPU | Reject prompts/images that violate fixed profile. |
| `llm_input_ids` creation | CPU | Replace multimodal placeholder token IDs with text pad token ID. |
| Token embedding | NPU | Weight-bearing op; must be quantized. |
| PLE token-identity embedding | NPU | If `hidden_size_per_layer_input > 0`. |
| Vision tower | NPU | Weight-bearing compute; prefill only. |
| `embed_vision` projection | NPU | RMSNorm + Linear; prefill only. |
| Multimodal fusion | NPU preferred | Must use fixed-slot fusion, not dynamic `masked_scatter`. |
| Position ID generation | CPU | Integer/control logic. |
| RoPE generation/slicing | CPU | Generate fixed tensors and pass to NPU. |
| Full/sliding/bidirectional mask generation | CPU | Shape/control logic. |
| Text decoder layers | NPU | Main quantized compute. |
| KV cache allocation | CPU | Runtime memory/state management. |
| KV cache update/write | CPU | Write returned K/V delta into fixed cache tensors. |
| Shared KV bookkeeping | CPU | Manage `shared_kv_states[layer_type]`. |
| Final norm + LM head | NPU | Weight-bearing compute; must be quantized. |
| Argmax/sampling/eos | CPU | Control logic, no weights. |
| Decode loop | CPU orchestrates NPU calls | Same pattern as Llama static runtime. |
| Detokenization / logging / metrics | CPU | Outside model compute. |

Rule of thumb:

> CPU may run control logic and tensor indexing/copy needed for runtime state. CPU should not run weight-bearing model compute in the target static inference path.

## Static inference flow

### Prefill flow

```text
[CPU]
  1. Load processor/tokenizer.
  2. Resize/pad image to fixed profile.
  3. Tokenize prompt with fixed max_seq padding.
  4. Validate batch_size == 1.
  5. Validate visual slots:
       visual_start_idx
       num_visual_tokens
       image token count
       max_seq
  6. Create llm_input_ids:
       replace image/video/audio placeholder token IDs with pad_token_id.
  7. Build valid_token_mask.
  8. Build position_ids.
  9. Build per-layer attention masks:
       full_attention mask
       sliding_attention mask
       bidirectional vision mask if required by config
 10. Build or gather RoPE tensors per layer type.
 11. Allocate fixed KV caches and shared-KV caches.

[NPU]
 12. text_embeds = token_embedding(llm_input_ids)
 13. image_embeds = vision_prefill(pixel_values, image_position_ids)
 14. fused_embeds = mm_fusion(text_embeds, image_embeds)
 15. per_layer_inputs = optional PLE projection(fused_embeds, per_layer_token_inputs)

[CPU + NPU layer loop]
 16. For each decoder layer i:
       layer_type = config.layer_types[i]
       mask = mask_by_layer_type[layer_type]
       rope = rope_by_layer_type[layer_type]

       if layer is non-shared KV:
           hidden, new_k, new_v, optional_shared = npu_prefill_layer_i(
               hidden, mask, rope, optional per_layer_input_i
           )
           CPU writes new_k/new_v into layer cache.
           If this layer stores shared KV, CPU updates shared_kv_cache[layer_type].

       if layer is shared KV:
           hidden = npu_prefill_shared_layer_i(
               hidden, mask, rope, shared_kv_cache[layer_type], optional per_layer_input_i
           )

[NPU]
 17. logits = lm_head(last_valid_hidden)

[CPU]
 18. next_token = argmax_or_sample(logits)
 19. Update runtime state:
       past_len
       generated tokens
       eos flag
```

### Decode flow

```text
Repeat until eos or max_new_tokens or max_seq is reached:

[CPU]
  1. Current token shape: (1, 1).
  2. Build decode position id for current past_len.
  3. Slice RoPE tensors for current position.
  4. Build full/sliding decode masks with static max_seq width.
  5. Prepare fixed cache tensors for each layer.

[NPU]
  6. hidden = token_embedding(next_token)
  7. optional per_layer_inputs for decode token.

[CPU + NPU layer loop]
  8. For each decoder layer i:
       layer_type = config.layer_types[i]
       mask = decode_mask_by_layer_type[layer_type]
       rope = decode_rope_by_layer_type[layer_type]

       if layer is non-shared KV:
           hidden, new_k, new_v, optional_shared_delta = npu_decode_layer_i(
               hidden,
               mask,
               rope,
               past_key_cache_i,
               past_value_cache_i,
               optional per_layer_input_i,
           )
           CPU writes new_k/new_v to cache_i[:, :, past_len:past_len+1, :].
           If this layer stores shared KV, CPU updates shared_kv_cache[layer_type]
           before later shared layers consume it.

       if layer is shared KV:
           hidden = npu_decode_shared_layer_i(
               hidden,
               mask,
               rope,
               shared_kv_cache[layer_type],
               optional per_layer_input_i,
           )

[NPU]
  9. logits = lm_head(hidden)

[CPU]
 10. next_token = argmax_or_sample(logits)
 11. past_len += 1
 12. Stop if eos or static window is full.
```

## Static shape contracts

Initial v0 contracts:

```text
batch_size = 1
prefill sequence length = max_seq
decode sequence length = 1
max_seq = fixed by config
image shape = fixed by config
num_visual_tokens = fixed by config
visual_start_idx = fixed by config
num_images = 1
num_audio = 0
num_video = 0
```

Recommended tensor shapes:

```text
input_ids:                 (1, max_seq)
llm_input_ids:             (1, max_seq)
valid_token_mask:          (1, max_seq)
pixel_values:              static image tensor from processor
image_position_ids:        fixed patch-position tensor
text_embeds prefill:       (1, max_seq, hidden_size)
image_embeds:              (1, num_visual_tokens, hidden_size)
fused_embeds:              (1, max_seq, hidden_size)
per_layer_inputs prefill:  (1, max_seq, num_layers, ple_dim), optional
per_layer_input_i prefill: (1, max_seq, ple_dim), optional
per_layer_input_i decode:  (1, 1, ple_dim), optional
prefill K/V:               (1, kv_heads, max_seq, head_dim)
decode past K/V:           (1, kv_heads, max_seq - 1, head_dim)
decode new K/V:            (1, kv_heads, 1, head_dim)
decode full K/V after CPU update: conceptually (1, kv_heads, max_seq, head_dim)
logits:                    (1, 1, vocab_size)
```

## Implementation phases

### Phase 1: dense text wrappers

Implement:

```text
quant_rmsnorm.py
quant_text_scaled_word_embedding.py
quant_text_mlp.py
quant_text_attention.py
quant_text_decoder_layer.py
quant_text_model.py
quant_for_causal_lm.py
```

Add smoke tests for text-only tiny model.

### Phase 2: vision and multimodal wrappers

Implement:

```text
quant_clippable_linear.py
quant_vision_patch_embedder.py
quant_vision_mlp.py
quant_vision_attention.py
quant_vision_encoder_layer.py
quant_vision_encoder.py
quant_vision_model.py
quant_multimodal_embedder.py
quant_model.py
quant_for_conditional_generation.py
```

Add image-text smoke tests.

### Phase 3: evaluation and comparison

- Add Gemma4 E2B LLaVA-Bench judge config.
- Generate FP and PTQ answer files.
- Compare against Qwen3-VL using the same sample set and judge.

### Phase 4: PTQ builder and recipe adapter

Implement:

```text
build_gemma4_ptq_config()
Gemma4Adapter
Gemma4 example configs
```

Run wrapper trace/parity and PTQ calibration.

### Phase 5: static export adapters

Implement:

```text
Gemma4TokenEmbeddingExportAdapter
Gemma4VisionPrefillExportAdapter
Gemma4MMFusionExportAdapter
Gemma4PLEProjectionExportAdapter, if needed
Gemma4TextDecoderLayerPrefillExportAdapter
Gemma4TextDecoderLayerDecodeExportAdapter
Gemma4LMHeadExportAdapter
```

Add shape-contract tests.

### Phase 6: static runtime prototype

Implement:

```text
static_gemma4_runtime.py
```

The runtime should:

- preprocess inputs on CPU;
- validate static profile;
- call NPU-compatible adapters;
- manage caches on CPU;
- verify logits against HF reference;
- support greedy generation.
