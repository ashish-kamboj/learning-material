# Quantization for LLMs

----------

## 1) What is quantization

**Quantization** reduces the numeric precision used to store and compute a model’s parameters and/or activations (for example from 16-bit floating point → 8-bit integer, 4-bit, 3-bit, etc.). This lowers model size, memory footprint, bandwidth and often inference cost/latency — at the expense of potential accuracy degradation unless specialized methods are used.

----------

## 2) High-level ways it’s applied

-   **Weight-only quantization**: only the model weights are quantized (activations left at higher precision). Common for inference-focused compression of LLMs.
    
-   **Activation + weight quantization**: both weights and activations are quantized — enables more memory/compute savings but is harder (needs better calibration / hardware support).
    
-   **Post-training quantization (PTQ)**: quantize a pretrained model without retraining; often uses a small calibration dataset. Fast and practical, but may degrade accuracy. 
    
-   **Quantization-aware training (QAT)**: include quantization effects during training so the model “learns” to be robust to low precision. Better accuracy but expensive.

----------

## 3) How quantization _actually_ works (core techniques)

-   **Uniform vs non-uniform quantization**: map continuous values to discrete levels. Uniform maps with equal step size; non-uniform (e.g., k-means, log, or learned) uses uneven levels to better fit distribution.
    
-   **Symmetric vs asymmetric**: symmetric quant maps around zero; asymmetric shifts range to use dynamic range better (important for activations).
    
-   **Per-tensor vs per-channel (row/column) scaling**: per-channel scales reduce quantization error for matrices with diverse ranges at the cost of storing more scale factors. Common for transformer weight matrices.
    
-   **Block/row-wise quantization & outlier handling**: libraries (e.g., bitsandbytes) use block-wise quantization and special handling for outlier columns so that heavy-tailed weights don’t ruin accuracy. 
    
-   **Advanced reconstruction / second-order methods**: methods like GPTQ use approximate second-order reconstruction to find quantized weights that minimize model-output error, enabling very low-bit (3–4 bit) weight-only quantization with small accuracy loss.
    

----------

## 4) Notable algorithms & libraries (practical)

-   **GPTQ (post-training, weight-only, 3–4 bit)** — shown to quantize 175B models to 3–4 bits with very small accuracy loss and practical single-GPU execution time (paper + implementation). Good for aggressive post-training compression.
    
-   **AWQ (Activation-aware Weight Quantization)** — protects a small fraction of “salient” weights based on activation statistics to improve low-bit (4-bit) weight quantization and generalize better across domains and instruction-tuned models.
    
-   **QLoRA (quantized fine-tuning)** — quantize the frozen base model (4-bit NF4 data type) and finetune low-rank adapters (LoRA). Enables finetuning very large models on a single 48GB GPU while preserving full 16-bit finetuning performance. Useful when you must finetune large LLMs but have limited GPU memory.
    
-   **bitsandbytes / LLM.int8()** — practical library support for 8-bit optimizers and 8-bit/4-bit inference kernels inside Hugging Face Transformers; applies block-wise quantization and special handling of outliers. Widely used in production/experimentation.
    
-   **Vendor runtimes (NVIDIA TensorRT, Intel, etc.)** — provide INT8 optimized kernels and builder flows for quantized inference; hardware support matters for speed.

----------

## 5) Advantages (why quantize LLMs)

-   **Much smaller model size** (bits per parameter reduced) → can fit much larger models on a single GPU or on-device. 
    
-   **Lower memory bandwidth and working set** → reduces host↔device transfers and enables larger batch sizes or longer context at same RAM.
    
-   **Faster inference and lower cost** when hardware has optimized low-precision kernels (e.g., INT8/TensorRT); also reduces energy consumption. Papers report multiple× speedups vs FP16 in many cases.
    
-   **Enables local / edge deployment** of large models that otherwise require multi-GPU servers (AWQ, GPTQ, TinyChat examples). 

----------

## 6) Disadvantages / risks

-   **Accuracy degradation** — naive quantization (especially low bits) can hurt perplexity, generation quality, and downstream task performance unless advanced methods (GPTQ/AWQ/QAT) are used.
    
-   **Task/model sensitivity** — some layers or heads are more sensitive; fine-grained decisions (which tensors to keep in higher precision) are often necessary.
    
-   **Hardware dependency** — speed gains require hardware-optimized kernels (Tensor cores, INT8 support). On hardware without support, quantized code may not be faster. 
    
-   **Implementation complexity & tooling** — methods like GPTQ, AWQ, or QLoRA need specialized tooling, careful calibration data, and sometimes kernel-level code. 
    
-   **Possible stability/regression in sensitive tasks** — small numeric differences can amplify for some generation prompts, leading to perceptible regressions. Test thoroughly.

----------

## 7) Practical rules-of-thumb — _When to use which approach_

-   **You need to deploy a very large model on limited hardware for inference** → try **weight-only PTQ** with GPTQ/AWQ (3–4 bit) + evaluation. Good starting point. 
    
-   **You must finetune a huge model but only have 1–2 GPUs (memory constrained)** → use **QLoRA** (4-bit NF4 + LoRA adapters) to finetune with minimal memory overhead while keeping base frozen.
    
-   **You need predictable, highest accuracy in a production critical task** → prefer **QAT** or keep higher-precision (FP16) for sensitive layers; consider hybrid: quantize some parts, keep others at FP16.
    
-   **If hardware supports INT8/TensorRT** → aim for INT8 optimized flows (via vendor runtimes) for best latency/cost improvements. If not, 4-bit weight-only methods could still reduce memory footprint even if speedups are modest. 
    
-   **Quick experimentation**: start with `bitsandbytes` / Hugging Face quantization recipes (LLM.int8(), 4-bit guides), validate on your evaluation set, then move to GPTQ/AWQ if you need more compression. 

----------

## 8) Recommended evaluation checklist (must do before shipping)

1.  **Use a representative calibration/eval dataset** (for PTQ).
    
2.  **Measure both perplexity and end-task metrics** (generation quality, downstream accuracy, hallucination rate).
    
3.  **Test on production-like prompts and long context** (quantization errors can increase with length).
    
4.  **Profile latency & memory on target hardware** (verify real speedups with vendor runtimes / kernels).
    
5.  **Consider hybrid precision** (keep sensitive layers in FP16 while quantizing others).    

----------

## 9) Actionable recommendation

If you want to run large LLMs in constrained memory **start** with library-supported 8-bit/4-bit methods (bitsandbytes + Hugging Face), evaluate on your tasks; if you need more compression with minimal accuracy loss, try **GPTQ/AWQ** for post-training weight quantization; if you must finetune large models on small GPUs use **QLoRA**. Always validate with representative tests on your target hardware. 	

----------