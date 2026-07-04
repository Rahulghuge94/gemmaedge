# GemmaEdge Architecture

This document describes the design, execution flow, and hardware optimization targets of the GemmaEdge inference engine.

---

## Objectives & Performance Goals

The primary goal of GemmaEdge is to run multimodal **Gemma 4 26B A4B** GGUF inference efficiently across two distinct execution profiles:
1. **Edge Fallback (Pixel 8a)**: Runs on CPU/RAM with strict constraints, fitting within an 8 GB memory footprint, optimizing thermal behavior, and utilizing ARM Neon SIMD registers.
2. **High-Throughput Host (Colab T4 GPU)**: Accelerates execution using customized CUDA kernels, parallel stream orchestration, and register-level SIMD operations to achieve over **10-13 tokens/sec** decode speed.

---

## Core Class Architecture

The engine splits model architecture definitions and memory ownership from active execution states:

- **`MappedFile`**: Manages direct read-only disk-to-memory mappings of GGUF weight files.
- **`GgufFile`**: Parses GGUF version 2/3 metadata, directories, and tensor records.
- **`Gemma4Model` & `Gemma4VisionModel`**: Represent text and vision projector configurations, holding weight maps and verifying layer architecture compatibility.
- **`LayerKvCache`**: Manages quantized 8-bit (`BlockQ8_0`) key and value token states. Supports fixed sliding-window cache boundaries using an $O(1)$ eviction ring-buffer.
- **`Gemma4Session`**: Owns the execution runtime, including token history, samplers, KV states, and the expert cache budget.
- **`CudaWorkspace`**: Controls CUDA backend device allocations, persistent stream configurations, and pre-cached event synchronization primitives.

---

## Weight Tiers & Expert Cache Management

GemmaEdge handles weights based on execution profiles to maximize memory utility:

### Spine Tensors (Resident)
Attention projections, layer normalization values, router parameters, and embeddings are kept permanently resident in GPU VRAM (or system RAM for CPU execution).

### Mixture of Experts (MoE) Tensors
Expert gate/up/down matrices remain in their original GGUF quantization (e.g., `Q4_K` or `Q6_K`). They are loaded dynamically or preloaded depending on the hardware profile:
- **Active Startup Preloading**: If the cache budget allows ($\ge 600\text{ MB}$), all 8 expert weights are pre-committed to device memory at startup, resulting in **zero cache misses** and zero runtime disk/PCIe transfer stalls.
- **Bounded-Budget LRU Cache**: For memory-constrained devices (like the Pixel 8a), the engine employs a byte-budgeted LRU cache that dynamically loads and evicts expert segments.

---

## Key-Value (KV) Cache Tiers & Storage

- **Quantized RAM Storage**: To conserve memory, key and value states are quantized and stored in RAM as 8-bit blocks (`BlockQ8_0`).
- **Sliding-Window Cache**: Windowed KV caches are managed as in-memory ring buffers.
- **Disk-Tiered Global Store**: A standalone `DiskKvStore` class is available to write and read checksummed, quantized 128-token global attention blocks to disk. (Full session integration is currently in progress).

---

## Parallel Execution & SIMD Optimization

- **Concurrent CUDA Streams**: The CUDA backend dispatches Q, K, and V projections in parallel streams. The dense feedforward path (main stream) executes concurrently with expert feedforward projections (across 8 independent CUDA streams), synchronizing via pre-allocated events.
- **CPU SIMD Primitives**: Features hand-optimized vector assembly loops (`dot_q8_block_neon`, `dot_q8_block_avx2`, and `accumulate_q8_block_*`) using Neon (ARM64) and AVX2/FMA (x86_64) to accelerate CPU attention score calculations and CPU fallback paths.

---

## Implementation Status & Roadmap

1. **Loader & Tensor Directory Parsing**: Direct GGUF loading and metadata inspection. (DONE)
2. **SIMD CPU Quantization Primitives**: Neon and AVX2 vector block-multiplication and lookup table (LUT) FP16-to-FP32 conversions. (DONE)
3. **Tokenizer & Chat Template Formatting**: BPE tokenizer with custom turn boundaries. (DONE)
4. **Execution Session & Prefill Loops**: Bounded generation, prefill latency metrics, and greedy sampling. (DONE)
5. **CUDA Acceleration Backend**: Custom matvec kernels, concurrent MoE stream routing, and cached events. (DONE)
6. **Android NDK Cross-Compilation**: `arm64-v8a` toolchain preset integration for edge devices. (DONE)
7. **Multimodal Projector Integration**: Projector weight validation (DONE); full session multimodal processing (IN PROGRESS).
8. **Disk-Based Global KV Block Streaming**: `DiskKvStore` implementation (DONE); session runtime integration (IN PROGRESS).

