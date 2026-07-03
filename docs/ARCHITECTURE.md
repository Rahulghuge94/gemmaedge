# GemmaEdge architecture

## Goal

Run multimodal Gemma 4 26B A4B GGUF inference on an 8 GB Android device without
requiring the Q4 model to fit in RAM. The engine optimizes sustained generation,
storage traffic, and thermal behavior rather than peak desktop throughput.

## Runtime split

`Engine` state owns immutable GGUF metadata, tensor views, execution backends,
and the expert cache. `VisionEngine` owns a separately loaded mmproj GGUF.
`Session` state owns tokens, sampler state, sliding KV, global KV block mappings,
and conversation checkpoints.

## Input files

GemmaEdge reads standard quantized GGUF files directly. It does not transcode
weights into a private model format and it never persistently expands Q4/Q8
tensors. A normal deployment contains:

- the quantized Gemma 4 text/MoE GGUF;
- the vision encoder/projector mmproj GGUF;
- optional cache/index sidecars containing no replacement weight data.

The mmproj file is opened lazily. Text-only sessions therefore do not pay its
resident-memory cost. Image embeddings are computed, projected into language
space, inserted as soft tokens, and then the vision working set can be released.

## Weight tiers

The resident spine contains embeddings/output weights, norms, attention
projections, and MoE routers. Expert gate/up/down regions remain in their
original GGUF quantization. The router runs before expert loading, allowing
asynchronous reads of only the selected expert slices. If a GGUF stores experts
as one packed tensor, a sidecar records byte ranges into that tensor; it does
not duplicate or rewrite the weights.

The cache is bounded by bytes rather than expert count because future
quantization formats and matrix padding can produce differently sized entries.
Admission combines recency with router probability. Telemetry must expose
bytes read per token; this is the primary Pixel optimization metric.

## KV tiers

Sliding-attention KV never reaches disk. The 1024-token windows remain resident.
Only five global-attention layers need growing storage.

Global KV is accumulated in RAM and sealed into 128-token blocks. Sealed blocks
are quantized, checksummed, appended to the session file, and entered into a
RAM block cache. The inference graph will request blocks asynchronously before
global attention begins.

## Correctness gates

Each implementation stage must compare against official Gemma outputs:

1. tokenizer bytes and token IDs;
2. embedding and per-layer hidden-state slices;
3. router top-k IDs and weights;
4. attention output;
5. final logits and greedy continuation.

Q4, KV quantization, reduced expert modes, and speculative decoding are tested
as separate changes so quality loss remains attributable.

## Near-term sequence

1. Direct GGUF loader and Gemma 4 tensor-name validation.
2. GGML Q4/Q8/K-quant CPU primitives and RMSNorm.
3. GGUF tokenizer metadata and chat rendering.
4. Exact single-token text graph.
5. Vision/mmproj graph and image preprocessing.
6. ARM kernels and Android NDK harness.
7. Vulkan compute backend.
8. Async expert streaming and disk global KV integration.
