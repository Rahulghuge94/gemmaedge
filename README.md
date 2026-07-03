# GemmaEdge

GemmaEdge is an experimental, model-specific GGUF inference engine for running
multimodal Gemma 4 26B A4B on memory-constrained Android devices. The primary
target is the Pixel 8a (8 GB RAM), using existing Q4/Q8 GGUF weights, a bounded
resident expert cache, and disk-tiered global KV state.

This repository currently contains the storage/runtime foundation:

- direct GGUF v2/v3 metadata and tensor-directory loading;
- a byte-budgeted expert cache with router-aware admission;
- an append-only, checksummed, block-based disk KV store;
- host-side tests that run before Android or GPU code is introduced.

It does not perform model inference yet.

GemmaEdge consumes the existing quantized text GGUF and multimodal projector
GGUF directly:

```sh
gemmaedge inspect gemma-4-26B-A4B-it-Q4_K_M.gguf
gemmaedge inspect mmproj-gemma-4-26B-A4B-it-BF16.gguf
```

The vision GGUF is opened lazily only when a request contains images.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Design constraints

- The model is never expanded to FP16 in persistent memory.
- Attention/router/output "spine" tensors are intended to stay resident.
- Routed experts are independently addressable and loaded on demand.
- Sliding-window KV remains in RAM; only global KV blocks are persisted.
- GGUF tensors are read in place without persistent dequantization.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the working design.

## Platforms

- Linux x86-64/aarch64: CLI, core tests and future CUDA/Vulkan backends.
- Android arm64-v8a: CLI plus JNI shared library, API level 28 or newer.
- Windows: host-side development and correctness tests.

Linux:

```sh
bash scripts/build_linux.sh
```

Android:

```sh
export ANDROID_NDK_HOME=/path/to/android-ndk
bash scripts/build_android.sh
```

For the initial hosted workflow, follow
[docs/COLAB.md](docs/COLAB.md).
