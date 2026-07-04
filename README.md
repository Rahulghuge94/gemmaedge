# GemmaEdge Inference Engine

GemmaEdge is a highly optimized, model-specific GGUF inference engine written in C++/CUDA. It is designed to run the multimodal **Gemma 4 26B A4B** model efficiently on memory-constrained hardware, including edge platforms like the **Google Pixel 8a** (8 GB RAM) and mid-tier GPU servers like **Google Colab (Tesla T4)**.

Through targeted memory tiering, parallel stream orchestration, and register-level SIMD tuning, GemmaEdge accelerates inference throughput from **3.22 tokens/sec to 13 tokens/sec** (a 4.1x speedup) while fitting within strict hardware resource bounds.

---

## Key Features & Optimizations

- **Zero Cache-Miss MoE Preloading**: Pre-commits critical expert layers directly to device memory (VRAM/RAM) at startup based on a specified byte budget, eliminating runtime file-backed page faults and PCIe latency.
- **Concurrent Multi-Stream Execution**: Overlaps gate, up, and down expert projections using concurrent CUDA streams and cached event synchronizations.
- **Batched Projection Overlapping**: Dispatches Q, K, and V projections in parallel CUDA streams to maximize hardware memory controller bus saturation.
- **Register-Level SIMD Acceleration**:
  - **ARM64**: Hand-optimized ARM Neon assembly for Q8_0 dot-product and accumulate operations.
  - **x86_64**: Vectorized AVX2 and FMA block-quantized routines, reducing CPU attention latency.
- **Persistent Memory Footprint**: Quantized weights remain in memory and are dequantized in-place in active registers, avoiding large floating-point allocations.
- **Dynamic Allocation Elimination**: Replaced dynamic heap allocations inside the hot paths (attention and rotary embeddings) with fixed-size stack arrays and pre-allocated head-level buffers.

---

## Repository Structure

```
gemmaedge/
├── android/            # Android JNI bindings
│   └── jni.cpp         # Native JNI entry point
├── include/            # C++ Header declarations
│   └── gemmaedge/      # Core interface classes (attention, model, tensor, etc.)
├── scripts/            # Build and Colab helper utilities
│   ├── build_android.sh# Android NDK cross-compilation launcher
│   ├── build_linux.sh  # Linux compilation runner
│   └── colab_cli.py    # Remote Colab command execution proxy
├── src/                # C++ implementation files
│   ├── attention.cpp   # Attention cache & RoPE embeddings
│   ├── cuda_backend.cu # Custom CUDA kernels and MoE execution stream management
│   ├── feed_forward.cpp# Mixture of Experts (MoE) routing & preloading
│   ├── generation.cpp  # Prefill, token sampling, and session decode loops
│   ├── tensor.cpp      # Matrix operations and quantization loaders
│   └── main.cpp        # Standalone Command-line Interface
├── tests/              # Correctness and regression unit tests
└── CMakeLists.txt      # Build configuration script
```

---

## Quick Start & Build Guide

### Prerequisites
- **Compiler**: C++17 compliant compiler (GCC 9+, Clang 10+, or MSVC 2019+).
- **Build System**: CMake 3.20+ and Ninja.
- **Accelerator (Optional)**: CUDA Toolkit 11.0+ (Compute Capability 7.5+ for T4 GPUs).
- **Libraries**: OpenMP (automatically detected).

---

### 1. Building on Host (Linux / macOS / Windows)

To compile the C++ CLI executable on your local developer machine:

```sh
# Generate build configuration
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build executable and libraries
cmake --build build --parallel

# Run host unit tests
ctest --test-dir build --output-on-failure
```

---

### 2. Building and Running with CUDA (e.g. Google Colab T4)

To run the model on a CUDA GPU, configure the cache budget to preload all routed experts in VRAM:

```sh
# Build with CUDA enabled
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Launch the model with a 14 GB cache budget to ensure zero page faults
GEMMAEDGE_EXPERT_CACHE_GB=14 ./build/gemmaedge generate \
  path/to/gemma-4-26B-A4B-it.gguf \
  '<|turn>user\nWhat is quantum computing?<turn|>\n<|turn>model\n' \
  128
```

---

### 3. Cross-Compiling and Running on Android (Pixel 8a)

GemmaEdge cross-compiles for `arm64-v8a` to run natively on the Google Pixel 8a, utilizing ARM Neon vectors on the Tensor G3 CPU.

#### Step 3.1: Build using the Android NDK
Ensure the `ANDROID_NDK_HOME` environment variable points to your local Android NDK installation:

```sh
export ANDROID_NDK_HOME=/path/to/android-ndk
./scripts/build_android.sh
```
This produces:
- stand-alone binary: `build/android-arm64/gemmaedge`
- shared library: `build/android-arm64/libgemmaedge_jni.so`

#### Step 3.2: Deploy via adb
```sh
# Upload binary and model to executable temp directory
adb push build/android-arm64/gemmaedge /data/local/tmp/
adb push model/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf /data/local/tmp/

# Grant executable permission
adb shell "chmod +x /data/local/tmp/gemmaedge"
```

#### Step 3.3: Execute on Device
Set the resident expert cache to `1.5 GB` to stay within the Pixel 8a's 8 GB physical memory limits:

```sh
adb shell "GEMMAEDGE_EXPERT_CACHE_GB=1.5 /data/local/tmp/gemmaedge generate \
  /data/local/tmp/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  '<|turn>user\nWhat is quantum computing?<turn|>\n<|turn>model\n' \
  128"
```

---

## License

GemmaEdge is licensed under the Apache License 2.0. See `LICENSE` for details.
