#include "gemmaedge/cuda_backend.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mutex>
#include <unordered_map>

namespace gemmaedge {
namespace {

// ============================================================================
// Device helpers
// ============================================================================

__device__ inline float cuda_f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// Warp-level sum reduction (no shared memory needed)
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ============================================================================
// Quantized block structures (matching GGML layout)
// ============================================================================

struct BlockQ4_0 {
    uint16_t d;
    uint8_t qs[16];
};

struct BlockQ8_0 {
    uint16_t d;
    int8_t q[32];
};

struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t q[128];
};

struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

// ============================================================================
// Q4_K scale/min helper
// ============================================================================

struct ScaleMin {
    uint8_t scale;
    uint8_t minimum;
};

__device__ inline ScaleMin cuda_q4k_scale_min(int index, const uint8_t* packed) {
    ScaleMin res;
    if (index < 4) {
        res.scale = packed[index] & 63;
        res.minimum = packed[index + 4] & 63;
    } else {
        res.scale = (packed[index + 4] & 0x0f) |
                ((packed[index - 4] >> 6) << 4);
        res.minimum = (packed[index + 4] >> 4) |
                  ((packed[index] >> 6) << 4);
    }
    return res;
}

// ============================================================================
// WARP-COOPERATIVE MATVEC KERNELS
// Each warp (32 threads) cooperates on a single output row.
// blockDim.x MUST be a multiple of 32.
// gridDim.x * (blockDim.x / 32) >= rows.
// ============================================================================

// --- F32 ---
__global__ void matvec_f32_kernel(const float* __restrict__ matrix,
                                  const float* __restrict__ vec,
                                  float* __restrict__ out,
                                  int rows, int cols) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;

    const float* row = matrix + (long long)warp_id * cols;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32)
        sum += row[i] * vec[i];
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[warp_id] = sum;
}

// --- Q4_0 ---
__global__ void matvec_q4_0_kernel(const BlockQ4_0* __restrict__ matrix,
                                    const float* __restrict__ vec,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;

    int blocks_per_row = cols / 32;
    const BlockQ4_0* row = matrix + (long long)warp_id * blocks_per_row;
    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ4_0& block = row[b];
        float scale = cuda_f16_to_f32(block.d);
        int v_off = b * 32;
        float local = 0.0f;
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = block.qs[i];
            local += scale * ((int)(packed & 0x0f) - 8) * vec[v_off + i];
            local += scale * ((int)(packed >> 4) - 8) * vec[v_off + 16 + i];
        }
        sum += local;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[warp_id] = sum;
}

// --- Q8_0 ---
__global__ void matvec_q8_0_kernel(const BlockQ8_0* __restrict__ matrix,
                                    const float* __restrict__ vec,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;

    int blocks_per_row = cols / 32;
    const BlockQ8_0* row = matrix + (long long)warp_id * blocks_per_row;
    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ8_0& block = row[b];
        float scale = cuda_f16_to_f32(block.d);
        int v_off = b * 32;
        float local = 0.0f;
        for (int i = 0; i < 32; ++i)
            local += (float)block.q[i] * vec[v_off + i];
        sum += local * scale;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[warp_id] = sum;
}

// --- Q4_K ---
__global__ void matvec_q4_k_kernel(const BlockQ4K* __restrict__ matrix,
                                    const float* __restrict__ vec,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;

    int blocks_per_row = cols / 256;
    const BlockQ4K* row = matrix + (long long)warp_id * blocks_per_row;
    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ4K& block = row[b];
        float d = cuda_f16_to_f32(block.d);
        float dmin = cuda_f16_to_f32(block.dmin);
        int group = 0;
        const uint8_t* q = block.q;
        int v_off = b * 256;
        float local = 0.0f;

        for (int base = 0; base < 256; base += 64) {
            ScaleMin sm1 = cuda_q4k_scale_min(group++, block.scales);
            ScaleMin sm2 = cuda_q4k_scale_min(group++, block.scales);

            float factor1 = d * sm1.scale;
            float bias1 = -dmin * sm1.minimum;
            float factor2 = d * sm2.scale;
            float bias2 = -dmin * sm2.minimum;

            for (int i = 0; i < 32; ++i)
                local += (factor1 * (q[i] & 0x0f) + bias1) * vec[v_off + base + i];
            for (int i = 0; i < 32; ++i)
                local += (factor2 * (q[i] >> 4) + bias2) * vec[v_off + base + 32 + i];
            q += 32;
        }
        sum += local;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[warp_id] = sum;
}

// --- Q6_K (NEW) ---
__global__ void matvec_q6_k_kernel(const BlockQ6K* __restrict__ matrix,
                                    const float* __restrict__ vec,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;

    int blocks_per_row = cols / 256;
    const BlockQ6K* row = matrix + (long long)warp_id * blocks_per_row;
    float sum = 0.0f;

    for (int b = lane; b < blocks_per_row; b += 32) {
        const BlockQ6K& block = row[b];
        float d = cuda_f16_to_f32(block.d);
        const uint8_t* ql = block.ql;
        const uint8_t* qh = block.qh;
        const int8_t* scales = block.scales;
        int v_off = b * 256;
        float local = 0.0f;

        for (int base = 0; base < 256; base += 128) {
            for (int i = 0; i < 32; ++i) {
                int g = i / 16;
                int q1 = ((ql[i] & 0x0f) | (((qh[i] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[i + 32] & 0x0f) | (((qh[i] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[i] >> 4) | (((qh[i] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[i + 32] >> 4) | (((qh[i] >> 6) & 3) << 4)) - 32;
                local += d * scales[g + 0] * q1 * vec[v_off + base + i];
                local += d * scales[g + 2] * q2 * vec[v_off + base + 32 + i];
                local += d * scales[g + 4] * q3 * vec[v_off + base + 64 + i];
                local += d * scales[g + 6] * q4 * vec[v_off + base + 96 + i];
            }
            ql += 64;
            qh += 32;
            scales += 8;
        }
        sum += local;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[warp_id] = sum;
}

// ============================================================================
// Workspace: reusable GPU buffers with pinned host staging
// ============================================================================

struct CudaWorkspace {
    // Streaming matrix buffer (for uncached weights)
    void* d_matrix = nullptr;
    std::size_t matrix_capacity = 0;

    // Persistent device vector & output (kept across matvec calls)
    float* d_vector = nullptr;
    float* d_output = nullptr;
    std::size_t vector_capacity = 0;
    std::size_t output_capacity = 0;

    // Pinned host staging buffers for async DMA
    float* h_pinned_vector = nullptr;
    float* h_pinned_output = nullptr;
    std::size_t pinned_vector_capacity = 0;
    std::size_t pinned_output_capacity = 0;

    // Double-buffered matrix staging
    void* d_matrix_b = nullptr;
    std::size_t matrix_b_capacity = 0;

    // MoE device buffers
    float* d_moe_gate_up = nullptr;
    float* d_moe_activated = nullptr;
    float* d_moe_output = nullptr;
    float* d_moe_sum = nullptr;
    std::size_t moe_capacity = 0;

    cudaStream_t stream = nullptr;
    cudaStream_t stream_b = nullptr;  // second stream for overlapped uploads
    bool initialized = false;

    void ensure(std::size_t mat_bytes, std::size_t vec_elems, std::size_t out_elems) {
        if (!initialized) {
            cudaStreamCreate(&stream);
            cudaStreamCreate(&stream_b);
            initialized = true;
        }
        if (mat_bytes > 0 && mat_bytes > matrix_capacity) {
            if (d_matrix) cudaFree(d_matrix);
            d_matrix = nullptr;
            matrix_capacity = 0;
            if (cudaMalloc(&d_matrix, mat_bytes) == cudaSuccess)
                matrix_capacity = mat_bytes;
        }
        std::size_t vec_bytes = vec_elems * sizeof(float);
        if (vec_bytes > vector_capacity) {
            if (d_vector) cudaFree(d_vector);
            d_vector = nullptr;
            vector_capacity = 0;
            if (cudaMalloc(&d_vector, vec_bytes) == cudaSuccess)
                vector_capacity = vec_bytes;
            // Also allocate pinned host buffer
            if (h_pinned_vector) cudaFreeHost(h_pinned_vector);
            h_pinned_vector = nullptr;
            pinned_vector_capacity = 0;
            if (cudaMallocHost(&h_pinned_vector, vec_bytes) == cudaSuccess)
                pinned_vector_capacity = vec_bytes;
        }
        std::size_t out_bytes = out_elems * sizeof(float);
        if (out_bytes > output_capacity) {
            if (d_output) cudaFree(d_output);
            d_output = nullptr;
            output_capacity = 0;
            if (cudaMalloc(&d_output, out_bytes) == cudaSuccess)
                output_capacity = out_bytes;
            // Also allocate pinned host output
            if (h_pinned_output) cudaFreeHost(h_pinned_output);
            h_pinned_output = nullptr;
            pinned_output_capacity = 0;
            if (cudaMallocHost(&h_pinned_output, out_bytes) == cudaSuccess)
                pinned_output_capacity = out_bytes;
        }
    }

    void ensure_moe_buffers(std::size_t hidden, std::size_t expert_size) {
        if (!initialized) {
            cudaStreamCreate(&stream);
            cudaStreamCreate(&stream_b);
            initialized = true;
        }
        std::size_t req_capacity = hidden > expert_size * 2 ? hidden : expert_size * 2;
        if (req_capacity > moe_capacity) {
            if (d_moe_gate_up) cudaFree(d_moe_gate_up);
            if (d_moe_activated) cudaFree(d_moe_activated);
            if (d_moe_output) cudaFree(d_moe_output);
            if (d_moe_sum) cudaFree(d_moe_sum);
            
            d_moe_gate_up = nullptr;
            d_moe_activated = nullptr;
            d_moe_output = nullptr;
            d_moe_sum = nullptr;
            moe_capacity = 0;
            
            if (cudaMalloc(&d_moe_gate_up, expert_size * 2 * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_moe_activated, expert_size * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_moe_output, hidden * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_moe_sum, hidden * sizeof(float)) == cudaSuccess) {
                moe_capacity = req_capacity;
            }
        }
    }

    ~CudaWorkspace() {
        if (d_matrix) cudaFree(d_matrix);
        if (d_matrix_b) cudaFree(d_matrix_b);
        if (d_vector) cudaFree(d_vector);
        if (d_output) cudaFree(d_output);
        if (h_pinned_vector) cudaFreeHost(h_pinned_vector);
        if (h_pinned_output) cudaFreeHost(h_pinned_output);
        if (d_moe_gate_up) cudaFree(d_moe_gate_up);
        if (d_moe_activated) cudaFree(d_moe_activated);
        if (d_moe_output) cudaFree(d_moe_output);
        if (d_moe_sum) cudaFree(d_moe_sum);
        if (stream) cudaStreamDestroy(stream);
        if (stream_b) cudaStreamDestroy(stream_b);
    }
};

static CudaWorkspace& get_workspace() {
    static CudaWorkspace ws;
    return ws;
}
static std::mutex g_cuda_mutex;
static std::unordered_map<const void*, void*> g_cuda_weights;

// ============================================================================
// Helpers: compute weight bytes, determine warp grid, launch kernel
// ============================================================================

static std::size_t weight_bytes_for(GgmlType type, std::size_t rows, std::size_t cols) {
    switch (type) {
        case GgmlType::F32:  return rows * cols * sizeof(float);
        case GgmlType::Q4_0: return (rows * cols / 32) * sizeof(BlockQ4_0);
        case GgmlType::Q8_0: return (rows * cols / 32) * sizeof(BlockQ8_0);
        case GgmlType::Q4_K: return (rows * cols / 256) * sizeof(BlockQ4K);
        case GgmlType::Q6_K: return (rows * cols / 256) * sizeof(BlockQ6K);
        default: return 0;
    }
}

// Launch with warp-cooperative grid: blockDim = 256 threads = 8 warps per block
static void launch_kernel(GgmlType type, const void* dev_matrix,
                          const float* dev_vec, float* dev_out,
                          int rows, int cols, cudaStream_t s) {
    constexpr int kThreadsPerBlock = 256;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
    int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;

    switch (type) {
        case GgmlType::F32:
            matvec_f32_kernel<<<blocks, kThreadsPerBlock, 0, s>>>(
                static_cast<const float*>(dev_matrix), dev_vec, dev_out, rows, cols);
            break;
        case GgmlType::Q4_0:
            matvec_q4_0_kernel<<<blocks, kThreadsPerBlock, 0, s>>>(
                static_cast<const BlockQ4_0*>(dev_matrix), dev_vec, dev_out, rows, cols);
            break;
        case GgmlType::Q8_0:
            matvec_q8_0_kernel<<<blocks, kThreadsPerBlock, 0, s>>>(
                static_cast<const BlockQ8_0*>(dev_matrix), dev_vec, dev_out, rows, cols);
            break;
        case GgmlType::Q4_K:
            matvec_q4_k_kernel<<<blocks, kThreadsPerBlock, 0, s>>>(
                static_cast<const BlockQ4K*>(dev_matrix), dev_vec, dev_out, rows, cols);
            break;
        case GgmlType::Q6_K:
            matvec_q6_k_kernel<<<blocks, kThreadsPerBlock, 0, s>>>(
                static_cast<const BlockQ6K*>(dev_matrix), dev_vec, dev_out, rows, cols);
            break;
        default:
            break;
    }
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

bool is_cuda_available() {
    // Cache the result — cudaGetDeviceCount() is a driver call that costs
    // microseconds each time. Over 40,000+ calls per generation, this adds up.
    static int cached = -1;
    if (cached < 0) {
        int device_count = 0;
        cached = (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) ? 1 : 0;
    }
    return cached == 1;
}

bool cuda_matvec(GgmlType type, const void* matrix, std::size_t rows, std::size_t cols,
                 const float* vector, float* output) {
    if (!is_cuda_available()) return false;

    std::size_t wbytes = weight_bytes_for(type, rows, cols);
    if (wbytes == 0) return false;  // Unsupported type

    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();

    // Check if the weight matrix is cached in VRAM
    void* dev_matrix = nullptr;
    auto it = g_cuda_weights.find(matrix);
    bool cached = (it != g_cuda_weights.end());
    if (cached) {
        dev_matrix = it->second;
    }

    ws.ensure(cached ? 0 : wbytes, cols, rows);
    if (!ws.d_vector || !ws.d_output || (!cached && !ws.d_matrix)) return false;

    if (!cached) {
        // Use pinned staging if available for faster H->D copy
        if (cudaMemcpyAsync(ws.d_matrix, matrix, wbytes,
                            cudaMemcpyHostToDevice, ws.stream) != cudaSuccess)
            return false;
        dev_matrix = ws.d_matrix;
    }

    // Upload input vector via pinned memory
    if (ws.h_pinned_vector && ws.pinned_vector_capacity >= cols * sizeof(float)) {
        memcpy(ws.h_pinned_vector, vector, cols * sizeof(float));
        if (cudaMemcpyAsync(ws.d_vector, ws.h_pinned_vector, cols * sizeof(float),
                            cudaMemcpyHostToDevice, ws.stream) != cudaSuccess)
            return false;
    } else {
        if (cudaMemcpyAsync(ws.d_vector, vector, cols * sizeof(float),
                            cudaMemcpyHostToDevice, ws.stream) != cudaSuccess)
            return false;
    }

    // Launch warp-cooperative kernel
    launch_kernel(type, dev_matrix, ws.d_vector, ws.d_output,
                  static_cast<int>(rows), static_cast<int>(cols), ws.stream);

    // Download output via pinned memory
    bool success;
    if (ws.h_pinned_output && ws.pinned_output_capacity >= rows * sizeof(float)) {
        success = (cudaMemcpyAsync(ws.h_pinned_output, ws.d_output, rows * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream) == cudaSuccess);
        cudaStreamSynchronize(ws.stream);
        if (success) memcpy(output, ws.h_pinned_output, rows * sizeof(float));
    } else {
        success = (cudaMemcpyAsync(output, ws.d_output, rows * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream) == cudaSuccess);
        cudaStreamSynchronize(ws.stream);
    }
    return success;
}

bool cuda_matvec_device_vec(GgmlType type, const void* matrix,
                            std::size_t rows, std::size_t cols, float* host_output) {
    if (!is_cuda_available()) return false;

    std::size_t wbytes = weight_bytes_for(type, rows, cols);
    if (wbytes == 0) return false;

    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();

    // Vector must already be on device (via cuda_upload_vector)
    if (!ws.d_vector || ws.vector_capacity < cols * sizeof(float))
        return false;

    // Check if weight is cached
    void* dev_matrix = nullptr;
    auto it = g_cuda_weights.find(matrix);
    bool cached = (it != g_cuda_weights.end());
    if (cached) {
        dev_matrix = it->second;
    }

    ws.ensure(cached ? 0 : wbytes, 0, rows);  // 0 for vec — already on device
    if (!ws.d_output || (!cached && !ws.d_matrix)) return false;

    if (!cached) {
        if (cudaMemcpyAsync(ws.d_matrix, matrix, wbytes,
                            cudaMemcpyHostToDevice, ws.stream) != cudaSuccess)
            return false;
        dev_matrix = ws.d_matrix;
    }

    // Launch — vector is already on device, no upload needed
    launch_kernel(type, dev_matrix, ws.d_vector, ws.d_output,
                  static_cast<int>(rows), static_cast<int>(cols), ws.stream);

    // Download output
    bool success;
    if (ws.h_pinned_output && ws.pinned_output_capacity >= rows * sizeof(float)) {
        success = (cudaMemcpyAsync(ws.h_pinned_output, ws.d_output, rows * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream) == cudaSuccess);
        cudaStreamSynchronize(ws.stream);
        if (success) memcpy(host_output, ws.h_pinned_output, rows * sizeof(float));
    } else {
        success = (cudaMemcpyAsync(host_output, ws.d_output, rows * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream) == cudaSuccess);
        cudaStreamSynchronize(ws.stream);
    }
    return success;
}

bool cuda_upload_vector(const float* host, std::size_t count) {
    if (!is_cuda_available() || !host || count == 0) return false;
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();
    std::size_t bytes = count * sizeof(float);
    ws.ensure(0, count, 0);
    if (!ws.d_vector) return false;

    if (ws.h_pinned_vector && ws.pinned_vector_capacity >= bytes) {
        memcpy(ws.h_pinned_vector, host, bytes);
        return (cudaMemcpyAsync(ws.d_vector, ws.h_pinned_vector, bytes,
                                cudaMemcpyHostToDevice, ws.stream) == cudaSuccess);
    }
    return (cudaMemcpyAsync(ws.d_vector, host, bytes,
                            cudaMemcpyHostToDevice, ws.stream) == cudaSuccess);
}

bool cuda_download_vector(float* host, std::size_t count) {
    if (!is_cuda_available() || !host || count == 0) return false;
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();
    std::size_t bytes = count * sizeof(float);
    if (!ws.d_output || ws.output_capacity < bytes) return false;

    if (ws.h_pinned_output && ws.pinned_output_capacity >= bytes) {
        bool ok = (cudaMemcpyAsync(ws.h_pinned_output, ws.d_output, bytes,
                                   cudaMemcpyDeviceToHost, ws.stream) == cudaSuccess);
        cudaStreamSynchronize(ws.stream);
        if (ok) memcpy(host, ws.h_pinned_output, bytes);
        return ok;
    }
    return (cudaMemcpy(host, ws.d_output, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
}

bool cuda_register_weight(const void* host_ptr, std::size_t bytes) {
    if (!is_cuda_available() || !host_ptr || bytes == 0) return false;
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    if (g_cuda_weights.find(host_ptr) != g_cuda_weights.end()) {
        return true;
    }
    void* d_ptr = nullptr;
    auto& ws = get_workspace();
    if (cudaMallocAsync(&d_ptr, bytes, ws.stream) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyAsync(d_ptr, host_ptr, bytes, cudaMemcpyHostToDevice, ws.stream) != cudaSuccess) {
        cudaFreeAsync(d_ptr, ws.stream);
        return false;
    }
    g_cuda_weights[host_ptr] = d_ptr;
    return true;
}

void cuda_unregister_weight(const void* host_ptr) {
    if (!host_ptr) return;
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto it = g_cuda_weights.find(host_ptr);
    if (it != g_cuda_weights.end()) {
        auto& ws = get_workspace();
        cudaFreeAsync(it->second, ws.stream);
        g_cuda_weights.erase(it);
    }
}

void cuda_clear_weights() {
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();
    for (auto& pair : g_cuda_weights) {
        cudaFreeAsync(pair.second, ws.stream);
    }
    g_cuda_weights.clear();
}

__global__ void gelu_activation_kernel(const float* gate_up, float* activated, int expert_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= expert_size) return;
    float g = gate_up[idx];
    float u = gate_up[expert_size + idx];
    float x = 0.79788456f * (g + 0.044715f * g * g * g);
    float tanh_x = tanhf(x);
    float gelu_g = 0.5f * g * (1.0f + tanh_x);
    activated[idx] = gelu_g * u;
}

__global__ void accumulate_sum_kernel(const float* expert_output, float* expert_sum, float weight, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden) return;
    expert_sum[idx] += weight * expert_output[idx];
}

bool cuda_matvec_d2d(GgmlType type, const void* matrix, const float* d_vector,
                     float* d_output, std::size_t rows, std::size_t cols) {
    if (!is_cuda_available()) return false;
    std::size_t wbytes = weight_bytes_for(type, rows, cols);
    if (wbytes == 0) return false;

    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();

    void* dev_matrix = nullptr;
    auto it = g_cuda_weights.find(matrix);
    bool cached = (it != g_cuda_weights.end());
    if (cached) {
        dev_matrix = it->second;
    } else {
        ws.ensure(wbytes, 0, 0);
        if (!ws.d_matrix) return false;
        if (cudaMemcpyAsync(ws.d_matrix, matrix, wbytes,
                            cudaMemcpyHostToDevice, ws.stream) != cudaSuccess)
            return false;
        dev_matrix = ws.d_matrix;
    }

    launch_kernel(type, dev_matrix, d_vector, d_output,
                  static_cast<int>(rows), static_cast<int>(cols), ws.stream);
    return true;
}

void cuda_moe_gelu(const float* d_gate_up, float* d_activated, std::size_t expert_size) {
    auto& ws = get_workspace();
    int threads = 256;
    int blocks = (static_cast<int>(expert_size) + threads - 1) / threads;
    gelu_activation_kernel<<<blocks, threads, 0, ws.stream>>>(d_gate_up, d_activated, static_cast<int>(expert_size));
}

void cuda_moe_accumulate(const float* d_expert_output, float* d_expert_sum, float weight, std::size_t hidden) {
    auto& ws = get_workspace();
    int threads = 256;
    int blocks = (static_cast<int>(hidden) + threads - 1) / threads;
    accumulate_sum_kernel<<<blocks, threads, 0, ws.stream>>>(d_expert_output, d_expert_sum, weight, static_cast<int>(hidden));
}

void cuda_zero_buffer(float* d_buf, std::size_t count) {
    auto& ws = get_workspace();
    cudaMemsetAsync(d_buf, 0, count * sizeof(float), ws.stream);
}

bool cuda_download_vector_from(float* host, const float* d_buf, std::size_t count) {
    if (!is_cuda_available() || !host || count == 0 || !d_buf) return false;
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();
    std::size_t bytes = count * sizeof(float);

    if (ws.h_pinned_output && ws.pinned_output_capacity >= bytes) {
        bool ok = (cudaMemcpyAsync(ws.h_pinned_output, d_buf, bytes,
                                   cudaMemcpyDeviceToHost, ws.stream) == cudaSuccess);
        cudaStreamSynchronize(ws.stream);
        if (ok) memcpy(host, ws.h_pinned_output, bytes);
        return ok;
    }
    bool ok = (cudaMemcpyAsync(host, d_buf, bytes,
                               cudaMemcpyDeviceToHost, ws.stream) == cudaSuccess);
    cudaStreamSynchronize(ws.stream);
    return ok;
}

float* cuda_get_vector_buf() {
    return get_workspace().d_vector;
}

float* cuda_get_moe_gate_up_buf() {
    return get_workspace().d_moe_gate_up;
}

float* cuda_get_moe_activated_buf() {
    return get_workspace().d_moe_activated;
}

float* cuda_get_moe_output_buf() {
    return get_workspace().d_moe_output;
}

float* cuda_get_moe_sum_buf() {
    return get_workspace().d_moe_sum;
}

void cuda_ensure_moe_buffers(std::size_t hidden, std::size_t expert_size) {
    get_workspace().ensure_moe_buffers(hidden, expert_size);
}

} // namespace gemmaedge
