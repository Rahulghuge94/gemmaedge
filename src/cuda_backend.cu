#include "gemmaedge/cuda_backend.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <mutex>
#include <unordered_map>

namespace gemmaedge {
namespace {

// ============================================================================
// Device helpers
// ============================================================================

__device__ inline float cuda_f16_to_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
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

__device__ inline ScaleMin cuda_q4k_scale_min_reg(int index, uint32_t meta_y, uint32_t meta_z, uint32_t meta_w) {
    ScaleMin res;
    uint8_t packed_val;
    uint8_t packed_val_next;
    if (index < 4) {
        packed_val = (meta_y >> (index * 8)) & 0xFFu;
        packed_val_next = (meta_z >> (index * 8)) & 0xFFu;
        res.scale = packed_val & 63;
        res.minimum = packed_val_next & 63;
    } else {
        int idx_shifted = index - 4;
        packed_val = (meta_w >> (idx_shifted * 8)) & 0xFFu;
        packed_val_next = (meta_y >> (idx_shifted * 8)) & 0xFFu;
        uint8_t packed_val_idx = (meta_z >> (idx_shifted * 8)) & 0xFFu;
        res.scale = (packed_val & 0x0f) | ((packed_val_next >> 6) << 4);
        res.minimum = (packed_val >> 4) | ((packed_val_idx >> 6) << 4);
    }
    return res;
}

__device__ inline int8_t extract_scale_q6(int idx, uint32_t s_x, uint32_t s_y, uint32_t s_z, uint32_t s_w) {
    uint32_t reg;
    if (idx < 4) {
        reg = s_x;
    } else if (idx < 8) {
        reg = s_y;
        idx -= 4;
    } else if (idx < 12) {
        reg = s_z;
        idx -= 8;
    } else {
        reg = s_w;
        idx -= 12;
    }
    return static_cast<int8_t>((reg >> (idx * 8)) & 0xFFu);
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
    float local = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const BlockQ4_0& block = row[b];
        uint16_t d_val;
        if (lane == 0) d_val = block.d;
        float scale = cuda_f16_to_f32(__shfl_sync(0xFFFFFFFF, d_val, 0));
        int v_off = b * 32;
        
        int val;
        if (lane < 16) {
            val = (block.qs[lane] & 0x0f) - 8;
        } else {
            val = (block.qs[lane - 16] >> 4) - 8;
        }
        local += scale * val * vec[v_off + lane];
    }
    float sum = warp_reduce_sum(local);
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
    float local = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const BlockQ8_0& block = row[b];
        uint16_t d_val;
        if (lane == 0) d_val = block.d;
        float scale = cuda_f16_to_f32(__shfl_sync(0xFFFFFFFF, d_val, 0));
        int v_off = b * 32;
        local += (float)block.q[lane] * vec[v_off + lane] * scale;
    }
    float sum = warp_reduce_sum(local);
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
    float local = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const BlockQ4K& block = row[b];
        
        // Coalesced metadata load (16 bytes in a single load)
        uint4 meta;
        if (lane == 0) {
            meta = *reinterpret_cast<const uint4*>(&block);
        }
        
        // Broadcast metadata registers to all threads in the warp
        uint32_t meta_x = __shfl_sync(0xFFFFFFFF, meta.x, 0);
        uint32_t meta_y = __shfl_sync(0xFFFFFFFF, meta.y, 0);
        uint32_t meta_z = __shfl_sync(0xFFFFFFFF, meta.z, 0);
        uint32_t meta_w = __shfl_sync(0xFFFFFFFF, meta.w, 0);

        float d = cuda_f16_to_f32(meta_x & 0xFFFFu);
        float dmin = cuda_f16_to_f32(meta_x >> 16);
        
        int v_off = b * 256;
        const uint8_t* q = block.q;
        int group = 0;

        for (int base = 0; base < 256; base += 64) {
            ScaleMin sm1 = cuda_q4k_scale_min_reg(group++, meta_y, meta_z, meta_w);
            ScaleMin sm2 = cuda_q4k_scale_min_reg(group++, meta_y, meta_z, meta_w);

            float factor1 = d * sm1.scale;
            float bias1 = -dmin * sm1.minimum;
            float factor2 = d * sm2.scale;
            float bias2 = -dmin * sm2.minimum;

            uint8_t q_val = q[lane];
            local += (factor1 * (q_val & 0x0f) + bias1) * vec[v_off + base + lane];
            local += (factor2 * (q_val >> 4) + bias2) * vec[v_off + base + 32 + lane];
            q += 32;
        }
    }
    float sum = warp_reduce_sum(local);
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
    float local = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const BlockQ6K& block = row[b];
        
        // Coalesced scales load (16 bytes)
        uint4 scales_val;
        uint16_t d_val;
        if (lane == 0) {
            scales_val = *reinterpret_cast<const uint4*>(block.scales);
            d_val = block.d;
        }
        
        // Broadcast
        uint32_t s_x = __shfl_sync(0xFFFFFFFF, scales_val.x, 0);
        uint32_t s_y = __shfl_sync(0xFFFFFFFF, scales_val.y, 0);
        uint32_t s_z = __shfl_sync(0xFFFFFFFF, scales_val.z, 0);
        uint32_t s_w = __shfl_sync(0xFFFFFFFF, scales_val.w, 0);
        uint16_t d_reg = __shfl_sync(0xFFFFFFFF, d_val, 0);
        
        float d = cuda_f16_to_f32(d_reg);
        const uint8_t* ql = block.ql;
        const uint8_t* qh = block.qh;
        int v_off = b * 256;

        int scale_group_offset = 0;
        for (int base = 0; base < 256; base += 128) {
            int g = lane / 16;
            
            // Extract scales from registers directly
            int scale_1_idx = scale_group_offset + g + 0;
            int scale_2_idx = scale_group_offset + g + 2;
            int scale_3_idx = scale_group_offset + g + 4;
            int scale_4_idx = scale_group_offset + g + 6;
            
            int8_t s1 = extract_scale_q6(scale_1_idx, s_x, s_y, s_z, s_w);
            int8_t s2 = extract_scale_q6(scale_2_idx, s_x, s_y, s_z, s_w);
            int8_t s3 = extract_scale_q6(scale_3_idx, s_x, s_y, s_z, s_w);
            int8_t s4 = extract_scale_q6(scale_4_idx, s_x, s_y, s_z, s_w);

            int q1 = ((ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            int q2 = ((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            int q3 = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            int q4 = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
            
            local += d * s1 * q1 * vec[v_off + base + lane];
            local += d * s2 * q2 * vec[v_off + base + 32 + lane];
            local += d * s3 * q3 * vec[v_off + base + 64 + lane];
            local += d * s4 * q4 * vec[v_off + base + 96 + lane];
            
            ql += 64;
            qh += 32;
            scale_group_offset += 8;
        }
    }
    float sum = warp_reduce_sum(local);
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
    float* d_moe_weights = nullptr;
    float* d_moe_input = nullptr;
    std::size_t moe_capacity = 0;

    // Dense FFN device buffers
    float* d_dense_gate_up = nullptr;
    float* d_dense_activated = nullptr;
    float* d_dense_output = nullptr;
    std::size_t dense_capacity = 0;

    cudaStream_t stream = nullptr;
    cudaStream_t stream_b = nullptr;  // second stream for overlapped uploads
    cudaStream_t streams[8] = {nullptr};
    cudaEvent_t norm_event = nullptr;
    cudaEvent_t dense_event = nullptr;
    cudaEvent_t exp_norm_event = nullptr;
    cudaEvent_t act_event = nullptr;
    cudaEvent_t expert_events[8] = {nullptr};
    bool initialized = false;

    void initialize_streams() {
        if (!initialized) {
            cudaStreamCreate(&stream);
            cudaStreamCreate(&stream_b);
            for (int i = 0; i < 8; ++i) {
                cudaStreamCreate(&streams[i]);
            }
            cudaEventCreateWithFlags(&norm_event, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&dense_event, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&exp_norm_event, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&act_event, cudaEventDisableTiming);
            for (int i = 0; i < 8; ++i) {
                cudaEventCreateWithFlags(&expert_events[i], cudaEventDisableTiming);
            }
            initialized = true;
        }
    }

    void ensure(std::size_t mat_bytes, std::size_t vec_elems, std::size_t out_elems) {
        initialize_streams();
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

    void ensure_moe_buffers(std::size_t hidden, std::size_t expert_size, std::size_t num_experts) {
        initialize_streams();
        std::size_t req_capacity = (hidden > expert_size * 2 ? hidden : expert_size * 2) * num_experts;
        if (req_capacity > moe_capacity) {
            if (d_moe_gate_up) cudaFree(d_moe_gate_up);
            if (d_moe_activated) cudaFree(d_moe_activated);
            if (d_moe_output) cudaFree(d_moe_output);
            if (d_moe_sum) cudaFree(d_moe_sum);
            if (d_moe_weights) cudaFree(d_moe_weights);
            if (d_moe_input) cudaFree(d_moe_input);
            
            d_moe_gate_up = nullptr;
            d_moe_activated = nullptr;
            d_moe_output = nullptr;
            d_moe_sum = nullptr;
            d_moe_weights = nullptr;
            d_moe_input = nullptr;
            moe_capacity = 0;
            
            if (cudaMalloc(&d_moe_gate_up, num_experts * expert_size * 2 * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_moe_activated, num_experts * expert_size * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_moe_output, num_experts * hidden * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_moe_sum, hidden * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_moe_weights, num_experts * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_moe_input, hidden * sizeof(float)) == cudaSuccess) {
                moe_capacity = req_capacity;
            }
        }
    }

    void ensure_dense_buffers(std::size_t hidden, std::size_t dense_size) {
        initialize_streams();
        std::size_t req_capacity = (dense_size * 2 > hidden ? dense_size * 2 : hidden);
        if (req_capacity > dense_capacity) {
            if (d_dense_gate_up) cudaFree(d_dense_gate_up);
            if (d_dense_activated) cudaFree(d_dense_activated);
            if (d_dense_output) cudaFree(d_dense_output);
            
            d_dense_gate_up = nullptr;
            d_dense_activated = nullptr;
            d_dense_output = nullptr;
            dense_capacity = 0;
            
            if (cudaMalloc(&d_dense_gate_up, dense_size * 2 * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_dense_activated, dense_size * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_dense_output, hidden * sizeof(float)) == cudaSuccess) {
                dense_capacity = req_capacity;
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
        if (d_moe_weights) cudaFree(d_moe_weights);
        if (d_moe_input) cudaFree(d_moe_input);
        if (d_dense_gate_up) cudaFree(d_dense_gate_up);
        if (d_dense_activated) cudaFree(d_dense_activated);
        if (d_dense_output) cudaFree(d_dense_output);
        if (stream) cudaStreamDestroy(stream);
        if (stream_b) cudaStreamDestroy(stream_b);
        for (int i = 0; i < 8; ++i) {
            if (streams[i]) cudaStreamDestroy(streams[i]);
        }
        if (norm_event) cudaEventDestroy(norm_event);
        if (dense_event) cudaEventDestroy(dense_event);
        if (exp_norm_event) cudaEventDestroy(exp_norm_event);
        if (act_event) cudaEventDestroy(act_event);
        for (int i = 0; i < 8; ++i) {
            if (expert_events[i]) cudaEventDestroy(expert_events[i]);
        }
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
    constexpr int kThreadsPerBlock = 64;
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

__global__ void gelu_activation_batch_kernel(const float* gate_up, float* activated, int expert_size, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = expert_size * num_experts;
    if (idx >= total_elements) return;
    
    int exp_idx = idx / expert_size;
    int local_idx = idx % expert_size;
    
    const float* exp_gate_up = gate_up + exp_idx * expert_size * 2;
    float* exp_activated = activated + exp_idx * expert_size;
    
    float g = exp_gate_up[local_idx];
    float u = exp_gate_up[expert_size + local_idx];
    float x = 0.79788456f * (g + 0.044715f * g * g * g);
    float tanh_x = tanhf(x);
    float gelu_g = 0.5f * g * (1.0f + tanh_x);
    exp_activated[local_idx] = gelu_g * u;
}

__global__ void accumulate_sum_batch_kernel(const float* expert_outputs, float* expert_sum, const float* weights, int hidden, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden) return;
    
    float sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        sum += weights[e] * expert_outputs[e * hidden + idx];
    }
    expert_sum[idx] += sum;
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

float* cuda_get_output_buf() {
    return get_workspace().d_output;
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

void cuda_ensure_moe_buffers(std::size_t hidden, std::size_t expert_size, std::size_t num_experts) {
    get_workspace().ensure_moe_buffers(hidden, expert_size, num_experts);
}

bool cuda_matvec_batch(const std::vector<CudaMatvecStep>& steps, const float* d_vector, std::size_t total_out_elems) {
    if (!is_cuda_available() || steps.empty()) return false;

    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();

    ws.ensure(0, 0, total_out_elems);
    if (!ws.d_output) return false;

    if (steps.size() == 1) {
        const auto& step = steps[0];
        std::size_t wbytes = weight_bytes_for(step.type, step.rows, step.cols);
        void* dev_matrix = nullptr;
        auto it = g_cuda_weights.find(step.matrix);
        if (it != g_cuda_weights.end()) {
            dev_matrix = it->second;
        } else {
            ws.ensure(wbytes, 0, 0);
            if (!ws.d_matrix) return false;
            if (cudaMemcpyAsync(ws.d_matrix, step.matrix, wbytes,
                                cudaMemcpyHostToDevice, ws.stream) != cudaSuccess)
                return false;
            dev_matrix = ws.d_matrix;
        }
        launch_kernel(step.type, dev_matrix, d_vector, ws.d_output + step.out_offset,
                      static_cast<int>(step.rows), static_cast<int>(step.cols), ws.stream);
    } else {
        std::vector<cudaStream_t> streams = {ws.stream, ws.stream_b, ws.streams[0]};
        for (std::size_t i = 0; i < steps.size(); ++i) {
            const auto& step = steps[i];
            cudaStream_t s = (i < streams.size()) ? streams[i] : ws.stream;
            void* dev_matrix = nullptr;
            auto it = g_cuda_weights.find(step.matrix);
            if (it != g_cuda_weights.end()) {
                dev_matrix = it->second;
            }
            if (dev_matrix) {
                launch_kernel(step.type, dev_matrix, d_vector, ws.d_output + step.out_offset,
                              static_cast<int>(step.rows), static_cast<int>(step.cols), s);
            }
        }
        
        // Sync helper streams back to ws.stream
        cudaEventRecord(ws.dense_event, ws.stream_b);
        cudaStreamWaitEvent(ws.stream, ws.dense_event, 0);
        
        cudaEventRecord(ws.act_event, ws.streams[0]);
        cudaStreamWaitEvent(ws.stream, ws.act_event, 0);
    }
    return true;
}

void cuda_moe_batch(const std::vector<CudaMatvecStep>& gate_steps,
                    const std::vector<CudaMatvecStep>& down_steps,
                    const float* host_weights,
                    float* host_expert_sum,
                    std::size_t hidden,
                    std::size_t expert_size,
                    std::size_t num_experts) {
    if (!is_cuda_available()) return;
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();

    ws.ensure_moe_buffers(hidden, expert_size, num_experts);
    if (!ws.d_moe_gate_up || !ws.d_moe_activated || !ws.d_moe_output || !ws.d_moe_sum || !ws.d_moe_weights) return;

    cuda_zero_buffer(ws.d_moe_sum, hidden);
    cudaMemcpyAsync(ws.d_moe_weights, host_weights, num_experts * sizeof(float), cudaMemcpyHostToDevice, ws.stream);

    for (std::size_t i = 0; i < num_experts; ++i) {
        auto step = gate_steps[i];
        std::size_t wbytes = weight_bytes_for(step.type, step.rows, step.cols);
        void* dev_matrix = nullptr;
        auto it = g_cuda_weights.find(step.matrix);
        if (it != g_cuda_weights.end()) {
            dev_matrix = it->second;
        } else {
            ws.ensure(wbytes, 0, 0);
            cudaMemcpyAsync(ws.d_matrix, step.matrix, wbytes, cudaMemcpyHostToDevice, ws.stream);
            dev_matrix = ws.d_matrix;
        }
        float* d_out = ws.d_moe_gate_up + i * expert_size * 2;
        launch_kernel(step.type, dev_matrix, ws.d_vector, d_out,
                      static_cast<int>(step.rows), static_cast<int>(step.cols), ws.stream);
    }

    int threads = 256;
    int total_elements = static_cast<int>(expert_size * num_experts);
    int blocks = (total_elements + threads - 1) / threads;
    gelu_activation_batch_kernel<<<blocks, threads, 0, ws.stream>>>(
        ws.d_moe_gate_up, ws.d_moe_activated, static_cast<int>(expert_size), static_cast<int>(num_experts));

    for (std::size_t i = 0; i < num_experts; ++i) {
        auto step = down_steps[i];
        std::size_t wbytes = weight_bytes_for(step.type, step.rows, step.cols);
        void* dev_matrix = nullptr;
        auto it = g_cuda_weights.find(step.matrix);
        if (it != g_cuda_weights.end()) {
            dev_matrix = it->second;
        } else {
            ws.ensure(wbytes, 0, 0);
            cudaMemcpyAsync(ws.d_matrix, step.matrix, wbytes, cudaMemcpyHostToDevice, ws.stream);
            dev_matrix = ws.d_matrix;
        }
        float* d_in = ws.d_moe_activated + i * expert_size;
        float* d_out = ws.d_moe_output + i * hidden;
        launch_kernel(step.type, dev_matrix, d_in, d_out,
                      static_cast<int>(step.rows), static_cast<int>(step.cols), ws.stream);
    }

    blocks = (static_cast<int>(hidden) + threads - 1) / threads;
    accumulate_sum_batch_kernel<<<blocks, threads, 0, ws.stream>>>(
        ws.d_moe_output, ws.d_moe_sum, ws.d_moe_weights, static_cast<int>(hidden), static_cast<int>(num_experts));

    std::size_t bytes = hidden * sizeof(float);
    if (ws.h_pinned_output && ws.pinned_output_capacity >= bytes) {
        cudaMemcpyAsync(ws.h_pinned_output, ws.d_moe_sum, bytes, cudaMemcpyDeviceToHost, ws.stream);
        cudaStreamSynchronize(ws.stream);
        memcpy(host_expert_sum, ws.h_pinned_output, bytes);
    } else {
        cudaMemcpyAsync(host_expert_sum, ws.d_moe_sum, bytes, cudaMemcpyDeviceToHost, ws.stream);
        cudaStreamSynchronize(ws.stream);
    }
}

__global__ void rms_norm_kernel(const float* input, const float* weight, float* output, float epsilon, int dim) {
    int idx = threadIdx.x;
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    for (int i = idx; i < dim; i += blockDim.x) {
        float val = input[i];
        sum += val * val;
    }
    
    sdata[idx] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }
    
    __shared__ float r_rms;
    if (idx == 0) {
        r_rms = rsqrtf(sdata[0] / dim + epsilon);
    }
    __syncthreads();
    
    for (int i = idx; i < dim; i += blockDim.x) {
        output[i] = input[i] * r_rms * (weight ? weight[i] : 1.0f);
    }
}

__global__ void add_vectors_kernel(const float* a, const float* b, float* out, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        out[idx] = a[idx] + b[idx];
    }
}

void cuda_ffn_execute(const float* host_ffn_input,
                      GgmlType dense_gate_type, const void* dense_gate_matrix,
                      GgmlType dense_up_type, const void* dense_up_matrix,
                      GgmlType dense_down_type, const void* dense_down_matrix,
                      const float* ffn_norm_weight,
                      const float* post_dense_norm_weight,
                      const float* pre_expert_norm_weight,
                      const float* post_expert_norm_weight,
                      const std::vector<CudaMatvecStep>& expert_gate_steps,
                      const std::vector<CudaMatvecStep>& expert_down_steps,
                      const float* host_expert_weights,
                      float* host_ffn_output,
                      std::size_t hidden,
                      std::size_t dense_size,
                      std::size_t expert_size,
                      std::size_t num_experts,
                      float epsilon) {
    if (!is_cuda_available()) return;
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto& ws = get_workspace();

    ws.ensure_dense_buffers(hidden, dense_size);
    ws.ensure_moe_buffers(hidden, expert_size, num_experts);

    // 1. Upload host_ffn_input to ws.d_vector
    cudaMemcpyAsync(ws.d_vector, host_ffn_input, hidden * sizeof(float), cudaMemcpyHostToDevice, ws.stream);

    // 2. Run ffn_norm (RMS Norm) on ws.d_vector -> ws.d_moe_input
    const float* d_ffn_norm_weight = nullptr;
    if (ffn_norm_weight) {
        auto it = g_cuda_weights.find(ffn_norm_weight);
        if (it != g_cuda_weights.end()) d_ffn_norm_weight = static_cast<const float*>(it->second);
    }
    
    rms_norm_kernel<<<1, 256, 256 * sizeof(float), ws.stream>>>(
        ws.d_vector, d_ffn_norm_weight, ws.d_moe_input, epsilon, static_cast<int>(hidden));

    // Sync stream to stream_b so Dense Up doesn't run before RMS Norm finishes
    cudaEventRecord(ws.norm_event, ws.stream);
    cudaStreamWaitEvent(ws.stream_b, ws.norm_event, 0);

    // 3. Dense Gate & Up Matvec (Executed concurrently on stream and stream_b)
    void* dev_matrix_gate = nullptr;
    {
        auto it = g_cuda_weights.find(dense_gate_matrix);
        if (it != g_cuda_weights.end()) {
            dev_matrix_gate = it->second;
        } else {
            std::size_t wbytes = weight_bytes_for(dense_gate_type, dense_size, hidden);
            ws.ensure(wbytes, 0, 0);
            cudaMemcpyAsync(ws.d_matrix, dense_gate_matrix, wbytes, cudaMemcpyHostToDevice, ws.stream);
            dev_matrix_gate = ws.d_matrix;
        }
        launch_kernel(dense_gate_type, dev_matrix_gate, ws.d_moe_input, ws.d_dense_gate_up,
                      static_cast<int>(dense_size), static_cast<int>(hidden), ws.stream);
    }
    void* dev_matrix_up = nullptr;
    {
        auto it = g_cuda_weights.find(dense_up_matrix);
        if (it != g_cuda_weights.end()) {
            dev_matrix_up = it->second;
        } else {
            std::size_t wbytes = weight_bytes_for(dense_up_type, dense_size, hidden);
            ws.ensure(wbytes, 0, 0);
            cudaMemcpyAsync(ws.d_matrix_b, dense_up_matrix, wbytes, cudaMemcpyHostToDevice, ws.stream_b);
            dev_matrix_up = ws.d_matrix_b;
        }
        launch_kernel(dense_up_type, dev_matrix_up, ws.d_moe_input, ws.d_dense_gate_up + dense_size,
                      static_cast<int>(dense_size), static_cast<int>(hidden), ws.stream_b);
    }

    // Sync stream_b to stream before activation
    cudaEventRecord(ws.dense_event, ws.stream_b);
    cudaStreamWaitEvent(ws.stream, ws.dense_event, 0);

    // 4. Dense GELU Activation
    int threads = 256;
    int blocks = (static_cast<int>(dense_size) + threads - 1) / threads;
    gelu_activation_kernel<<<blocks, threads, 0, ws.stream>>>(
        ws.d_dense_gate_up, ws.d_dense_activated, static_cast<int>(dense_size));

    // 5. Dense Down Step
    {
        void* dev_matrix = nullptr;
        auto it = g_cuda_weights.find(dense_down_matrix);
        if (it != g_cuda_weights.end()) {
            dev_matrix = it->second;
        } else {
            std::size_t wbytes = weight_bytes_for(dense_down_type, hidden, dense_size);
            ws.ensure(wbytes, 0, 0);
            cudaMemcpyAsync(ws.d_matrix, dense_down_matrix, wbytes, cudaMemcpyHostToDevice, ws.stream);
            dev_matrix = ws.d_matrix;
        }
        launch_kernel(dense_down_type, dev_matrix, ws.d_dense_activated, ws.d_dense_output,
                      static_cast<int>(hidden), static_cast<int>(dense_size), ws.stream);
    }

    // 6. Post Dense Norm (RMS Norm) on ws.d_dense_output -> ws.d_dense_output
    const float* d_post_dense_norm_weight = nullptr;
    if (post_dense_norm_weight) {
        auto it = g_cuda_weights.find(post_dense_norm_weight);
        if (it != g_cuda_weights.end()) d_post_dense_norm_weight = static_cast<const float*>(it->second);
    }
    rms_norm_kernel<<<1, 256, 256 * sizeof(float), ws.stream>>>(
        ws.d_dense_output, d_post_dense_norm_weight, ws.d_dense_output, epsilon, static_cast<int>(hidden));

    // ========================================================================
    // MoE Expert Path (Executed concurrently with the dense path)
    // ========================================================================
    if (num_experts > 0) {
        // 7. Pre Expert Norm (RMS Norm) on ws.d_vector (host_ffn_input) -> ws.d_moe_input
        const float* d_pre_expert_norm_weight = nullptr;
        if (pre_expert_norm_weight) {
            auto it = g_cuda_weights.find(pre_expert_norm_weight);
            if (it != g_cuda_weights.end()) d_pre_expert_norm_weight = static_cast<const float*>(it->second);
        }
        rms_norm_kernel<<<1, 256, 256 * sizeof(float), ws.stream>>>(
            ws.d_vector, d_pre_expert_norm_weight, ws.d_moe_input, epsilon, static_cast<int>(hidden));

        // Sync stream to all expert streams so expert matvecs don't run before Pre Expert Norm finishes
        cudaEventRecord(ws.exp_norm_event, ws.stream);
        for (std::size_t i = 0; i < num_experts; ++i) {
            cudaStreamWaitEvent(ws.streams[i], ws.exp_norm_event, 0);
        }

        // 8. Expert Gate & Up steps (Executed concurrently in streams[0..7])
        cuda_zero_buffer(ws.d_moe_sum, hidden);
        cudaMemcpyAsync(ws.d_moe_weights, host_expert_weights, num_experts * sizeof(float), cudaMemcpyHostToDevice, ws.stream);

        for (std::size_t i = 0; i < num_experts; ++i) {
            auto step = expert_gate_steps[i];
            std::size_t wbytes = weight_bytes_for(step.type, step.rows, step.cols);
            void* dev_matrix = nullptr;
            auto it = g_cuda_weights.find(step.matrix);
            if (it != g_cuda_weights.end()) {
                dev_matrix = it->second;
            } else {
                ws.ensure(wbytes, 0, 0);
                cudaMemcpyAsync(ws.d_matrix, step.matrix, wbytes, cudaMemcpyHostToDevice, ws.streams[i]);
                dev_matrix = ws.d_matrix;
            }
            float* d_out = ws.d_moe_gate_up + i * expert_size * 2;
            launch_kernel(step.type, dev_matrix, ws.d_moe_input, d_out,
                          static_cast<int>(step.rows), static_cast<int>(step.cols), ws.streams[i]);
        }

        // Sync all expert streams back to ws.stream before activation
        for (std::size_t i = 0; i < num_experts; ++i) {
            cudaEventRecord(ws.expert_events[i], ws.streams[i]);
            cudaStreamWaitEvent(ws.stream, ws.expert_events[i], 0);
        }

        // 9. Expert GELU Activation
        int total_elements = static_cast<int>(expert_size * num_experts);
        blocks = (total_elements + threads - 1) / threads;
        gelu_activation_batch_kernel<<<blocks, threads, 0, ws.stream>>>(
            ws.d_moe_gate_up, ws.d_moe_activated, static_cast<int>(expert_size), static_cast<int>(num_experts));

        // Sync stream to all expert streams so expert down matvecs don't run before activation finishes
        cudaEventRecord(ws.act_event, ws.stream);
        for (std::size_t i = 0; i < num_experts; ++i) {
            cudaStreamWaitEvent(ws.streams[i], ws.act_event, 0);
        }

        // 10. Expert Down steps (Executed concurrently in streams[0..7])
        for (std::size_t i = 0; i < num_experts; ++i) {
            auto step = expert_down_steps[i];
            std::size_t wbytes = weight_bytes_for(step.type, step.rows, step.cols);
            void* dev_matrix = nullptr;
            auto it = g_cuda_weights.find(step.matrix);
            if (it != g_cuda_weights.end()) {
                dev_matrix = it->second;
            } else {
                ws.ensure(wbytes, 0, 0);
                cudaMemcpyAsync(ws.d_matrix, step.matrix, wbytes, cudaMemcpyHostToDevice, ws.streams[i]);
                dev_matrix = ws.d_matrix;
            }
            float* d_in = ws.d_moe_activated + i * expert_size;
            float* d_out = ws.d_moe_output + i * hidden;
            launch_kernel(step.type, dev_matrix, d_in, d_out,
                          static_cast<int>(step.rows), static_cast<int>(step.cols), ws.streams[i]);
        }

        // Sync all expert streams back to ws.stream before accumulate sum
        for (std::size_t i = 0; i < num_experts; ++i) {
            cudaEventRecord(ws.expert_events[i], ws.streams[i]);
            cudaStreamWaitEvent(ws.stream, ws.expert_events[i], 0);
        }

        // 11. Expert Accumulate Sum
        blocks = (static_cast<int>(hidden) + threads - 1) / threads;
        accumulate_sum_batch_kernel<<<blocks, threads, 0, ws.stream>>>(
            ws.d_moe_output, ws.d_moe_sum, ws.d_moe_weights, static_cast<int>(hidden), static_cast<int>(num_experts));

        // 11.5 Post Expert Norm (RMS Norm) on ws.d_moe_sum -> ws.d_moe_sum
        const float* d_post_expert_norm_weight = nullptr;
        if (post_expert_norm_weight) {
            auto it = g_cuda_weights.find(post_expert_norm_weight);
            if (it != g_cuda_weights.end()) d_post_expert_norm_weight = static_cast<const float*>(it->second);
        }
        rms_norm_kernel<<<1, 256, 256 * sizeof(float), ws.stream>>>(
            ws.d_moe_sum, d_post_expert_norm_weight, ws.d_moe_sum, epsilon, static_cast<int>(hidden));

        // 12. Combine Dense Output and MoE Sum
        add_vectors_kernel<<<blocks, threads, 0, ws.stream>>>(
            ws.d_dense_output, ws.d_moe_sum, ws.d_dense_output, static_cast<int>(hidden));
    }

    // 13. Download output and sync once
    std::size_t bytes = hidden * sizeof(float);
    if (ws.h_pinned_output && ws.pinned_output_capacity >= bytes) {
        cudaMemcpyAsync(ws.h_pinned_output, ws.d_dense_output, bytes, cudaMemcpyDeviceToHost, ws.stream);
        cudaStreamSynchronize(ws.stream);
        memcpy(host_ffn_output, ws.h_pinned_output, bytes);
    } else {
        cudaMemcpyAsync(host_ffn_output, ws.d_dense_output, bytes, cudaMemcpyDeviceToHost, ws.stream);
        cudaStreamSynchronize(ws.stream);
    }
}

} // namespace gemmaedge
