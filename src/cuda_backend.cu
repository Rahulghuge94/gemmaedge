#include "gemmaedge/cuda_backend.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <unordered_map>
#include <mutex>
#include <iostream>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

namespace gemmaedge {
namespace {

// Half-precision float conversion on device
__device__ inline float cuda_f16_to_f32(uint16_t h) {
    uint32_t w = (h & 0x7FFF) << 13;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = w & 0x7F800000;
    if (exp == 0x7F800000) {
        w += 0x38000000;
    } else if (exp != 0) {
        w += 0x38000000;
    } else if (w != 0) {
        w += 0x38000000;
        uint32_t e = w & 0x7F800000;
        e -= 0x00800000;
        w &= ~0x7F800000;
        w |= e;
    }
    uint32_t result = sign | w;
    return *(float*)&result;
}

// 1. F32 Matvec Kernel
__global__ void matvec_f32_kernel(const float* __restrict__ matrix,
                                  const float* __restrict__ vec,
                                  float* __restrict__ out,
                                  int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    double sum = 0.0;
    int offset = row * cols;
    for (int col = 0; col < cols; ++col) {
        sum += (double)matrix[offset + col] * (double)vec[col];
    }
    out[row] = (float)sum;
}

// 1.5 Q4_0 Matvec Kernel
struct BlockQ4_0 {
    uint16_t d;
    uint8_t qs[16];
};

__global__ void matvec_q4_0_kernel(const BlockQ4_0* __restrict__ matrix,
                                    const float* __restrict__ vec,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    double sum = 0.0;
    int blocks_per_row = cols / 32;
    int offset = row * blocks_per_row;
    
    for (int b = 0; b < blocks_per_row; ++b) {
        BlockQ4_0 block = matrix[offset + b];
        float scale = cuda_f16_to_f32(block.d);
        int v_offset = b * 32;
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = block.qs[i];
            sum += (double)scale * (double)((int)(packed & 0x0f) - 8) * (double)vec[v_offset + i];
            sum += (double)scale * (double)((int)(packed >> 4) - 8) * (double)vec[v_offset + 16 + i];
        }
    }
    out[row] = (float)sum;
}

// 2. Q8_0 Matvec Kernel
struct BlockQ8_0 {
    uint16_t d;
    int8_t q[32];
};

__global__ void matvec_q8_0_kernel(const BlockQ8_0* __restrict__ matrix,
                                    const float* __restrict__ vec,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    double sum = 0.0;
    int blocks_per_row = cols / 32;
    int offset = row * blocks_per_row;
    
    for (int b = 0; b < blocks_per_row; ++b) {
        BlockQ8_0 block = matrix[offset + b];
        float scale = cuda_f16_to_f32(block.d);
        int v_offset = b * 32;
        for (int i = 0; i < 32; ++i) {
            sum += (double)scale * (double)block.q[i] * (double)vec[v_offset + i];
        }
    }
    out[row] = (float)sum;
}

// 3. Q4_K Matvec Kernel
struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t q[128];
};

__device__ inline void cuda_q4k_scale_min(int index, const uint8_t* packed,
                                          uint8_t& scale, uint8_t& minimum) {
    if (index < 4) {
        scale = packed[index] & 63;
        minimum = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 0x0f) |
                ((packed[index - 4] >> 6) << 4);
        minimum = (packed[index + 4] >> 4) |
                  ((packed[index] >> 6) << 4);
    }
}

__global__ void matvec_q4_k_kernel(const BlockQ4K* __restrict__ matrix,
                                    const float* __restrict__ vec,
                                    float* __restrict__ out,
                                    int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    double sum = 0.0;
    int blocks_per_row = cols / 256;
    int offset = row * blocks_per_row;
    
    for (int b = 0; b < blocks_per_row; ++b) {
        BlockQ4K block = matrix[offset + b];
        float d = cuda_f16_to_f32(block.d);
        float dmin = cuda_f16_to_f32(block.dmin);
        int group = 0;
        const uint8_t* q = block.q;
        int v_offset = b * 256;
        
        for (int base = 0; base < 256; base += 64) {
            uint8_t scale1, min1, scale2, min2;
            cuda_q4k_scale_min(group++, block.scales, scale1, min1);
            cuda_q4k_scale_min(group++, block.scales, scale2, min2);
            
            float factor1 = d * scale1;
            float bias1 = -dmin * min1;
            float factor2 = d * scale2;
            float bias2 = -dmin * min2;
            
            for (int i = 0; i < 32; ++i) {
                sum += (double)(factor1 * (q[i] & 0x0f) + bias1) * (double)vec[v_offset + base + i];
            }
            for (int i = 0; i < 32; ++i) {
                sum += (double)(factor2 * (q[i] >> 4) + bias2) * (double)vec[v_offset + base + 32 + i];
            }
            q += 32;
        }
    }
    out[row] = (float)sum;
}

// Global cache for GPU weights to prevent re-copying large layers
static std::unordered_map<const void*, void*> g_gpu_weights;
static std::mutex g_gpu_mutex;

void* get_gpu_matrix(const void* host_ptr, std::size_t bytes) {
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    auto it = g_gpu_weights.find(host_ptr);
    if (it != g_gpu_weights.end()) {
        return it->second;
    }
    void* dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, bytes) == cudaSuccess) {
        if (cudaMemcpy(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice) == cudaSuccess) {
#if defined(__linux__) || defined(__APPLE__)
            uintptr_t page_size = 4096;
            uintptr_t addr = reinterpret_cast<uintptr_t>(host_ptr);
            uintptr_t aligned_addr = addr & ~(page_size - 1);
            uintptr_t aligned_size = (addr + bytes - aligned_addr + page_size - 1) & ~(page_size - 1);
            madvise(reinterpret_cast<void*>(aligned_addr), aligned_size, MADV_DONTNEED);
#endif
            g_gpu_weights[host_ptr] = dev_ptr;
            return dev_ptr;
        }
        cudaFree(dev_ptr);
    }
    return nullptr;
}

} // namespace

bool is_cuda_available() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return false;
    }
    return device_count > 0;
}

bool cuda_matvec(GgmlType type, const void* matrix, std::size_t rows, std::size_t cols,
                 const float* vector, float* output) {
    if (!is_cuda_available()) return false;

    // Determine weight size in bytes
    std::size_t weight_bytes = 0;
    if (type == GgmlType::F32) {
        weight_bytes = rows * cols * sizeof(float);
    } else if (type == GgmlType::Q4_0) {
        weight_bytes = (rows * cols / 32) * sizeof(BlockQ4_0);
    } else if (type == GgmlType::Q8_0) {
        weight_bytes = (rows * cols / 32) * sizeof(BlockQ8_0);
    } else if (type == GgmlType::Q4_K) {
        weight_bytes = (rows * cols / 256) * sizeof(BlockQ4K);
    } else {
        return false; // Fallback to CPU for unsupported types
    }

    // Get or allocate/copy weight matrix in GPU memory
    void* d_matrix = get_gpu_matrix(matrix, weight_bytes);
    if (!d_matrix) return false;

    // Allocate vector and output buffers on GPU
    float* d_vector = nullptr;
    float* d_output = nullptr;
    
    if (cudaMalloc(&d_vector, cols * sizeof(float)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&d_output, rows * sizeof(float)) != cudaSuccess) {
        cudaFree(d_vector);
        return false;
    }

    // Copy input vector to GPU
    if (cudaMemcpy(d_vector, vector, cols * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_vector);
        cudaFree(d_output);
        return false;
    }

    // Launch CUDA kernel
    int threads_per_block = 256;
    int blocks = (static_cast<int>(rows) + threads_per_block - 1) / threads_per_block;

    if (type == GgmlType::F32) {
        matvec_f32_kernel<<<blocks, threads_per_block>>>(
            static_cast<const float*>(d_matrix), d_vector, d_output, static_cast<int>(rows), static_cast<int>(cols));
    } else if (type == GgmlType::Q4_0) {
        matvec_q4_0_kernel<<<blocks, threads_per_block>>>(
            static_cast<const BlockQ4_0*>(d_matrix), d_vector, d_output, static_cast<int>(rows), static_cast<int>(cols));
    } else if (type == GgmlType::Q8_0) {
        matvec_q8_0_kernel<<<blocks, threads_per_block>>>(
            static_cast<const BlockQ8_0*>(d_matrix), d_vector, d_output, static_cast<int>(rows), static_cast<int>(cols));
    } else if (type == GgmlType::Q4_K) {
        matvec_q4_k_kernel<<<blocks, threads_per_block>>>(
            static_cast<const BlockQ4K*>(d_matrix), d_vector, d_output, static_cast<int>(rows), static_cast<int>(cols));
    }

    cudaDeviceSynchronize();

    // Copy output back to host
    bool success = (cudaMemcpy(output, d_output, rows * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaFree(d_vector);
    cudaFree(d_output);

    return success;
}

} // namespace gemmaedge
