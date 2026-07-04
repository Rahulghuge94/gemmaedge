#pragma once

#include "gemmaedge/gguf.h"
#include <cstddef>
#include <cstdint>

namespace gemmaedge {

// Checks if CUDA GPU device is available and initialized.
bool is_cuda_available();

// Performs matrix-vector multiplication on the GPU via CUDA.
// Supports F32, Q4_0, Q8_0, Q4_K, and Q6_K formats.
// Returns true if successfully executed, false if fallback to CPU is required.
bool cuda_matvec(GgmlType type, const void* matrix, std::size_t rows, std::size_t cols,
                 const float* vector, float* output);

// Registers a weight matrix on the GPU, allocating memory and copying its contents.
bool cuda_register_weight(const void* host_ptr, std::size_t bytes);

// Unregisters a weight matrix, freeing its GPU memory.
void cuda_unregister_weight(const void* host_ptr);

// Clears all registered weights from GPU memory.
void cuda_clear_weights();

// --- Persistent device vector API ---
// Upload a float vector from host to a persistent device buffer.
// Subsequent cuda_matvec_device_vec calls use this buffer as the input vector.
bool cuda_upload_vector(const float* host, std::size_t count);

// Download the persistent device output buffer to host memory.
bool cuda_download_vector(float* host, std::size_t count);

// Matrix-vector multiply using the already-uploaded device-resident input vector.
// Writes result to the persistent device output buffer (retrievable via cuda_download_vector).
// This avoids per-call H->D vector upload and D->H output download.
bool cuda_matvec_device_vec(GgmlType type, const void* matrix,
                            std::size_t rows, std::size_t cols, float* host_output);

// --- MoE End-to-End GPU Execution API ---
bool cuda_matvec_d2d(GgmlType type, const void* matrix, const float* d_vector,
                     float* d_output, std::size_t rows, std::size_t cols);
void cuda_moe_gelu(const float* d_gate_up, float* d_activated, std::size_t expert_size);
void cuda_moe_accumulate(const float* d_expert_output, float* d_expert_sum, float weight, std::size_t hidden);
void cuda_zero_buffer(float* d_buf, std::size_t count);
bool cuda_download_vector_from(float* host, const float* d_buf, std::size_t count);

float* cuda_get_vector_buf();
float* cuda_get_moe_gate_up_buf();
float* cuda_get_moe_activated_buf();
float* cuda_get_moe_output_buf();
float* cuda_get_moe_sum_buf();
void cuda_ensure_moe_buffers(std::size_t hidden, std::size_t expert_size);

} // namespace gemmaedge
