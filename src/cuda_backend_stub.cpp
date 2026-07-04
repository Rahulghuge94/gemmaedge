#include "gemmaedge/cuda_backend.h"

namespace gemmaedge {

bool is_cuda_available() {
    return false;
}

bool cuda_matvec(GgmlType type, const void* matrix, std::size_t rows, std::size_t cols,
                 const float* vector, float* output) {
    (void)type; (void)matrix; (void)rows; (void)cols; (void)vector; (void)output;
    return false;
}

bool cuda_matvec_device_vec(GgmlType type, const void* matrix,
                            std::size_t rows, std::size_t cols, float* host_output) {
    (void)type; (void)matrix; (void)rows; (void)cols; (void)host_output;
    return false;
}

bool cuda_upload_vector(const float* host, std::size_t count) {
    (void)host; (void)count;
    return false;
}

bool cuda_download_vector(float* host, std::size_t count) {
    (void)host; (void)count;
    return false;
}

bool cuda_register_weight(const void* host_ptr, std::size_t bytes) {
    (void)host_ptr; (void)bytes;
    return false;
}

void cuda_unregister_weight(const void* host_ptr) {
    (void)host_ptr;
}

void cuda_clear_weights() {
}

bool cuda_matvec_batch(const std::vector<CudaMatvecStep>& steps, const float* d_vector, std::size_t total_out_elems) {
    (void)steps; (void)d_vector; (void)total_out_elems;
    return false;
}

void cuda_moe_batch(const std::vector<CudaMatvecStep>& gate_steps,
                    const std::vector<CudaMatvecStep>& down_steps,
                    const float* host_weights,
                    float* host_expert_sum,
                    std::size_t hidden,
                    std::size_t expert_size,
                    std::size_t num_experts) {
    (void)gate_steps; (void)down_steps; (void)host_weights; (void)host_expert_sum;
    (void)hidden; (void)expert_size; (void)num_experts;
}

void cuda_ensure_moe_buffers(std::size_t hidden, std::size_t expert_size, std::size_t num_experts) {
    (void)hidden; (void)expert_size; (void)num_experts;
}

float* cuda_get_output_buf() {
    return nullptr;
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
    (void)host_ffn_input;
    (void)dense_gate_type; (void)dense_gate_matrix;
    (void)dense_up_type; (void)dense_up_matrix;
    (void)dense_down_type; (void)dense_down_matrix;
    (void)ffn_norm_weight;
    (void)post_dense_norm_weight;
    (void)pre_expert_norm_weight;
    (void)post_expert_norm_weight;
    (void)expert_gate_steps;
    (void)expert_down_steps;
    (void)host_expert_weights;
    (void)host_ffn_output;
    (void)hidden;
    (void)dense_size;
    (void)expert_size;
    (void)num_experts;
    (void)epsilon;
}

} // namespace gemmaedge
