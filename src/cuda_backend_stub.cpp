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

} // namespace gemmaedge
