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

} // namespace gemmaedge
