#pragma once

#include "gemmaedge/gguf.h"
#include <cstddef>
#include <cstdint>

namespace gemmaedge {

// Checks if CUDA GPU device is available and initialized.
bool is_cuda_available();

// Performs matrix-vector multiplication on the GPU via CUDA.
// Supports F32, Q8_0, and Q4_K formats.
// Returns true if successfully executed, false if fallback to CPU is required.
bool cuda_matvec(GgmlType type, const void* matrix, std::size_t rows, std::size_t cols,
                 const float* vector, float* output);

} // namespace gemmaedge
