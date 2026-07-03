#pragma once

#include <cstddef>
#include <cstdint>

namespace gemmaedge {

// Checks if Vulkan device is initialized and available for compute acceleration.
bool is_vulkan_available();

// Attempts to execute a matrix-vector multiplication on the Vulkan GPU device.
// Returns true if successfully executed, false if fallback to CPU is required.
bool vulkan_matvec(const void* matrix, std::size_t rows, std::size_t cols,
                   const float* vector, float* output);

} // namespace gemmaedge
