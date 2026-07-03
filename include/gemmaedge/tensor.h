#pragma once

#include "gemmaedge/gguf.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gemmaedge {

constexpr std::size_t kQ4BlockSize = 32;

#pragma pack(push, 1)
struct Q4Block {
    std::uint16_t scale_f16;
    std::uint8_t quants[kQ4BlockSize / 2];
};
#pragma pack(pop)

static_assert(sizeof(Q4Block) == 18, "Q4 block layout must remain stable");

float f16_to_f32(std::uint16_t value) noexcept;
std::uint16_t f32_to_f16(float value) noexcept;
float bf16_to_f32(std::uint16_t value) noexcept;
std::uint16_t f32_to_bf16(float value) noexcept;

Q4Block quantize_q4_block(const float* values);
void dequantize_q4_block(const Q4Block& block, float* output);

// Matrix is row-major and each row is independently Q4-blocked. cols must be
// divisible by kQ4BlockSize. The result contains `rows` float values.
void q4_matvec(const Q4Block* matrix, std::size_t rows, std::size_t cols,
               const float* vector, float* output);

// Reference GGML-compatible matrix-vector path. Matrix dimensions follow GGUF
// order: dimensions[0] is input width and dimensions[1] is output rows.
void ggml_matvec(GgmlType type, const void* matrix, std::size_t rows,
                 std::size_t cols, const float* vector, float* output);
void ggml_dequantize_row(GgmlType type, const void* row, std::size_t cols,
                         float* output);

void rms_norm(const float* input, const float* weight, std::size_t size,
              float epsilon, float* output);
void softmax(float* values, std::size_t size);
std::vector<std::pair<std::uint32_t, float>>
top_k(const float* values, std::size_t size, std::size_t k);
float gelu_tanh(float value) noexcept;

} // namespace gemmaedge
