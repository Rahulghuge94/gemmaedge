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
struct BlockQ8_0 {
    std::uint16_t d; // scale
    std::int8_t q[32]; // quants
};
#pragma pack(pop)

static_assert(sizeof(Q4Block) == 18, "Q4 block layout must remain stable");
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block layout must remain stable");

#if defined(__ARM_NEON)
#include <arm_neon.h>

inline float dot_q8_block_neon(const BlockQ8_0& block, const float* x) {
    const float d = f16_to_f32(block.d);
    int8x16_t q0 = vld1q_s8(block.q);
    int8x16_t q1 = vld1q_s8(block.q + 16);

    int16x8_t q0_lo = vmovl_s8(vget_low_s8(q0));
    int16x8_t q0_hi = vmovl_s8(vget_high_s8(q0));
    int16x8_t q1_lo = vmovl_s8(vget_low_s8(q1));
    int16x8_t q1_hi = vmovl_s8(vget_high_s8(q1));

    float32x4_t q0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_lo)));
    float32x4_t q0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_lo)));
    float32x4_t q0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_hi)));
    float32x4_t q0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_hi)));

    float32x4_t q1_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_lo)));
    float32x4_t q1_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_lo)));
    float32x4_t q1_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_hi)));
    float32x4_t q1_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_hi)));

    float32x4_t x0 = vld1q_f32(x);
    float32x4_t x1 = vld1q_f32(x + 4);
    float32x4_t x2 = vld1q_f32(x + 8);
    float32x4_t x3 = vld1q_f32(x + 12);
    float32x4_t x4 = vld1q_f32(x + 16);
    float32x4_t x5 = vld1q_f32(x + 20);
    float32x4_t x6 = vld1q_f32(x + 24);
    float32x4_t x7 = vld1q_f32(x + 28);

    float32x4_t sum = vmulq_f32(q0_0, x0);
    sum = vmlaq_f32(sum, q0_1, x1);
    sum = vmlaq_f32(sum, q0_2, x2);
    sum = vmlaq_f32(sum, q0_3, x3);
    sum = vmlaq_f32(sum, q1_0, x4);
    sum = vmlaq_f32(sum, q1_1, x5);
    sum = vmlaq_f32(sum, q1_2, x6);
    sum = vmlaq_f32(sum, q1_3, x7);

    float total = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                  vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
    return total * d;
}

inline void accumulate_q8_block_neon(const BlockQ8_0& block, float scale_prob, float* dest) {
    float32x4_t scale_prob_vec = vdupq_n_f32(scale_prob);
    int8x16_t q0 = vld1q_s8(block.q);
    int8x16_t q1 = vld1q_s8(block.q + 16);

    int16x8_t q0_lo = vmovl_s8(vget_low_s8(q0));
    int16x8_t q0_hi = vmovl_s8(vget_high_s8(q0));
    int16x8_t q1_lo = vmovl_s8(vget_low_s8(q1));
    int16x8_t q1_hi = vmovl_s8(vget_high_s8(q1));

    float32x4_t q0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_lo)));
    float32x4_t q0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_lo)));
    float32x4_t q0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_hi)));
    float32x4_t q0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_hi)));
    float32x4_t q1_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_lo)));
    float32x4_t q1_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_lo)));
    float32x4_t q1_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_hi)));
    float32x4_t q1_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_hi)));

    vst1q_f32(dest,      vmlaq_f32(vld1q_f32(dest),      q0_0, scale_prob_vec));
    vst1q_f32(dest + 4,  vmlaq_f32(vld1q_f32(dest + 4),  q0_1, scale_prob_vec));
    vst1q_f32(dest + 8,  vmlaq_f32(vld1q_f32(dest + 8),  q0_2, scale_prob_vec));
    vst1q_f32(dest + 12, vmlaq_f32(vld1q_f32(dest + 12), q0_3, scale_prob_vec));
    vst1q_f32(dest + 16, vmlaq_f32(vld1q_f32(dest + 16), q1_0, scale_prob_vec));
    vst1q_f32(dest + 20, vmlaq_f32(vld1q_f32(dest + 20), q1_1, scale_prob_vec));
    vst1q_f32(dest + 24, vmlaq_f32(vld1q_f32(dest + 24), q1_2, scale_prob_vec));
    vst1q_f32(dest + 28, vmlaq_f32(vld1q_f32(dest + 28), q1_3, scale_prob_vec));
}
#endif

float f16_to_f32(std::uint16_t value) noexcept;
std::uint16_t f32_to_f16(float value) noexcept;
float bf16_to_f32(std::uint16_t value) noexcept;
std::uint16_t f32_to_bf16(float value) noexcept;

Q4Block quantize_q4_block(const float* values);
void dequantize_q4_block(const Q4Block& block, float* output);
BlockQ8_0 quantize_q8_block(const float* values);
void dequantize_q8_block(const BlockQ8_0& block, float* output);

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
