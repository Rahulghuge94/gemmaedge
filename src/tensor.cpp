#include "gemmaedge/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gemmaedge {

float f16_to_f32(std::uint16_t value) noexcept {
    const std::uint32_t sign = (value & 0x8000u) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x3ffu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t f32_to_f16(float value) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t normalized = (mantissa | 0x800000u) >>
                                         static_cast<unsigned>(1 - exponent);
        return static_cast<std::uint16_t>(
            sign | ((normalized + 0x1000u) >> 13));
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(
            sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10) |
        ((mantissa + 0x1000u) >> 13));
}

float bf16_to_f32(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t f32_to_bf16(float value) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round-to-nearest-even before truncation.
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

Q4Block quantize_q4_block(const float* values) {
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < kQ4BlockSize; ++i)
        max_abs = std::max(max_abs, std::abs(values[i]));
    const float scale = max_abs == 0.0f ? 0.0f : max_abs / 7.0f;
    Q4Block block{};
    block.scale_f16 = f32_to_f16(scale);
    for (std::size_t i = 0; i < kQ4BlockSize / 2; ++i) {
        const auto quant = [scale](float x) {
            if (scale == 0.0f) return 0;
            return std::max(-8, std::min(7,
                static_cast<int>(std::nearbyint(x / scale))));
        };
        const int low = quant(values[i]) + 8;
        const int high = quant(values[i + kQ4BlockSize / 2]) + 8;
        block.quants[i] =
            static_cast<std::uint8_t>(low | (high << 4));
    }
    return block;
}

void dequantize_q4_block(const Q4Block& block, float* output) {
    const float scale = f16_to_f32(block.scale_f16);
    for (std::size_t i = 0; i < kQ4BlockSize / 2; ++i) {
        const auto packed = block.quants[i];
        output[i] = (static_cast<int>(packed & 0xfu) - 8) * scale;
        output[i + kQ4BlockSize / 2] =
            (static_cast<int>(packed >> 4) - 8) * scale;
    }
}

void q4_matvec(const Q4Block* matrix, std::size_t rows, std::size_t cols,
               const float* vector, float* output) {
    if (cols % kQ4BlockSize != 0)
        throw std::invalid_argument("Q4 matvec columns must be block aligned");
    const std::size_t blocks_per_row = cols / kQ4BlockSize;
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const auto& block = matrix[row * blocks_per_row + b];
            const float scale = f16_to_f32(block.scale_f16);
            const float* x = vector + b * kQ4BlockSize;
            for (std::size_t i = 0; i < kQ4BlockSize / 2; ++i) {
                const auto packed = block.quants[i];
                sum += (static_cast<int>(packed & 0xfu) - 8) * scale * x[i];
                sum += (static_cast<int>(packed >> 4) - 8) * scale *
                       x[i + kQ4BlockSize / 2];
            }
        }
        output[row] = sum;
    }
}

#pragma pack(push, 1)
struct BlockQ8_0 {
    std::uint16_t d;
    std::int8_t q[32];
};
struct BlockQ5_0 {
    std::uint16_t d;
    std::uint8_t qh[4];
    std::uint8_t q[16];
};
struct BlockQ4K {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t q[128];
};
struct BlockQ6K {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    std::uint16_t d;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ8_0) == 34, "GGML Q8_0 layout mismatch");
static_assert(sizeof(BlockQ5_0) == 22, "GGML Q5_0 layout mismatch");
static_assert(sizeof(BlockQ4K) == 144, "GGML Q4_K layout mismatch");
static_assert(sizeof(BlockQ6K) == 210, "GGML Q6_K layout mismatch");

namespace {

void q4k_scale_min(int index, const std::uint8_t* packed,
                   std::uint8_t& scale, std::uint8_t& minimum) {
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

float dot_q4k(const BlockQ4K& block, const float* x) {
    const float d = f16_to_f32(block.d);
    const float dmin = f16_to_f32(block.dmin);
    float sum = 0.0f;
    int group = 0;
    const std::uint8_t* q = block.q;
    for (int base = 0; base < 256; base += 64) {
        std::uint8_t scale1, min1, scale2, min2;
        q4k_scale_min(group++, block.scales, scale1, min1);
        q4k_scale_min(group++, block.scales, scale2, min2);
        for (int i = 0; i < 32; ++i)
            sum += (d * scale1 * (q[i] & 0x0f) - dmin * min1) *
                   x[base + i];
        for (int i = 0; i < 32; ++i)
            sum += (d * scale2 * (q[i] >> 4) - dmin * min2) *
                   x[base + 32 + i];
        q += 32;
    }
    return sum;
}

float dot_q6k(const BlockQ6K& block, const float* x) {
    const float d = f16_to_f32(block.d);
    float sum = 0.0f;
    const std::uint8_t* ql = block.ql;
    const std::uint8_t* qh = block.qh;
    const std::int8_t* scales = block.scales;
    for (int base = 0; base < 256; base += 128) {
        for (int i = 0; i < 32; ++i) {
            const int group = i / 16;
            const int q1 = ((ql[i] & 0x0f) | (((qh[i] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[i + 32] & 0x0f) | (((qh[i] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[i] >> 4) | (((qh[i] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[i + 32] >> 4) | (((qh[i] >> 6) & 3) << 4)) - 32;
            sum += d * scales[group + 0] * q1 * x[base + i];
            sum += d * scales[group + 2] * q2 * x[base + 32 + i];
            sum += d * scales[group + 4] * q3 * x[base + 64 + i];
            sum += d * scales[group + 6] * q4 * x[base + 96 + i];
        }
        ql += 64;
        qh += 32;
        scales += 8;
    }
    return sum;
}

} // namespace

void ggml_matvec(GgmlType type, const void* matrix, std::size_t rows,
                 std::size_t cols, const float* vector, float* output) {
    if (!matrix || !vector || !output)
        throw std::invalid_argument("null GGML matvec input");
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        switch (type) {
            case GgmlType::F32: {
                const auto* weights = static_cast<const float*>(matrix) + row * cols;
                for (std::size_t i = 0; i < cols; ++i) sum += weights[i] * vector[i];
                break;
            }
            case GgmlType::F16:
            case GgmlType::BF16: {
                const auto* weights = static_cast<const std::uint16_t*>(matrix) + row * cols;
                for (std::size_t i = 0; i < cols; ++i) {
                    const float weight = type == GgmlType::F16
                        ? f16_to_f32(weights[i]) : bf16_to_f32(weights[i]);
                    sum += weight * vector[i];
                }
                break;
            }
            case GgmlType::Q4_0: {
                if (cols % 32) throw std::invalid_argument("unaligned Q4_0 row");
                const auto* blocks = static_cast<const Q4Block*>(matrix) +
                    row * (cols / 32);
                q4_matvec(blocks, 1, cols, vector, &sum);
                break;
            }
            case GgmlType::Q8_0: {
                if (cols % 32) throw std::invalid_argument("unaligned Q8_0 row");
                const auto* blocks = static_cast<const BlockQ8_0*>(matrix) +
                    row * (cols / 32);
                for (std::size_t b = 0; b < cols / 32; ++b) {
                    const float d = f16_to_f32(blocks[b].d);
                    for (std::size_t i = 0; i < 32; ++i)
                        sum += d * blocks[b].q[i] * vector[b * 32 + i];
                }
                break;
            }
            case GgmlType::Q5_0: {
                if (cols % 32) throw std::invalid_argument("unaligned Q5_0 row");
                const auto* blocks = static_cast<const BlockQ5_0*>(matrix) +
                    row * (cols / 32);
                for (std::size_t b = 0; b < cols / 32; ++b) {
                    const float d = f16_to_f32(blocks[b].d);
                    std::uint32_t high = 0;
                    std::memcpy(&high, blocks[b].qh, sizeof(high));
                    for (std::size_t i = 0; i < 16; ++i) {
                        const int hi0 = ((high >> i) << 4) & 0x10;
                        const int hi1 = (high >> (i + 12)) & 0x10;
                        const int q0 = ((blocks[b].q[i] & 0x0f) | hi0) - 16;
                        const int q1 = ((blocks[b].q[i] >> 4) | hi1) - 16;
                        sum += d * q0 * vector[b * 32 + i];
                        sum += d * q1 * vector[b * 32 + 16 + i];
                    }
                }
                break;
            }
            case GgmlType::Q4_K: {
                if (cols % 256) throw std::invalid_argument("unaligned Q4_K row");
                const auto* blocks = static_cast<const BlockQ4K*>(matrix) +
                    row * (cols / 256);
                for (std::size_t b = 0; b < cols / 256; ++b)
                    sum += dot_q4k(blocks[b], vector + b * 256);
                break;
            }
            case GgmlType::Q6_K: {
                if (cols % 256) throw std::invalid_argument("unaligned Q6_K row");
                const auto* blocks = static_cast<const BlockQ6K*>(matrix) +
                    row * (cols / 256);
                for (std::size_t b = 0; b < cols / 256; ++b)
                    sum += dot_q6k(blocks[b], vector + b * 256);
                break;
            }
            default:
                throw std::runtime_error("GGML matvec type is not implemented");
        }
        output[row] = sum;
    }
}

void ggml_dequantize_row(GgmlType type, const void* row, std::size_t cols,
                         float* output) {
    if (!row || !output) throw std::invalid_argument("null dequantize row");
    switch (type) {
        case GgmlType::F32:
            std::memcpy(output, row, cols * sizeof(float));
            return;
        case GgmlType::F16:
        case GgmlType::BF16: {
            const auto* values = static_cast<const std::uint16_t*>(row);
            for (std::size_t i = 0; i < cols; ++i)
                output[i] = type == GgmlType::F16
                    ? f16_to_f32(values[i]) : bf16_to_f32(values[i]);
            return;
        }
        case GgmlType::Q4_0: {
            if (cols % 32) throw std::invalid_argument("unaligned Q4_0 row");
            const auto* blocks = static_cast<const Q4Block*>(row);
            for (std::size_t b = 0; b < cols / 32; ++b)
                dequantize_q4_block(blocks[b], output + b * 32);
            return;
        }
        case GgmlType::Q5_0: {
            if (cols % 32) throw std::invalid_argument("unaligned Q5_0 row");
            const auto* blocks = static_cast<const BlockQ5_0*>(row);
            for (std::size_t b = 0; b < cols / 32; ++b) {
                const float d = f16_to_f32(blocks[b].d);
                std::uint32_t high = 0;
                std::memcpy(&high, blocks[b].qh, sizeof(high));
                for (std::size_t i = 0; i < 16; ++i) {
                    const int hi0 = ((high >> i) << 4) & 0x10;
                    const int hi1 = (high >> (i + 12)) & 0x10;
                    output[b * 32 + i] =
                        (((blocks[b].q[i] & 0x0f) | hi0) - 16) * d;
                    output[b * 32 + 16 + i] =
                        (((blocks[b].q[i] >> 4) | hi1) - 16) * d;
                }
            }
            return;
        }
        case GgmlType::Q8_0: {
            if (cols % 32) throw std::invalid_argument("unaligned Q8_0 row");
            const auto* blocks = static_cast<const BlockQ8_0*>(row);
            for (std::size_t b = 0; b < cols / 32; ++b) {
                const float d = f16_to_f32(blocks[b].d);
                for (std::size_t i = 0; i < 32; ++i)
                    output[b * 32 + i] = blocks[b].q[i] * d;
            }
            return;
        }
        case GgmlType::Q4_K: {
            if (cols % 256) throw std::invalid_argument("unaligned Q4_K row");
            const auto* blocks = static_cast<const BlockQ4K*>(row);
            for (std::size_t b = 0; b < cols / 256; ++b) {
                const float d = f16_to_f32(blocks[b].d);
                const float dmin = f16_to_f32(blocks[b].dmin);
                int group = 0;
                const std::uint8_t* q = blocks[b].q;
                for (int base = 0; base < 256; base += 64) {
                    std::uint8_t s1, m1, s2, m2;
                    q4k_scale_min(group++, blocks[b].scales, s1, m1);
                    q4k_scale_min(group++, blocks[b].scales, s2, m2);
                    for (int i = 0; i < 32; ++i)
                        output[b * 256 + base + i] =
                            d * s1 * (q[i] & 0x0f) - dmin * m1;
                    for (int i = 0; i < 32; ++i)
                        output[b * 256 + base + 32 + i] =
                            d * s2 * (q[i] >> 4) - dmin * m2;
                    q += 32;
                }
            }
            return;
        }
        case GgmlType::Q6_K: {
            if (cols % 256) throw std::invalid_argument("unaligned Q6_K row");
            const auto* blocks = static_cast<const BlockQ6K*>(row);
            for (std::size_t b = 0; b < cols / 256; ++b) {
                const float d = f16_to_f32(blocks[b].d);
                const std::uint8_t* ql = blocks[b].ql;
                const std::uint8_t* qh = blocks[b].qh;
                const std::int8_t* scales = blocks[b].scales;
                for (int base = 0; base < 256; base += 128) {
                    for (int i = 0; i < 32; ++i) {
                        const int g = i / 16;
                        const int q1 = ((ql[i] & 15) | (((qh[i] >> 0) & 3) << 4)) - 32;
                        const int q2 = ((ql[i + 32] & 15) | (((qh[i] >> 2) & 3) << 4)) - 32;
                        const int q3 = ((ql[i] >> 4) | (((qh[i] >> 4) & 3) << 4)) - 32;
                        const int q4 = ((ql[i + 32] >> 4) | (((qh[i] >> 6) & 3) << 4)) - 32;
                        output[b * 256 + base + i] = d * scales[g] * q1;
                        output[b * 256 + base + 32 + i] = d * scales[g + 2] * q2;
                        output[b * 256 + base + 64 + i] = d * scales[g + 4] * q3;
                        output[b * 256 + base + 96 + i] = d * scales[g + 6] * q4;
                    }
                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
            }
            return;
        }
        default:
            throw std::runtime_error("GGML row type is not implemented");
    }
}

void rms_norm(const float* input, const float* weight, std::size_t size,
              float epsilon, float* output) {
    double squares = 0.0;
    for (std::size_t i = 0; i < size; ++i)
        squares += static_cast<double>(input[i]) * input[i];
    const float inverse = 1.0f /
        std::sqrt(static_cast<float>(squares / size) + epsilon);
    for (std::size_t i = 0; i < size; ++i)
        output[i] = input[i] * inverse * (weight ? weight[i] : 1.0f);
}

void softmax(float* values, std::size_t size) {
    if (size == 0) return;
    const float maximum = *std::max_element(values, values + size);
    double sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::exp(values[i] - maximum);
        sum += values[i];
    }
    const float inverse = static_cast<float>(1.0 / sum);
    for (std::size_t i = 0; i < size; ++i) values[i] *= inverse;
}

std::vector<std::pair<std::uint32_t, float>>
top_k(const float* values, std::size_t size, std::size_t k) {
    k = std::min(k, size);
    std::vector<std::pair<std::uint32_t, float>> result;
    result.reserve(size);
    for (std::size_t i = 0; i < size; ++i)
        result.emplace_back(static_cast<std::uint32_t>(i), values[i]);
    std::partial_sort(result.begin(), result.begin() + k, result.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    result.resize(k);
    return result;
}

float gelu_tanh(float value) noexcept {
    constexpr float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * value *
        (1.0f + std::tanh(k * (value + 0.044715f * value * value * value)));
}

} // namespace gemmaedge
