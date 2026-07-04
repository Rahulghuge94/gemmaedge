#include "gemmaedge/attention.h"
#include "gemmaedge/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gemmaedge {
namespace {

constexpr std::uint32_t kTieredKvMagic = 0x31564b47; // "GKV1"
constexpr std::uint32_t kTieredKvQ8 = 1;

template <typename T>
void append_scalar(std::vector<std::uint8_t>& output, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(value));
}

template <typename T>
T payload_scalar(const std::uint8_t*& cursor, const std::uint8_t* end) {
    if (static_cast<std::size_t>(end - cursor) < sizeof(T))
        throw std::runtime_error("truncated tiered KV payload");
    T value;
    std::memcpy(&value, cursor, sizeof(value));
    cursor += sizeof(value);
    return value;
}

} // namespace

void apply_rope(float* states, std::size_t head_count, std::size_t head_dim,
                std::size_t rotary_dim, std::uint64_t position,
                float frequency_base, const float* frequency_factors) {
    if (!states || rotary_dim == 0 || rotary_dim > head_dim ||
        (rotary_dim & 1u) != 0 || frequency_base <= 0.0f)
        throw std::invalid_argument("invalid RoPE arguments");
    const std::size_t half = rotary_dim / 2;
    if (half > 256) throw std::invalid_argument("rotary_dim too large");

    float cos_vals[256];
    float sin_vals[256];
    for (std::size_t i = 0; i < half; ++i) {
        float frequency =
            std::pow(frequency_base,
                     -2.0f * static_cast<float>(i) /
                         static_cast<float>(rotary_dim));
        if (frequency_factors) frequency /= frequency_factors[i];
        const float angle = static_cast<float>(position) * frequency;
        cos_vals[i] = std::cos(angle);
        sin_vals[i] = std::sin(angle);
    }

    for (std::size_t head = 0; head < head_count; ++head) {
        float* row = states + head * head_dim;
        for (std::size_t i = 0; i < half; ++i) {
            const float cosine = cos_vals[i];
            const float sine = sin_vals[i];
            const float first = row[i];
            const float second = row[i + half];
            row[i] = first * cosine - second * sine;
            row[i + half] = second * cosine + first * sine;
        }
    }
}

LayerKvCache::LayerKvCache(AttentionConfig config) : config_(config) {
    if (config_.query_heads == 0 || config_.kv_heads == 0 ||
        config_.head_dim == 0 ||
        config_.query_heads % config_.kv_heads != 0)
        throw std::invalid_argument("invalid grouped-query attention shape");
    if (config_.head_dim % 32 != 0)
        throw std::invalid_argument("head_dim must be a multiple of 32");

    // Pre-allocate ring buffer for sliding window layers (fixed memory)
    if (config_.sliding_window) {
        capacity_ = config_.sliding_window;
        const auto row_blocks = row_values() / 32;
        positions_.resize(capacity_);
        keys_.resize(capacity_ * row_blocks);
        values_.resize(capacity_ * row_blocks);
    }
}

std::size_t LayerKvCache::row_values() const noexcept {
    return static_cast<std::size_t>(config_.kv_heads) * config_.head_dim;
}

void LayerKvCache::append(std::uint64_t position, const float* keys,
                          const float* values) {
    if (!keys || !values) throw std::invalid_argument("null KV row");

    const auto width = row_values();
    const std::size_t row_blocks = width / 32;

    if (config_.sliding_window) {
        // Ring buffer: overwrite at cursor, O(1) eviction
        positions_[cursor_] = position;
        for (std::size_t b = 0; b < row_blocks; ++b) {
            keys_[cursor_ * row_blocks + b] = quantize_q8_block(keys + b * 32);
            values_[cursor_ * row_blocks + b] = quantize_q8_block(values + b * 32);
        }
        cursor_ = (cursor_ + 1) % capacity_;
        if (count_ < capacity_) ++count_;
    } else {
        // Global attention: append (grows dynamically)
        positions_.push_back(position);
        for (std::size_t b = 0; b < row_blocks; ++b) {
            keys_.push_back(quantize_q8_block(keys + b * 32));
            values_.push_back(quantize_q8_block(values + b * 32));
        }
        ++count_;
        capacity_ = count_;
    }
}

void LayerKvCache::attend(const float* queries, float* output) const {
    if (!queries || !output) throw std::invalid_argument("null attention buffer");
    const std::size_t query_values =
        static_cast<std::size_t>(config_.query_heads) * config_.head_dim;
    std::fill(output, output + query_values, 0.0f);
    if (count_ == 0) return;

    const std::size_t width = row_values();
    const std::size_t row_blocks = width / 32;
    const std::size_t group = config_.query_heads / config_.kv_heads;
    const std::size_t blocks_per_head = config_.head_dim / 32;

    std::vector<float> scores_all(config_.query_heads * count_);

    #pragma omp parallel for schedule(static)
    for (std::size_t qh = 0; qh < config_.query_heads; ++qh) {
        float* scores = scores_all.data() + qh * count_;
        const std::size_t kvh = qh / group;
        const float* query = queries + qh * config_.head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < count_; ++token) {
            const std::size_t slot = ring_index(token);
            const BlockQ8_0* key_row = keys_.data() + slot * row_blocks + kvh * blocks_per_head;
            double dot = 0.0;
            for (std::size_t b = 0; b < blocks_per_head; ++b) {
                const auto& block = key_row[b];
#if defined(__ARM_NEON)
                dot += dot_q8_block_neon(block, query + b * 32);
#elif defined(__AVX2__)
                dot += dot_q8_block_avx2(block, query + b * 32);
#else
                const float scale = f16_to_f32(block.d);
                const float* q_block = query + b * 32;
                float sum = 0.0f;
                for (std::size_t i = 0; i < 32; ++i) {
                    sum += block.q[i] * q_block[i];
                }
                dot += sum * scale;
#endif
            }
            scores[token] = static_cast<float>(dot) * config_.scale;
            maximum = std::max(maximum, scores[token]);
        }
        double denominator = 0.0;
        for (std::size_t t = 0; t < count_; ++t) {
            scores[t] = std::exp(scores[t] - maximum);
            denominator += scores[t];
        }
        const float inverse = static_cast<float>(1.0 / denominator);
        float* destination = output + qh * config_.head_dim;
        for (std::size_t token = 0; token < count_; ++token) {
            const float probability = scores[token] * inverse;
            const std::size_t slot = ring_index(token);
            const BlockQ8_0* val_row = values_.data() + slot * row_blocks + kvh * blocks_per_head;
            for (std::size_t b = 0; b < blocks_per_head; ++b) {
                const auto& block = val_row[b];
                const float scale = f16_to_f32(block.d);
                float* dest_block = destination + b * 32;
                const float scale_prob = scale * probability;
#if defined(__ARM_NEON)
                accumulate_q8_block_neon(block, scale_prob, dest_block);
#elif defined(__AVX2__)
                accumulate_q8_block_avx2(block, scale_prob, dest_block);
#else
                for (std::size_t i = 0; i < 32; ++i) {
                    dest_block[i] += scale_prob * block.q[i];
                }
#endif
            }
        }
    }
}

void LayerKvCache::clear() {
    if (!config_.sliding_window) {
        // Global attention: release memory
        positions_.clear();
        keys_.clear();
        values_.clear();
        capacity_ = 0;
    }
    // Ring buffer: just reset cursors, keep memory allocated
    cursor_ = 0;
    count_ = 0;
}

std::vector<std::uint8_t> LayerKvCache::serialize_f32() const {
    const std::size_t width = row_values();
    const std::size_t row_blocks = width / 32;
    const std::size_t float_count = count_ * width * 2;
    std::vector<std::uint8_t> result(float_count * sizeof(float));
    float* output = reinterpret_cast<float*>(result.data());
    
    // Dequantize keys in age order
    for (std::size_t t = 0; t < count_; ++t) {
        const std::size_t slot = ring_index(t);
        for (std::size_t b = 0; b < row_blocks; ++b) {
            dequantize_q8_block(keys_[slot * row_blocks + b],
                                output + (t * row_blocks + b) * 32);
        }
    }
    // Dequantize values in age order
    float* val_output = output + count_ * width;
    for (std::size_t t = 0; t < count_; ++t) {
        const std::size_t slot = ring_index(t);
        for (std::size_t b = 0; b < row_blocks; ++b) {
            dequantize_q8_block(values_[slot * row_blocks + b],
                                val_output + (t * row_blocks + b) * 32);
        }
    }
    return result;
}

void LayerKvCache::restore_f32(const std::uint8_t* payload, std::size_t bytes,
                               std::uint64_t first_position,
                               std::size_t token_count) {
    const auto width = row_values();
    const auto expected = token_count * width * sizeof(float) * 2;
    if (!payload || bytes != expected)
        throw std::invalid_argument("invalid serialized KV payload");
    if (config_.sliding_window && token_count > config_.sliding_window)
        throw std::invalid_argument("serialized KV exceeds sliding window");

    clear();
    const std::size_t row_blocks = width / 32;

    if (config_.sliding_window) {
        // Ring buffer: write sequentially from slot 0
        for (std::size_t i = 0; i < token_count; ++i)
            positions_[i] = first_position + i;
        cursor_ = token_count % capacity_;
        count_ = token_count;
    } else {
        positions_.resize(token_count);
        for (std::size_t i = 0; i < token_count; ++i)
            positions_[i] = first_position + i;
        keys_.resize(token_count * row_blocks);
        values_.resize(token_count * row_blocks);
        count_ = token_count;
        capacity_ = token_count;
    }

    const float* float_payload = reinterpret_cast<const float*>(payload);
    for (std::size_t block_idx = 0; block_idx < token_count * row_blocks; ++block_idx) {
        keys_[block_idx] = quantize_q8_block(float_payload + block_idx * 32);
    }
    const float* val_payload = float_payload + token_count * width;
    for (std::size_t block_idx = 0; block_idx < token_count * row_blocks; ++block_idx) {
        values_[block_idx] = quantize_q8_block(val_payload + block_idx * 32);
    }
}

KvBlock LayerKvCache::to_disk_block(std::uint32_t layer) const {
    if (count_ == 0) throw std::runtime_error("cannot spill empty KV cache");
    // Verify contiguity in age order
    const std::uint64_t first = positions_[ring_index(0)];
    for (std::size_t i = 1; i < count_; ++i) {
        if (positions_[ring_index(i)] != first + i)
            throw std::runtime_error("disk KV block positions are not contiguous");
    }
    if (first > std::numeric_limits<std::uint32_t>::max() ||
        count_ > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("disk KV block index overflow");

    const std::size_t row_blocks = row_values() / 32;
    std::vector<std::uint8_t> payload;
    payload.reserve(16 + count_ * row_blocks * sizeof(BlockQ8_0) * 2);
    append_scalar(payload, kTieredKvMagic);
    append_scalar(payload, kTieredKvQ8);
    append_scalar(payload, static_cast<std::uint32_t>(count_));
    append_scalar(payload, static_cast<std::uint32_t>(row_values()));

    // Write keys in age order
    for (std::size_t t = 0; t < count_; ++t) {
        const std::size_t slot = ring_index(t);
        const auto* key_bytes = reinterpret_cast<const std::uint8_t*>(
            keys_.data() + slot * row_blocks);
        payload.insert(payload.end(), key_bytes,
                       key_bytes + row_blocks * sizeof(BlockQ8_0));
    }
    // Write values in age order
    for (std::size_t t = 0; t < count_; ++t) {
        const std::size_t slot = ring_index(t);
        const auto* val_bytes = reinterpret_cast<const std::uint8_t*>(
            values_.data() + slot * row_blocks);
        payload.insert(payload.end(), val_bytes,
                       val_bytes + row_blocks * sizeof(BlockQ8_0));
    }

    return {{layer, static_cast<std::uint32_t>(first)},
            static_cast<std::uint32_t>(count_), std::move(payload)};
}

void LayerKvCache::restore_disk_block(const KvBlock& block) {
    const std::uint8_t* cursor = block.bytes.data();
    const std::uint8_t* end = cursor + block.bytes.size();
    const auto magic = payload_scalar<std::uint32_t>(cursor, end);
    const auto precision = payload_scalar<std::uint32_t>(cursor, end);
    const auto tokens = payload_scalar<std::uint32_t>(cursor, end);
    const auto width = payload_scalar<std::uint32_t>(cursor, end);
    if (magic != kTieredKvMagic || precision != kTieredKvQ8 ||
        tokens != block.token_count || width != row_values())
        throw std::runtime_error("incompatible tiered KV payload");

    const std::size_t row_blocks = width / 32;
    const std::size_t expected_bytes = tokens * row_blocks * sizeof(BlockQ8_0) * 2;
    if (static_cast<std::size_t>(end - cursor) < expected_bytes)
        throw std::runtime_error("truncated tiered KV payload data");

    clear();
    if (!config_.sliding_window) {
        keys_.resize(tokens * row_blocks);
        values_.resize(tokens * row_blocks);
    }

    std::memcpy(keys_.data(), cursor, tokens * row_blocks * sizeof(BlockQ8_0));
    cursor += tokens * row_blocks * sizeof(BlockQ8_0);

    std::memcpy(values_.data(), cursor, tokens * row_blocks * sizeof(BlockQ8_0));
    cursor += tokens * row_blocks * sizeof(BlockQ8_0);

    if (cursor != end) throw std::runtime_error("trailing tiered KV bytes");

    if (config_.sliding_window) {
        for (std::uint32_t i = 0; i < tokens; ++i)
            positions_[i] = block.key.token_start + i;
        cursor_ = tokens % capacity_;
    } else {
        positions_.resize(tokens);
        for (std::uint32_t i = 0; i < tokens; ++i)
            positions_[i] = block.key.token_start + i;
        capacity_ = tokens;
    }
    count_ = tokens;
}

} // namespace gemmaedge
