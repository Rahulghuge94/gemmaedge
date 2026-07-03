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

void append_q8(std::vector<std::uint8_t>& output,
               const std::vector<float>& values) {
    if (values.size() % 32 != 0)
        throw std::runtime_error("KV width is not Q8 block aligned");
    for (std::size_t at = 0; at < values.size(); at += 32) {
        float maximum = 0.0f;
        for (std::size_t i = 0; i < 32; ++i)
            maximum = std::max(maximum, std::abs(values[at + i]));
        const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
        append_scalar(output, f32_to_f16(scale));
        for (std::size_t i = 0; i < 32; ++i) {
            const int quantized = scale == 0.0f ? 0 :
                static_cast<int>(std::nearbyint(values[at + i] / scale));
            output.push_back(static_cast<std::uint8_t>(
                static_cast<std::int8_t>(
                    std::max(-127, std::min(127, quantized)))));
        }
    }
}

std::vector<float> read_q8(const std::uint8_t*& cursor,
                           const std::uint8_t* end, std::size_t values) {
    std::vector<float> result(values);
    if (values % 32 != 0)
        throw std::runtime_error("invalid Q8 KV value count");
    for (std::size_t at = 0; at < values; at += 32) {
        const float scale = f16_to_f32(
            payload_scalar<std::uint16_t>(cursor, end));
        if (static_cast<std::size_t>(end - cursor) < 32)
            throw std::runtime_error("truncated Q8 KV block");
        for (std::size_t i = 0; i < 32; ++i)
            result[at + i] =
                static_cast<std::int8_t>(cursor[i]) * scale;
        cursor += 32;
    }
    return result;
}

} // namespace

void apply_rope(float* states, std::size_t head_count, std::size_t head_dim,
                std::size_t rotary_dim, std::uint64_t position,
                float frequency_base, const float* frequency_factors) {
    if (!states || rotary_dim == 0 || rotary_dim > head_dim ||
        (rotary_dim & 1u) != 0 || frequency_base <= 0.0f)
        throw std::invalid_argument("invalid RoPE arguments");
    const std::size_t half = rotary_dim / 2;
    for (std::size_t head = 0; head < head_count; ++head) {
        float* row = states + head * head_dim;
        for (std::size_t i = 0; i < half; ++i) {
            float frequency =
                std::pow(frequency_base,
                         -2.0f * static_cast<float>(i) /
                             static_cast<float>(rotary_dim));
            if (frequency_factors) frequency /= frequency_factors[i];
            const float angle = static_cast<float>(position) * frequency;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
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
}

std::size_t LayerKvCache::row_values() const noexcept {
    return static_cast<std::size_t>(config_.kv_heads) * config_.head_dim;
}

void LayerKvCache::append(std::uint64_t position, const float* keys,
                          const float* values) {
    if (!keys || !values) throw std::invalid_argument("null KV row");
    if (!positions_.empty() && position <= positions_.back())
        throw std::invalid_argument("KV positions must increase");

    const auto width = row_values();
    positions_.push_back(position);
    keys_.insert(keys_.end(), keys, keys + width);
    values_.insert(values_.end(), values, values + width);

    if (config_.sliding_window &&
        positions_.size() > config_.sliding_window) {
        positions_.erase(positions_.begin());
        keys_.erase(keys_.begin(), keys_.begin() +
                    static_cast<std::ptrdiff_t>(width));
        values_.erase(values_.begin(), values_.begin() +
                      static_cast<std::ptrdiff_t>(width));
    }
}

void LayerKvCache::attend(const float* queries, float* output) const {
    if (!queries || !output) throw std::invalid_argument("null attention buffer");
    const std::size_t query_values =
        static_cast<std::size_t>(config_.query_heads) * config_.head_dim;
    std::fill(output, output + query_values, 0.0f);
    if (positions_.empty()) return;

    const std::size_t width = row_values();
    const std::size_t group = config_.query_heads / config_.kv_heads;
    std::vector<float> scores(positions_.size());

    for (std::size_t qh = 0; qh < config_.query_heads; ++qh) {
        const std::size_t kvh = qh / group;
        const float* query = queries + qh * config_.head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < positions_.size(); ++token) {
            const float* key = keys_.data() + token * width +
                               kvh * config_.head_dim;
            double dot = 0.0;
            for (std::size_t i = 0; i < config_.head_dim; ++i)
                dot += static_cast<double>(query[i]) * key[i];
            scores[token] = static_cast<float>(dot) * config_.scale;
            maximum = std::max(maximum, scores[token]);
        }
        double denominator = 0.0;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        const float inverse = static_cast<float>(1.0 / denominator);
        float* destination = output + qh * config_.head_dim;
        for (std::size_t token = 0; token < positions_.size(); ++token) {
            const float probability = scores[token] * inverse;
            const float* value = values_.data() + token * width +
                                 kvh * config_.head_dim;
            for (std::size_t i = 0; i < config_.head_dim; ++i)
                destination[i] += probability * value[i];
        }
    }
}

void LayerKvCache::clear() {
    positions_.clear();
    keys_.clear();
    values_.clear();
}

std::vector<std::uint8_t> LayerKvCache::serialize_f32() const {
    const std::size_t float_count = keys_.size() + values_.size();
    std::vector<std::uint8_t> result(float_count * sizeof(float));
    const std::size_t key_bytes = keys_.size() * sizeof(float);
    std::memcpy(result.data(), keys_.data(), key_bytes);
    std::memcpy(result.data() + key_bytes, values_.data(),
                values_.size() * sizeof(float));
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
    positions_.reserve(token_count);
    for (std::size_t i = 0; i < token_count; ++i)
        positions_.push_back(first_position + i);
    keys_.resize(token_count * width);
    values_.resize(token_count * width);
    const auto half = token_count * width * sizeof(float);
    std::memcpy(keys_.data(), payload, half);
    std::memcpy(values_.data(), payload + half, half);
}

KvBlock LayerKvCache::to_disk_block(std::uint32_t layer) const {
    if (positions_.empty()) throw std::runtime_error("cannot spill empty KV cache");
    for (std::size_t i = 1; i < positions_.size(); ++i) {
        if (positions_[i] != positions_[0] + i)
            throw std::runtime_error("disk KV block positions are not contiguous");
    }
    if (positions_[0] > std::numeric_limits<std::uint32_t>::max() ||
        positions_.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("disk KV block index overflow");
    std::vector<std::uint8_t> payload;
    payload.reserve(16 + (keys_.size() + values_.size()) * 34 / 32);
    append_scalar(payload, kTieredKvMagic);
    append_scalar(payload, kTieredKvQ8);
    append_scalar(payload, static_cast<std::uint32_t>(positions_.size()));
    append_scalar(payload, static_cast<std::uint32_t>(row_values()));
    append_q8(payload, keys_);
    append_q8(payload, values_);
    return {{layer, static_cast<std::uint32_t>(positions_[0])},
            static_cast<std::uint32_t>(positions_.size()), std::move(payload)};
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
    keys_ = read_q8(cursor, end, static_cast<std::size_t>(tokens) * width);
    values_ = read_q8(cursor, end, static_cast<std::size_t>(tokens) * width);
    if (cursor != end) throw std::runtime_error("trailing tiered KV bytes");
    positions_.clear();
    positions_.reserve(tokens);
    for (std::uint32_t i = 0; i < tokens; ++i)
        positions_.push_back(block.key.token_start + i);
}

} // namespace gemmaedge
