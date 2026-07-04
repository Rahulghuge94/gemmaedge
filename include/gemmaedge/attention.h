#pragma once

#include "gemmaedge/kv_store.h"
#include "gemmaedge/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gemmaedge {

// Gemma uses split-half RoPE: the first half of each head is paired with the
// second half. `frequency_factors`, when present, contains rotary_dim / 2 GGUF
// proportional-RoPE factors.
void apply_rope(float* states, std::size_t head_count, std::size_t head_dim,
                std::size_t rotary_dim, std::uint64_t position,
                float frequency_base,
                const float* frequency_factors = nullptr);

struct AttentionConfig {
    std::uint32_t query_heads{0};
    std::uint32_t kv_heads{0};
    std::uint32_t head_dim{0};
    std::uint32_t sliding_window{0}; // zero means global attention
    float scale{1.0f};
};

class LayerKvCache {
public:
    explicit LayerKvCache(AttentionConfig config);

    void append(std::uint64_t position, const float* keys, const float* values);
    void attend(const float* queries, float* output) const;
    void clear();

    std::size_t token_count() const noexcept { return count_; }
    std::uint64_t first_position() const noexcept {
        if (count_ == 0) return 0;
        if (config_.sliding_window) {
            // Ring buffer: oldest entry is at cursor_ (if full)
            std::size_t oldest = (count_ == capacity_) ? cursor_ : 0;
            return positions_[oldest];
        }
        return positions_[0];
    }
    std::uint64_t last_position() const noexcept {
        if (count_ == 0) return 0;
        if (config_.sliding_window) {
            std::size_t newest = (cursor_ == 0 ? capacity_ : cursor_) - 1;
            return positions_[newest];
        }
        return positions_[count_ - 1];
    }
    const AttentionConfig& config() const noexcept { return config_; }

    // Contiguous payload used by the disk tier. Layout is all K rows followed
    // by all V rows, both oldest-to-newest.
    std::vector<std::uint8_t> serialize_f32() const;
    void restore_f32(const std::uint8_t* payload, std::size_t bytes,
                     std::uint64_t first_position, std::size_t token_count);
    KvBlock to_disk_block(std::uint32_t layer) const;
    void restore_disk_block(const KvBlock& block);

private:
    std::size_t row_values() const noexcept;

    // Returns the linear index for the i-th token in age order (0=oldest)
    std::size_t ring_index(std::size_t i) const noexcept {
        if (!config_.sliding_window) return i;
        return (cursor_ + capacity_ - count_ + i) % capacity_;
    }

    AttentionConfig config_;
    std::size_t capacity_{0};    // Pre-allocated slots (= sliding_window, or grows)
    std::size_t cursor_{0};      // Next write position (ring buffer index)
    std::size_t count_{0};       // Number of tokens currently stored
    std::vector<std::uint64_t> positions_;
    std::vector<BlockQ8_0> keys_;
    std::vector<BlockQ8_0> values_;
};

} // namespace gemmaedge
