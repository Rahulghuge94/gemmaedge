#pragma once

#include "gemmaedge/kv_store.h"

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

    std::size_t token_count() const noexcept { return positions_.size(); }
    std::uint64_t first_position() const noexcept {
        return positions_.empty() ? 0 : positions_.front();
    }
    std::uint64_t last_position() const noexcept {
        return positions_.empty() ? 0 : positions_.back();
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

    AttentionConfig config_;
    std::vector<std::uint64_t> positions_;
    std::vector<float> keys_;
    std::vector<float> values_;
};

} // namespace gemmaedge
