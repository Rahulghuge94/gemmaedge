#pragma once

#include "gemmaedge/expert_cache.h"
#include "gemmaedge/gemma4_model.h"
#include "gemmaedge/mapped_file.h"

#include <cstdint>
#include <vector>

namespace gemmaedge {

struct RoutedExpert {
    std::uint32_t expert{0};
    float probability{0.0f};
    float weight{0.0f};
};

std::vector<RoutedExpert>
route_top_k(const float* hidden, std::size_t hidden_size,
            const float* router_scale, const float* router_matrix,
            std::size_t expert_count, std::size_t experts_used,
            const float* per_expert_scale, float epsilon);

// Scratch-buffer overload: avoids per-call heap allocations (critical for edge).
// `scratch_norm` must be at least `hidden_size` floats.
// `scratch_probs` must be at least `expert_count` floats.
std::vector<RoutedExpert>
route_top_k(const float* hidden, std::size_t hidden_size,
            const float* router_scale, const float* router_matrix,
            std::size_t expert_count, std::size_t experts_used,
            const float* per_expert_scale, float epsilon,
            float* scratch_norm, float* scratch_probs);

class Gemma4FeedForward {
public:
    Gemma4FeedForward(const Gemma4Model& model, const MappedFile& weights,
                      std::uint64_t expert_cache_bytes);

    // Executes one Gemma 4 decoder layer's complete feed-forward section:
    // shared dense branch + routed top-8 branch + final norm/residual/scale.
    void forward(std::uint32_t layer, const float* input, float* output,
                 std::vector<RoutedExpert>* routing = nullptr);

    // Same as above but uses pre-allocated scratch buffers from a ScratchArena
    // to avoid per-call heap allocations (critical for mobile).
    void forward(std::uint32_t layer, const float* input, float* output,
                 struct ScratchArena& scratch,
                 std::vector<RoutedExpert>* routing = nullptr);

    const ExpertCacheStats& cache_stats() const noexcept {
        return expert_cache_.stats();
    }

    void preload_all_experts();

private:
    ExpertView load_expert(const ExpertKey& key) const;
    const std::uint8_t* tensor_data(const GgufTensor& tensor) const;
    const float* f32_data(const GgufTensor& tensor) const;
    void matvec(const GgufTensor& tensor, const float* input,
                float* output) const;
    void upload_input(const float* input, std::size_t count) const;
    void matvec_device_vec(const GgufTensor& tensor, const float* input, float* output) const;

    const Gemma4Model& model_;
    const MappedFile& weights_;
    ExpertCache expert_cache_;
};

} // namespace gemmaedge

