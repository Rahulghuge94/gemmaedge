#pragma once

#include "gemmaedge/attention.h"
#include "gemmaedge/feed_forward.h"
#include "gemmaedge/gemma4_model.h"
#include "gemmaedge/mapped_file.h"
#include "gemmaedge/tokenizer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace gemmaedge {

// Pre-allocated workspace that lives for the lifetime of a session, avoiding
// per-token heap churn.  Sized once at construction from the model config.
struct ScratchArena {
    // Attention scratch
    std::vector<float> normalized;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attended;
    std::vector<float> projected;
    std::vector<float> layer_output;

    // FFN scratch
    std::vector<float> dense_input;
    std::vector<float> dense_gate;
    std::vector<float> dense_up;
    std::vector<float> dense_output;
    std::vector<float> expert_input;
    std::vector<float> expert_sum;
    std::vector<float> gate_up;
    std::vector<float> activated;
    std::vector<float> expert_output;
    std::vector<float> combined;

    void resize(const Gemma4Config& config);
};

struct SamplingConfig {
    float temperature{1.0f};
    std::uint32_t top_k{64};
    float top_p{0.95f};
    float min_p{0.0f};
    std::uint64_t seed{0};
};

struct GenerationConfig {
    std::uint32_t max_new_tokens{128};
    SamplingConfig sampling{};
    std::vector<TokenId> stop_tokens{1, 106}; // <eos>, <turn|>
};

TokenId sample_token(const std::vector<float>& logits,
                     const SamplingConfig& config,
                     std::mt19937_64& random);

class Gemma4Session {
public:
    Gemma4Session(const Gemma4Model& model, const MappedFile& weights,
                  std::uint64_t expert_cache_bytes);

    // Evaluates one language token and returns logits for the following token.
    const std::vector<float>& evaluate(TokenId token, bool skip_logits = false);

    // Evaluates an already projected 2816-wide embedding, used for image soft
    // tokens. Unlike token embeddings, it is not multiplied by sqrt(hidden).
    const std::vector<float>& evaluate_embedding(const float* embedding, bool skip_logits = false);

    const std::vector<float>& prefill(const std::vector<TokenId>& prompt);
    std::vector<TokenId> generate(
        const std::vector<TokenId>& prompt, const GenerationConfig& config,
        const std::function<void(TokenId)>& emit = {});

    void reset();
    std::uint64_t position() const noexcept { return position_; }
    const ExpertCacheStats& expert_cache_stats() const noexcept {
        return feed_forward_.cache_stats();
    }

private:
    const std::vector<float>& forward(float* hidden, bool skip_logits = false);
    const std::uint8_t* tensor_data(const GgufTensor& tensor) const;
    const float* f32_data(const GgufTensor& tensor) const;
    void matvec(const GgufTensor& tensor, const float* input,
                float* output) const;

    const Gemma4Model& model_;
    const MappedFile& weights_;
    Gemma4FeedForward feed_forward_;
    std::vector<std::unique_ptr<LayerKvCache>> kv_;
    std::vector<float> logits_;
    std::uint64_t position_{0};
    ScratchArena scratch_;
};

} // namespace gemmaedge
