#pragma once

#include "gemmaedge/gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gemmaedge {

struct Gemma4Config {
    std::uint32_t context_length{0};
    std::uint32_t hidden_size{0};
    std::uint32_t layer_count{0};
    std::uint32_t attention_heads{0};
    std::uint32_t sliding_window{0};
    std::uint32_t dense_intermediate{0};
    std::uint32_t expert_intermediate{0};
    std::uint32_t expert_count{0};
    std::uint32_t experts_used{0};
    std::uint32_t vocab_size{0};
    std::uint32_t local_head_dim{0};
    std::uint32_t global_head_dim{0};
    float rms_epsilon{1e-6f};
    float final_logit_softcap{30.0f};
    float local_rope_base{10000.0f};
    float global_rope_base{1000000.0f};
    std::vector<bool> sliding_layers;
    std::vector<std::uint32_t> kv_heads;
};

struct Gemma4LayerTensors {
    const GgufTensor* attention_norm{};
    const GgufTensor* q{};
    const GgufTensor* k{};
    const GgufTensor* v{};
    const GgufTensor* attention_output{};
    const GgufTensor* q_norm{};
    const GgufTensor* k_norm{};
    const GgufTensor* post_attention_norm{};

    const GgufTensor* dense_gate{};
    const GgufTensor* dense_up{};
    const GgufTensor* dense_down{};
    const GgufTensor* ffn_norm{};
    const GgufTensor* post_ffn_norm{};

    const GgufTensor* router{};
    const GgufTensor* router_scale{};
    const GgufTensor* expert_gate_up{};
    const GgufTensor* expert_down{};
    const GgufTensor* expert_output_scale{};
    const GgufTensor* pre_expert_norm{};
    const GgufTensor* post_dense_norm{};
    const GgufTensor* post_expert_norm{};
    const GgufTensor* layer_scale{};
};

class Gemma4Model {
public:
    explicit Gemma4Model(const GgufFile& gguf);

    const Gemma4Config& config() const noexcept { return config_; }
    const GgufTensor& token_embedding() const noexcept { return *token_embedding_; }
    const GgufTensor& output_norm() const noexcept { return *output_norm_; }
    const GgufTensor& output() const noexcept { return *output_; }
    const GgufTensor* rope_frequencies() const noexcept {
        return rope_frequencies_;
    }
    const std::vector<Gemma4LayerTensors>& layers() const noexcept {
        return layers_;
    }

private:
    const GgufTensor* required(const std::string& name) const;
    const GgufTensor* optional(const std::string& name) const noexcept;
    void require_shape(const GgufTensor* tensor,
                       std::initializer_list<std::uint64_t> shape) const;

    const GgufFile& gguf_;
    Gemma4Config config_;
    const GgufTensor* token_embedding_{};
    const GgufTensor* output_norm_{};
    const GgufTensor* output_{};
    const GgufTensor* rope_frequencies_{};
    std::vector<Gemma4LayerTensors> layers_;
};

struct Gemma4VisionConfig {
    std::uint32_t image_size{0};
    std::uint32_t patch_size{0};
    std::uint32_t hidden_size{0};
    std::uint32_t projection_size{0};
    std::uint32_t intermediate_size{0};
    std::uint32_t layer_count{0};
    std::uint32_t attention_heads{0};
    float norm_epsilon{1e-6f};
};

struct Gemma4VisionLayerTensors {
    const GgufTensor* input_norm{};
    const GgufTensor* q{};
    const GgufTensor* k{};
    const GgufTensor* v{};
    const GgufTensor* output{};
    const GgufTensor* q_norm{};
    const GgufTensor* k_norm{};
    const GgufTensor* attention_post_norm{};
    const GgufTensor* ffn_norm{};
    const GgufTensor* gate{};
    const GgufTensor* up{};
    const GgufTensor* down{};
    const GgufTensor* ffn_post_norm{};
};

class Gemma4VisionModel {
public:
    explicit Gemma4VisionModel(const GgufFile& gguf);

    const Gemma4VisionConfig& config() const noexcept { return config_; }
    const std::vector<Gemma4VisionLayerTensors>& layers() const noexcept {
        return layers_;
    }
    const GgufTensor& patch_embedding() const noexcept { return *patch_embedding_; }
    const GgufTensor& position_embedding() const noexcept { return *position_embedding_; }
    const GgufTensor& standardize_scale() const noexcept { return *standardize_scale_; }
    const GgufTensor& standardize_bias() const noexcept { return *standardize_bias_; }
    const GgufTensor& projector() const noexcept { return *projector_; }

private:
    const GgufTensor* required(const std::string& name) const;
    const GgufFile& gguf_;
    Gemma4VisionConfig config_;
    const GgufTensor* patch_embedding_{};
    const GgufTensor* position_embedding_{};
    const GgufTensor* standardize_scale_{};
    const GgufTensor* standardize_bias_{};
    const GgufTensor* projector_{};
    std::vector<Gemma4VisionLayerTensors> layers_;
};

} // namespace gemmaedge
