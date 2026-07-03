#include "gemmaedge/gemma4_model.h"

#include <stdexcept>
#include <iostream>

namespace gemmaedge {
namespace {

const GgufMetadata& meta(const GgufFile& gguf, const char* key) {
    const auto* value = gguf.meta(key);
    if (!value) throw std::runtime_error(std::string("missing Gemma metadata: ") + key);
    return *value;
}

std::uint32_t u32(const GgufFile& gguf, const char* key) {
    return static_cast<std::uint32_t>(meta(gguf, key).unsigned_value);
}

float f32(const GgufFile& gguf, const char* key) {
    return static_cast<float>(meta(gguf, key).float_value);
}

std::string layer_name(std::uint32_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

} // namespace

Gemma4Model::Gemma4Model(const GgufFile& gguf) : gguf_(gguf) {
    if (meta(gguf, "general.architecture").text != "gemma4")
        throw std::runtime_error("GGUF architecture is not gemma4");

    config_.context_length = u32(gguf, "gemma4.context_length");
    config_.hidden_size = u32(gguf, "gemma4.embedding_length");
    config_.layer_count = u32(gguf, "gemma4.block_count");
    config_.attention_heads = u32(gguf, "gemma4.attention.head_count");
    config_.sliding_window = u32(gguf, "gemma4.attention.sliding_window");
    config_.dense_intermediate = u32(gguf, "gemma4.feed_forward_length");
    config_.expert_intermediate =
        u32(gguf, "gemma4.expert_feed_forward_length");
    config_.expert_count = u32(gguf, "gemma4.expert_count");
    config_.experts_used = u32(gguf, "gemma4.expert_used_count");
    config_.rms_epsilon = f32(gguf, "gemma4.attention.layer_norm_rms_epsilon");
    config_.final_logit_softcap =
        f32(gguf, "gemma4.final_logit_softcapping");
    config_.local_rope_base = f32(gguf, "gemma4.rope.freq_base_swa");
    config_.global_rope_base = f32(gguf, "gemma4.rope.freq_base");
    config_.local_head_dim = u32(gguf, "gemma4.attention.key_length_swa");
    config_.global_head_dim = u32(gguf, "gemma4.attention.key_length");

    const auto& tokens = meta(gguf, "tokenizer.ggml.tokens").strings;
    config_.vocab_size = static_cast<std::uint32_t>(tokens.size());
    const auto& pattern =
        meta(gguf, "gemma4.attention.sliding_window_pattern").unsigned_values;
    const auto& heads =
        meta(gguf, "gemma4.attention.head_count_kv").signed_values;
    if (pattern.size() != config_.layer_count ||
        heads.size() != config_.layer_count)
        throw std::runtime_error("Gemma per-layer metadata length mismatch");
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        config_.sliding_layers.push_back(pattern[i] != 0);
        config_.kv_heads.push_back(static_cast<std::uint32_t>(heads[i]));
    }

    if (config_.layer_count != 30 || config_.hidden_size != 2816 ||
        config_.expert_count != 128 || config_.experts_used != 8)
        throw std::runtime_error("GGUF is not Gemma 4 26B A4B");

    std::cout << "[Model Info] Config Loaded:"
              << "\n  layers: " << config_.layer_count
              << "\n  hidden_size: " << config_.hidden_size
              << "\n  attention_heads: " << config_.attention_heads
              << "\n  sliding_window: " << config_.sliding_window
              << "\n  local_head_dim: " << config_.local_head_dim
              << "\n  global_head_dim: " << config_.global_head_dim
              << "\n  vocab_size: " << config_.vocab_size
              << std::endl;

    token_embedding_ = required("token_embd.weight");
    output_norm_ = required("output_norm.weight");
    output_ = optional("output.weight");
    if (!output_) output_ = token_embedding_;
    rope_frequencies_ = optional("rope_freqs.weight");
    require_shape(token_embedding_, {config_.hidden_size, config_.vocab_size});
    require_shape(output_norm_, {config_.hidden_size});

    layers_.reserve(config_.layer_count);
    for (std::uint32_t i = 0; i < config_.layer_count; ++i) {
        Gemma4LayerTensors layer;
        layer.attention_norm = required(layer_name(i, "attn_norm.weight"));
        layer.q = required(layer_name(i, "attn_q.weight"));
        layer.k = required(layer_name(i, "attn_k.weight"));
        layer.v = optional(layer_name(i, "attn_v.weight"));
        layer.attention_output = required(layer_name(i, "attn_output.weight"));
        layer.q_norm = required(layer_name(i, "attn_q_norm.weight"));
        layer.k_norm = required(layer_name(i, "attn_k_norm.weight"));
        layer.post_attention_norm =
            required(layer_name(i, "post_attention_norm.weight"));
        layer.dense_gate = required(layer_name(i, "ffn_gate.weight"));
        layer.dense_up = required(layer_name(i, "ffn_up.weight"));
        layer.dense_down = required(layer_name(i, "ffn_down.weight"));
        layer.ffn_norm = required(layer_name(i, "ffn_norm.weight"));
        layer.post_ffn_norm = required(layer_name(i, "post_ffw_norm.weight"));
        layer.router = required(layer_name(i, "ffn_gate_inp.weight"));
        layer.router_scale = required(layer_name(i, "ffn_gate_inp.scale"));
        layer.expert_gate_up =
            required(layer_name(i, "ffn_gate_up_exps.weight"));
        layer.expert_down = required(layer_name(i, "ffn_down_exps.weight"));
        layer.expert_output_scale =
            required(layer_name(i, "ffn_down_exps.scale"));
        layer.pre_expert_norm =
            required(layer_name(i, "pre_ffw_norm_2.weight"));
        layer.post_dense_norm =
            required(layer_name(i, "post_ffw_norm_1.weight"));
        layer.post_expert_norm =
            required(layer_name(i, "post_ffw_norm_2.weight"));
        layer.layer_scale =
            optional(layer_name(i, "layer_output_scale.weight"));

        const auto head_dim = config_.sliding_layers[i]
            ? config_.local_head_dim : config_.global_head_dim;
        require_shape(layer.q,
                      {config_.hidden_size,
                       static_cast<std::uint64_t>(head_dim) *
                           config_.attention_heads});
        require_shape(layer.expert_gate_up,
                      {config_.hidden_size,
                       static_cast<std::uint64_t>(config_.expert_intermediate) * 2,
                       config_.expert_count});
        require_shape(layer.expert_down,
                      {config_.expert_intermediate, config_.hidden_size,
                       config_.expert_count});
        layers_.push_back(layer);
    }
}

const GgufTensor* Gemma4Model::required(const std::string& name) const {
    const auto* tensor = gguf_.tensor(name);
    if (!tensor) throw std::runtime_error("missing Gemma tensor: " + name);
    return tensor;
}

const GgufTensor* Gemma4Model::optional(const std::string& name) const noexcept {
    return gguf_.tensor(name);
}

void Gemma4Model::require_shape(
    const GgufTensor* tensor,
    std::initializer_list<std::uint64_t> shape) const {
    if (!tensor || tensor->dimensions != std::vector<std::uint64_t>(shape))
        throw std::runtime_error("unexpected tensor shape: " +
                                 (tensor ? tensor->name : std::string("<null>")));
}

Gemma4VisionModel::Gemma4VisionModel(const GgufFile& gguf) : gguf_(gguf) {
    if (meta(gguf, "general.architecture").text != "clip" ||
        meta(gguf, "clip.vision.projector_type").text != "gemma4v")
        throw std::runtime_error("GGUF is not a Gemma 4 vision projector");
    config_.image_size = u32(gguf, "clip.vision.image_size");
    config_.patch_size = u32(gguf, "clip.vision.patch_size");
    config_.hidden_size = u32(gguf, "clip.vision.embedding_length");
    config_.projection_size = u32(gguf, "clip.vision.projection_dim");
    config_.intermediate_size =
        u32(gguf, "clip.vision.feed_forward_length");
    config_.layer_count = u32(gguf, "clip.vision.block_count");
    config_.attention_heads = u32(gguf, "clip.vision.attention.head_count");
    config_.norm_epsilon =
        f32(gguf, "clip.vision.attention.layer_norm_epsilon");
    if (config_.image_size != 224 || config_.patch_size != 16 ||
        config_.hidden_size != 1152 || config_.projection_size != 2816 ||
        config_.layer_count != 27)
        throw std::runtime_error("unexpected Gemma 4 vision shape");

    patch_embedding_ = required("v.patch_embd.weight");
    position_embedding_ = required("v.position_embd.weight");
    standardize_scale_ = required("v.std_scale");
    standardize_bias_ = required("v.std_bias");
    projector_ = required("mm.input_projection.weight");

    layers_.reserve(config_.layer_count);
    for (std::uint32_t i = 0; i < config_.layer_count; ++i) {
        const auto prefix = "v.blk." + std::to_string(i) + ".";
        Gemma4VisionLayerTensors layer;
        layer.input_norm = required(prefix + "ln1.weight");
        layer.q = required(prefix + "attn_q.weight");
        layer.k = required(prefix + "attn_k.weight");
        layer.v = required(prefix + "attn_v.weight");
        layer.output = required(prefix + "attn_out.weight");
        layer.q_norm = required(prefix + "attn_q_norm.weight");
        layer.k_norm = required(prefix + "attn_k_norm.weight");
        layer.attention_post_norm =
            required(prefix + "attn_post_norm.weight");
        layer.ffn_norm = required(prefix + "ln2.weight");
        layer.gate = required(prefix + "ffn_gate.weight");
        layer.up = required(prefix + "ffn_up.weight");
        layer.down = required(prefix + "ffn_down.weight");
        layer.ffn_post_norm = required(prefix + "ffn_post_norm.weight");
        layers_.push_back(layer);
    }
}

const GgufTensor*
Gemma4VisionModel::required(const std::string& name) const {
    const auto* tensor = gguf_.tensor(name);
    if (!tensor) throw std::runtime_error("missing vision tensor: " + name);
    return tensor;
}

} // namespace gemmaedge
