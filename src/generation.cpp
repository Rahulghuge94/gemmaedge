#include "gemmaedge/generation.h"

#include "gemmaedge/tensor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gemmaedge {

TokenId sample_token(const std::vector<float>& logits,
                     const SamplingConfig& config,
                     std::mt19937_64& random) {
    if (logits.empty()) throw std::invalid_argument("cannot sample empty logits");
    if (config.temperature <= 0.0f) {
        return static_cast<TokenId>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<std::pair<TokenId, float>> candidates;
    candidates.reserve(logits.size());
    const float maximum = *std::max_element(logits.begin(), logits.end());
    double denominator = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float probability =
            std::exp((logits[i] - maximum) / config.temperature);
        candidates.emplace_back(static_cast<TokenId>(i), probability);
        denominator += probability;
    }
    for (auto& item : candidates)
        item.second = static_cast<float>(item.second / denominator);
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
    if (config.top_k && candidates.size() > config.top_k)
        candidates.resize(config.top_k);

    const float minimum = candidates.front().second * config.min_p;
    float cumulative = 0.0f;
    std::size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
        if (candidates[keep].second < minimum && keep > 0) break;
        cumulative += candidates[keep].second;
        if (cumulative >= config.top_p && keep > 0) {
            ++keep;
            break;
        }
    }
    candidates.resize(std::max<std::size_t>(1, keep));
    float mass = 0.0f;
    for (const auto& item : candidates) mass += item.second;
    std::uniform_real_distribution<float> distribution(0.0f, mass);
    float choice = distribution(random);
    for (const auto& item : candidates) {
        choice -= item.second;
        if (choice <= 0.0f) return item.first;
    }
    return candidates.back().first;
}

Gemma4Session::Gemma4Session(
    const Gemma4Model& model, const MappedFile& weights,
    std::uint64_t expert_cache_bytes)
    : model_(model), weights_(weights),
      feed_forward_(model, weights, expert_cache_bytes),
      logits_(model.config().vocab_size) {
    const auto& config = model_.config();
    kv_.reserve(config.layer_count);
    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        const bool sliding = config.sliding_layers[layer];
        const auto head_dim = sliding ? config.local_head_dim : config.global_head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        kv_.push_back(std::make_unique<LayerKvCache>(AttentionConfig{
            config.attention_heads, config.kv_heads[layer],
            head_dim, sliding ? config.sliding_window : 0, scale}));
    }
}

const std::uint8_t*
Gemma4Session::tensor_data(const GgufTensor& tensor) const {
    return weights_.view(tensor.absolute_offset, tensor.bytes);
}

const float* Gemma4Session::f32_data(const GgufTensor& tensor) const {
    if (tensor.type != GgmlType::F32)
        throw std::runtime_error("expected F32 control tensor: " + tensor.name);
    return reinterpret_cast<const float*>(tensor_data(tensor));
}

void Gemma4Session::matvec(const GgufTensor& tensor, const float* input,
                           float* output) const {
    if (tensor.dimensions.size() != 2)
        throw std::runtime_error("generation matrix is not 2D: " + tensor.name);
    ggml_matvec(tensor.type, tensor_data(tensor), tensor.dimensions[1],
                tensor.dimensions[0], input, output);
}

const std::vector<float>& Gemma4Session::evaluate(TokenId token) {
    const auto& embedding = model_.token_embedding();
    const auto& config = model_.config();
    if (token < 0 || static_cast<std::uint32_t>(token) >= config.vocab_size)
        throw std::out_of_range("token outside vocabulary");
    const std::uint64_t row_bytes = embedding.bytes / config.vocab_size;
    const auto* row = weights_.view(
        embedding.absolute_offset + row_bytes * static_cast<std::uint32_t>(token),
        row_bytes);
    std::vector<float> hidden(config.hidden_size);
    ggml_dequantize_row(embedding.type, row, config.hidden_size, hidden.data());
    const float scale = std::sqrt(static_cast<float>(config.hidden_size));
    for (float& value : hidden) value *= scale;
    return forward(hidden.data());
}

const std::vector<float>&
Gemma4Session::evaluate_embedding(const float* embedding) {
    if (!embedding) throw std::invalid_argument("null language embedding");
    std::vector<float> hidden(
        embedding, embedding + model_.config().hidden_size);
    return forward(hidden.data());
}

const std::vector<float>& Gemma4Session::forward(float* hidden) {
    const auto& config = model_.config();
    std::vector<float> normalized(config.hidden_size);
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attended;
    std::vector<float> projected(config.hidden_size);
    std::vector<float> layer_output(config.hidden_size);

    const float* global_factors = nullptr;
    if (model_.rope_frequencies())
        global_factors = f32_data(*model_.rope_frequencies());

    for (std::uint32_t layer_index = 0;
         layer_index < config.layer_count; ++layer_index) {
        const auto& layer = model_.layers()[layer_index];
        const bool sliding = config.sliding_layers[layer_index];
        const std::size_t head_dim =
            sliding ? config.local_head_dim : config.global_head_dim;
        const std::size_t query_values =
            static_cast<std::size_t>(config.attention_heads) * head_dim;
        const std::size_t kv_values =
            static_cast<std::size_t>(config.kv_heads[layer_index]) * head_dim;
        q.resize(query_values);
        k.resize(kv_values);
        v.resize(kv_values);
        attended.resize(query_values);

        rms_norm(hidden, f32_data(*layer.attention_norm), config.hidden_size,
                 config.rms_epsilon, normalized.data());
        matvec(*layer.q, normalized.data(), q.data());
        matvec(*layer.k, normalized.data(), k.data());
        if (layer.v) matvec(*layer.v, normalized.data(), v.data());
        else std::copy(k.begin(), k.end(), v.begin());

        for (std::uint32_t head = 0; head < config.attention_heads; ++head)
            rms_norm(q.data() + static_cast<std::size_t>(head) * head_dim,
                     f32_data(*layer.q_norm), head_dim, config.rms_epsilon,
                     q.data() + static_cast<std::size_t>(head) * head_dim);
        for (std::uint32_t head = 0;
             head < config.kv_heads[layer_index]; ++head) {
            rms_norm(k.data() + static_cast<std::size_t>(head) * head_dim,
                     f32_data(*layer.k_norm), head_dim, config.rms_epsilon,
                     k.data() + static_cast<std::size_t>(head) * head_dim);
            rms_norm(v.data() + static_cast<std::size_t>(head) * head_dim,
                     nullptr, head_dim, config.rms_epsilon,
                     v.data() + static_cast<std::size_t>(head) * head_dim);
        }

        const float rope_base =
            sliding ? config.local_rope_base : config.global_rope_base;
        const float* factors = sliding ? nullptr : global_factors;
        apply_rope(q.data(), config.attention_heads, head_dim, head_dim,
                   position_, rope_base, factors);
        apply_rope(k.data(), config.kv_heads[layer_index], head_dim, head_dim,
                   position_, rope_base, factors);
        kv_[layer_index]->append(position_, k.data(), v.data());
        kv_[layer_index]->attend(q.data(), attended.data());
        matvec(*layer.attention_output, attended.data(), projected.data());
        rms_norm(projected.data(), f32_data(*layer.post_attention_norm),
                 config.hidden_size, config.rms_epsilon, projected.data());
        for (std::size_t i = 0; i < config.hidden_size; ++i)
            layer_output[i] = hidden[i] + projected[i];

        feed_forward_.forward(layer_index, layer_output.data(), hidden);
    }

    rms_norm(hidden, f32_data(model_.output_norm()), config.hidden_size,
             config.rms_epsilon, normalized.data());
    matvec(model_.output(), normalized.data(), logits_.data());
    if (config.final_logit_softcap > 0.0f) {
        for (float& logit : logits_)
            logit = config.final_logit_softcap *
                    std::tanh(logit / config.final_logit_softcap);
    }
    ++position_;
    return logits_;
}

const std::vector<float>&
Gemma4Session::prefill(const std::vector<TokenId>& prompt) {
    if (prompt.empty()) throw std::invalid_argument("prompt is empty");
    for (const auto token : prompt) evaluate(token);
    return logits_;
}

std::vector<TokenId> Gemma4Session::generate(
    const std::vector<TokenId>& prompt, const GenerationConfig& config,
    const std::function<void(TokenId)>& emit) {
    prefill(prompt);
    std::mt19937_64 random(config.sampling.seed);
    std::vector<TokenId> generated;
    generated.reserve(config.max_new_tokens);
    for (std::uint32_t i = 0; i < config.max_new_tokens; ++i) {
        const TokenId token = sample_token(logits_, config.sampling, random);
        generated.push_back(token);
        if (emit) emit(token);
        if (std::find(config.stop_tokens.begin(), config.stop_tokens.end(),
                      token) != config.stop_tokens.end())
            break;
        evaluate(token);
    }
    return generated;
}

void Gemma4Session::reset() {
    for (auto& cache : kv_) cache->clear();
    std::fill(logits_.begin(), logits_.end(), 0.0f);
    position_ = 0;
}

} // namespace gemmaedge
