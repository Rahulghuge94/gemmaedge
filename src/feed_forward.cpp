#include "gemmaedge/feed_forward.h"

#include "gemmaedge/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace gemmaedge {

std::vector<RoutedExpert>
route_top_k(const float* hidden, std::size_t hidden_size,
            const float* router_scale, const float* router_matrix,
            std::size_t expert_count, std::size_t experts_used,
            const float* per_expert_scale, float epsilon) {
    if (!hidden || !router_scale || !router_matrix || !per_expert_scale ||
        hidden_size == 0 || expert_count == 0 || experts_used == 0 ||
        experts_used > expert_count)
        throw std::invalid_argument("invalid Gemma router arguments");

    std::vector<float> normalized(hidden_size);
    rms_norm(hidden, nullptr, hidden_size, epsilon, normalized.data());
    const float root_scale = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    for (std::size_t i = 0; i < hidden_size; ++i)
        normalized[i] *= router_scale[i] * root_scale;

    std::vector<float> probabilities(expert_count);
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        double sum = 0.0;
        const float* row = router_matrix + expert * hidden_size;
        for (std::size_t i = 0; i < hidden_size; ++i)
            sum += static_cast<double>(row[i]) * normalized[i];
        probabilities[expert] = static_cast<float>(sum);
    }
    softmax(probabilities.data(), probabilities.size());
    const auto selected = top_k(probabilities.data(), probabilities.size(),
                                experts_used);
    float selected_sum = 0.0f;
    for (const auto& item : selected) selected_sum += item.second;

    std::vector<RoutedExpert> result;
    result.reserve(selected.size());
    for (const auto& item : selected) {
        const float normalized_weight = item.second / selected_sum;
        result.push_back(
            {item.first, item.second,
             normalized_weight * per_expert_scale[item.first]});
    }
    return result;
}

Gemma4FeedForward::Gemma4FeedForward(
    const Gemma4Model& model, const MappedFile& weights,
    std::uint64_t expert_cache_bytes)
    : model_(model), weights_(weights),
      expert_cache_(expert_cache_bytes,
                    [this](const ExpertKey& key) {
                        return load_expert(key);
                    }) {}

const std::uint8_t*
Gemma4FeedForward::tensor_data(const GgufTensor& tensor) const {
    return weights_.view(tensor.absolute_offset, tensor.bytes);
}

const float*
Gemma4FeedForward::f32_data(const GgufTensor& tensor) const {
    if (tensor.type != GgmlType::F32)
        throw std::runtime_error("control tensor is not F32: " + tensor.name);
    return reinterpret_cast<const float*>(tensor_data(tensor));
}

void Gemma4FeedForward::matvec(const GgufTensor& tensor,
                              const float* input, float* output) const {
    if (tensor.dimensions.size() != 2)
        throw std::runtime_error("matvec tensor is not 2D: " + tensor.name);
    ggml_matvec(tensor.type, tensor_data(tensor),
                static_cast<std::size_t>(tensor.dimensions[1]),
                static_cast<std::size_t>(tensor.dimensions[0]),
                input, output);
}

ExpertCache::Bytes
Gemma4FeedForward::load_expert(const ExpertKey& key) const {
    if (key.layer >= model_.layers().size())
        throw std::out_of_range("expert layer outside model");
    const auto& layer = model_.layers()[key.layer];
    const GgufTensor* tensor = nullptr;
    if (key.role == TensorRole::ExpertGateUp)
        tensor = layer.expert_gate_up;
    else if (key.role == TensorRole::ExpertDown)
        tensor = layer.expert_down;
    else
        throw std::invalid_argument("unsupported expert tensor role");
    if (!tensor || tensor->dimensions.size() != 3 ||
        key.expert >= tensor->dimensions[2])
        throw std::out_of_range("expert index outside tensor");

    const std::uint64_t slice_bytes = tensor->bytes / tensor->dimensions[2];
    const auto* source = weights_.view(
        tensor->absolute_offset + slice_bytes * key.expert, slice_bytes);
    return ExpertCache::Bytes(source, source + slice_bytes);
}

void Gemma4FeedForward::forward(
    std::uint32_t layer_index, const float* input, float* output,
    std::vector<RoutedExpert>* routing) {
    if (!input || !output || layer_index >= model_.layers().size())
        throw std::invalid_argument("invalid feed-forward invocation");
    const auto& config = model_.config();
    const auto& layer = model_.layers()[layer_index];
    const std::size_t hidden = config.hidden_size;
    const std::size_t dense_size = config.dense_intermediate;
    const std::size_t expert_size = config.expert_intermediate;

    std::vector<float> dense_input(hidden);
    rms_norm(input, f32_data(*layer.ffn_norm), hidden, config.rms_epsilon,
             dense_input.data());
    std::vector<float> dense_gate(dense_size);
    std::vector<float> dense_up(dense_size);
    matvec(*layer.dense_gate, dense_input.data(), dense_gate.data());
    matvec(*layer.dense_up, dense_input.data(), dense_up.data());
    for (std::size_t i = 0; i < dense_size; ++i)
        dense_gate[i] = gelu_tanh(dense_gate[i]) * dense_up[i];
    std::vector<float> dense_output(hidden);
    matvec(*layer.dense_down, dense_gate.data(), dense_output.data());
    rms_norm(dense_output.data(), f32_data(*layer.post_dense_norm), hidden,
             config.rms_epsilon, dense_output.data());

    const auto selected = route_top_k(
        input, hidden, f32_data(*layer.router_scale),
        f32_data(*layer.router), config.expert_count, config.experts_used,
        f32_data(*layer.expert_output_scale), config.rms_epsilon);
    if (routing) *routing = selected;

    std::vector<float> expert_input(hidden);
    rms_norm(input, f32_data(*layer.pre_expert_norm), hidden,
             config.rms_epsilon, expert_input.data());
    std::vector<float> expert_sum(hidden, 0.0f);
    std::vector<float> gate_up(expert_size * 2);
    std::vector<float> activated(expert_size);
    std::vector<float> expert_output(hidden);

    for (const auto& selected_expert : selected) {
        const ExpertKey gate_key{layer_index, selected_expert.expert,
                                 TensorRole::ExpertGateUp};
        const ExpertKey down_key{layer_index, selected_expert.expert,
                                 TensorRole::ExpertDown};
        const auto gate_bytes =
            expert_cache_.get(gate_key, selected_expert.probability);
        const auto down_bytes =
            expert_cache_.get(down_key, selected_expert.probability);

        ggml_matvec(layer.expert_gate_up->type, gate_bytes->data(),
                    expert_size * 2, hidden, expert_input.data(),
                    gate_up.data());
        for (std::size_t i = 0; i < expert_size; ++i)
            activated[i] = gelu_tanh(gate_up[i]) *
                           gate_up[expert_size + i];
        ggml_matvec(layer.expert_down->type, down_bytes->data(), hidden,
                    expert_size, activated.data(), expert_output.data());
        for (std::size_t i = 0; i < hidden; ++i)
            expert_sum[i] += selected_expert.weight * expert_output[i];
    }
    rms_norm(expert_sum.data(), f32_data(*layer.post_expert_norm), hidden,
             config.rms_epsilon, expert_sum.data());

    std::vector<float> combined(hidden);
    for (std::size_t i = 0; i < hidden; ++i)
        combined[i] = dense_output[i] + expert_sum[i];
    rms_norm(combined.data(), f32_data(*layer.post_ffn_norm), hidden,
             config.rms_epsilon, combined.data());

    float layer_scale = 1.0f;
    if (layer.layer_scale)
        layer_scale = f32_data(*layer.layer_scale)[0];
    for (std::size_t i = 0; i < hidden; ++i)
        output[i] = (input[i] + combined[i]) * layer_scale;
}

} // namespace gemmaedge

