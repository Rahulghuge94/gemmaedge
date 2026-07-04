#include "gemmaedge/feed_forward.h"

#include "gemmaedge/generation.h"
#include "gemmaedge/tensor.h"
#include "gemmaedge/cuda_backend.h"

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
                        ExpertView view = load_expert(key);
                        if (is_cuda_available()) {
                            cuda_register_weight(view.data, view.size);
                        }
                        return view;
                    },
                    [this](const ExpertKey& key, const ExpertView& view) {
                        (void)key;
                        if (is_cuda_available()) {
                            cuda_unregister_weight(view.data);
                        }
                        weights_.advise_dont_need(view.offset, view.size);
                    }) {
    if (is_cuda_available() && expert_cache_bytes >= 600 * 1024 * 1024ULL) {
        preload_all_experts();
    }
}

void Gemma4FeedForward::preload_all_experts() {
    const auto& config = model_.config();
    for (std::uint32_t layer = 0; layer < model_.layers().size(); ++layer) {
        for (std::uint32_t expert = 0; expert < config.expert_count; ++expert) {
            ExpertKey gate_key{layer, expert, TensorRole::ExpertGateUp};
            ExpertKey down_key{layer, expert, TensorRole::ExpertDown};
            expert_cache_.get(gate_key, 0.0f);
            expert_cache_.get(down_key, 0.0f);
        }
    }
}

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

void Gemma4FeedForward::upload_input(const float* input, std::size_t count) const {
    cuda_upload_vector(input, count);
}

void Gemma4FeedForward::matvec_device_vec(const GgufTensor& tensor, const float* input, float* output) const {
    if (tensor.dimensions.size() != 2)
        throw std::runtime_error("matvec tensor is not 2D: " + tensor.name);
    if (is_cuda_available() &&
        cuda_matvec_device_vec(tensor.type, tensor_data(tensor),
                               static_cast<std::size_t>(tensor.dimensions[1]),
                               static_cast<std::size_t>(tensor.dimensions[0]),
                               output)) {
        return;
    }
    // Fallback to CPU/standard GPU path
    matvec(tensor, input, output);
}

ExpertView
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
    const std::uint64_t absolute_offset = tensor->absolute_offset + slice_bytes * key.expert;
    weights_.advise_will_need(absolute_offset, slice_bytes);
    const auto* source = weights_.view(absolute_offset, slice_bytes);
    return {source, slice_bytes, absolute_offset};
}

// Original forward() — allocates scratch vectors per call (backward compat).
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

        ggml_matvec(layer.expert_gate_up->type, gate_bytes.data,
                    expert_size * 2, hidden, expert_input.data(),
                    gate_up.data());
        for (std::size_t i = 0; i < expert_size; ++i)
            activated[i] = gelu_tanh(gate_up[i]) *
                           gate_up[expert_size + i];
        ggml_matvec(layer.expert_down->type, down_bytes.data, hidden,
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

// Scratch-based forward() — uses pre-allocated ScratchArena buffers.
void Gemma4FeedForward::forward(
    std::uint32_t layer_index, const float* input, float* output,
    ScratchArena& s, std::vector<RoutedExpert>* routing) {
    if (!input || !output || layer_index >= model_.layers().size())
        throw std::invalid_argument("invalid feed-forward invocation");
    const auto& config = model_.config();
    const auto& layer = model_.layers()[layer_index];
    const std::size_t hidden = config.hidden_size;
    const std::size_t dense_size = config.dense_intermediate;
    const std::size_t expert_size = config.expert_intermediate;

    if (is_cuda_available()) {
        const auto selected = route_top_k(
            input, hidden, f32_data(*layer.router_scale),
            f32_data(*layer.router), config.expert_count, config.experts_used,
            f32_data(*layer.expert_output_scale), config.rms_epsilon);
        if (routing) *routing = selected;

        // Prefetch all selected expert weights in parallel from OS page cache
        for (const auto& selected_expert : selected) {
            const ExpertKey gate_key{layer_index, selected_expert.expert, TensorRole::ExpertGateUp};
            const ExpertKey down_key{layer_index, selected_expert.expert, TensorRole::ExpertDown};
            if (!expert_cache_.contains(gate_key)) {
                const auto& l = model_.layers()[layer_index];
                const GgufTensor* tensor = l.expert_gate_up;
                const std::uint64_t slice_bytes = tensor->bytes / tensor->dimensions[2];
                const std::uint64_t absolute_offset = tensor->absolute_offset + slice_bytes * selected_expert.expert;
                weights_.advise_will_need(absolute_offset, slice_bytes);
            }
            if (!expert_cache_.contains(down_key)) {
                const auto& l = model_.layers()[layer_index];
                const GgufTensor* tensor = l.expert_down;
                const std::uint64_t slice_bytes = tensor->bytes / tensor->dimensions[2];
                const std::uint64_t absolute_offset = tensor->absolute_offset + slice_bytes * selected_expert.expert;
                weights_.advise_will_need(absolute_offset, slice_bytes);
            }
        }

        const std::size_t num_experts = selected.size();
        std::vector<CudaMatvecStep> expert_gate_steps;
        std::vector<CudaMatvecStep> expert_down_steps;
        expert_gate_steps.reserve(num_experts);
        expert_down_steps.reserve(num_experts);
        
        std::vector<float> host_expert_weights;
        host_expert_weights.reserve(num_experts);

        for (const auto& selected_expert : selected) {
            const ExpertKey gate_key{layer_index, selected_expert.expert,
                                     TensorRole::ExpertGateUp};
            const ExpertKey down_key{layer_index, selected_expert.expert,
                                     TensorRole::ExpertDown};
            const auto gate_bytes =
                expert_cache_.get(gate_key, selected_expert.probability);
            const auto down_bytes =
                expert_cache_.get(down_key, selected_expert.probability);

            expert_gate_steps.push_back({layer.expert_gate_up->type, gate_bytes.data, 0, expert_size * 2, hidden});
            expert_down_steps.push_back({layer.expert_down->type, down_bytes.data, 0, hidden, expert_size});
            host_expert_weights.push_back(selected_expert.weight);
        }

        cuda_ffn_execute(input,
                         layer.dense_gate->type, tensor_data(*layer.dense_gate),
                         layer.dense_up->type, tensor_data(*layer.dense_up),
                         layer.dense_down->type, tensor_data(*layer.dense_down),
                         f32_data(*layer.ffn_norm),
                         f32_data(*layer.post_dense_norm),
                         f32_data(*layer.pre_expert_norm),
                         f32_data(*layer.post_expert_norm),
                         expert_gate_steps,
                         expert_down_steps,
                         host_expert_weights.data(),
                         s.combined.data(),
                         hidden,
                         dense_size,
                         expert_size,
                         num_experts,
                         config.rms_epsilon);
    } else {
        rms_norm(input, f32_data(*layer.ffn_norm), hidden, config.rms_epsilon,
                 s.dense_input.data());
        matvec(*layer.dense_gate, s.dense_input.data(), s.dense_gate.data());
        matvec(*layer.dense_up, s.dense_input.data(), s.dense_up.data());
        for (std::size_t i = 0; i < dense_size; ++i)
            s.dense_gate[i] = gelu_tanh(s.dense_gate[i]) * s.dense_up[i];
        matvec(*layer.dense_down, s.dense_gate.data(), s.dense_output.data());
        rms_norm(s.dense_output.data(), f32_data(*layer.post_dense_norm), hidden,
                 config.rms_epsilon, s.dense_output.data());

        const auto selected = route_top_k(
            input, hidden, f32_data(*layer.router_scale),
            f32_data(*layer.router), config.expert_count, config.experts_used,
            f32_data(*layer.expert_output_scale), config.rms_epsilon);
        if (routing) *routing = selected;

        rms_norm(input, f32_data(*layer.pre_expert_norm), hidden,
                 config.rms_epsilon, s.expert_input.data());
                 
        std::fill(s.expert_sum.begin(),
                  s.expert_sum.begin() + static_cast<std::ptrdiff_t>(hidden), 0.0f);

        for (const auto& selected_expert : selected) {
            const ExpertKey gate_key{layer_index, selected_expert.expert,
                                     TensorRole::ExpertGateUp};
            const ExpertKey down_key{layer_index, selected_expert.expert,
                                     TensorRole::ExpertDown};
            const auto gate_bytes =
                expert_cache_.get(gate_key, selected_expert.probability);
            const auto down_bytes =
                expert_cache_.get(down_key, selected_expert.probability);

            ggml_matvec(layer.expert_gate_up->type, gate_bytes.data,
                        expert_size * 2, hidden, s.expert_input.data(),
                        s.gate_up.data());

            for (std::size_t i = 0; i < expert_size; ++i)
                s.activated[i] = gelu_tanh(s.gate_up[i]) *
                                 s.gate_up[expert_size + i];

            ggml_matvec(layer.expert_down->type, down_bytes.data, hidden,
                        expert_size, s.activated.data(), s.expert_output.data());

            for (std::size_t i = 0; i < hidden; ++i)
                s.expert_sum[i] += selected_expert.weight * s.expert_output[i];
        }
        rms_norm(s.expert_sum.data(), f32_data(*layer.post_expert_norm), hidden,
                 config.rms_epsilon, s.expert_sum.data());

        for (std::size_t i = 0; i < hidden; ++i)
            s.combined[i] = s.dense_output[i] + s.expert_sum[i];
    }
    rms_norm(s.combined.data(), f32_data(*layer.post_ffn_norm), hidden,
             config.rms_epsilon, s.combined.data());

    float layer_scale = 1.0f;
    if (layer.layer_scale)
        layer_scale = f32_data(*layer.layer_scale)[0];
    for (std::size_t i = 0; i < hidden; ++i)
        output[i] = (input[i] + s.combined[i]) * layer_scale;
}

} // namespace gemmaedge
