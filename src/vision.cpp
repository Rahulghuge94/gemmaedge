#include "gemmaedge/vision.h"

#include "gemmaedge/attention.h"
#include "gemmaedge/tensor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gemmaedge {
namespace {

float bilinear_sample(const RgbImage& image, float x, float y,
                      std::uint32_t channel) {
    x = std::max(0.0f, std::min(x, static_cast<float>(image.width - 1)));
    y = std::max(0.0f, std::min(y, static_cast<float>(image.height - 1)));
    const auto x0 = static_cast<std::uint32_t>(std::floor(x));
    const auto y0 = static_cast<std::uint32_t>(std::floor(y));
    const auto x1 = std::min(x0 + 1, image.width - 1);
    const auto y1 = std::min(y0 + 1, image.height - 1);
    const float fx = x - x0;
    const float fy = y - y0;
    const auto pixel = [&](std::uint32_t px, std::uint32_t py) {
        return image.pixels[(static_cast<std::size_t>(py) * image.width + px) *
                            3 + channel];
    };
    const float top = pixel(x0, y0) * (1.0f - fx) + pixel(x1, y0) * fx;
    const float bottom = pixel(x0, y1) * (1.0f - fx) + pixel(x1, y1) * fx;
    return top * (1.0f - fy) + bottom * fy;
}

void apply_vision_rope(float* states, std::uint32_t heads,
                       std::uint32_t head_dim, std::uint32_t x,
                       std::uint32_t y) {
    const std::size_t spatial = head_dim / 2;
    for (std::uint32_t head = 0; head < heads; ++head) {
        float* row = states + static_cast<std::size_t>(head) * head_dim;
        apply_rope(row, 1, spatial, spatial, x, 100.0f);
        apply_rope(row + spatial, 1, spatial, spatial, y, 100.0f);
    }
}

} // namespace

PreparedImage prepare_gemma4_image(
    const RgbImage& image, std::uint32_t patch_size, std::uint32_t pool_size,
    std::uint32_t min_tokens, std::uint32_t max_tokens) {
    if (image.width == 0 || image.height == 0 ||
        image.pixels.size() !=
            static_cast<std::size_t>(image.width) * image.height * 3 ||
        patch_size == 0 || pool_size == 0 || min_tokens == 0 ||
        max_tokens < min_tokens)
        throw std::invalid_argument("invalid Gemma vision image");

    const float aspect = static_cast<float>(image.width) / image.height;
    const float native_tokens =
        static_cast<float>(image.width) * image.height /
        static_cast<float>(patch_size * pool_size * patch_size * pool_size);
    const auto target = static_cast<std::uint32_t>(std::max(
        static_cast<float>(min_tokens),
        std::min(static_cast<float>(max_tokens), std::round(native_tokens))));

    std::uint32_t pooled_x = 1;
    std::uint32_t pooled_y = min_tokens;
    float best_score = std::numeric_limits<float>::infinity();
    for (std::uint32_t y = 1; y <= max_tokens; ++y) {
        for (std::uint32_t x = 1; x <= max_tokens / y; ++x) {
            const auto count = x * y;
            if (count < min_tokens || count > max_tokens) continue;
            const float aspect_error =
                std::abs(std::log((static_cast<float>(x) / y) / aspect));
            const float token_error =
                std::abs(static_cast<float>(count) - target) /
                static_cast<float>(max_tokens);
            const float score = aspect_error + token_error * 0.25f;
            if (score < best_score) {
                best_score = score;
                pooled_x = x;
                pooled_y = y;
            }
        }
    }

    PreparedImage result;
    result.patches_x = pooled_x * pool_size;
    result.patches_y = pooled_y * pool_size;
    result.width = result.patches_x * patch_size;
    result.height = result.patches_y * patch_size;
    result.pixels.resize(
        static_cast<std::size_t>(result.width) * result.height * 3);
    const float sx = static_cast<float>(image.width) / result.width;
    const float sy = static_cast<float>(image.height) / result.height;
    for (std::uint32_t y = 0; y < result.height; ++y) {
        for (std::uint32_t x = 0; x < result.width; ++x) {
            const float source_x = (x + 0.5f) * sx - 0.5f;
            const float source_y = (y + 0.5f) * sy - 0.5f;
            for (std::uint32_t c = 0; c < 3; ++c) {
                result.pixels[
                    (static_cast<std::size_t>(y) * result.width + x) * 3 + c] =
                    bilinear_sample(image, source_x, source_y, c) * 2.0f - 1.0f;
            }
        }
    }
    return result;
}

const std::uint8_t*
Gemma4VisionEncoder::tensor_data(const GgufTensor& tensor) const {
    return weights_.view(tensor.absolute_offset, tensor.bytes);
}

const float*
Gemma4VisionEncoder::f32_data(const GgufTensor& tensor) const {
    if (tensor.type != GgmlType::F32)
        throw std::runtime_error("vision control tensor is not F32");
    return reinterpret_cast<const float*>(tensor_data(tensor));
}

void Gemma4VisionEncoder::matvec(
    const GgufTensor& tensor, const float* input, float* output) const {
    if (tensor.dimensions.size() != 2)
        throw std::runtime_error("vision matrix is not 2D");
    ggml_matvec(tensor.type, tensor_data(tensor), tensor.dimensions[1],
                tensor.dimensions[0], input, output);
}

VisionOutput Gemma4VisionEncoder::encode(const RgbImage& image) {
    const auto& config = model_.config();
    const auto prepared =
        prepare_gemma4_image(image, config.patch_size, 3, 40, 280);
    const std::size_t tokens =
        static_cast<std::size_t>(prepared.patches_x) * prepared.patches_y;
    const std::size_t hidden = config.hidden_size;
    const std::size_t heads = config.attention_heads;
    const std::size_t head_dim = hidden / heads;

    const auto& patch_weight = model_.patch_embedding();
    const std::size_t patch_values =
        static_cast<std::size_t>(config.patch_size) * config.patch_size * 3;
    const float* patch_matrix = f32_data(patch_weight);
    std::vector<float> states(tokens * hidden);
    std::vector<float> patch(patch_values);
    for (std::uint32_t py = 0; py < prepared.patches_y; ++py) {
        for (std::uint32_t px = 0; px < prepared.patches_x; ++px) {
            std::size_t at = 0;
            // GGUF convolution order: x, y, channel.
            for (std::uint32_t c = 0; c < 3; ++c)
                for (std::uint32_t y = 0; y < config.patch_size; ++y)
                    for (std::uint32_t x = 0; x < config.patch_size; ++x)
                        patch[at++] = prepared.pixels[
                            (static_cast<std::size_t>(
                                 py * config.patch_size + y) *
                                 prepared.width +
                             px * config.patch_size + x) *
                                3 +
                            c];
            const std::size_t token =
                static_cast<std::size_t>(py) * prepared.patches_x + px;
            ggml_matvec(GgmlType::F32, patch_matrix, hidden, patch_values,
                        patch.data(), states.data() + token * hidden);
        }
    }

    const auto& position = model_.position_embedding();
    const float* position_data = f32_data(position);
    const std::size_t position_count = position.dimensions[1];
    const float* x_table = position_data;
    const float* y_table = position_data + position_count * hidden;
    if (prepared.patches_x > position_count ||
        prepared.patches_y > position_count)
        throw std::runtime_error("image grid exceeds position table");
    for (std::uint32_t py = 0; py < prepared.patches_y; ++py) {
        for (std::uint32_t px = 0; px < prepared.patches_x; ++px) {
            float* token = states.data() +
                (static_cast<std::size_t>(py) * prepared.patches_x + px) *
                    hidden;
            const float* x_embedding = x_table + px * hidden;
            const float* y_embedding = y_table + py * hidden;
            for (std::size_t i = 0; i < hidden; ++i)
                token[i] += x_embedding[i] + y_embedding[i];
        }
    }

    std::vector<float> normalized(hidden);
    std::vector<float> q(tokens * hidden);
    std::vector<float> k(tokens * hidden);
    std::vector<float> v(tokens * hidden);
    std::vector<float> attended(tokens * hidden);
    std::vector<float> projected(hidden);
    std::vector<float> ffn_gate(config.intermediate_size);
    std::vector<float> ffn_up(config.intermediate_size);
    std::vector<float> ffn_down(hidden);

    for (std::uint32_t layer_index = 0;
         layer_index < config.layer_count; ++layer_index) {
        const auto& layer = model_.layers()[layer_index];
        for (std::size_t token = 0; token < tokens; ++token) {
            const float* input = states.data() + token * hidden;
            rms_norm(input, f32_data(*layer.input_norm), hidden,
                     config.norm_epsilon, normalized.data());
            matvec(*layer.q, normalized.data(), q.data() + token * hidden);
            matvec(*layer.k, normalized.data(), k.data() + token * hidden);
            matvec(*layer.v, normalized.data(), v.data() + token * hidden);
            for (std::size_t head = 0; head < heads; ++head) {
                rms_norm(q.data() + token * hidden + head * head_dim,
                         f32_data(*layer.q_norm), head_dim,
                         config.norm_epsilon,
                         q.data() + token * hidden + head * head_dim);
                rms_norm(k.data() + token * hidden + head * head_dim,
                         f32_data(*layer.k_norm), head_dim,
                         config.norm_epsilon,
                         k.data() + token * hidden + head * head_dim);
                rms_norm(v.data() + token * hidden + head * head_dim,
                         nullptr, head_dim, config.norm_epsilon,
                         v.data() + token * hidden + head * head_dim);
            }
            const auto x = static_cast<std::uint32_t>(
                token % prepared.patches_x);
            const auto y = static_cast<std::uint32_t>(
                token / prepared.patches_x);
            apply_vision_rope(q.data() + token * hidden, heads, head_dim, x, y);
            apply_vision_rope(k.data() + token * hidden, heads, head_dim, x, y);
        }

        LayerKvCache full_attention({
            static_cast<std::uint32_t>(heads),
            static_cast<std::uint32_t>(heads),
            static_cast<std::uint32_t>(head_dim), 0,
            1.0f});
        for (std::size_t token = 0; token < tokens; ++token)
            full_attention.append(token, k.data() + token * hidden,
                                  v.data() + token * hidden);
        for (std::size_t token = 0; token < tokens; ++token) {
            full_attention.attend(q.data() + token * hidden,
                                  attended.data() + token * hidden);
            matvec(*layer.output, attended.data() + token * hidden,
                   projected.data());
            rms_norm(projected.data(),
                     f32_data(*layer.attention_post_norm), hidden,
                     config.norm_epsilon, projected.data());
            float* state = states.data() + token * hidden;
            for (std::size_t i = 0; i < hidden; ++i)
                state[i] += projected[i];

            rms_norm(state, f32_data(*layer.ffn_norm), hidden,
                     config.norm_epsilon, normalized.data());
            matvec(*layer.gate, normalized.data(), ffn_gate.data());
            matvec(*layer.up, normalized.data(), ffn_up.data());
            for (std::size_t i = 0; i < config.intermediate_size; ++i)
                ffn_gate[i] = gelu_tanh(ffn_gate[i]) * ffn_up[i];
            matvec(*layer.down, ffn_gate.data(), ffn_down.data());
            rms_norm(ffn_down.data(), f32_data(*layer.ffn_post_norm), hidden,
                     config.norm_epsilon, ffn_down.data());
            for (std::size_t i = 0; i < hidden; ++i)
                state[i] += ffn_down[i];
        }
    }

    constexpr std::uint32_t pool = 3;
    const std::uint32_t out_x = prepared.patches_x / pool;
    const std::uint32_t out_y = prepared.patches_y / pool;
    const std::size_t output_tokens =
        static_cast<std::size_t>(out_x) * out_y;
    std::vector<float> pooled(output_tokens * hidden, 0.0f);
    const float pool_scale = std::sqrt(static_cast<float>(hidden)) / 9.0f;
    for (std::uint32_t oy = 0; oy < out_y; ++oy) {
        for (std::uint32_t ox = 0; ox < out_x; ++ox) {
            float* destination = pooled.data() +
                (static_cast<std::size_t>(oy) * out_x + ox) * hidden;
            for (std::uint32_t dy = 0; dy < pool; ++dy)
                for (std::uint32_t dx = 0; dx < pool; ++dx) {
                    const float* source = states.data() +
                        (static_cast<std::size_t>(oy * pool + dy) *
                             prepared.patches_x +
                         ox * pool + dx) *
                            hidden;
                    for (std::size_t i = 0; i < hidden; ++i)
                        destination[i] += source[i] * pool_scale;
                }
        }
    }

    const float* bias = f32_data(model_.standardize_bias());
    const float* scale = f32_data(model_.standardize_scale());
    VisionOutput result;
    result.token_count = static_cast<std::uint32_t>(output_tokens);
    result.embedding_size = config.projection_size;
    result.embeddings.resize(output_tokens * config.projection_size);
    for (std::size_t token = 0; token < output_tokens; ++token) {
        float* row = pooled.data() + token * hidden;
        for (std::size_t i = 0; i < hidden; ++i)
            row[i] = (row[i] - bias[i]) * scale[i];
        rms_norm(row, nullptr, hidden, config.norm_epsilon, normalized.data());
        matvec(model_.projector(), normalized.data(),
               result.embeddings.data() + token * config.projection_size);
    }
    return result;
}

} // namespace gemmaedge
