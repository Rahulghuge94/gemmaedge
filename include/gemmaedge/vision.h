#pragma once

#include "gemmaedge/gemma4_model.h"
#include "gemmaedge/mapped_file.h"

#include <cstdint>
#include <vector>

namespace gemmaedge {

struct RgbImage {
    std::uint32_t width{0};
    std::uint32_t height{0};
    // Interleaved RGB float values in [0, 1].
    std::vector<float> pixels;
};

struct PreparedImage {
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::uint32_t patches_x{0};
    std::uint32_t patches_y{0};
    std::vector<float> pixels;
};

struct VisionOutput {
    std::uint32_t token_count{0};
    std::uint32_t embedding_size{0};
    // Token-major language-space embeddings.
    std::vector<float> embeddings;
};

PreparedImage prepare_gemma4_image(const RgbImage& image,
                                   std::uint32_t patch_size = 16,
                                   std::uint32_t pool_size = 3,
                                   std::uint32_t min_tokens = 40,
                                   std::uint32_t max_tokens = 280);

class Gemma4VisionEncoder {
public:
    Gemma4VisionEncoder(const Gemma4VisionModel& model,
                        const MappedFile& weights)
        : model_(model), weights_(weights) {}

    VisionOutput encode(const RgbImage& image);

private:
    const std::uint8_t* tensor_data(const GgufTensor& tensor) const;
    const float* f32_data(const GgufTensor& tensor) const;
    void matvec(const GgufTensor& tensor, const float* input,
                float* output) const;

    const Gemma4VisionModel& model_;
    const MappedFile& weights_;
};

} // namespace gemmaedge

