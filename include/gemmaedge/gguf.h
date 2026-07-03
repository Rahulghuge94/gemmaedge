#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace gemmaedge {

constexpr std::uint32_t kGgufMagic = 0x46554747; // "GGUF"

enum class GgufValueType : std::uint32_t {
    Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3, Uint32 = 4, Int32 = 5,
    Float32 = 6, Bool = 7, String = 8, Array = 9, Uint64 = 10,
    Int64 = 11, Float64 = 12,
};

enum class GgmlType : std::uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12,
    Q5_K = 13, Q6_K = 14, Q8_K = 15, IQ2_XXS = 16, IQ2_XS = 17,
    IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21,
    IQ2_S = 22, IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26,
    I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30, TQ1_0 = 34,
    TQ2_0 = 35, MXFP4 = 39, NVFP4 = 40, Q1_0 = 41,
};

struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    GgmlType type{GgmlType::F32};
    std::uint64_t relative_offset{0};
    std::uint64_t absolute_offset{0};
    std::uint64_t bytes{0};
};

struct GgufMetadata {
    GgufValueType type{};
    std::string text;
    std::uint64_t unsigned_value{0};
    std::int64_t signed_value{0};
    double float_value{0.0};
    std::vector<std::string> strings;
    std::vector<std::uint64_t> unsigned_values;
    std::vector<std::int64_t> signed_values;
    std::vector<double> float_values;
};

class GgufFile {
public:
    explicit GgufFile(const std::filesystem::path& path,
                      bool require_tensor_payloads = true);

    std::uint32_t version() const noexcept { return version_; }
    std::uint64_t alignment() const noexcept { return alignment_; }
    std::uint64_t data_offset() const noexcept { return data_offset_; }
    const std::vector<GgufTensor>& tensors() const noexcept { return tensors_; }
    const std::unordered_map<std::string, GgufMetadata>& metadata() const noexcept {
        return metadata_;
    }
    const GgufTensor* tensor(const std::string& name) const noexcept;
    const GgufMetadata* meta(const std::string& key) const noexcept;
    std::vector<std::uint8_t> read(const GgufTensor& tensor);

private:
    std::filesystem::path path_;
    std::ifstream stream_;
    std::uint32_t version_{0};
    std::uint64_t alignment_{32};
    std::uint64_t data_offset_{0};
    std::unordered_map<std::string, GgufMetadata> metadata_;
    std::vector<GgufTensor> tensors_;
};

std::uint64_t ggml_tensor_bytes(GgmlType type,
                                const std::vector<std::uint64_t>& dimensions);
const char* ggml_type_name(GgmlType type) noexcept;

} // namespace gemmaedge
