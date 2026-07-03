#include "gemmaedge/gguf.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace gemmaedge {
namespace {

template <typename T>
T read_scalar(std::istream& input) {
    static_assert(std::is_trivially_copyable<T>::value, "scalar required");
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated GGUF");
    return value;
}

std::string read_string(std::istream& input) {
    const auto size = read_scalar<std::uint64_t>(input);
    if (size > (1ULL << 30)) throw std::runtime_error("unreasonable GGUF string");
    std::string result(static_cast<std::size_t>(size), '\0');
    input.read(result.data(), static_cast<std::streamsize>(size));
    if (!input) throw std::runtime_error("truncated GGUF string");
    return result;
}

std::uint64_t scalar_size(GgufValueType type) {
    switch (type) {
        case GgufValueType::Uint8:
        case GgufValueType::Int8:
        case GgufValueType::Bool: return 1;
        case GgufValueType::Uint16:
        case GgufValueType::Int16: return 2;
        case GgufValueType::Uint32:
        case GgufValueType::Int32:
        case GgufValueType::Float32: return 4;
        case GgufValueType::Uint64:
        case GgufValueType::Int64:
        case GgufValueType::Float64: return 8;
        default: return 0;
    }
}

GgufMetadata read_value(std::istream& input, GgufValueType type) {
    GgufMetadata value;
    value.type = type;
    switch (type) {
        case GgufValueType::Uint8:
            value.unsigned_value = read_scalar<std::uint8_t>(input); break;
        case GgufValueType::Int8:
            value.signed_value = read_scalar<std::int8_t>(input); break;
        case GgufValueType::Uint16:
            value.unsigned_value = read_scalar<std::uint16_t>(input); break;
        case GgufValueType::Int16:
            value.signed_value = read_scalar<std::int16_t>(input); break;
        case GgufValueType::Uint32:
            value.unsigned_value = read_scalar<std::uint32_t>(input); break;
        case GgufValueType::Int32:
            value.signed_value = read_scalar<std::int32_t>(input); break;
        case GgufValueType::Uint64:
            value.unsigned_value = read_scalar<std::uint64_t>(input); break;
        case GgufValueType::Int64:
            value.signed_value = read_scalar<std::int64_t>(input); break;
        case GgufValueType::Float32:
            value.float_value = read_scalar<float>(input); break;
        case GgufValueType::Float64:
            value.float_value = read_scalar<double>(input); break;
        case GgufValueType::Bool:
            value.unsigned_value = read_scalar<std::uint8_t>(input) != 0; break;
        case GgufValueType::String:
            value.text = read_string(input); break;
        case GgufValueType::Array: {
            const auto element_type =
                static_cast<GgufValueType>(read_scalar<std::uint32_t>(input));
            const auto count = read_scalar<std::uint64_t>(input);
            if (count > (1ULL << 32))
                throw std::runtime_error("unreasonable GGUF array");
            if (element_type == GgufValueType::String) {
                value.strings.reserve(static_cast<std::size_t>(count));
                for (std::uint64_t i = 0; i < count; ++i)
                    value.strings.push_back(read_string(input));
            } else {
                value.unsigned_values.reserve(static_cast<std::size_t>(count));
                value.signed_values.reserve(static_cast<std::size_t>(count));
                value.float_values.reserve(static_cast<std::size_t>(count));
                for (std::uint64_t i = 0; i < count; ++i) {
                    switch (element_type) {
                        case GgufValueType::Uint8:
                            value.unsigned_values.push_back(read_scalar<std::uint8_t>(input)); break;
                        case GgufValueType::Uint16:
                            value.unsigned_values.push_back(read_scalar<std::uint16_t>(input)); break;
                        case GgufValueType::Uint32:
                            value.unsigned_values.push_back(read_scalar<std::uint32_t>(input)); break;
                        case GgufValueType::Uint64:
                            value.unsigned_values.push_back(read_scalar<std::uint64_t>(input)); break;
                        case GgufValueType::Bool:
                            value.unsigned_values.push_back(read_scalar<std::uint8_t>(input) != 0); break;
                        case GgufValueType::Int8:
                            value.signed_values.push_back(read_scalar<std::int8_t>(input)); break;
                        case GgufValueType::Int16:
                            value.signed_values.push_back(read_scalar<std::int16_t>(input)); break;
                        case GgufValueType::Int32:
                            value.signed_values.push_back(read_scalar<std::int32_t>(input)); break;
                        case GgufValueType::Int64:
                            value.signed_values.push_back(read_scalar<std::int64_t>(input)); break;
                        case GgufValueType::Float32:
                            value.float_values.push_back(read_scalar<float>(input)); break;
                        case GgufValueType::Float64:
                            value.float_values.push_back(read_scalar<double>(input)); break;
                        default:
                            throw std::runtime_error("nested/unknown GGUF array");
                    }
                }
            }
            break;
        }
        default:
            throw std::runtime_error("unknown GGUF metadata type");
    }
    return value;
}

struct TypeLayout {
    std::uint64_t block;
    std::uint64_t bytes;
    const char* name;
};

TypeLayout layout(GgmlType type) {
    switch (type) {
        case GgmlType::F32: return {1, 4, "F32"};
        case GgmlType::F16: return {1, 2, "F16"};
        case GgmlType::BF16: return {1, 2, "BF16"};
        case GgmlType::I8: return {1, 1, "I8"};
        case GgmlType::I16: return {1, 2, "I16"};
        case GgmlType::I32: return {1, 4, "I32"};
        case GgmlType::I64: return {1, 8, "I64"};
        case GgmlType::F64: return {1, 8, "F64"};
        case GgmlType::Q4_0: return {32, 18, "Q4_0"};
        case GgmlType::Q4_1: return {32, 20, "Q4_1"};
        case GgmlType::Q5_0: return {32, 22, "Q5_0"};
        case GgmlType::Q5_1: return {32, 24, "Q5_1"};
        case GgmlType::Q8_0: return {32, 34, "Q8_0"};
        case GgmlType::Q8_1: return {32, 40, "Q8_1"};
        case GgmlType::Q2_K: return {256, 84, "Q2_K"};
        case GgmlType::Q3_K: return {256, 110, "Q3_K"};
        case GgmlType::Q4_K: return {256, 144, "Q4_K"};
        case GgmlType::Q5_K: return {256, 176, "Q5_K"};
        case GgmlType::Q6_K: return {256, 210, "Q6_K"};
        case GgmlType::Q8_K: return {256, 292, "Q8_K"};
        case GgmlType::IQ4_NL: return {32, 18, "IQ4_NL"};
        case GgmlType::IQ4_XS: return {256, 136, "IQ4_XS"};
        case GgmlType::MXFP4: return {32, 17, "MXFP4"};
        default: return {0, 0, "unsupported"};
    }
}

} // namespace

std::uint64_t ggml_tensor_bytes(
    GgmlType type, const std::vector<std::uint64_t>& dimensions) {
    if (dimensions.empty()) return 0;
    const auto info = layout(type);
    if (info.block == 0) throw std::runtime_error("unsupported GGML type");
    if (dimensions[0] % info.block != 0)
        throw std::runtime_error("GGML row is not block aligned");
    std::uint64_t rows = 1;
    for (std::size_t i = 1; i < dimensions.size(); ++i) {
        if (dimensions[i] != 0 &&
            rows > std::numeric_limits<std::uint64_t>::max() / dimensions[i])
            throw std::runtime_error("GGUF tensor size overflow");
        rows *= dimensions[i];
    }
    return (dimensions[0] / info.block) * info.bytes * rows;
}

const char* ggml_type_name(GgmlType type) noexcept {
    return layout(type).name;
}

GgufFile::GgufFile(const std::filesystem::path& path,
                   bool require_tensor_payloads)
    : path_(path), stream_(path, std::ios::binary) {
    if (!stream_) throw std::runtime_error("cannot open GGUF: " + path.string());
    if (read_scalar<std::uint32_t>(stream_) != kGgufMagic)
        throw std::runtime_error("not a GGUF file");
    version_ = read_scalar<std::uint32_t>(stream_);
    if (version_ < 2 || version_ > 3)
        throw std::runtime_error("unsupported GGUF version");
    const auto tensor_count = read_scalar<std::uint64_t>(stream_);
    const auto metadata_count = read_scalar<std::uint64_t>(stream_);
    if (tensor_count > (1ULL << 24) || metadata_count > (1ULL << 24))
        throw std::runtime_error("unreasonable GGUF counts");

    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        auto key = read_string(stream_);
        const auto type =
            static_cast<GgufValueType>(read_scalar<std::uint32_t>(stream_));
        metadata_.emplace(std::move(key), read_value(stream_, type));
    }
    if (const auto* item = meta("general.alignment"))
        alignment_ = item->unsigned_value;
    if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0)
        throw std::runtime_error("invalid GGUF alignment");

    tensors_.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensor tensor;
        tensor.name = read_string(stream_);
        const auto rank = read_scalar<std::uint32_t>(stream_);
        if (rank == 0 || rank > 4) throw std::runtime_error("invalid GGUF rank");
        tensor.dimensions.reserve(rank);
        for (std::uint32_t d = 0; d < rank; ++d)
            tensor.dimensions.push_back(read_scalar<std::uint64_t>(stream_));
        tensor.type = static_cast<GgmlType>(
            read_scalar<std::uint32_t>(stream_));
        tensor.relative_offset = read_scalar<std::uint64_t>(stream_);
        tensor.bytes = ggml_tensor_bytes(tensor.type, tensor.dimensions);
        tensors_.push_back(std::move(tensor));
    }

    const auto descriptor_end = static_cast<std::uint64_t>(stream_.tellg());
    data_offset_ = (descriptor_end + alignment_ - 1) & ~(alignment_ - 1);
    stream_.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::uint64_t>(stream_.tellg());
    for (auto& tensor : tensors_) {
        tensor.absolute_offset = data_offset_ + tensor.relative_offset;
        if (require_tensor_payloads &&
            (tensor.absolute_offset > file_size ||
             tensor.bytes > file_size - tensor.absolute_offset))
            throw std::runtime_error("GGUF tensor outside file: " + tensor.name);
    }
}

const GgufTensor* GgufFile::tensor(const std::string& name) const noexcept {
    const auto found = std::find_if(tensors_.begin(), tensors_.end(),
        [&](const GgufTensor& item) { return item.name == name; });
    return found == tensors_.end() ? nullptr : &*found;
}

const GgufMetadata* GgufFile::meta(const std::string& key) const noexcept {
    const auto found = metadata_.find(key);
    return found == metadata_.end() ? nullptr : &found->second;
}

std::vector<std::uint8_t> GgufFile::read(const GgufTensor& tensor) {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.bytes));
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(tensor.absolute_offset));
    stream_.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream_) throw std::runtime_error("failed reading GGUF tensor");
    return bytes;
}

} // namespace gemmaedge
