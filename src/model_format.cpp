#include "gemmaedge/model_format.h"

#include <array>
#include <stdexcept>
#include <type_traits>

namespace gemmaedge {
namespace {

template <typename T>
T read_scalar(std::istream& in) {
    static_assert(std::is_trivially_copyable<T>::value, "scalar required");
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated GemmaEdge model file");
    return value;
}

} // namespace

std::uint64_t checksum64(const void* data, std::size_t size) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return static_cast<std::uint64_t>(crc ^ 0xffffffffu);
}

ModelFile::ModelFile(const std::filesystem::path& path)
    : stream_(path, std::ios::binary) {
    if (!stream_) throw std::runtime_error("cannot open model: " + path.string());

    header_.magic = read_scalar<std::uint32_t>(stream_);
    header_.version = read_scalar<std::uint32_t>(stream_);
    header_.layer_count = read_scalar<std::uint32_t>(stream_);
    header_.expert_count = read_scalar<std::uint32_t>(stream_);
    header_.top_k = read_scalar<std::uint32_t>(stream_);
    header_.hidden_size = read_scalar<std::uint32_t>(stream_);
    header_.expert_intermediate_size = read_scalar<std::uint32_t>(stream_);
    header_.vocab_size = read_scalar<std::uint32_t>(stream_);
    header_.directory_offset = read_scalar<std::uint64_t>(stream_);
    header_.directory_entries = read_scalar<std::uint64_t>(stream_);

    if (header_.magic != kModelMagic || header_.version != kModelVersion)
        throw std::runtime_error("unsupported GemmaEdge model format");
    if (header_.layer_count != 30 || header_.expert_count != 128 ||
        header_.top_k != 8 || header_.hidden_size != 2816)
        throw std::runtime_error("model is not Gemma 4 26B A4B");
    if (header_.directory_entries > 30ULL * 128ULL * 3ULL + 4096ULL)
        throw std::runtime_error("unreasonable tensor directory size");

    stream_.seekg(static_cast<std::streamoff>(header_.directory_offset));
    for (std::uint64_t i = 0; i < header_.directory_entries; ++i) {
        TensorEntry entry;
        entry.role = static_cast<TensorRole>(read_scalar<std::uint32_t>(stream_));
        entry.quant = static_cast<QuantType>(read_scalar<std::uint32_t>(stream_));
        entry.layer = read_scalar<std::uint32_t>(stream_);
        entry.expert = read_scalar<std::uint32_t>(stream_);
        entry.offset = read_scalar<std::uint64_t>(stream_);
        entry.size = read_scalar<std::uint64_t>(stream_);
        entry.checksum = read_scalar<std::uint64_t>(stream_);
        const auto rank = read_scalar<std::uint32_t>(stream_);
        if (rank > 8) throw std::runtime_error("invalid tensor rank");
        entry.shape.reserve(rank);
        for (std::uint32_t dim = 0; dim < rank; ++dim)
            entry.shape.push_back(read_scalar<std::uint64_t>(stream_));
        const auto name_size = read_scalar<std::uint32_t>(stream_);
        if (name_size > 4096) throw std::runtime_error("invalid tensor name");
        entry.name.resize(name_size);
        stream_.read(entry.name.data(), name_size);
        if (!stream_) throw std::runtime_error("truncated tensor directory");
        entries_.push_back(std::move(entry));
    }
}

std::vector<std::uint8_t> ModelFile::read(const TensorEntry& entry) {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(entry.size));
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(entry.offset));
    stream_.read(reinterpret_cast<char*>(result.data()),
                 static_cast<std::streamsize>(result.size()));
    if (!stream_) throw std::runtime_error("truncated tensor payload");
    if (checksum64(result.data(), result.size()) != entry.checksum)
        throw std::runtime_error("tensor checksum mismatch: " + entry.name);
    return result;
}

const TensorEntry* ModelFile::find_expert(std::uint32_t layer,
                                         std::uint32_t expert,
                                         TensorRole role) const noexcept {
    for (const auto& entry : entries_) {
        if (entry.layer == layer && entry.expert == expert && entry.role == role)
            return &entry;
    }
    return nullptr;
}

} // namespace gemmaedge
