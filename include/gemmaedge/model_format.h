#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gemmaedge {

constexpr std::uint32_t kModelMagic = 0x34454747; // "GGE4"
constexpr std::uint32_t kModelVersion = 2;

enum class QuantType : std::uint32_t {
    F16 = 1,
    Q8_0 = 2,
    Q4_0 = 3,
};

enum class TensorRole : std::uint32_t {
    Resident = 1,
    ExpertGate = 2,
    ExpertUp = 3,
    ExpertDown = 4,
    ExpertGateUp = 5,
};

struct ModelHeader {
    std::uint32_t magic{kModelMagic};
    std::uint32_t version{kModelVersion};
    std::uint32_t layer_count{30};
    std::uint32_t expert_count{128};
    std::uint32_t top_k{8};
    std::uint32_t hidden_size{2816};
    std::uint32_t expert_intermediate_size{704};
    std::uint32_t vocab_size{262144};
    std::uint64_t directory_offset{0};
    std::uint64_t directory_entries{0};
};

struct TensorEntry {
    TensorRole role{TensorRole::Resident};
    QuantType quant{QuantType::Q4_0};
    std::uint32_t layer{0};
    std::uint32_t expert{0};
    std::uint64_t offset{0};
    std::uint64_t size{0};
    std::uint64_t checksum{0};
    std::vector<std::uint64_t> shape;
    std::string name;
};

class ModelFile {
public:
    explicit ModelFile(const std::filesystem::path& path);

    const ModelHeader& header() const noexcept { return header_; }
    const std::vector<TensorEntry>& entries() const noexcept { return entries_; }
    std::vector<std::uint8_t> read(const TensorEntry& entry);
    const TensorEntry* find_expert(std::uint32_t layer, std::uint32_t expert,
                                   TensorRole role) const noexcept;

private:
    std::ifstream stream_;
    ModelHeader header_{};
    std::vector<TensorEntry> entries_;
};

std::uint64_t checksum64(const void* data, std::size_t size) noexcept;

} // namespace gemmaedge
