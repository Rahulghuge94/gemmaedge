#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace gemmaedge {

constexpr std::uint32_t kKvMagic = 0x564b4547; // "GEKV"
constexpr std::uint32_t kKvVersion = 1;

struct KvBlockKey {
    std::uint32_t layer;
    std::uint32_t token_start;

    bool operator==(const KvBlockKey& other) const noexcept {
        return layer == other.layer && token_start == other.token_start;
    }
};

struct KvBlockKeyHash {
    std::size_t operator()(const KvBlockKey& key) const noexcept;
};

struct KvBlock {
    KvBlockKey key{};
    std::uint32_t token_count{0};
    std::vector<std::uint8_t> bytes;
};

class DiskKvStore {
public:
    DiskKvStore(const std::filesystem::path& path, std::uint64_t model_fingerprint,
                std::uint32_t block_tokens, bool create);

    void append(const KvBlock& block);
    KvBlock read(const KvBlockKey& key);
    bool contains(const KvBlockKey& key) const;
    std::size_t block_count() const noexcept { return index_.size(); }
    std::uint32_t block_tokens() const noexcept { return block_tokens_; }
    void flush();

private:
    struct Location {
        std::uint64_t offset;
        std::uint64_t payload_size;
        std::uint32_t token_count;
        std::uint64_t checksum;
    };

    void create_file();
    void open_existing();
    void rebuild_index();

    std::filesystem::path path_;
    std::fstream stream_;
    std::uint64_t model_fingerprint_;
    std::uint32_t block_tokens_;
    std::unordered_map<KvBlockKey, Location, KvBlockKeyHash> index_;
};

} // namespace gemmaedge

