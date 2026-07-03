#pragma once

#include "gemmaedge/model_format.h"

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace gemmaedge {

struct ExpertKey {
    std::uint32_t layer;
    std::uint32_t expert;
    TensorRole role;

    bool operator==(const ExpertKey& other) const noexcept {
        return layer == other.layer && expert == other.expert && role == other.role;
    }
};

struct ExpertKeyHash {
    std::size_t operator()(const ExpertKey& key) const noexcept;
};

struct ExpertCacheStats {
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t admissions{0};
    std::uint64_t evictions{0};
    std::uint64_t bytes_loaded{0};
    std::uint64_t bytes_resident{0};
};

class ExpertCache {
public:
    using Bytes = std::vector<std::uint8_t>;
    using Loader = std::function<Bytes(const ExpertKey&)>;

    ExpertCache(std::uint64_t byte_budget, Loader loader);

    std::shared_ptr<const Bytes> get(const ExpertKey& key, float router_probability);
    bool contains(const ExpertKey& key) const;
    void clear();

    std::uint64_t budget() const noexcept { return byte_budget_; }
    const ExpertCacheStats& stats() const noexcept { return stats_; }

private:
    struct Slot {
        std::shared_ptr<Bytes> bytes;
        float score;
        std::list<ExpertKey>::iterator lru;
    };

    void touch(Slot& slot, const ExpertKey& key, float router_probability);
    void evict_until_fit(std::uint64_t incoming);

    std::uint64_t byte_budget_;
    Loader loader_;
    std::list<ExpertKey> lru_;
    std::unordered_map<ExpertKey, Slot, ExpertKeyHash> slots_;
    ExpertCacheStats stats_{};
};

} // namespace gemmaedge

