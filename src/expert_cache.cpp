#include "gemmaedge/expert_cache.h"

#include <algorithm>
#include <stdexcept>

namespace gemmaedge {

std::size_t ExpertKeyHash::operator()(const ExpertKey& key) const noexcept {
    std::size_t h = key.layer;
    h = h * 1315423911u + key.expert;
    h = h * 31u + static_cast<std::uint32_t>(key.role);
    return h;
}

ExpertCache::ExpertCache(std::uint64_t byte_budget, Loader loader, Evictor evictor)
    : byte_budget_(byte_budget), loader_(std::move(loader)), evictor_(std::move(evictor)) {
    if (!loader_) throw std::invalid_argument("expert cache requires a loader");
}

void ExpertCache::touch(Slot& slot, const ExpertKey& key,
                        float router_probability) {
    lru_.erase(slot.lru);
    lru_.push_front(key);
    slot.lru = lru_.begin();
    slot.score = slot.score * 0.875f + std::max(0.0f, router_probability);
}

void ExpertCache::evict_until_fit(std::uint64_t incoming) {
    while (!lru_.empty() && stats_.bytes_resident + incoming > byte_budget_) {
        // Prefer an old, weakly-routed entry from the oldest eight slots.
        auto victim = std::prev(lru_.end());
        auto cursor = victim;
        float weakest = slots_.at(*victim).score;
        for (int i = 0; i < 7 && cursor != lru_.begin(); ++i) {
            --cursor;
            const float score = slots_.at(*cursor).score;
            if (score < weakest) {
                victim = cursor;
                weakest = score;
            }
        }
        auto slot = slots_.find(*victim);
        if (evictor_) {
            evictor_(*victim, slot->second.view);
        }
        stats_.bytes_resident -= slot->second.view.size;
        slots_.erase(slot);
        lru_.erase(victim);
        ++stats_.evictions;
    }
}

ExpertView
ExpertCache::get(const ExpertKey& key, float router_probability) {
    auto found = slots_.find(key);
    if (found != slots_.end()) {
        ++stats_.hits;
        touch(found->second, key, router_probability);
        return found->second.view;
    }

    ++stats_.misses;
    ExpertView view = loader_(key);
    stats_.bytes_loaded += view.size;
    if (view.size > byte_budget_ || byte_budget_ == 0)
        return view;

    evict_until_fit(view.size);
    lru_.push_front(key);
    Slot slot{view, std::max(0.0f, router_probability), lru_.begin()};
    slots_.emplace(key, std::move(slot));
    stats_.bytes_resident += view.size;
    ++stats_.admissions;
    return view;
}

bool ExpertCache::contains(const ExpertKey& key) const {
    return slots_.find(key) != slots_.end();
}

void ExpertCache::clear() {
    slots_.clear();
    lru_.clear();
    stats_.bytes_resident = 0;
}

} // namespace gemmaedge

