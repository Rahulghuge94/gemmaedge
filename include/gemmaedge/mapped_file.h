#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace gemmaedge {

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    const std::uint8_t* data() const noexcept {
        return static_cast<const std::uint8_t*>(data_);
    }
    std::uint64_t size() const noexcept { return size_; }
    const std::uint8_t* view(std::uint64_t offset,
                             std::uint64_t bytes) const;

    // Hint that a region is likely to be needed soon or can be evicted. These
    // are performance hints; failure is non-fatal on kernels/filesystems that
    // do not implement them.
    void advise_will_need(std::uint64_t offset, std::uint64_t bytes) const noexcept;
    void advise_dont_need(std::uint64_t offset, std::uint64_t bytes) const noexcept;

private:
    void close() noexcept;

    void* data_{nullptr};
    std::uint64_t size_{0};
#ifdef _WIN32
    void* file_{nullptr};
    void* mapping_{nullptr};
#else
    int fd_{-1};
#endif
};

} // namespace gemmaedge

