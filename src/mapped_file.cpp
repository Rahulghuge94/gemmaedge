#include "gemmaedge/mapped_file.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace gemmaedge {

MappedFile::MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
    const auto handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                    nullptr, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot open mapped file");
    file_ = handle;
    LARGE_INTEGER length{};
    if (!GetFileSizeEx(handle, &length) || length.QuadPart <= 0) {
        close();
        throw std::runtime_error("cannot size mapped file");
    }
    size_ = static_cast<std::uint64_t>(length.QuadPart);
    mapping_ = CreateFileMappingW(handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_) {
        close();
        throw std::runtime_error("cannot create file mapping");
    }
    data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
    if (!data_) {
        close();
        throw std::runtime_error("cannot map file");
    }
#else
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0)
        throw std::runtime_error("cannot open mapped file: " +
                                 std::string(std::strerror(errno)));
    struct stat info {};
    if (fstat(fd_, &info) != 0 || info.st_size <= 0) {
        close();
        throw std::runtime_error("cannot size mapped file");
    }
    size_ = static_cast<std::uint64_t>(info.st_size);
    data_ = mmap(nullptr, static_cast<std::size_t>(size_), PROT_READ,
                 MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        close();
        throw std::runtime_error("cannot mmap file: " +
                                 std::string(std::strerror(errno)));
    }
#endif
}

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept {
    *this = std::move(other);
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this == &other) return *this;
    close();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
#ifdef _WIN32
    file_ = other.file_;
    mapping_ = other.mapping_;
    other.file_ = nullptr;
    other.mapping_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    return *this;
}

const std::uint8_t* MappedFile::view(std::uint64_t offset,
                                     std::uint64_t bytes) const {
    if (offset > size_ || bytes > size_ - offset)
        throw std::out_of_range("mapped file view outside file");
    return data() + offset;
}

void MappedFile::advise_will_need(std::uint64_t offset,
                                  std::uint64_t bytes) const noexcept {
    if (offset > size_ || bytes > size_ - offset) return;
#ifdef _WIN32
    (void)offset;
    (void)bytes;
#else
    const auto page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    const auto start = offset & ~(page - 1);
    const auto end = (offset + bytes + page - 1) & ~(page - 1);
    madvise(const_cast<std::uint8_t*>(data() + start),
            static_cast<std::size_t>(std::min(end, size_) - start),
            MADV_WILLNEED);
#endif
}

void MappedFile::advise_dont_need(std::uint64_t offset,
                                  std::uint64_t bytes) const noexcept {
    if (offset > size_ || bytes > size_ - offset) return;
#ifdef _WIN32
    (void)offset;
    (void)bytes;
#else
    const auto page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    const auto start = offset & ~(page - 1);
    const auto end = (offset + bytes + page - 1) & ~(page - 1);
    madvise(const_cast<std::uint8_t*>(data() + start),
            static_cast<std::size_t>(std::min(end, size_) - start),
            MADV_DONTNEED);
#endif
}

void MappedFile::close() noexcept {
#ifdef _WIN32
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
    if (file_) CloseHandle(static_cast<HANDLE>(file_));
    data_ = nullptr;
    mapping_ = nullptr;
    file_ = nullptr;
#else
    if (data_) munmap(data_, static_cast<std::size_t>(size_));
    if (fd_ >= 0) ::close(fd_);
    data_ = nullptr;
    fd_ = -1;
#endif
    size_ = 0;
}

} // namespace gemmaedge
