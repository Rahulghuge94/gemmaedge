#include "gemmaedge/kv_store.h"

#include "gemmaedge/model_format.h"

#include <stdexcept>
#include <type_traits>

namespace gemmaedge {
namespace {

template <typename T>
void write_scalar(std::ostream& out, T value) {
    static_assert(std::is_trivially_copyable<T>::value, "scalar required");
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("KV store write failed");
}

template <typename T>
T read_scalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("KV store read failed");
    return value;
}

constexpr std::uint64_t kFileHeaderBytes =
    sizeof(std::uint32_t) * 3 + sizeof(std::uint64_t);
constexpr std::uint64_t kBlockHeaderBytes =
    sizeof(std::uint32_t) * 3 + sizeof(std::uint64_t) * 2;

} // namespace

std::size_t KvBlockKeyHash::operator()(const KvBlockKey& key) const noexcept {
    return static_cast<std::size_t>(key.layer) * 2654435761u + key.token_start;
}

DiskKvStore::DiskKvStore(const std::filesystem::path& path,
                         std::uint64_t model_fingerprint,
                         std::uint32_t block_tokens, bool create)
    : path_(path), model_fingerprint_(model_fingerprint),
      block_tokens_(block_tokens) {
    if (block_tokens == 0) throw std::invalid_argument("zero KV block size");
    if (create) create_file();
    else open_existing();
}

void DiskKvStore::create_file() {
    stream_.open(path_, std::ios::binary | std::ios::in | std::ios::out |
                            std::ios::trunc);
    if (!stream_) throw std::runtime_error("cannot create KV store");
    write_scalar(stream_, kKvMagic);
    write_scalar(stream_, kKvVersion);
    write_scalar(stream_, block_tokens_);
    write_scalar(stream_, model_fingerprint_);
    stream_.flush();
}

void DiskKvStore::open_existing() {
    stream_.open(path_, std::ios::binary | std::ios::in | std::ios::out);
    if (!stream_) throw std::runtime_error("cannot open KV store");
    const auto magic = read_scalar<std::uint32_t>(stream_);
    const auto version = read_scalar<std::uint32_t>(stream_);
    const auto block_tokens = read_scalar<std::uint32_t>(stream_);
    const auto fingerprint = read_scalar<std::uint64_t>(stream_);
    if (magic != kKvMagic || version != kKvVersion)
        throw std::runtime_error("unsupported KV store format");
    if (fingerprint != model_fingerprint_)
        throw std::runtime_error("KV store belongs to another model");
    if (block_tokens != block_tokens_)
        throw std::runtime_error("KV block size mismatch");
    rebuild_index();
}

void DiskKvStore::rebuild_index() {
    index_.clear();
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(kFileHeaderBytes));
    while (stream_.peek() != std::char_traits<char>::eof()) {
        const auto layer = read_scalar<std::uint32_t>(stream_);
        const auto token_start = read_scalar<std::uint32_t>(stream_);
        const auto token_count = read_scalar<std::uint32_t>(stream_);
        const auto payload_size = read_scalar<std::uint64_t>(stream_);
        const auto checksum = read_scalar<std::uint64_t>(stream_);
        const auto offset = static_cast<std::uint64_t>(stream_.tellg());
        index_[{layer, token_start}] =
            {offset, payload_size, token_count, checksum};
        stream_.seekg(static_cast<std::streamoff>(payload_size), std::ios::cur);
        if (!stream_) throw std::runtime_error("truncated KV block");
    }
    stream_.clear();
}

void DiskKvStore::append(const KvBlock& block) {
    if (block.token_count == 0 || block.token_count > block_tokens_)
        throw std::invalid_argument("invalid KV token count");
    if (contains(block.key))
        throw std::invalid_argument("KV block already exists");

    stream_.clear();
    stream_.seekp(0, std::ios::end);
    write_scalar(stream_, block.key.layer);
    write_scalar(stream_, block.key.token_start);
    write_scalar(stream_, block.token_count);
    write_scalar(stream_, static_cast<std::uint64_t>(block.bytes.size()));
    const auto checksum = checksum64(block.bytes.data(), block.bytes.size());
    write_scalar(stream_, checksum);
    const auto payload_offset = static_cast<std::uint64_t>(stream_.tellp());
    stream_.write(reinterpret_cast<const char*>(block.bytes.data()),
                  static_cast<std::streamsize>(block.bytes.size()));
    if (!stream_) throw std::runtime_error("KV payload write failed");
    index_[block.key] = {payload_offset, block.bytes.size(),
                         block.token_count, checksum};
}

KvBlock DiskKvStore::read(const KvBlockKey& key) {
    const auto found = index_.find(key);
    if (found == index_.end()) throw std::out_of_range("KV block not found");
    KvBlock block;
    block.key = key;
    block.token_count = found->second.token_count;
    block.bytes.resize(static_cast<std::size_t>(found->second.payload_size));
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(found->second.offset));
    stream_.read(reinterpret_cast<char*>(block.bytes.data()),
                 static_cast<std::streamsize>(block.bytes.size()));
    if (!stream_) throw std::runtime_error("KV payload read failed");
    if (checksum64(block.bytes.data(), block.bytes.size()) !=
        found->second.checksum)
        throw std::runtime_error("KV block checksum mismatch");
    return block;
}

bool DiskKvStore::contains(const KvBlockKey& key) const {
    return index_.find(key) != index_.end();
}

void DiskKvStore::flush() {
    stream_.flush();
    if (!stream_) throw std::runtime_error("KV store flush failed");
}

} // namespace gemmaedge

