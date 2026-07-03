#include "gemmaedge/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace gemmaedge {
namespace {

constexpr std::int64_t kTokenNormal = 1;
constexpr std::int64_t kTokenControl = 3;
constexpr std::int64_t kTokenUserDefined = 4;
constexpr std::int64_t kTokenByte = 6;
constexpr char kMergeSeparator = '\x1f';

const GgufMetadata& require_meta(const GgufFile& gguf, const char* key) {
    const auto* value = gguf.meta(key);
    if (!value) throw std::runtime_error(std::string("missing GGUF metadata: ") + key);
    return *value;
}

TokenId meta_token(const GgufFile& gguf, const char* key) {
    return static_cast<TokenId>(require_meta(gguf, key).unsigned_value);
}

std::string merge_key(const std::string& left, const std::string& right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key += left;
    key += kMergeSeparator;
    key += right;
    return key;
}

bool is_special_type(std::int64_t type) {
    return type == kTokenControl || type == kTokenUserDefined;
}

} // namespace

Gemma4Tokenizer::Gemma4Tokenizer(const GgufFile& gguf)
    : Gemma4Tokenizer(
          require_meta(gguf, "tokenizer.ggml.tokens").strings,
          require_meta(gguf, "tokenizer.ggml.merges").strings,
          require_meta(gguf, "tokenizer.ggml.token_type").signed_values,
          meta_token(gguf, "tokenizer.ggml.bos_token_id"),
          meta_token(gguf, "tokenizer.ggml.eos_token_id"),
          meta_token(gguf, "tokenizer.ggml.unknown_token_id"),
          meta_token(gguf, "tokenizer.ggml.padding_token_id")) {
    const auto& model = require_meta(gguf, "tokenizer.ggml.model").text;
    if (model != "gemma4")
        throw std::runtime_error("GGUF tokenizer is not gemma4");
}

Gemma4Tokenizer::Gemma4Tokenizer(
    std::vector<std::string> tokens, std::vector<std::string> merges,
    std::vector<std::int64_t> token_types, TokenId bos, TokenId eos,
    TokenId unknown, TokenId padding)
    : tokens_(std::move(tokens)), token_types_(std::move(token_types)),
      bos_(bos), eos_(eos), unknown_(unknown), padding_(padding) {
    if (tokens_.empty() || token_types_.size() != tokens_.size())
        throw std::invalid_argument("invalid tokenizer vocabulary");

    std::fill(std::begin(byte_tokens_), std::end(byte_tokens_), unknown_);
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
        token_to_id_.emplace(tokens_[i], static_cast<TokenId>(i));
        const auto type = token_types_[i];
        if (is_special_type(type)) special_tokens_.push_back(tokens_[i]);
        if (type == kTokenByte && tokens_[i].size() == 6 &&
            tokens_[i].compare(0, 3, "<0x") == 0 && tokens_[i].back() == '>') {
            unsigned byte = 0;
            if (std::sscanf(tokens_[i].c_str() + 3, "%2x", &byte) == 1)
                byte_tokens_[byte] = static_cast<TokenId>(i);
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) {
                  return a.size() > b.size();
              });

    std::cout << "[Tokenizer Info] Special tokens count: " << special_tokens_.size() << std::endl;
    for (std::size_t i = 0; i < special_tokens_.size(); ++i) {
        std::cout << "  Special token " << i << ": " << special_tokens_[i] << std::endl;
    }

    for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
        const auto split = merges[rank].find(' ');
        if (split == std::string::npos) continue;
        merge_rank_.emplace(
            merge_key(merges[rank].substr(0, split),
                      merges[rank].substr(split + 1)),
            rank);
    }
}

TokenId Gemma4Tokenizer::find(const std::string& token) const noexcept {
    const auto found = token_to_id_.find(token);
    return found == token_to_id_.end() ? -1 : found->second;
}

std::string Gemma4Tokenizer::escape_spaces(const std::string& text) {
    static const std::string marker = "\xE2\x96\x81";
    std::string result;
    result.reserve(text.size() + 8);
    for (const char ch : text) {
        if (ch == ' ') result += marker;
        else result += ch;
    }
    return result;
}

std::vector<std::string>
Gemma4Tokenizer::utf8_symbols(const std::string& text) {
    std::vector<std::string> result;
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        std::size_t length = 1;
        if ((lead & 0xe0u) == 0xc0u) length = 2;
        else if ((lead & 0xf0u) == 0xe0u) length = 3;
        else if ((lead & 0xf8u) == 0xf0u) length = 4;
        length = std::min(length, text.size() - i);
        result.emplace_back(text.substr(i, length));
        i += length;
    }
    return result;
}

void Gemma4Tokenizer::encode_bpe_span(
    const std::string& span, std::vector<TokenId>& out) const {
    if (span.empty()) return;
    auto symbols = utf8_symbols(span);
    while (symbols.size() > 1) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t best = symbols.size();
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto found = merge_rank_.find(merge_key(symbols[i], symbols[i + 1]));
            if (found != merge_rank_.end() && found->second < best_rank) {
                best_rank = found->second;
                best = i;
            }
        }
        if (best == symbols.size()) break;
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }

    for (const auto& symbol : symbols) {
        const auto token = find(symbol);
        if (token >= 0) {
            out.push_back(token);
        } else {
            for (const unsigned char byte : symbol)
                out.push_back(byte_tokens_[byte]);
        }
    }
}

void Gemma4Tokenizer::encode_raw(
    const std::string& text, std::vector<TokenId>& out) const {
    const auto escaped = escape_spaces(text);
    for (std::size_t begin = 0; begin < escaped.size();) {
        const bool newline = escaped[begin] == '\n';
        std::size_t end = begin + 1;
        while (end < escaped.size() && (escaped[end] == '\n') == newline) ++end;
        const auto span = escaped.substr(begin, end - begin);
        if (newline) {
            const auto exact = find(span);
            if (exact >= 0) out.push_back(exact);
            else encode_bpe_span(span, out);
        } else {
            encode_bpe_span(span, out);
        }
        begin = end;
    }
}

std::vector<TokenId> Gemma4Tokenizer::encode(
    const std::string& text, bool add_bos, bool parse_special) const {
    std::vector<TokenId> result;
    if (add_bos && bos_ >= 0) result.push_back(bos_);
    if (!parse_special) {
        encode_raw(text, result);
        return result;
    }

    std::size_t raw_start = 0;
    std::size_t pos = 0;
    while (pos < text.size()) {
        TokenId matched = -1;
        std::size_t matched_size = 0;
        if (text[pos] == '<') {
            for (const auto& special : special_tokens_) {
                if (special.size() > matched_size &&
                    text.compare(pos, special.size(), special) == 0) {
                    matched = find(special);
                    matched_size = special.size();
                    break;
                }
            }
        }
        if (matched >= 0) {
            encode_raw(text.substr(raw_start, pos - raw_start), result);
            result.push_back(matched);
            pos += matched_size;
            raw_start = pos;
        } else {
            ++pos;
        }
    }
    encode_raw(text.substr(raw_start), result);
    return result;
}

std::string Gemma4Tokenizer::decode(TokenId token,
                                    bool render_special) const {
    if (token < 0 || static_cast<std::size_t>(token) >= tokens_.size())
        return {};
    const auto& piece = tokens_[static_cast<std::size_t>(token)];
    const auto type = token_types_[static_cast<std::size_t>(token)];
    if (type == kTokenByte && piece.size() == 6) {
        unsigned byte = 0;
        if (std::sscanf(piece.c_str() + 3, "%2x", &byte) == 1)
            return std::string(1, static_cast<char>(byte));
    }
    if (is_special_type(type) && !render_special) return {};
    static const std::string marker = "\xE2\x96\x81";
    std::string result = piece;
    for (std::size_t at = 0; (at = result.find(marker, at)) != std::string::npos;) {
        result.replace(at, marker.size(), " ");
        ++at;
    }
    return result;
}

std::string Gemma4Tokenizer::decode(
    const std::vector<TokenId>& tokens, bool render_special) const {
    std::string result;
    for (const auto token : tokens) result += decode(token, render_special);
    return result;
}

std::string render_gemma4_chat(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt,
    bool enable_thinking) {
    std::string output = "<bos>";
    for (const auto& message : messages) {
        const char* role = message.role == ChatRole::System ? "system" :
                           message.role == ChatRole::User ? "user" : "model";
        output += "<|turn>";
        output += role;
        output += '\n';
        if (message.role == ChatRole::System && enable_thinking)
            output += "<|think|>\n";
        for (const auto& part : message.parts) {
            if (part.type == ChatPart::Type::Image) output += "<|image|>";
            else output += part.text;
        }
        output += "<turn|>\n";
    }
    if (add_generation_prompt) {
        output += "<|turn>model\n";
        if (!enable_thinking)
            output += "<|channel>thought\n<channel|>";
    }
    return output;
}

} // namespace gemmaedge

