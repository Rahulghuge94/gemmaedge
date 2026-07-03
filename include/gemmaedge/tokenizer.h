#pragma once

#include "gemmaedge/gguf.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gemmaedge {

using TokenId = std::int32_t;

class Gemma4Tokenizer {
public:
    explicit Gemma4Tokenizer(const GgufFile& gguf);
    Gemma4Tokenizer(std::vector<std::string> tokens,
                    std::vector<std::string> merges,
                    std::vector<std::int64_t> token_types,
                    TokenId bos, TokenId eos, TokenId unknown,
                    TokenId padding);

    std::vector<TokenId> encode(const std::string& text,
                                bool add_bos = true,
                                bool parse_special = true) const;
    std::string decode(TokenId token,
                       bool render_special = false) const;
    std::string decode(const std::vector<TokenId>& tokens,
                       bool render_special = false) const;

    TokenId bos() const noexcept { return bos_; }
    TokenId eos() const noexcept { return eos_; }
    TokenId unknown() const noexcept { return unknown_; }
    TokenId padding() const noexcept { return padding_; }
    std::size_t size() const noexcept { return tokens_.size(); }
    TokenId find(const std::string& token) const noexcept;

private:
    void encode_raw(const std::string& text, std::vector<TokenId>& out) const;
    void encode_bpe_span(const std::string& span,
                         std::vector<TokenId>& out) const;
    static std::vector<std::string> utf8_symbols(const std::string& text);
    static std::string escape_spaces(const std::string& text);

    std::vector<std::string> tokens_;
    std::vector<std::int64_t> token_types_;
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::unordered_map<std::string, std::uint32_t> merge_rank_;
    std::vector<std::string> special_tokens_;
    TokenId byte_tokens_[256]{};
    TokenId bos_{-1};
    TokenId eos_{-1};
    TokenId unknown_{-1};
    TokenId padding_{-1};
};

enum class ChatRole {
    System,
    User,
    Assistant,
};

struct ChatPart {
    enum class Type { Text, Image };
    Type type{Type::Text};
    std::string text;
};

struct ChatMessage {
    ChatRole role{ChatRole::User};
    std::vector<ChatPart> parts;
};

// Minimal exact Gemma 4 turn renderer for text/image chat. Tool schema support
// is intentionally kept outside this primitive renderer.
std::string render_gemma4_chat(const std::vector<ChatMessage>& messages,
                               bool add_generation_prompt = true,
                               bool enable_thinking = false);

} // namespace gemmaedge

