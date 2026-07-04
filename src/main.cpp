#include "gemmaedge/kv_store.h"
#include "gemmaedge/gguf.h"
#include "gemmaedge/gemma4_model.h"
#include "gemmaedge/tokenizer.h"
#include "gemmaedge/generation.h"
#include "gemmaedge/mapped_file.h"
#include "gemmaedge/model_format.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <algorithm>

int main(int argc, char** argv) {
    try {
        if (argc == 3 && (std::string(argv[1]) == "inspect" ||
                          std::string(argv[1]) == "inspect-header")) {
            const bool header_only = std::string(argv[1]) == "inspect-header";
            gemmaedge::GgufFile model(argv[2], !header_only);
            std::cout << "GGUF v" << model.version() << '\n'
                      << "alignment: " << model.alignment() << '\n'
                      << "metadata: " << model.metadata().size() << '\n'
                      << "tensors: " << model.tensors().size() << '\n';
            if (const auto* architecture = model.meta("general.architecture"))
                std::cout << "architecture: " << architecture->text << '\n';
            std::uint64_t bytes = 0;
            for (const auto& tensor : model.tensors()) bytes += tensor.bytes;
            std::cout << "tensor bytes: " << bytes << '\n';
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "list-tensors") {
            gemmaedge::GgufFile model(argv[2], false);
            for (const auto& tensor : model.tensors()) {
                std::cout << tensor.name << '\t'
                          << gemmaedge::ggml_type_name(tensor.type) << '\t';
                for (std::size_t i = 0; i < tensor.dimensions.size(); ++i) {
                    if (i) std::cout << 'x';
                    std::cout << tensor.dimensions[i];
                }
                std::cout << '\t' << tensor.absolute_offset << '\t'
                          << tensor.bytes << '\n';
            }
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "list-metadata") {
            gemmaedge::GgufFile model(argv[2], false);
            for (const auto& pair : model.metadata()) {
                const auto& value = pair.second;
                std::cout << pair.first << '\t';
                if (!value.text.empty()) std::cout << value.text;
                else if (!value.strings.empty())
                    std::cout << "string[" << value.strings.size() << ']';
                else if (!value.float_values.empty())
                    std::cout << "float[" << value.float_values.size() << ']';
                else if (!value.signed_values.empty())
                    std::cout << "int[" << value.signed_values.size() << ']';
                else if (!value.unsigned_values.empty())
                    std::cout << "uint[" << value.unsigned_values.size() << ']';
                else if (value.float_value != 0.0)
                    std::cout << std::setprecision(12) << value.float_value;
                else if (value.signed_value != 0)
                    std::cout << value.signed_value;
                else std::cout << value.unsigned_value;
                std::cout << '\n';
            }
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "validate-model") {
            gemmaedge::GgufFile file(argv[2], false);
            const auto* architecture = file.meta("general.architecture");
            if (architecture && architecture->text == "clip") {
                gemmaedge::Gemma4VisionModel model(file);
                const auto& config = model.config();
                std::cout << "valid Gemma 4 vision projector\n"
                          << "layers: " << config.layer_count << '\n'
                          << "image: " << config.image_size << "x"
                          << config.image_size << '\n'
                          << "projection: " << config.projection_size << '\n';
            } else {
                gemmaedge::Gemma4Model model(file);
                const auto& config = model.config();
                std::cout << "valid Gemma 4 26B A4B\n"
                          << "layers: " << config.layer_count << '\n'
                          << "vocabulary: " << config.vocab_size << '\n'
                          << "experts: " << config.expert_count << " top-"
                          << config.experts_used << '\n';
            }
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "tokenize") {
            gemmaedge::GgufFile file(argv[2], false);
            gemmaedge::Gemma4Tokenizer tokenizer(file);
            const auto tokens = tokenizer.encode(argv[3]);
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                if (i) std::cout << ' ';
                std::cout << tokens[i];
            }
            std::cout << '\n';
            return 0;
        }
        if ((argc == 4 || argc == 5) &&
            std::string(argv[1]) == "generate") {
            gemmaedge::GgufFile file(argv[2]);
            gemmaedge::Gemma4Model model(file);
            gemmaedge::Gemma4Tokenizer tokenizer(file);
            gemmaedge::MappedFile weights(argv[2]);
            std::uint64_t expert_cache_bytes = 6ULL * 1024ULL * 1024ULL * 1024ULL; // 6 GB default
            if (const char* env_cache = std::getenv("GEMMAEDGE_EXPERT_CACHE_GB")) {
                try {
                    expert_cache_bytes = std::stoull(env_cache) * 1024ULL * 1024ULL * 1024ULL;
                } catch (...) {
                    // fallback to 6 GB default
                }
            }
            gemmaedge::Gemma4Session session(model, weights, expert_cache_bytes);
            const auto prompt = tokenizer.encode(argv[3], true, true);
            gemmaedge::GenerationConfig config;
            if (argc == 5)
                config.max_new_tokens = static_cast<std::uint32_t>(
                    std::max(1, std::atoi(argv[4])));
             std::cout << "Starting prefill/generation with " << prompt.size() << " prompt tokens:" << std::endl;
             for (std::size_t i = 0; i < prompt.size(); ++i) {
                 std::cout << "  Prompt Token " << i << " (ID " << prompt[i] << "): '"
                           << tokenizer.decode(prompt[i], true) << "'" << std::endl;
             }
            const auto prefill_start = std::chrono::steady_clock::now();
            session.prefill(prompt);
            const auto prefill_end = std::chrono::steady_clock::now();
            const double prefill_ms = std::chrono::duration_cast<std::chrono::microseconds>(prefill_end - prefill_start).count() / 1000.0;

            std::mt19937_64 random(config.sampling.seed);
            std::vector<gemmaedge::TokenId> generated;
            generated.reserve(config.max_new_tokens);

            const auto decode_start = std::chrono::steady_clock::now();
            for (std::uint32_t i = 0; i < config.max_new_tokens; ++i) {
                const gemmaedge::TokenId token = gemmaedge::sample_token(session.logits(), config.sampling, random);
                if (std::find(config.stop_tokens.begin(), config.stop_tokens.end(),
                              token) != config.stop_tokens.end())
                    break;
                generated.push_back(token);
                std::cout << tokenizer.decode(token, false) << std::flush;
                session.evaluate(token);
            }
            std::cout << std::endl;
            const auto decode_end = std::chrono::steady_clock::now();
            const double decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count() / 1000.0;
            const double total_ms = prefill_ms + decode_ms;

            const auto stats = session.expert_cache_stats();
            std::cout << "\n\n=== Context & Speed Report ==="
                      << "\n- Input context size  : " << prompt.size() << " tokens"
                      << "\n- Output context size : " << generated.size() << " tokens"
                      << "\n- Prefill latency     : " << prefill_ms << " ms"
                      << "\n- Prefill speed       : " << (prompt.size() / (prefill_ms / 1000.0)) << " tokens/sec"
                      << "\n- Decode latency      : " << decode_ms << " ms"
                      << "\n- Decode speed        : " << (generated.size() / (decode_ms / 1000.0)) << " tokens/sec"
                      << "\n- Total latency       : " << total_ms << " ms"
                      << "\n- Total speed         : " << ((prompt.size() + generated.size()) / (total_ms / 1000.0)) << " tokens/sec"
                      << "\n- Cache Hits          : " << stats.hits
                      << "\n- Cache Misses        : " << stats.misses
                      << "\n- Cache Admissions    : " << stats.admissions
                      << "\n- Cache Evictions     : " << stats.evictions
                      << "\n- Cache Resident Bytes: " << stats.bytes_resident
                      << "\n==============================\n";
            return 0;
        }
        std::cout << "GemmaEdge 0.1 storage-core\n"
                  << "usage: gemmaedge inspect MODEL.gguf\n"
                  << "       gemmaedge inspect-header PARTIAL.gguf\n"
                  << "       gemmaedge list-tensors MODEL.gguf\n"
                  << "       gemmaedge list-metadata MODEL.gguf\n"
                  << "       gemmaedge validate-model MODEL.gguf\n"
                  << "       gemmaedge tokenize MODEL.gguf TEXT\n"
                  << "       gemmaedge generate MODEL.gguf PROMPT [TOKENS]\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
