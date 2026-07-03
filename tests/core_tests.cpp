#include "gemmaedge/expert_cache.h"
#include "gemmaedge/gguf.h"
#include "gemmaedge/kv_store.h"
#include "gemmaedge/mapped_file.h"
#include "gemmaedge/tensor.h"
#include "gemmaedge/tokenizer.h"
#include "gemmaedge/attention.h"
#include "gemmaedge/feed_forward.h"
#include "gemmaedge/vision.h"
#include "gemmaedge/generation.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_expert_cache() {
    using namespace gemmaedge;
    int loads = 0;
    ExpertCache cache(24, [&](const ExpertKey& key) {
        ++loads;
        return ExpertCache::Bytes(8, static_cast<std::uint8_t>(key.expert));
    });

    const ExpertKey a{0, 1, TensorRole::ExpertGate};
    const ExpertKey b{0, 2, TensorRole::ExpertGate};
    const ExpertKey c{0, 3, TensorRole::ExpertGate};
    const ExpertKey d{0, 4, TensorRole::ExpertGate};
    cache.get(a, 0.9f);
    cache.get(b, 0.2f);
    cache.get(c, 0.3f);
    cache.get(a, 0.9f);
    cache.get(d, 0.8f);

    require(loads == 4, "cache did not reuse resident expert");
    require(cache.contains(a), "hot expert was evicted");
    require(cache.stats().hits == 1, "cache hit count incorrect");
    require(cache.stats().evictions == 1, "cache eviction count incorrect");
    require(cache.stats().bytes_resident <= cache.budget(),
            "cache exceeded byte budget");
}

void test_kv_store() {
    using namespace gemmaedge;
    const auto path = std::filesystem::temp_directory_path() /
                      "gemmaedge_core_test.gekv";
    std::filesystem::remove(path);
    {
        DiskKvStore store(path, 0x12345678, 128, true);
        KvBlock block{{5, 0}, 3, {1, 2, 3, 4, 5, 6}};
        store.append(block);
        store.flush();
        require(store.contains({5, 0}), "new KV block missing");
        require(store.read({5, 0}).bytes == block.bytes,
                "KV round trip failed");
    }
    {
        DiskKvStore reopened(path, 0x12345678, 128, false);
        require(reopened.block_count() == 1, "KV index rebuild failed");
        require(reopened.read({5, 0}).token_count == 3,
                "KV token count lost");
    }
    std::filesystem::remove(path);
}

void test_tensor_primitives() {
    using namespace gemmaedge;
    std::vector<float> values(kQ4BlockSize);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = (static_cast<int>(i) - 16) * 0.125f;
    const auto block = quantize_q4_block(values.data());
    std::vector<float> restored(kQ4BlockSize);
    dequantize_q4_block(block, restored.data());
    for (std::size_t i = 0; i < values.size(); ++i)
        require(std::abs(values[i] - restored[i]) < 0.16f,
                "Q4 round-trip error too large");

    std::vector<float> matrix_values(kQ4BlockSize, 1.0f);
    const auto ones = quantize_q4_block(matrix_values.data());
    float dot = 0.0f;
    q4_matvec(&ones, 1, kQ4BlockSize, matrix_values.data(), &dot);
    require(std::abs(dot - 32.0f) < 0.1f, "Q4 matvec incorrect");

    float input[] = {3.0f, 4.0f};
    float output[] = {0.0f, 0.0f};
    rms_norm(input, nullptr, 2, 0.0f, output);
    require(std::abs(output[0] - 0.848528f) < 1e-5f,
            "RMS norm incorrect");

    float probabilities[] = {1.0f, 2.0f, 3.0f};
    softmax(probabilities, 3);
    require(std::abs(probabilities[0] + probabilities[1] +
                     probabilities[2] - 1.0f) < 1e-6f,
            "softmax does not sum to one");
    const auto selected = top_k(probabilities, 3, 2);
    require(selected[0].first == 2 && selected[1].first == 1,
            "top-k ordering incorrect");
}

void test_ggml_layouts() {
    using namespace gemmaedge;
    require(ggml_tensor_bytes(GgmlType::Q4_0, {32, 2}) == 36,
            "Q4_0 byte layout incorrect");
    require(ggml_tensor_bytes(GgmlType::Q8_0, {32, 2}) == 68,
            "Q8_0 byte layout incorrect");
    require(ggml_tensor_bytes(GgmlType::Q4_K, {256, 2}) == 288,
            "Q4_K byte layout incorrect");
    require(std::string(ggml_type_name(GgmlType::BF16)) == "BF16",
            "GGML type name incorrect");
}

void test_mapped_file() {
    using namespace gemmaedge;
    const auto path = std::filesystem::temp_directory_path() /
                      "gemmaedge_map_test.bin";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const char bytes[] = {'g', 'e', 'm', 'm', 'a'};
        output.write(bytes, sizeof(bytes));
    }
    {
        MappedFile mapping(path);
        require(mapping.size() == 5, "mapped file size incorrect");
        require(mapping.view(1, 3)[0] == 'e', "mapped file contents incorrect");
        bool rejected = false;
        try {
            mapping.view(4, 2);
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        require(rejected, "out-of-range mapped view accepted");
    }
    std::filesystem::remove(path);
}

void test_tokenizer() {
    using namespace gemmaedge;
    std::vector<std::string> tokens = {
        "<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>",
        "\xE2\x96\x81", "h", "i", "\xE2\x96\x81h", "\xE2\x96\x81hi",
        "\n", "<0x21>"
    };
    std::vector<std::int64_t> types(tokens.size(), 1);
    types[0] = types[1] = types[2] = types[3] = types[4] = types[5] = 3;
    types[12] = 6;
    Gemma4Tokenizer tokenizer(
        tokens,
        {"\xE2\x96\x81 h", "\xE2\x96\x81h i"},
        types, 2, 1, 3, 0);
    const auto ids = tokenizer.encode(" hi\n!", true, true);
    require(ids.size() == 4, "tokenizer produced unexpected token count");
    require(ids[0] == 2 && ids[1] == 10 && ids[2] == 11 && ids[3] == 12,
            "tokenizer BPE/byte fallback incorrect");
    require(tokenizer.decode({10, 11, 12}) == " hi\n!",
            "tokenizer decode incorrect");

    const auto rendered = render_gemma4_chat({
        {ChatRole::User, {{ChatPart::Type::Text, "look"},
                          {ChatPart::Type::Image, ""}}}
    });
    require(rendered.find("<|image|>") != std::string::npos,
            "chat renderer lost image placeholder");
}

void test_rope_attention_kv() {
    using namespace gemmaedge;
    float rope[] = {1.0f, 0.0f};
    apply_rope(rope, 1, 2, 2, 1, 1.0f);
    require(std::abs(rope[0] - std::cos(1.0f)) < 1e-6f &&
            std::abs(rope[1] - std::sin(1.0f)) < 1e-6f,
            "RoPE rotation incorrect");

    LayerKvCache cache({2, 1, 2, 2, 1.0f});
    const float key0[] = {1.0f, 0.0f};
    const float val0[] = {2.0f, 0.0f};
    const float key1[] = {0.0f, 1.0f};
    const float val1[] = {0.0f, 4.0f};
    cache.append(0, key0, val0);
    cache.append(1, key1, val1);
    const float queries[] = {10.0f, 0.0f, 0.0f, 10.0f};
    float output[4]{};
    cache.attend(queries, output);
    require(output[0] > 1.99f && output[1] < 0.01f,
            "GQA head zero attended to wrong value");
    require(output[2] < 0.01f && output[3] > 3.99f,
            "GQA head one attended to wrong value");

    const float key2[] = {1.0f, 1.0f};
    const float val2[] = {6.0f, 6.0f};
    cache.append(2, key2, val2);
    require(cache.token_count() == 2 && cache.first_position() == 1,
            "sliding KV window did not evict oldest row");
    float expected_after_eviction[4]{};
    cache.attend(queries, expected_after_eviction);

    const auto payload = cache.serialize_f32();
    LayerKvCache restored({2, 1, 2, 2, 1.0f});
    restored.restore_f32(payload.data(), payload.size(), 1, 2);
    float restored_output[4]{};
    restored.attend(queries, restored_output);
    for (std::size_t i = 0; i < 4; ++i)
        require(std::abs(restored_output[i] - expected_after_eviction[i]) < 1e-6f,
                "KV restore changed attention output");

    bool q8_rejected_unaligned = false;
    try {
        (void)cache.to_disk_block(7);
    } catch (const std::runtime_error&) {
        q8_rejected_unaligned = true;
    }
    require(q8_rejected_unaligned,
            "Q8 disk KV accepted a non-production unaligned head width");

    LayerKvCache disk_ready({1, 1, 32, 0, 1.0f});
    float disk_key[32]{};
    float disk_value[32]{};
    disk_key[0] = 0.75f;
    disk_value[3] = -1.25f;
    disk_ready.append(8, disk_key, disk_value);
    const auto q8_block = disk_ready.to_disk_block(7);
    LayerKvCache disk_restored({1, 1, 32, 0, 1.0f});
    disk_restored.restore_disk_block(q8_block);
    require(disk_restored.token_count() == 1 &&
            disk_restored.first_position() == 8,
            "Q8 disk KV restore failed");
}

void test_router_top_k() {
    using namespace gemmaedge;
    const float hidden[] = {1.0f, 0.0f};
    const float scale[] = {1.0f, 1.0f};
    const float router[] = {
        3.0f, 0.0f,
        2.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
    };
    const float expert_scale[] = {1.0f, 2.0f, 1.0f, 1.0f};
    const auto selected = route_top_k(
        hidden, 2, scale, router, 4, 2, expert_scale, 1e-6f);
    require(selected.size() == 2 && selected[0].expert == 0 &&
            selected[1].expert == 1,
            "router selected wrong experts");
    require(selected[0].probability > selected[1].probability,
            "router probabilities are not ordered");
    const float normalized_before_scale =
        selected[0].weight + selected[1].weight / 2.0f;
    require(std::abs(normalized_before_scale - 1.0f) < 1e-5f,
            "router top-k normalization/scaling incorrect");
}

void test_image_preprocessing() {
    using namespace gemmaedge;
    RgbImage image;
    image.width = 2;
    image.height = 1;
    image.pixels = {0.0f, 0.5f, 1.0f, 1.0f, 0.5f, 0.0f};
    const auto prepared = prepare_gemma4_image(image, 2, 3, 4, 4);
    require(prepared.patches_x % 3 == 0 &&
            prepared.patches_y % 3 == 0,
            "prepared image grid is not pool aligned");
    require((prepared.patches_x / 3) * (prepared.patches_y / 3) == 4,
            "prepared image token budget incorrect");
    for (const float value : prepared.pixels)
        require(value >= -1.0001f && value <= 1.0001f,
                "prepared pixel is outside [-1,1]");
}

void test_sampling() {
    using namespace gemmaedge;
    std::mt19937_64 random(7);
    SamplingConfig greedy;
    greedy.temperature = 0.0f;
    require(sample_token({-1.0f, 3.0f, 2.0f}, greedy, random) == 1,
            "greedy sampler selected wrong token");

    SamplingConfig top_one;
    top_one.temperature = 1.0f;
    top_one.top_k = 1;
    top_one.top_p = 1.0f;
    require(sample_token({4.0f, 1.0f, 0.0f}, top_one, random) == 0,
            "top-k sampler selected excluded token");
}

void write_scalar_helper(std::ostream& out, const char* data, std::size_t bytes) {
    out.write(data, static_cast<std::streamsize>(bytes));
}

template <typename T>
void write_scalar(std::ostream& out, T val) {
    write_scalar_helper(out, reinterpret_cast<const char*>(&val), sizeof(val));
}

void write_string(std::ostream& out, const std::string& str) {
    const std::uint64_t len = str.size();
    write_scalar(out, len);
    write_scalar_helper(out, str.data(), len);
}

void write_tensor_header(std::ostream& out, const std::string& name,
                         const std::vector<std::uint64_t>& dims,
                         std::uint32_t type, std::uint64_t offset) {
    write_string(out, name);
    write_scalar(out, static_cast<std::uint32_t>(dims.size()));
    for (const auto d : dims) {
        write_scalar(out, d);
    }
    write_scalar(out, type);
    write_scalar(out, offset);
}

void create_mock_gguf_file(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to create mock GGUF");

    write_scalar<std::uint32_t>(out, 0x46554747); // magic: "GGUF"
    write_scalar<std::uint32_t>(out, 3); // version
    
    const std::uint64_t tensor_count = 2 + 30 * 20; // 602
    write_scalar(out, tensor_count);

    const std::uint64_t metadata_count = 27;
    write_scalar(out, metadata_count);

    auto write_meta_str = [&](const std::string& k, const std::string& v) {
        write_string(out, k);
        write_scalar<std::uint32_t>(out, 8); // String
        write_string(out, v);
    };
    auto write_meta_u32 = [&](const std::string& k, std::uint32_t v) {
        write_string(out, k);
        write_scalar<std::uint32_t>(out, 4); // Uint32
        write_scalar(out, v);
    };
    auto write_meta_f32 = [&](const std::string& k, float v) {
        write_string(out, k);
        write_scalar<std::uint32_t>(out, 6); // Float32
        write_scalar(out, v);
    };

    write_meta_str("general.architecture", "gemma4");
    write_meta_u32("general.alignment", 32);
    write_meta_u32("gemma4.context_length", 1024);
    write_meta_u32("gemma4.embedding_length", 2816);
    write_meta_u32("gemma4.block_count", 30);
    write_meta_u32("gemma4.attention.head_count", 8);
    write_meta_u32("gemma4.attention.sliding_window", 1024);
    write_meta_u32("gemma4.feed_forward_length", 8);
    write_meta_u32("gemma4.expert_feed_forward_length", 8);
    write_meta_u32("gemma4.expert_count", 128);
    write_meta_u32("gemma4.expert_used_count", 8);
    write_meta_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
    write_meta_f32("gemma4.final_logit_softcapping", 30.0f);
    write_meta_f32("gemma4.rope.freq_base_swa", 10000.0f);
    write_meta_f32("gemma4.rope.freq_base", 1000000.0f);
    write_meta_u32("gemma4.attention.key_length_swa", 16);
    write_meta_u32("gemma4.attention.key_length", 16);

    write_string(out, "gemma4.attention.sliding_window_pattern");
    write_scalar<std::uint32_t>(out, 9); // Array
    write_scalar<std::uint32_t>(out, 10); // Uint64
    write_scalar<std::uint64_t>(out, 30);
    for (int i = 0; i < 30; ++i) write_scalar<std::uint64_t>(out, 0);

    write_string(out, "gemma4.attention.head_count_kv");
    write_scalar<std::uint32_t>(out, 9); // Array
    write_scalar<std::uint32_t>(out, 11); // Int64
    write_scalar<std::uint64_t>(out, 30);
    for (int i = 0; i < 30; ++i) write_scalar<std::int64_t>(out, 8);

    write_string(out, "tokenizer.ggml.tokens");
    write_scalar<std::uint32_t>(out, 9); // Array
    write_scalar<std::uint32_t>(out, 8); // String
    write_scalar<std::uint64_t>(out, 5);
    write_string(out, "<pad>");
    write_string(out, "<eos>");
    write_string(out, "<bos>");
    write_string(out, "<unk>");
    write_string(out, "<turn>");

    write_string(out, "tokenizer.ggml.merges");
    write_scalar<std::uint32_t>(out, 9); // Array
    write_scalar<std::uint32_t>(out, 8); // String
    write_scalar<std::uint64_t>(out, 0);

    write_string(out, "tokenizer.ggml.token_type");
    write_scalar<std::uint32_t>(out, 9); // Array
    write_scalar<std::uint32_t>(out, 11); // Int64
    write_scalar<std::uint64_t>(out, 5);
    for (int i = 0; i < 5; ++i) write_scalar<std::int64_t>(out, 3);

    write_meta_u32("tokenizer.ggml.bos_token_id", 2);
    write_meta_u32("tokenizer.ggml.eos_token_id", 1);
    write_meta_u32("tokenizer.ggml.unknown_token_id", 3);
    write_meta_u32("tokenizer.ggml.padding_token_id", 0);
    write_meta_str("tokenizer.ggml.model", "gemma4");

    auto write_tensor = [&](const std::string& name, const std::vector<std::uint64_t>& dims) {
        write_tensor_header(out, name, dims, 0, 0);
    };

    write_tensor("token_embd.weight", {2816, 5});
    write_tensor("output_norm.weight", {2816});

    for (int i = 0; i < 30; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        write_tensor(prefix + "attn_norm.weight", {2816});
        write_tensor(prefix + "attn_q.weight", {2816, 128});
        write_tensor(prefix + "attn_k.weight", {2816, 128});
        write_tensor(prefix + "attn_output.weight", {128, 2816});
        write_tensor(prefix + "attn_q_norm.weight", {128});
        write_tensor(prefix + "attn_k_norm.weight", {128});
        write_tensor(prefix + "post_attention_norm.weight", {2816});
        write_tensor(prefix + "ffn_gate.weight", {2816, 8});
        write_tensor(prefix + "ffn_up.weight", {2816, 8});
        write_tensor(prefix + "ffn_down.weight", {8, 2816});
        write_tensor(prefix + "ffn_norm.weight", {2816});
        write_tensor(prefix + "post_ffw_norm.weight", {2816});
        write_tensor(prefix + "ffn_gate_inp.weight", {2816, 128});
        write_tensor(prefix + "ffn_gate_inp.scale", {2816});
        write_tensor(prefix + "ffn_gate_up_exps.weight", {2816, 16, 128});
        write_tensor(prefix + "ffn_down_exps.weight", {8, 2816, 128});
        write_tensor(prefix + "ffn_down_exps.scale", {128});
        write_tensor(prefix + "pre_ffw_norm_2.weight", {2816});
        write_tensor(prefix + "post_ffw_norm_1.weight", {2816});
        write_tensor(prefix + "post_ffw_norm_2.weight", {2816});
    }

    const std::uint64_t descriptor_end = static_cast<std::uint64_t>(out.tellp());
    const std::uint64_t data_offset = (descriptor_end + 31) & ~31;
    for (std::uint64_t i = descriptor_end; i < data_offset; ++i) out.put(0);

    const std::size_t floats_needed = 2816 * 16 * 128;
    std::vector<float> data_chunk(65536, 1.0f);
    std::size_t floats_written = 0;
    while (floats_written < floats_needed) {
        const std::size_t chunk = std::min(data_chunk.size(), floats_needed - floats_written);
        out.write(reinterpret_cast<const char*>(data_chunk.data()),
                  static_cast<std::streamsize>(chunk * sizeof(float)));
        floats_written += chunk;
    }
}

void test_generation_autoregressive() {
    using namespace gemmaedge;
    const auto path = std::filesystem::temp_directory_path() / "gemmaedge_mock_model.gguf";
    std::filesystem::remove(path);
    
    create_mock_gguf_file(path);
    
    {
        GgufFile file(path);
        Gemma4Model model(file);
        Gemma4Tokenizer tokenizer(file);
        MappedFile weights(path);
        
        Gemma4Session session(model, weights, 512ULL * 1024ULL * 1024ULL);
        
        const std::vector<TokenId> prompt = {2, 0};
        
        GenerationConfig config;
        config.max_new_tokens = 3;
        config.stop_tokens = {1, 106};
        config.sampling.temperature = 0.0f; // greedy
        
        std::vector<TokenId> generated;
        session.generate(prompt, config, [&](TokenId token) {
            generated.push_back(token);
        });
        
        require(generated.size() == 3, "autoregressive generation token count mismatch");
    }
    
    std::filesystem::remove(path);
}

} // namespace

int main() {
    try {
        test_expert_cache();
        test_kv_store();
        test_tensor_primitives();
        test_ggml_layouts();
        test_mapped_file();
        test_tokenizer();
        test_rope_attention_kv();
        test_router_top_k();
        test_image_preprocessing();
        test_sampling();
        test_generation_autoregressive();
        std::cout << "all core tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
