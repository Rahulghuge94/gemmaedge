#include "gemmaedge/tensor.h"
#include "gemmaedge/vulkan_backend.h"
#include "gemmaedge/cuda_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <climits>
#include <stdexcept>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <thread>
#include <functional>

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define GEMMAEDGE_AVX2 1
#endif

namespace gemmaedge {

// A lightweight thread pool for parallel matrix-vector multiplication.
class ThreadPool {
public:
    ThreadPool(std::size_t threads) : stop(false) {
        for (std::size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker: workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

static ThreadPool& get_thread_pool() {
    static unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
#if defined(GEMMAEDGE_ANDROID)
    // Cap at 4 threads for mobile/Android big cores to optimize thermal/power efficiency
    static ThreadPool pool(std::min(4u, num_threads));
#else
    // Desktop: use all available cores for maximum throughput
    static ThreadPool pool(num_threads);
#endif
    return pool;
}

#include <fstream>

static int get_cpu_temperature_celsius() {
#ifdef GEMMAEDGE_LINUX
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    if (temp_file.is_open()) {
        int temp_raw = 0;
        if (temp_file >> temp_raw) {
            // Millidegrees Celsius
            return temp_raw / 1000;
        }
    }
#endif
    return 35; // Default safe temperature
}

template <typename F>
static void parallel_for(std::size_t start, std::size_t end, F&& loop_body) {
    auto& pool = get_thread_pool();
    std::size_t total = end - start;
    if (total == 0) return;
    
    if (total < 16) {
        for (std::size_t i = start; i < end; ++i) {
            loop_body(i);
        }
        return;
    }

    std::size_t num_workers = std::thread::hardware_concurrency();
    if (num_workers == 0) num_workers = 4;
#if defined(GEMMAEDGE_ANDROID)
    num_workers = std::min(4u, static_cast<unsigned int>(num_workers));
#endif

    // Dynamic thermal awareness: scale down concurrency under high load/heat.
    const int temp_c = get_cpu_temperature_celsius();
    if (temp_c > 75) {
        num_workers = std::min<std::size_t>(2, num_workers);
    }
    if (temp_c > 85) {
        num_workers = 1;
    }

    std::size_t chunk_size = (total + num_workers - 1) / num_workers;
    std::vector<std::future<void>> futures;
    futures.reserve(num_workers);

    for (std::size_t w = 0; w < num_workers; ++w) {
        std::size_t chunk_start = start + w * chunk_size;
        std::size_t chunk_end = std::min(end, chunk_start + chunk_size);
        if (chunk_start >= end) break;
        
        futures.push_back(pool.enqueue([chunk_start, chunk_end, &loop_body]() {
            for (std::size_t i = chunk_start; i < chunk_end; ++i) {
                loop_body(i);
            }
        }));
    }

    for (auto& fut : futures) {
        fut.wait();
    }
}

namespace {
struct F16ToF32Table {
    float table[65536];
    F16ToF32Table() {
        for (std::uint32_t i = 0; i < 65536; ++i) {
            std::uint16_t value = static_cast<std::uint16_t>(i);
            const std::uint32_t sign = (value & 0x8000u) << 16;
            std::uint32_t exponent = (value >> 10) & 0x1fu;
            std::uint32_t mantissa = value & 0x3ffu;
            std::uint32_t bits;
            if (exponent == 0) {
                if (mantissa == 0) {
                    bits = sign;
                } else {
                    exponent = 1;
                    while ((mantissa & 0x400u) == 0) {
                        mantissa <<= 1;
                        --exponent;
                    }
                    mantissa &= 0x3ffu;
                    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
                }
            } else if (exponent == 31) {
                bits = sign | 0x7f800000u | (mantissa << 13);
            } else {
                bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
            }
            float result;
            std::memcpy(&result, &bits, sizeof(result));
            table[i] = result;
        }
    }
};

const F16ToF32Table g_f16_to_f32_table;
} // namespace

float f16_to_f32(std::uint16_t value) noexcept {
    return g_f16_to_f32_table.table[value];
}

std::uint16_t f32_to_f16(float value) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == -127) {
        return static_cast<std::uint16_t>(sign);
    }
    if (exponent > 15) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (exponent < -14) {
        const std::int32_t shift = -14 - exponent;
        if (shift > 24) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        mantissa >>= shift;
        return static_cast<std::uint16_t>(sign | (mantissa >> 13));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent + 15) << 10) |
        (mantissa >> 13));
}

float bf16_to_f32(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t f32_to_bf16(float value) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

Q4Block quantize_q4_block(const float* values) {
    float maximum = 0.0f;
    for (std::size_t i = 0; i < kQ4BlockSize; ++i)
        maximum = std::max(maximum, std::abs(values[i]));
    const float scale = maximum == 0.0f ? 0.0f : maximum / 7.0f;
    Q4Block block{};
    block.scale_f16 = f32_to_f16(scale);
    for (std::size_t i = 0; i < kQ4BlockSize / 2; ++i) {
        const int q0 = scale == 0.0f ? 0 :
            static_cast<int>(std::nearbyint(values[i] / scale)) + 8;
        const int q1 = scale == 0.0f ? 0 :
            static_cast<int>(std::nearbyint(values[i + kQ4BlockSize / 2] / scale)) + 8;
        const auto val0 = static_cast<std::uint8_t>(std::max(0, std::min(15, q0)));
        const auto val1 = static_cast<std::uint8_t>(std::max(0, std::min(15, q1)));
        block.quants[i] = val0 | (val1 << 4);
    }
    return block;
}

void dequantize_q4_block(const Q4Block& block, float* output) {
    const float scale = f16_to_f32(block.scale_f16);
    for (std::size_t i = 0; i < kQ4BlockSize / 2; ++i) {
        const std::uint8_t packed = block.quants[i];
        output[i] = (static_cast<int>(packed & 0x0fu) - 8) * scale;
        output[i + kQ4BlockSize / 2] =
            (static_cast<int>(packed >> 4) - 8) * scale;
    }
}

void q4_matvec(const Q4Block* matrix, std::size_t rows, std::size_t cols,
               const float* x, float* output) {
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        const auto* blocks = matrix + row * (cols / kQ4BlockSize);
        for (std::size_t b = 0; b < cols / kQ4BlockSize; ++b) {
            const float scale = f16_to_f32(blocks[b].scale_f16);
            const float* vec_slice = x + b * kQ4BlockSize;
            for (std::size_t i = 0; i < kQ4BlockSize / 2; ++i) {
                const std::uint8_t packed = blocks[b].quants[i];
                sum += (static_cast<int>(packed & 0x0f) - 8) * scale *
                       vec_slice[i];
                sum += (static_cast<int>(packed >> 4) - 8) * scale *
                       vec_slice[i + kQ4BlockSize / 2];
            }
        }
        output[row] = sum;
    }
}

BlockQ8_0 quantize_q8_block(const float* values) {
    float maximum = 0.0f;
    for (std::size_t i = 0; i < 32; ++i)
        maximum = std::max(maximum, std::abs(values[i]));
    const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    BlockQ8_0 block{};
    block.d = f32_to_f16(scale);
    for (std::size_t i = 0; i < 32; ++i) {
        const int quantized = scale == 0.0f ? 0 :
            static_cast<int>(std::nearbyint(values[i] / scale));
        block.q[i] = static_cast<std::int8_t>(
            std::max(-127, std::min(127, quantized)));
    }
    return block;
}

void dequantize_q8_block(const BlockQ8_0& block, float* output) {
    const float scale = f16_to_f32(block.d);
    for (std::size_t i = 0; i < 32; ++i) {
        output[i] = block.q[i] * scale;
    }
}

#pragma pack(push, 1)
struct BlockQ5_0 {
    std::uint16_t d;
    std::uint8_t qh[4];
    std::uint8_t q[16];
};
struct BlockQ4K {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t q[128];
};
struct BlockQ6K {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    std::uint16_t d;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ5_0) == 22, "GGML Q5_0 layout mismatch");
static_assert(sizeof(BlockQ4K) == 144, "GGML Q4_K layout mismatch");
static_assert(sizeof(BlockQ6K) == 210, "GGML Q6_K layout mismatch");

namespace {

void q4k_scale_min(int index, const std::uint8_t* packed,
                   std::uint8_t& scale, std::uint8_t& minimum) {
    if (index < 4) {
        scale = packed[index] & 63;
        minimum = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 0x0f) |
                ((packed[index - 4] >> 6) << 4);
        minimum = (packed[index + 4] >> 4) |
                  ((packed[index] >> 6) << 4);
    }
}

#if defined(__ARM_NEON)
inline float dot_q4k_neon(const BlockQ4K& block, const float* x) {
    const float d = f16_to_f32(block.d);
    const float dmin = f16_to_f32(block.dmin);
    float sum = 0.0f;
    int group = 0;
    const std::uint8_t* q = block.q;

    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    for (int base = 0; base < 256; base += 64) {
        std::uint8_t scale1, min1, scale2, min2;
        q4k_scale_min(group++, block.scales, scale1, min1);
        q4k_scale_min(group++, block.scales, scale2, min2);

        const float factor1 = d * scale1;
        const float bias1 = -dmin * min1;
        const float factor2 = d * scale2;
        const float bias2 = -dmin * min2;

        float32x4_t f1_vec = vdupq_n_f32(factor1);
        float32x4_t b1_vec = vdupq_n_f32(bias1);
        float32x4_t f2_vec = vdupq_n_f32(factor2);
        float32x4_t b2_vec = vdupq_n_f32(bias2);

        uint8x16_t packed0 = vld1q_u8(q);
        uint8x16_t packed1 = vld1q_u8(q + 16);
        uint8x16_t mask = vdupq_n_u8(0x0F);

        uint8x16_t low_nibbles0 = vandq_u8(packed0, mask);
        uint8x16_t high_nibbles0 = vshrq_n_u8(packed0, 4);

        uint8x16_t low_nibbles1 = vandq_u8(packed1, mask);
        uint8x16_t high_nibbles1 = vshrq_n_u8(packed1, 4);

        uint16x8_t low_lo0 = vmovl_u8(vget_low_u8(low_nibbles0));
        uint16x8_t low_hi0 = vmovl_u8(vget_high_u8(low_nibbles0));
        float32x4_t l0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(low_lo0)));
        float32x4_t l1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(low_lo0)));
        float32x4_t l2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(low_hi0)));
        float32x4_t l3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(low_hi0)));
        l0 = vmlaq_f32(b1_vec, l0, f1_vec);
        l1 = vmlaq_f32(b1_vec, l1, f1_vec);
        l2 = vmlaq_f32(b1_vec, l2, f1_vec);
        l3 = vmlaq_f32(b1_vec, l3, f1_vec);

        uint16x8_t low_lo1 = vmovl_u8(vget_low_u8(low_nibbles1));
        uint16x8_t low_hi1 = vmovl_u8(vget_high_u8(low_nibbles1));
        float32x4_t l4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(low_lo1)));
        float32x4_t l5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(low_lo1)));
        float32x4_t l6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(low_hi1)));
        float32x4_t l7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(low_hi1)));
        l4 = vmlaq_f32(b1_vec, l4, f1_vec);
        l5 = vmlaq_f32(b1_vec, l5, f1_vec);
        l6 = vmlaq_f32(b1_vec, l6, f1_vec);
        l7 = vmlaq_f32(b1_vec, l7, f1_vec);

        uint16x8_t high_lo0 = vmovl_u8(vget_low_u8(high_nibbles0));
        uint16x8_t high_hi0 = vmovl_u8(vget_high_u8(high_nibbles0));
        float32x4_t h0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(high_lo0)));
        float32x4_t h1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(high_lo0)));
        float32x4_t h2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(high_hi0)));
        float32x4_t h3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(high_hi0)));
        h0 = vmlaq_f32(b2_vec, h0, f2_vec);
        h1 = vmlaq_f32(b2_vec, h1, f2_vec);
        h2 = vmlaq_f32(b2_vec, h2, f2_vec);
        h3 = vmlaq_f32(b2_vec, h3, f2_vec);

        uint16x8_t high_lo1 = vmovl_u8(vget_low_u8(high_nibbles1));
        uint16x8_t high_hi1 = vmovl_u8(vget_high_u8(high_nibbles1));
        float32x4_t h4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(high_lo1)));
        float32x4_t h5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(high_lo1)));
        float32x4_t h6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(high_hi1)));
        float32x4_t h7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(high_hi1)));
        h4 = vmlaq_f32(b2_vec, h4, f2_vec);
        h5 = vmlaq_f32(b2_vec, h5, f2_vec);
        h6 = vmlaq_f32(b2_vec, h6, f2_vec);
        h7 = vmlaq_f32(b2_vec, h7, f2_vec);

        sum_vec = vmlaq_f32(sum_vec, l0, vld1q_f32(x + base));
        sum_vec = vmlaq_f32(sum_vec, l1, vld1q_f32(x + base + 4));
        sum_vec = vmlaq_f32(sum_vec, l2, vld1q_f32(x + base + 8));
        sum_vec = vmlaq_f32(sum_vec, l3, vld1q_f32(x + base + 12));

        sum_vec = vmlaq_f32(sum_vec, l4, vld1q_f32(x + base + 16));
        sum_vec = vmlaq_f32(sum_vec, l5, vld1q_f32(x + base + 20));
        sum_vec = vmlaq_f32(sum_vec, l6, vld1q_f32(x + base + 24));
        sum_vec = vmlaq_f32(sum_vec, l7, vld1q_f32(x + base + 28));

        sum_vec = vmlaq_f32(sum_vec, h0, vld1q_f32(x + base + 32));
        sum_vec = vmlaq_f32(sum_vec, h1, vld1q_f32(x + base + 36));
        sum_vec = vmlaq_f32(sum_vec, h2, vld1q_f32(x + base + 40));
        sum_vec = vmlaq_f32(sum_vec, h3, vld1q_f32(x + base + 44));

        sum_vec = vmlaq_f32(sum_vec, h4, vld1q_f32(x + base + 48));
        sum_vec = vmlaq_f32(sum_vec, h5, vld1q_f32(x + base + 52));
        sum_vec = vmlaq_f32(sum_vec, h6, vld1q_f32(x + base + 56));
        sum_vec = vmlaq_f32(sum_vec, h7, vld1q_f32(x + base + 60));

        q += 32;
    }

    sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
          vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
    return sum;
}
#endif

float dot_q4k(const BlockQ4K& block, const float* x) {
#if defined(__ARM_NEON)
    return dot_q4k_neon(block, x);
#elif defined(GEMMAEDGE_AVX2)
    const float d = f16_to_f32(block.d);
    const float dmin = f16_to_f32(block.dmin);
    __m256 sum_vec = _mm256_setzero_ps();
    int group = 0;
    const std::uint8_t* q = block.q;
    for (int base = 0; base < 256; base += 64) {
        std::uint8_t scale1, min1, scale2, min2;
        q4k_scale_min(group++, block.scales, scale1, min1);
        q4k_scale_min(group++, block.scales, scale2, min2);
        __m256 f1 = _mm256_set1_ps(d * scale1);
        __m256 b1 = _mm256_set1_ps(-dmin * min1);
        __m256 f2 = _mm256_set1_ps(d * scale2);
        __m256 b2 = _mm256_set1_ps(-dmin * min2);
        // Low nibbles (32 values)
        for (int i = 0; i < 32; i += 8) {
            __m256 xv = _mm256_loadu_ps(x + base + i);
            // Extract low nibbles from q[i..i+7]
            __m256 qv = _mm256_set_ps(
                (float)(q[i+7] & 0x0f), (float)(q[i+6] & 0x0f),
                (float)(q[i+5] & 0x0f), (float)(q[i+4] & 0x0f),
                (float)(q[i+3] & 0x0f), (float)(q[i+2] & 0x0f),
                (float)(q[i+1] & 0x0f), (float)(q[i+0] & 0x0f));
            __m256 dq = _mm256_fmadd_ps(qv, f1, b1);
            sum_vec = _mm256_fmadd_ps(dq, xv, sum_vec);
        }
        // High nibbles (32 values)
        for (int i = 0; i < 32; i += 8) {
            __m256 xv = _mm256_loadu_ps(x + base + 32 + i);
            __m256 qv = _mm256_set_ps(
                (float)(q[i+7] >> 4), (float)(q[i+6] >> 4),
                (float)(q[i+5] >> 4), (float)(q[i+4] >> 4),
                (float)(q[i+3] >> 4), (float)(q[i+2] >> 4),
                (float)(q[i+1] >> 4), (float)(q[i+0] >> 4));
            __m256 dq = _mm256_fmadd_ps(qv, f2, b2);
            sum_vec = _mm256_fmadd_ps(dq, xv, sum_vec);
        }
        q += 32;
    }
    // Horizontal sum of 8 floats
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
#else
    const float d = f16_to_f32(block.d);
    const float dmin = f16_to_f32(block.dmin);
    float sum = 0.0f;
    int group = 0;
    const std::uint8_t* q = block.q;
    for (int base = 0; base < 256; base += 64) {
        std::uint8_t scale1, min1, scale2, min2;
        q4k_scale_min(group++, block.scales, scale1, min1);
        q4k_scale_min(group++, block.scales, scale2, min2);
        for (int i = 0; i < 32; ++i)
            sum += (d * scale1 * (q[i] & 0x0f) - dmin * min1) *
                   x[base + i];
        for (int i = 0; i < 32; ++i)
            sum += (d * scale2 * (q[i] >> 4) - dmin * min2) *
                   x[base + 32 + i];
        q += 32;
    }
    return sum;
#endif
}

float dot_q6k(const BlockQ6K& block, const float* x) {
    const float d = f16_to_f32(block.d);
    float sum = 0.0f;
    const std::uint8_t* ql = block.ql;
    const std::uint8_t* qh = block.qh;
    const std::int8_t* scales = block.scales;
    for (int base = 0; base < 256; base += 128) {
        for (int i = 0; i < 32; ++i) {
            const int group = i / 16;
            const int q1 = ((ql[i] & 0x0f) | (((qh[i] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[i + 32] & 0x0f) | (((qh[i] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[i] >> 4) | (((qh[i] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[i + 32] >> 4) | (((qh[i] >> 6) & 3) << 4)) - 32;
            sum += d * scales[group + 0] * q1 * x[base + i];
            sum += d * scales[group + 2] * q2 * x[base + 32 + i];
            sum += d * scales[group + 4] * q3 * x[base + 64 + i];
            sum += d * scales[group + 6] * q4 * x[base + 96 + i];
        }
        ql += 64;
        qh += 32;
        scales += 8;
    }
    return sum;
}

} // namespace


void ggml_matvec(GgmlType type, const void* matrix, std::size_t rows,
                 std::size_t cols, const float* vector, float* output) {
    if (!matrix || !vector || !output)
        throw std::invalid_argument("null GGML matvec input");

    if (is_cuda_available()) {
        if (cuda_matvec(type, matrix, rows, cols, vector, output)) {
            return;
        }
    }

    if (is_vulkan_available()) {
        if (vulkan_matvec(matrix, rows, cols, vector, output)) {
            return;
        }
    }

    parallel_for(0, rows, [&](std::size_t row) {
        float sum = 0.0f;
        switch (type) {
            case GgmlType::F32: {
                const auto* weights = static_cast<const float*>(matrix) + row * cols;
                for (std::size_t i = 0; i < cols; ++i) sum += weights[i] * vector[i];
                break;
            }
            case GgmlType::F16:
            case GgmlType::BF16: {
                const auto* weights = static_cast<const std::uint16_t*>(matrix) + row * cols;
                for (std::size_t i = 0; i < cols; ++i) {
                    const float weight = type == GgmlType::F16
                        ? f16_to_f32(weights[i]) : bf16_to_f32(weights[i]);
                    sum += weight * vector[i];
                }
                break;
            }
            case GgmlType::Q4_0: {
                if (cols % 32) throw std::invalid_argument("unaligned Q4_0 row");
                const auto* blocks = static_cast<const Q4Block*>(matrix) +
                    row * (cols / 32);
                q4_matvec(blocks, 1, cols, vector, &sum);
                break;
            }
            case GgmlType::Q8_0: {
                if (cols % 32) throw std::invalid_argument("unaligned Q8_0 row");
                const auto* blocks = static_cast<const BlockQ8_0*>(matrix) +
                    row * (cols / 32);
                for (std::size_t b = 0; b < cols / 32; ++b) {
#if defined(__ARM_NEON)
                    sum += dot_q8_block_neon(blocks[b], vector + b * 32);
#elif defined(__AVX2__)
                    sum += dot_q8_block_avx2(blocks[b], vector + b * 32);
#else
                    const float d = f16_to_f32(blocks[b].d);
                    for (std::size_t i = 0; i < 32; ++i)
                        sum += d * blocks[b].q[i] * vector[b * 32 + i];
#endif
                }
                break;
            }
            case GgmlType::Q5_0: {
                if (cols % 32) throw std::invalid_argument("unaligned Q5_0 row");
                const auto* blocks = static_cast<const BlockQ5_0*>(matrix) +
                    row * (cols / 32);
                for (std::size_t b = 0; b < cols / 32; ++b) {
                    const float d = f16_to_f32(blocks[b].d);
                    std::uint32_t high = 0;
                    std::memcpy(&high, blocks[b].qh, sizeof(high));
                    for (std::size_t i = 0; i < 16; ++i) {
                        const int hi0 = ((high >> i) << 4) & 0x10;
                        const int hi1 = (high >> (i + 12)) & 0x10;
                        const int q0 = ((blocks[b].q[i] & 0x0f) | hi0) - 16;
                        const int q1 = ((blocks[b].q[i] >> 4) | hi1) - 16;
                        sum += d * q0 * vector[b * 32 + i];
                        sum += d * q1 * vector[b * 32 + 16 + i];
                    }
                }
                break;
            }
            case GgmlType::Q4_K: {
                if (cols % 256) throw std::invalid_argument("unaligned Q4_K row");
                const auto* blocks = static_cast<const BlockQ4K*>(matrix) +
                    row * (cols / 256);
                for (std::size_t b = 0; b < cols / 256; ++b)
                    sum += dot_q4k(blocks[b], vector + b * 256);
                break;
            }
            case GgmlType::Q6_K: {
                if (cols % 256) throw std::invalid_argument("unaligned Q6_K row");
                const auto* blocks = static_cast<const BlockQ6K*>(matrix) +
                    row * (cols / 256);
                for (std::size_t b = 0; b < cols / 256; ++b)
                    sum += dot_q6k(blocks[b], vector + b * 256);
                break;
            }
            default:
                throw std::runtime_error("GGML row type is not implemented");
        }
        output[row] = sum;
    });
}

void ggml_dequantize_row(GgmlType type, const void* row, std::size_t cols,
                         float* output) {
    if (!row || !output) throw std::invalid_argument("null GGML dequantize inputs");
    switch (type) {
        case GgmlType::F32: {
            std::memcpy(output, row, cols * sizeof(float));
            return;
        }
        case GgmlType::F16:
        case GgmlType::BF16: {
            const auto* weights = static_cast<const std::uint16_t*>(row);
            for (std::size_t i = 0; i < cols; ++i)
                output[i] = type == GgmlType::F16
                    ? f16_to_f32(weights[i]) : bf16_to_f32(weights[i]);
            return;
        }
        case GgmlType::Q4_0: {
            if (cols % 32) throw std::invalid_argument("unaligned Q4_0 row");
            const auto* blocks = static_cast<const Q4Block*>(row);
            for (std::size_t b = 0; b < cols / 32; ++b)
                dequantize_q4_block(blocks[b], output + b * 32);
            return;
        }
        case GgmlType::Q5_0: {
            if (cols % 32) throw std::invalid_argument("unaligned Q5_0 row");
            const auto* blocks = static_cast<const BlockQ5_0*>(row);
            for (std::size_t b = 0; b < cols / 32; ++b) {
                const float d = f16_to_f32(blocks[b].d);
                std::uint32_t high = 0;
                std::memcpy(&high, blocks[b].qh, sizeof(high));
                for (std::size_t i = 0; i < 16; ++i) {
                    const int hi0 = ((high >> i) << 4) & 0x10;
                    const int hi1 = (high >> (i + 12)) & 0x10;
                    output[b * 32 + i] =
                        (((blocks[b].q[i] & 0x0f) | hi0) - 16) * d;
                    output[b * 32 + 16 + i] =
                        (((blocks[b].q[i] >> 4) | hi1) - 16) * d;
                }
            }
            return;
        }
        case GgmlType::Q8_0: {
            if (cols % 32) throw std::invalid_argument("unaligned Q8_0 row");
            const auto* blocks = static_cast<const BlockQ8_0*>(row);
            for (std::size_t b = 0; b < cols / 32; ++b) {
                const float d = f16_to_f32(blocks[b].d);
                for (std::size_t i = 0; i < 32; ++i)
                    output[b * 32 + i] = blocks[b].q[i] * d;
            }
            return;
        }
        case GgmlType::Q4_K: {
            if (cols % 256) throw std::invalid_argument("unaligned Q4_K row");
            const auto* blocks = static_cast<const BlockQ4K*>(row);
            for (std::size_t b = 0; b < cols / 256; ++b) {
                const float d = f16_to_f32(blocks[b].d);
                const float dmin = f16_to_f32(blocks[b].dmin);
                int group = 0;
                const std::uint8_t* q = blocks[b].q;
                for (int base = 0; base < 256; base += 64) {
                    std::uint8_t s1, m1, s2, m2;
                    q4k_scale_min(group++, blocks[b].scales, s1, m1);
                    q4k_scale_min(group++, blocks[b].scales, s2, m2);
                    for (int i = 0; i < 32; ++i)
                        output[b * 256 + base + i] =
                            d * s1 * (q[i] & 0x0f) - dmin * m1;
                    for (int i = 0; i < 32; ++i)
                        output[b * 256 + base + 32 + i] =
                            d * s2 * (q[i] >> 4) - dmin * m2;
                    q += 32;
                }
            }
            return;
        }
        case GgmlType::Q6_K: {
            if (cols % 256) throw std::invalid_argument("unaligned Q6_K row");
            const auto* blocks = static_cast<const BlockQ6K*>(row);
            for (std::size_t b = 0; b < cols / 256; ++b) {
                const float d = f16_to_f32(blocks[b].d);
                const std::uint8_t* ql = blocks[b].ql;
                const std::uint8_t* qh = blocks[b].qh;
                const std::int8_t* scales = blocks[b].scales;
                for (int base = 0; base < 256; base += 128) {
                    for (int i = 0; i < 32; ++i) {
                        const int g = i / 16;
                        const int q1 = ((ql[i] & 15) | (((qh[i] >> 0) & 3) << 4)) - 32;
                        const int q2 = ((ql[i + 32] & 15) | (((qh[i] >> 2) & 3) << 4)) - 32;
                        const int q3 = ((ql[i] >> 4) | (((qh[i] >> 4) & 3) << 4)) - 32;
                        const int q4 = ((ql[i + 32] >> 4) | (((qh[i] >> 6) & 3) << 4)) - 32;
                        output[b * 256 + base + i] = d * scales[g] * q1;
                        output[b * 256 + base + 32 + i] = d * scales[g + 2] * q2;
                        output[b * 256 + base + 64 + i] = d * scales[g + 4] * q3;
                        output[b * 256 + base + 96 + i] = d * scales[g + 6] * q4;
                    }
                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
            }
            return;
        }
        default:
            throw std::runtime_error("GGML row type is not implemented");
    }
}

void rms_norm(const float* input, const float* weight, std::size_t size,
              float epsilon, float* output) {
#if defined(__ARM_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t in = vld1q_f32(input + i);
        sum_vec = vmlaq_f32(sum_vec, in, in);
    }
    float sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
    for (; i < size; ++i) {
        sum += input[i] * input[i];
    }
    const float inverse = 1.0f / std::sqrt(sum / size + epsilon);
    float32x4_t inv_vec = vdupq_n_f32(inverse);

    i = 0;
    if (weight) {
        for (; i + 3 < size; i += 4) {
            float32x4_t in = vld1q_f32(input + i);
            float32x4_t w = vld1q_f32(weight + i);
            float32x4_t res = vmulq_f32(vmulq_f32(in, inv_vec), w);
            vst1q_f32(output + i, res);
        }
    } else {
        for (; i + 3 < size; i += 4) {
            float32x4_t in = vld1q_f32(input + i);
            float32x4_t res = vmulq_f32(in, inv_vec);
            vst1q_f32(output + i, res);
        }
    }
    for (; i < size; ++i) {
        output[i] = input[i] * inverse * (weight ? weight[i] : 1.0f);
    }
#elif defined(GEMMAEDGE_AVX2)
    // AVX2 vectorized rms_norm
    __m256 sq_vec = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 in = _mm256_loadu_ps(input + i);
        sq_vec = _mm256_fmadd_ps(in, in, sq_vec);
    }
    // Horizontal sum of sq_vec
    __m128 lo = _mm256_castps256_ps128(sq_vec);
    __m128 hi = _mm256_extractf128_ps(sq_vec, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    float squares = _mm_cvtss_f32(s1);
    for (; i < size; ++i)
        squares += input[i] * input[i];
    const float inverse = 1.0f / std::sqrt(squares / static_cast<float>(size) + epsilon);
    __m256 inv_vec = _mm256_set1_ps(inverse);

    i = 0;
    if (weight) {
        for (; i + 7 < size; i += 8) {
            __m256 in = _mm256_loadu_ps(input + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 res = _mm256_mul_ps(_mm256_mul_ps(in, inv_vec), w);
            _mm256_storeu_ps(output + i, res);
        }
    } else {
        for (; i + 7 < size; i += 8) {
            __m256 in = _mm256_loadu_ps(input + i);
            __m256 res = _mm256_mul_ps(in, inv_vec);
            _mm256_storeu_ps(output + i, res);
        }
    }
    for (; i < size; ++i)
        output[i] = input[i] * inverse * (weight ? weight[i] : 1.0f);
#else
    double squares = 0.0;
    for (std::size_t i = 0; i < size; ++i)
        squares += static_cast<double>(input[i]) * input[i];
    const float inverse = 1.0f /
        std::sqrt(static_cast<float>(squares / size) + epsilon);
    for (std::size_t i = 0; i < size; ++i)
        output[i] = input[i] * inverse * (weight ? weight[i] : 1.0f);
#endif
}

void softmax(float* values, std::size_t size) {
    if (size == 0) return;
    const float maximum = *std::max_element(values, values + size);
    double sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = std::exp(values[i] - maximum);
        sum += values[i];
    }
    const float inverse = static_cast<float>(1.0 / sum);
    for (std::size_t i = 0; i < size; ++i) values[i] *= inverse;
}

std::vector<std::pair<std::uint32_t, float>>
top_k(const float* values, std::size_t size, std::size_t k) {
    k = std::min(k, size);
    std::vector<std::pair<std::uint32_t, float>> result;
    result.reserve(size);
    for (std::size_t i = 0; i < size; ++i)
        result.emplace_back(static_cast<std::uint32_t>(i), values[i]);
    std::partial_sort(result.begin(), result.begin() + k, result.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    result.resize(k);
    return result;
}

float gelu_tanh(float value) noexcept {
    constexpr float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * value *
        (1.0f + std::tanh(k * (value + 0.044715f * value * value * value)));
}

} // namespace gemmaedge
