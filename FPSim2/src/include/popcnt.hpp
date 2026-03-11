#pragma once
/**
 * SIMD-optimized popcount for fingerprint similarity.
 *
 * Provides CommonBitsCount(a, b, n) which computes popcount(a[0..n] & b[0..n]).
 *
 * Build options (environment variables in setup.py):
 *
 * FPSIM2_ARCH - Architecture selection:
 *   - default: Scalar POPCNT, works on all x86-64 CPUs (Nehalem+)
 *   - avx512: AVX-512 VPOPCNTDQ (Ice Lake+, Zen4+), best performance
 *   - native: Auto-detect via -march=native
 *
 * FPSIM2_FP_SIZE - Fixed fingerprint size (in uint64s) for zero-overhead:
 *   - unset: Runtime size (default, flexible)
 *   - 4: 256-bit fingerprints - single SIMD op
 *   - 8: 512-bit fingerprints - single SIMD op
 *   - 16: 1024-bit fingerprints
 *   - 32: 2048-bit fingerprints
 *
 * Usage:
 *   # Maximum compatibility (default for wheels)
 *   pip install .
 *
 *   # AVX-512 + fixed 256-bit fingerprints (zero overhead)
 *   FPSIM2_ARCH=avx512 FPSIM2_FP_SIZE=4 pip install .
 *
 * ARM64 always uses NEON (mandatory on ARMv8).
 */

#include <cstdint>
#include <cstddef>

// ============================================================================
// Scalar popcount intrinsic (always available)
// ============================================================================

#if defined(_MSC_VER)
#include <intrin.h>
static inline uint64_t popcntll(uint64_t x) { return __popcnt64(x); }
#else
static inline uint64_t popcntll(uint64_t x) { return __builtin_popcountll(x); }
#endif

// ============================================================================
// Platform detection
// ============================================================================

#if defined(__aarch64__) || defined(_M_ARM64)
#define FPSIM2_ARM64
#elif defined(__x86_64__) || defined(_M_X64)
#define FPSIM2_X86_64
#endif

// ============================================================================
// Template-based CommonBitsCount for compile-time known sizes
// Uses if constexpr for zero-overhead dispatch
// ============================================================================

#if defined(FPSIM2_ARM64)
#include <arm_neon.h>

template <size_t N>
static inline uint64_t CommonBitsCountFixed(const uint64_t *a, const uint64_t *b)
{
    uint64x2_t accum = vdupq_n_u64(0);
    size_t i = 0;

    // Compiler will unroll this for small N
    for (; i + 2 <= N; i += 2)
    {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vc = vandq_u64(va, vb);
        uint8x16_t cnt = vcntq_u8(vreinterpretq_u8_u64(vc));
        accum = vaddq_u64(accum, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(cnt))));
    }

    uint64_t result = vgetq_lane_u64(accum, 0) + vgetq_lane_u64(accum, 1);
    if constexpr (N % 2 == 1)
    {
        result += popcntll(a[N - 1] & b[N - 1]);
    }
    return result;
}

static inline uint64_t CommonBitsCount(const uint64_t *a, const uint64_t *b, [[maybe_unused]] size_t n)
{
#ifdef FPSIM2_FP_SIZE
    return CommonBitsCountFixed<FPSIM2_FP_SIZE>(a, b);
#else
    uint64x2_t accum = vdupq_n_u64(0);
    size_t i = 0;
    for (; i + 2 <= n; i += 2)
    {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vc = vandq_u64(va, vb);
        uint8x16_t cnt = vcntq_u8(vreinterpretq_u8_u64(vc));
        accum = vaddq_u64(accum, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(cnt))));
    }
    uint64_t result = vgetq_lane_u64(accum, 0) + vgetq_lane_u64(accum, 1);
    for (; i < n; ++i)
    {
        result += popcntll(a[i] & b[i]);
    }
    return result;
#endif
}

#elif defined(FPSIM2_X86_64) && defined(__AVX512VPOPCNTDQ__)
// ---------------------------------------------------------------------------
// x86-64 AVX-512 VPOPCNTDQ: All CPUs with this also have AVX512VL (Ice Lake+)
// ---------------------------------------------------------------------------
#include <immintrin.h>

// Fixed-size template: compiler fully unrolls, no branches
template <size_t N>
static inline uint64_t CommonBitsCountFixed(const uint64_t *a, const uint64_t *b)
{
    if constexpr (N == 4)
    {
        // Single 256-bit operation - optimal for 256-bit fingerprints
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));
        __m256i vc = _mm256_and_si256(va, vb);
        __m256i cnt = _mm256_popcnt_epi64(vc);
        __m128i lo = _mm256_castsi256_si128(cnt);
        __m128i hi = _mm256_extracti128_si256(cnt, 1);
        __m128i sum = _mm_add_epi64(lo, hi);
        return static_cast<uint64_t>(_mm_extract_epi64(sum, 0)) +
               static_cast<uint64_t>(_mm_extract_epi64(sum, 1));
    }
    else if constexpr (N == 8)
    {
        // Single 512-bit operation - optimal for 512-bit fingerprints
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b));
        __m512i vc = _mm512_and_si512(va, vb);
        return _mm512_reduce_add_epi64(_mm512_popcnt_epi64(vc));
    }
    else if constexpr (N == 16)
    {
        // Two 512-bit operations - 1024-bit fingerprints
        __m512i va0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a));
        __m512i vb0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b));
        __m512i va1 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + 8));
        __m512i vb1 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + 8));
        __m512i cnt0 = _mm512_popcnt_epi64(_mm512_and_si512(va0, vb0));
        __m512i cnt1 = _mm512_popcnt_epi64(_mm512_and_si512(va1, vb1));
        return _mm512_reduce_add_epi64(_mm512_add_epi64(cnt0, cnt1));
    }
    else if constexpr (N == 32)
    {
        // Four 512-bit operations - 2048-bit fingerprints
        __m512i accum = _mm512_setzero_si512();
        for (size_t i = 0; i < 32; i += 8)
        {
            __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + i));
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + i));
            accum = _mm512_add_epi64(accum, _mm512_popcnt_epi64(_mm512_and_si512(va, vb)));
        }
        return _mm512_reduce_add_epi64(accum);
    }
    else
    {
        // Generic: 512-bit chunks + 256-bit remainder
        __m512i accum512 = _mm512_setzero_si512();
        size_t i = 0;
        for (; i + 8 <= N; i += 8)
        {
            __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + i));
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + i));
            accum512 = _mm512_add_epi64(accum512, _mm512_popcnt_epi64(_mm512_and_si512(va, vb)));
        }
        uint64_t result = _mm512_reduce_add_epi64(accum512);

        if constexpr ((N % 8) >= 4)
        {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
            __m256i cnt = _mm256_popcnt_epi64(_mm256_and_si256(va, vb));
            __m128i lo = _mm256_castsi256_si128(cnt);
            __m128i hi = _mm256_extracti128_si256(cnt, 1);
            __m128i sum = _mm_add_epi64(lo, hi);
            result += static_cast<uint64_t>(_mm_extract_epi64(sum, 0)) +
                      static_cast<uint64_t>(_mm_extract_epi64(sum, 1));
            i += 4;
        }
        for (; i < N; ++i)
        {
            result += popcntll(a[i] & b[i]);
        }
        return result;
    }
}

static inline uint64_t CommonBitsCount(const uint64_t *a, const uint64_t *b, [[maybe_unused]] size_t n)
{
#ifdef FPSIM2_FP_SIZE
    return CommonBitsCountFixed<FPSIM2_FP_SIZE>(a, b);
#else
    // Runtime dispatch version
    size_t i = 0;
    if (n >= 8)
    {
        __m512i accum = _mm512_setzero_si512();
        for (; i + 8 <= n; i += 8)
        {
            __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + i));
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + i));
            accum = _mm512_add_epi64(accum, _mm512_popcnt_epi64(_mm512_and_si512(va, vb)));
        }
        uint64_t result = _mm512_reduce_add_epi64(accum);
        for (; i < n; ++i)
        {
            result += popcntll(a[i] & b[i]);
        }
        return result;
    }

    __m256i accum256 = _mm256_setzero_si256();
    for (; i + 4 <= n; i += 4)
    {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
        accum256 = _mm256_add_epi64(accum256, _mm256_popcnt_epi64(_mm256_and_si256(va, vb)));
    }
    __m128i lo = _mm256_castsi256_si128(accum256);
    __m128i hi = _mm256_extracti128_si256(accum256, 1);
    __m128i sum = _mm_add_epi64(lo, hi);
    uint64_t result = static_cast<uint64_t>(_mm_extract_epi64(sum, 0)) +
                      static_cast<uint64_t>(_mm_extract_epi64(sum, 1));
    for (; i < n; ++i)
    {
        result += popcntll(a[i] & b[i]);
    }
    return result;
#endif
}

#else
// ---------------------------------------------------------------------------
// Scalar POPCNT: Works on all x86-64 CPUs (default for wheels)
// ---------------------------------------------------------------------------

template <size_t N>
static inline uint64_t CommonBitsCountFixed(const uint64_t *a, const uint64_t *b)
{
    uint64_t result = 0;
    for (size_t i = 0; i < N; ++i)
    {
        result += popcntll(a[i] & b[i]);
    }
    return result;
}

static inline uint64_t CommonBitsCount(const uint64_t *a, const uint64_t *b, [[maybe_unused]] size_t n)
{
#ifdef FPSIM2_FP_SIZE
    return CommonBitsCountFixed<FPSIM2_FP_SIZE>(a, b);
#else
    uint64_t result = 0;
    for (size_t i = 0; i < n; ++i)
    {
        result += popcntll(a[i] & b[i]);
    }
    return result;
#endif
}

#endif
