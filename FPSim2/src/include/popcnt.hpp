#pragma once

#include <cstdint>

#if defined(_MSC_VER) // windows intel

#include <nmmintrin.h>

static inline uint64_t popcntll(const uint64_t X) {
    return _mm_popcnt_u64(X);
}

#else // unix (linux, osx) intel / arm

static inline uint64_t popcntll(const uint64_t X) {
    return __builtin_popcountll(X);
}

#endif

// AVX512 support for vectorized popcount
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__)

#include <immintrin.h>

static inline uint64_t common_popcnt_avx512(const uint64_t* qptr, 
                                             const uint64_t* dbptr,
                                             const size_t start,
                                             const size_t end) {
    const size_t len = end - start;
    const size_t vec_end = start + (len / 8) * 8;
    
    __m512i sum = _mm512_setzero_si512();
    
    for (size_t j = start; j < vec_end; j += 8) {
        __m512i q = _mm512_loadu_si512((__m512i*)&qptr[j]);
        __m512i d = _mm512_loadu_si512((__m512i*)&dbptr[j]);
        __m512i and_result = _mm512_and_si512(q, d);
        __m512i popcnt = _mm512_popcnt_epi64(and_result);
        sum = _mm512_add_epi64(sum, popcnt);
    }
    
    // Horizontal sum of 8 uint64_t values
    uint64_t result = _mm512_reduce_add_epi64(sum);
    
    // Handle remaining elements
    for (size_t j = vec_end; j < end; j++) {
        result += popcntll(qptr[j] & dbptr[j]);
    }
    
    return result;
}

#define HAS_AVX512_POPCNT 1

#else

#define HAS_AVX512_POPCNT 0

#endif

// Generic common popcount function - uses AVX512 if available
static inline uint64_t common_popcnt(const uint64_t* __restrict__ qptr,
                                      const uint64_t* __restrict__ dbptr,
                                      const size_t start,
                                      const size_t end) {
#if HAS_AVX512_POPCNT
    return common_popcnt_avx512(qptr, dbptr, start, end);
#else
    uint64_t result = 0;
    size_t j = start;
    
    // Unroll by 4 for better instruction-level parallelism
    const size_t unroll_end = start + ((end - start) / 4) * 4;
    for (; j < unroll_end; j += 4) {
        result += popcntll(qptr[j] & dbptr[j]);
        result += popcntll(qptr[j+1] & dbptr[j+1]);
        result += popcntll(qptr[j+2] & dbptr[j+2]);
        result += popcntll(qptr[j+3] & dbptr[j+3]);
    }
    // Handle remainder
    for (; j < end; j++) {
        result += popcntll(qptr[j] & dbptr[j]);
    }
    return result;
#endif
}
