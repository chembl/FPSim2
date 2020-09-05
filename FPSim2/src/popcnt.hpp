#include <cstdint>

#if defined(_MSC_VER) && defined(_WIN64) // windows intel
#include <nmmintrin.h>

static inline uint64_t popcntll(const uint64_t X) {
    return _mm_popcnt_u64(X);
}
#else // unix many architectures
static inline uint64_t popcntll(const uint64_t X) {
    return __builtin_popcountll(X);
}
#endif
