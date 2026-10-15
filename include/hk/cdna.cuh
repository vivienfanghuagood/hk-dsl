/**
 * @file
 * @brief What hk's translation calls that include/cdna3 and include/cdna4 do
 * not define.
 *
 * The rdna3 tree in the fork grew these alongside hk; the CDNA trees are
 * upstream HipKittens and never needed them. Kept in hk's include/ rather than
 * in the submodule, so the CDNA trees stay byte-identical to upstream and a
 * kernel written against them by hand sees no hk symbol. Included after
 * kittens.cuh, on CDNA targets only (hk/target: extra_includes).
 *
 * Everything here is the rdna3 definition with the two facts that differ
 * changed: a wave is 64 lanes, and gfx9's lgkmcnt is 4 bits (0..15), not 6.
 * What is NOT here -- the bounded/gathered/scattered row loads, global
 * atomics, Philox, load_w4, the LDS granule set -- is lane-layout code that
 * has to be rewritten against the CDNA register layouts, not copied.
 */

#pragma once

#if !defined(KITTENS_CDNA3) && !defined(KITTENS_CDNA4)
#error "hk/cdna.cuh is for KITTENS_CDNA3 / KITTENS_CDNA4 builds"
#endif

#include <stdexcept>
#include <string>

namespace kittens {

static_assert(WARP_THREADS == 64, "hk/cdna.cuh assumes wave64");

/**
 * @brief hk's launchers pass a row stride; the CDNA gl has none.
 *
 * include/rdna3's gl carries one (a slice of the last axis is a gl there);
 * upstream's rows are always `cols` apart. Accept exactly the cases that
 * agree -- -1 is the launchers' "contiguous" -- and refuse a real stride,
 * which read as packed rows would be the wrong elements, not a slow read.
 */
template<ducks::gl::all GL, bool safe=true>
__host__ inline GL make_gl(uint64_t data, int b, int d, int r, int c, int row_stride) {
    if (row_stride >= 0 && row_stride != c)
        throw std::invalid_argument(
            "hk: this tensor's rows are " + std::to_string(row_stride) +
            " elements apart, not " + std::to_string(c) + "; a CDNA gl has no "
            "row stride yet -- call .contiguous()");
    return make_gl<GL, safe>(data, b, d, r, c);
}

/**
 * @brief The warp ID, in a scalar register.
 *
 * `warpid()` keeps `threadIdx.x >> 6` in a VGPR; the compiler cannot see it is
 * wave-uniform, so a branch on it becomes divergent and every value the branch
 * needs is pinned in a vector register. Same reason as on gfx11.
 */
__device__ __forceinline__ int warpid_uniform() {
    return __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_THREADS);
}

/**
 * @brief Wait until at most `N` LDS operations are still outstanding.
 *
 * gfx9 encodes lgkmcnt in 4 bits. A count the pipeline pass derives above 15
 * would wrap to a smaller wait -- an under-wait, so a wrong answer -- and is
 * refused here at compile time instead.
 */
template<int N=0> __device__ inline void lds_wait() {
    static_assert(N >= 0 && N <= 15, "lgkmcnt is 4 bits on gfx9");
    asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(N) : "memory");
}

/** @brief Wait until at most `N` vector-memory loads are still outstanding. */
template<int N=0> __device__ inline void vm_wait() {
    static_assert(N >= 0 && N <= 63, "vmcnt is 6 bits on gfx9");
    asm volatile("s_waitcnt vmcnt(%0)" :: "i"(N) : "memory");
}

namespace detail {
typedef float hk_bind_v4f __attribute__((ext_vector_type(4)));
// 128-bit groups first, then single dwords: on wave64 a 16x16 bf16 tile is
// 8 bytes a lane, which the rdna3 form (whole 128-bit groups only) rejects.
template<int I, int N, typename V, typename T> __device__ inline void hk_bind_regs(T &x) {
    if constexpr (I < N) {
        V v;
        __builtin_memcpy(&v, (const char *)&x + I * sizeof(V), sizeof(v));
        asm volatile("" : "+v"(v));
        __builtin_memcpy((char *)&x + I * sizeof(V), &v, sizeof(v));
        hk_bind_regs<I + 1, N, V>(x);
    }
}
}  // namespace detail

/**
 * @brief Re-establish the register dependence that s_waitcnt does not carry.
 *
 * include/cdna4's shared->register loads are `asm volatile("ds_read_b128 ..."
 * : "=v"(...))` with no wait, exactly like include/rdna3's: LLVM believes the
 * destination is ready when the asm ends, and the MFMA builtin is pure, so
 * nothing in the IR keeps it below a separate `s_waitcnt` asm. The hazard is
 * a property of the IR, not of RDNA3, and so is the fix.
 */
template<typename T> __device__ inline void lds_bind(T &x) {
    static_assert(sizeof(T) % sizeof(float) == 0, "lds_bind takes whole dwords");
    if constexpr (sizeof(T) % sizeof(detail::hk_bind_v4f) == 0)
        detail::hk_bind_regs<0, sizeof(T) / sizeof(detail::hk_bind_v4f), detail::hk_bind_v4f>(x);
    else
        detail::hk_bind_regs<0, sizeof(T) / sizeof(float), float>(x);
}

/** @brief `lds_wait<N>()` followed by `lds_bind` on each tile it retires. */
template<int N=0, typename... Ts> __device__ inline void lds_wait_for(Ts&... xs) {
    lds_wait<N>();
    (lds_bind(xs), ...);
}

namespace base_ops {
/** @brief 1/sqrt(x), one v_rsq_f32. ~1 ulp. */
struct rsqrt {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float  rsqrt::op<float> (const float &x ) { return __builtin_amdgcn_rsqf(x); }
template<> __device__ inline float2 rsqrt::op<float2>(const float2 &x) {
    return float2{__builtin_amdgcn_rsqf(x.x), __builtin_amdgcn_rsqf(x.y)};
}
/** @brief x * sigmoid(x), through the hardware exp2 and rcp. */
struct silu {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float silu::op<float>(const float &x) {
    return x * __builtin_amdgcn_rcpf(1.f + __builtin_amdgcn_exp2f(-x * 1.44269504088896340736f));
}
template<> __device__ inline float2 silu::op<float2>(const float2 &x) {
    return float2{silu::op<float>(x.x), silu::op<float>(x.y)};
}
/** @brief 1 where x is positive, else 0. */
struct step {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float  step::op<float> (const float &x ) { return x > 0.f ? 1.f : 0.f; }
template<> __device__ inline float2 step::op<float2>(const float2 &x) {
    return float2{x.x > 0.f ? 1.f : 0.f, x.y > 0.f ? 1.f : 0.f};
}
}  // namespace base_ops

template<ducks::rt::all T> __device__ static inline void rsqrt(T &dst, const T &src) { unary_map<base_ops::rsqrt, T>(dst, src); }
template<ducks::rt::all T> __device__ static inline void silu (T &dst, const T &src) { unary_map<base_ops::silu,  T>(dst, src); }
template<ducks::rt::all T> __device__ static inline void step (T &dst, const T &src) { unary_map<base_ops::step,  T>(dst, src); }
template<ducks::rv::all T> __device__ static inline void rsqrt(T &dst, const T &src) { unary_op<base_ops::rsqrt, T>(dst, src); }
template<ducks::rv::all T> __device__ static inline void silu (T &dst, const T &src) { unary_op<base_ops::silu,  T>(dst, src); }
template<ducks::rv::all T> __device__ static inline void step (T &dst, const T &src) { unary_op<base_ops::step,  T>(dst, src); }

}  // namespace kittens
