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
 * The scalar helpers are the rdna3 definitions with the two facts that differ
 * changed: a wave is 64 lanes, and gfx9's lgkmcnt is 4 bits (0..15), not 6.
 * The register-tile memory ops (load/store on a strided_gl, the bounded,
 * gathered and scattered rows, atomics, Philox) are rewritten against the
 * CDNA lane layouts through one coordinate function, hk_frag_coord. Not here:
 * load_w4 and the LDS granule set, which hk only uses on RDNA3.
 */

#pragma once

#if !defined(KITTENS_CDNA3) && !defined(KITTENS_CDNA4)
#error "hk/cdna.cuh is for KITTENS_CDNA3 / KITTENS_CDNA4 builds"
#endif

#include <stdexcept>
#include <string>

namespace kittens {

static_assert(WARP_THREADS == 64, "hk/cdna.cuh assumes wave64");

/*
 * int8 as a storage type, as in the fork's rdna3 tree: a packing so a
 * gl<int8> instantiates, and the two conversions a quantizer needs. There is
 * no rt<int8>; everything between a load and a store stays float.
 *
 * include/cdna4 already has a packing for int8_t: its fp8e8m0 (MX scale) is
 * ROCm's __amd_scale_t, which is int8_t. That packing (unpacked int8_t, one
 * a slot) is all a gl needs, so on CDNA4 only the conversions are added; it
 * defines none for fp8e8m0, and its constants<>::one() (0x7F, a unit scale)
 * is not one hk's int8 path reads.
 */
using int8   = int8_t;
using int8_4 = char4;
namespace base_types {
#if defined(KITTENS_CDNA3)
template<> struct packing<int8> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = int8;
    using packed_type = int8_4;
};
template<> struct packing<int8_4> {
    static __device__ inline constexpr int num() { return 4; }
    using unpacked_type = int8;
    using packed_type = int8_4;
};
template<> struct constants<int8> {
    static __device__ inline constexpr int8 zero() { return 0; }
    static __device__ inline constexpr int8 one()  { return 1; }
};
#endif
/** float -> int8 rounds to nearest-even and saturates; a C cast truncates and wraps. */
template<> struct convertor<int8, float> {
    static __host__ __device__ inline int8 convert(const float &u) {
        float r = __builtin_rintf(u);
        r = r < -128.f ? -128.f : (r > 127.f ? 127.f : r);
        return static_cast<int8>(r);
    }
};
template<> struct convertor<float, int8> {
    static __host__ __device__ inline float convert(const int8 &u) { return static_cast<float>(u); }
};
}  // namespace base_types

/**
 * @brief The global hk passes on CDNA: upstream's gl plus one row stride.
 *
 * include/cdna{3,4}'s gl has rows exactly `cols` apart. A slice of the last
 * axis -- x[..., I:] of an (M, 2I) tensor, the halves of a fused gate/up
 * output -- has cols() == I and rows 2I apart, and read as packed rows it
 * is the wrong elements, not a slow read. This is that gl with
 * `row_stride_internal`, the field the fork's rdna3 gl has: the outer axes
 * stay compact over the rows (depth stride = rows * row stride). Everything
 * the CDNA library reaches a global through -- operator[], idx(), shape<>,
 * stride<> -- is stride-aware, so group (DMA) loads and vector loads take it
 * unchanged; register-tile loads and stores go through the overloads below,
 * which size their buffer descriptor by the tile's rows.
 *
 * Field names and order are the code-object launcher's (scaffold.GL_FIELDS).
 */
template<typename _T> struct strided_gl {
    using identifier = ducks::gl::identifier;
    using T     = typename base_types::packing<_T>::unpacked_type;
    using T2    = typename base_types::packing<_T>::packed_type;
    using dtype = T;
    static constexpr int __b__ = -1, __d__ = -1, __r__ = -1, __c__ = -1;

    T* raw_ptr;
    ducks::gl::runtime_dim batch_internal, depth_internal, rows_internal, cols_internal;
    int row_stride_internal;

    __host__ inline strided_gl(T *data, size_t b, size_t d, size_t r, size_t c, int row_stride = -1)
        : raw_ptr(data), batch_internal(b), depth_internal(d), rows_internal(r), cols_internal(c),
          row_stride_internal(row_stride < 0 ? int(c) : row_stride) {}

    __host__ __device__ inline int batch() const { return batch_internal; }
    __host__ __device__ inline int depth() const { return depth_internal; }
    __host__ __device__ inline int rows() const { return rows_internal; }
    __host__ __device__ inline int cols() const { return cols_internal; }
    __host__ __device__ inline int row_stride() const { return row_stride_internal; }

    __device__ inline int64_t idx(const coord<ducks::default_type> &i) const {
        return ((int64_t(i.b) * depth() + i.d) * rows() + i.r) * row_stride() + i.c;
    }
    __device__ inline T& operator[](const coord<ducks::default_type> &i) const { return raw_ptr[idx(i)]; }
    template<int axis> __device__ inline size_t shape() const {
        static_assert(axis >= 0 && axis <= 3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis == 0) return size_t(batch());
        else if constexpr (axis == 1) return size_t(depth());
        else if constexpr (axis == 2) return size_t(rows());
        else return size_t(cols());
    }
    template<int axis> __device__ inline size_t stride() const {
        static_assert(axis >= 0 && axis <= 3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis == 0) return size_t(depth()) * rows() * row_stride();
        else if constexpr (axis == 1) return size_t(rows()) * row_stride();
        else if constexpr (axis == 2) return size_t(row_stride());
        else return 1;
    }
};
template<typename T> struct is_strided_gl : std::false_type {};
template<typename T> struct is_strided_gl<strided_gl<T>> : std::true_type {};

/**
 * @brief make_gl with hk's launchers' row stride (-1: contiguous).
 *
 * For a strided_gl the stride is kept, after the same check rdna3's make_gl
 * makes. Any other gl -- a hand-written kernel's upstream gl -- has nowhere
 * to keep one, so a real stride is refused rather than read as packed rows.
 */
template<ducks::gl::all GL, bool safe=true>
__host__ inline GL make_gl(uint64_t data, int b, int d, int r, int c, int row_stride) {
    if constexpr (is_strided_gl<GL>::value) {
        if (row_stride >= 0 && row_stride < c)
            throw std::runtime_error("Row stride shorter than a row.");
        return GL(reinterpret_cast<typename GL::dtype *>(data), b, d, r, c, row_stride);
    } else {
        if (row_stride >= 0 && row_stride != c)
            throw std::invalid_argument(
                "hk: this tensor's rows are " + std::to_string(row_stride) +
                " elements apart, not " + std::to_string(c) + "; this gl has no "
                "row stride -- call .contiguous()");
        return make_gl<GL, safe>(data, b, d, r, c);
    }
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

#if defined(KITTENS_CDNA3)
/* include/cdna3 has no GELU; include/cdna4's tanh form, verbatim. */
namespace base_ops {
namespace hk_gelu_detail {
constexpr float SQRT_2_OVER_PI   = 0.7978845608028654f;
constexpr float GELU_COEFF       = 0.044715f;
constexpr float GELU_INNER_COEFF = GELU_COEFF * SQRT_2_OVER_PI;
constexpr float DGELU_COEFF      = 3.0f * GELU_COEFF * SQRT_2_OVER_PI;
static __device__ inline float fast_tanh(float x) {
    x = fmaxf(fminf(x, 20.f), -20.f);
    float e2x = __builtin_amdgcn_exp2f(x * 2.8853900817779268f);
    return (e2x - 1.0f) * __frcp_rn(e2x + 1.0f);
}
}  // namespace hk_gelu_detail
struct gelu {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float gelu::op<float>(const float &x) {
    using namespace hk_gelu_detail;
    return x * (0.5f + 0.5f * fast_tanh(x * (SQRT_2_OVER_PI + GELU_INNER_COEFF * x * x)));
}
template<> __device__ inline float2 gelu::op<float2>(const float2 &x) {
    return float2{gelu::op<float>(x.x), gelu::op<float>(x.y)};
}
struct dgelu {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float dgelu::op<float>(const float &x) {
    using namespace hk_gelu_detail;
    float t = fast_tanh(SQRT_2_OVER_PI * x * (1.f + GELU_COEFF * x * x));
    return 0.5f * x * ((1.f - t * t) * (SQRT_2_OVER_PI + DGELU_COEFF * x * x)) + 0.5f * (1.f + t);
}
template<> __device__ inline float2 dgelu::op<float2>(const float2 &x) {
    return float2{dgelu::op<float>(x.x), dgelu::op<float>(x.y)};
}
}  // namespace base_ops
template<ducks::rt::all T> __device__ static inline void gelu (T &dst, const T &src) { unary_map<base_ops::gelu,  T>(dst, src); }
template<ducks::rt::all T> __device__ static inline void dgelu(T &dst, const T &src) { unary_map<base_ops::dgelu, T>(dst, src); }

/**
 * @brief fp32 -> bf16 tile copy that rounds to nearest even.
 *
 * include/cdna3's convertor<bf16_2, float2> truncates, so every cast of an
 * accumulator to bf16 is biased toward zero by up to one ulp. More
 * specialized than its copy(rt_base<T>&, const rt_base<U>&), which its tile
 * copy reaches by ADL, so a cast picks this.
 */
template<ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<bf16, layout> &dst, const rt_base<float, layout> &src) {
    #pragma unroll
    for (int k = 0; k < dst.packed_per_thread; k++)
        dst.data[k] = bf16_2{__float2bfloat16(src.data[k].x), __float2bfloat16(src.data[k].y)};
}
#endif

/* ------------------------------------------------------------------------
 * Register tiles <-> global memory, through a buffer descriptor per tile.
 * ---------------------------------------------------------------------- */

namespace detail {

typedef int32_t hk_i32x4 __attribute__((ext_vector_type(4)));

// The raw buffer intrinsics under names of hk's own: include/cdna3 declares
// only the 64- and 128-bit loads, and a second declaration under the same
// C++ name would collide where a tree does have it.
__device__ uint8_t     hk_buf_ld8  (hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.load.i8");
__device__ uint16_t    hk_buf_ld16 (hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.load.i16");
__device__ uint32_t    hk_buf_ld32 (hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ uint64_t    hk_buf_ld64 (hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.load.i64");
__device__ __uint128_t hk_buf_ld128(hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.load.i128");
__device__ void hk_buf_st8  (uint8_t,     hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.store.i8");
__device__ void hk_buf_st16 (uint16_t,    hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.store.i16");
__device__ void hk_buf_st32 (uint32_t,    hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.store.i32");
__device__ void hk_buf_st64 (uint64_t,    hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.store.i64");
__device__ void hk_buf_st128(__uint128_t, hk_i32x4, uint32_t, uint32_t, uint32_t) __asm("llvm.amdgcn.raw.buffer.store.i128");

/**
 * A raw buffer descriptor over `bytes` from `base`. Word 3 is the one
 * include/cdna{3,4}'s register loads use (DATA_FORMAT 32): with a zero data
 * format gfx9 treats the buffer as invalid. Past `bytes` the hardware reads
 * zero and drops stores, which is the whole of the row predication below.
 *
 * Every caller's descriptor is the same in all lanes (one tile, one tensor),
 * and readfirstlane says so: a descriptor LLVM cannot prove uniform becomes a
 * waterfall loop around every buffer access (measured: gfx942's weighted
 * RMSNorm, a readfirstlane/saveexec loop per load).
 */
__device__ inline hk_i32x4 hk_srsrc(const void *base, uint64_t bytes) {
    struct { uint64_t ptr; uint32_t range; uint32_t config; } d{
        reinterpret_cast<uint64_t>(base),
        bytes > 0xFFFFFFFFull ? 0xFFFFFFFFu : uint32_t(bytes), 0x00020000u};
    hk_i32x4 r;
    __builtin_memcpy(&r, &d, sizeof(r));
    return hk_i32x4{__builtin_amdgcn_readfirstlane(r.x), __builtin_amdgcn_readfirstlane(r.y),
                    __builtin_amdgcn_readfirstlane(r.z), __builtin_amdgcn_readfirstlane(r.w)};
}

/**
 * Where a register tile's elements live, for both CDNA trees: a base tile
 * is R x C, and each lane holds K runs of S elements that are contiguous
 * along the reduction axis (columns in the row layout, rows in the col
 * layout), the runs G apart. include/cdna3 is always one 16x16 base tile,
 * S = 4; include/cdna4 names the shape in the tile type (rt_16x32: S = 8).
 */
template<typename RT> struct hk_frag {
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    static_assert(base_types::packing<typename RT::dtype>::num() == 2,
                  "hk's CDNA memory ops take bf16, fp16 and fp32 register tiles");
    static constexpr bool row = std::is_same_v<typename RT::layout, ducks::rt_layout::row>;
#if defined(KITTENS_CDNA4)
    static constexpr int R = RT::base_tile_rows, C = RT::base_tile_cols;
    static constexpr int S = RT::base_tile_stride, K = RT::base_tile_num_strides;
    static constexpr int G = RT::base_tile_elements_per_stride_group;
#else
    static constexpr int R = RT::tile_size_row, C = RT::tile_size_col;
    static constexpr int S = 4, K = 1, G = 16;
    static_assert(R == 16 && C == 16, "include/cdna3's 16-bit and fp32 base tile is 16x16");
#endif
    static constexpr int E = S * K;  ///< elements a lane holds per base tile
    static_assert(E * WARP_THREADS == R * C, "lane layout does not cover the base tile");
};

/** (row, col) within its base tile of element `e` of lane `lane`'s share. */
template<typename RT> __device__ inline int2 hk_frag_coord(int e, int lane) {
    using F = hk_frag<RT>;
    const int k = e / F::S, s = e % F::S;
    if constexpr (F::row) return {lane % F::R, F::S * (lane / F::R) + k * F::G + s};
    else                  return {F::S * (lane / F::C) + k * F::G + s, lane % F::C};
}

/**
 * T -> U as a store narrows it. float -> bf16 rounds to nearest even: both
 * CDNA trees' scalar convertor<bf16, float> truncates (a bias toward zero of
 * up to one bf16 ulp, which fails a bit-exact residual on gfx950), and
 * include/cdna3's packed one does too. __float2bfloat16 is an fptrunc, a
 * v_cvt_pk_bf16_f32 on gfx950.
 */
template<typename U, typename T> __device__ inline U hk_cvt(const T &v) {
    if constexpr (std::is_same_v<U, bf16> && std::is_same_v<T, float>) return __float2bfloat16(v);
    else return base_types::convertor<U, T>::convert(v);
}

template<typename RT> __device__ inline auto hk_get(const RT &t, int i, int j, int e) {
    return (e & 1) ? t.tiles[i][j].data[e >> 1].y : t.tiles[i][j].data[e >> 1].x;
}
template<typename RT, typename V> __device__ inline void hk_set(RT &t, int i, int j, int e, V v) {
    if (e & 1) t.tiles[i][j].data[e >> 1].y = v;
    else       t.tiles[i][j].data[e >> 1].x = v;
}

template<typename U> __device__ inline U hk_buf_load_elem(hk_i32x4 r, uint32_t off) {
    U v;
    if constexpr (sizeof(U) == 1) { const uint8_t  x = hk_buf_ld8 (r, off, 0, 0); __builtin_memcpy(&v, &x, 1); }
    else if constexpr (sizeof(U) == 2) { const uint16_t x = hk_buf_ld16(r, off, 0, 0); __builtin_memcpy(&v, &x, 2); }
    else { static_assert(sizeof(U) == 4, "8-, 16- or 32-bit elements"); const uint32_t x = hk_buf_ld32(r, off, 0, 0); __builtin_memcpy(&v, &x, 4); }
    return v;
}
template<typename U> __device__ inline void hk_buf_store_elem(hk_i32x4 r, uint32_t off, U v) {
    if constexpr (sizeof(U) == 1) { uint8_t  x; __builtin_memcpy(&x, &v, 1); hk_buf_st8 (x, r, off, 0, 0); }
    else if constexpr (sizeof(U) == 2) { uint16_t x; __builtin_memcpy(&x, &v, 2); hk_buf_st16(x, r, off, 0, 0); }
    else { static_assert(sizeof(U) == 4, "8-, 16- or 32-bit elements"); uint32_t x; __builtin_memcpy(&x, &v, 4); hk_buf_st32(x, r, off, 0, 0); }
}

/** `N` contiguous bytes at `off`, in the widest accesses that tile them. */
template<int N> __device__ inline void hk_buf_load_bytes(hk_i32x4 r, uint32_t off, void *dst) {
    if constexpr (N % 16 == 0) {
        #pragma unroll
        for (int q = 0; q < N / 16; q++) {
            const __uint128_t x = hk_buf_ld128(r, off + 16 * q, 0, 0);
            __builtin_memcpy((char *)dst + 16 * q, &x, 16);
        }
    }
    else if constexpr (N == 8) { const uint64_t x = hk_buf_ld64(r, off, 0, 0); __builtin_memcpy(dst, &x, 8); }
    else if constexpr (N == 4) { const uint32_t x = hk_buf_ld32(r, off, 0, 0); __builtin_memcpy(dst, &x, 4); }
    else if constexpr (N == 2) { const uint16_t x = hk_buf_ld16(r, off, 0, 0); __builtin_memcpy(dst, &x, 2); }
    else static_assert(N == 16, "a run of 2, 4, 8 or a multiple of 16 bytes");
}
template<int N> __device__ inline void hk_buf_store_bytes(hk_i32x4 r, uint32_t off, const void *src) {
    if constexpr (N % 16 == 0) {
        #pragma unroll
        for (int q = 0; q < N / 16; q++) {
            __uint128_t x; __builtin_memcpy(&x, (const char *)src + 16 * q, 16);
            hk_buf_st128(x, r, off + 16 * q, 0, 0);
        }
    }
    else if constexpr (N == 8) { uint64_t x; __builtin_memcpy(&x, src, 8); hk_buf_st64(x, r, off, 0, 0); }
    else if constexpr (N == 4) { uint32_t x; __builtin_memcpy(&x, src, 4); hk_buf_st32(x, r, off, 0, 0); }
    else if constexpr (N == 2) { uint16_t x; __builtin_memcpy(&x, src, 2); hk_buf_st16(x, r, off, 0, 0); }
    else static_assert(N == 16, "a run of 2, 4, 8 or a multiple of 16 bytes");
}

/**
 * The descriptor for one tile: based at the tile's first element, spanning
 * its rows that are inside the tensor and before `limit`. Sized by the
 * tile's own rows, so it never wraps however large the tensor -- unlike a
 * descriptor sized by the whole tensor, whose 32-bit byte count does past
 * 4 GiB -- and a strided tensor's rows are counted at their real pitch.
 */
template<int ROWS, typename GL, typename C4>
__device__ inline hk_i32x4 hk_tile_srsrc(const GL &g, const C4 &unit, const void *base, int limit) {
    const int left = int(g.template shape<2>()) - unit.r;
    int valid = left < ROWS ? left : ROWS;
    valid = limit < valid ? limit : valid;
    valid = valid < 0 ? 0 : valid;
    return hk_srsrc(base, uint64_t(valid) * g.template stride<2>() * sizeof(typename GL::dtype));
}

template<typename RT, typename GL, typename COORD>
__device__ inline void hk_load_tile(RT &dst, const GL &src, const COORD &idx, int limit) {
    using F = hk_frag<RT>;
    using T = typename F::T;
    using U = typename GL::dtype;
    const auto unit = idx.template unit_coord<2, 3>();
    const U *base = (const U *)&src[unit];
    const uint32_t rs = uint32_t(src.template stride<2>());
    const hk_i32x4 r = hk_tile_srsrc<RT::rows>(src, unit, base, limit);
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            if constexpr (F::row) {
                // Each run is S contiguous elements of one row: one access.
                #pragma unroll
                for (int k = 0; k < F::K; k++) {
                    const int2 c = hk_frag_coord<RT>(k * F::S, lane);
                    U run[F::S];
                    hk_buf_load_bytes<F::S * sizeof(U)>(
                        r, ((i * F::R + c.x) * rs + j * F::C + c.y) * sizeof(U), run);
                    #pragma unroll
                    for (int s = 0; s < F::S; s++)
                        hk_set(dst, i, j, k * F::S + s, base_types::convertor<T, U>::convert(run[s]));
                }
            } else {
                // A run goes down a column: one element per row.
                #pragma unroll
                for (int e = 0; e < F::E; e++) {
                    const int2 c = hk_frag_coord<RT>(e, lane);
                    hk_set(dst, i, j, e, base_types::convertor<T, U>::convert(hk_buf_load_elem<U>(
                        r, ((i * F::R + c.x) * rs + j * F::C + c.y) * sizeof(U))));
                }
            }
        }
    }
}

template<typename RT, typename GL, typename COORD>
__device__ inline void hk_store_tile(const GL &dst, const RT &src, const COORD &idx, int limit) {
    using F = hk_frag<RT>;
    using T = typename F::T;
    using U = typename GL::dtype;
    const auto unit = idx.template unit_coord<2, 3>();
    U *base = (U *)&dst[unit];
    const uint32_t rs = uint32_t(dst.template stride<2>());
    const hk_i32x4 r = hk_tile_srsrc<RT::rows>(dst, unit, base, limit);
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            if constexpr (F::row) {
                #pragma unroll
                for (int k = 0; k < F::K; k++) {
                    const int2 c = hk_frag_coord<RT>(k * F::S, lane);
                    U run[F::S];
                    #pragma unroll
                    for (int s = 0; s < F::S; s++)
                        run[s] = hk_cvt<U, T>(hk_get(src, i, j, k * F::S + s));
                    hk_buf_store_bytes<F::S * sizeof(U)>(
                        r, ((i * F::R + c.x) * rs + j * F::C + c.y) * sizeof(U), run);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < F::E; e++) {
                    const int2 c = hk_frag_coord<RT>(e, lane);
                    hk_buf_store_elem<U>(r, ((i * F::R + c.x) * rs + j * F::C + c.y) * sizeof(U),
                                         hk_cvt<U, T>(hk_get(src, i, j, e)));
                }
            }
        }
    }
}

}  // namespace detail

/**
 * @brief load() of a register tile from a strided_gl. Rows of the tile past
 * the tensor's last row read as zero (as include/rdna3's load does), and the
 * rows are `row_stride` apart.
 *
 * More specialized than include/cdna{3,4}'s load(RT&, const GL&, ...), so it
 * is the one a strided_gl picks. Same accesses as the library's on its fast
 * path: one buffer_load of a lane's run (dwordx4 for an rt_16x32 bf16 run).
 */
template<ducks::rt::all RT, typename U, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const strided_gl<U> &src, const COORD &idx) {
    detail::hk_load_tile(dst, src, idx, RT::rows);
}
/** @brief store() of a register tile to a strided_gl; rows past the tensor's last are dropped. */
template<ducks::rt::all RT, typename U, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const strided_gl<U> &dst, const RT &src, const COORD &idx) {
    detail::hk_store_tile(dst, src, idx, RT::rows);
}

/**
 * @brief load(), predicated on the tensor's rows and on `limit` rows from
 * the tile's first: rows past either read as zero and their memory is never
 * touched. No branch and no masked lane: the descriptor ends there.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_bounded(RT &dst, const GL &src, const COORD &idx,
                                           int limit = RT::rows) {
    detail::hk_load_tile(dst, src, idx, limit);
}
/** @brief store(), predicated the same way: rows past either are not written. */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_bounded(const GL &dst, const RT &src, const COORD &idx,
                                            int limit = RT::rows) {
    detail::hk_store_tile(dst, src, idx, limit);
}

/*
 * fp32 register tiles <-> LDS tiles. include/cdna4's load(rt, st) and
 * store(st, rt) take 16-bit tiles only (static_assert "Unsupported type"),
 * and a cross-warp combine of fp32 partial sums -- the decode GEMV's -- is
 * exactly this. Plain LDS accesses, one element at a time through the
 * fragment's coordinates and the shared tile's own swizzle: the compiler
 * sees them and waits for them, which is what hk-lds-pipeline takes an fp32
 * tile's store and load to be. More specialized than the library's
 * templates, so an fp32 pair picks these.
 */
namespace detail {
template<typename ST> __device__ inline float *hk_st_elem(ST &s, int r, int c) {
    const int sub = (r / ST::underlying_subtile_rows) * ST::underlying_subtiles_per_row +
                    c / ST::underlying_subtile_cols;
    const uint32_t off = ST::swizzle({r % ST::underlying_subtile_rows, c % ST::underlying_subtile_cols});
    return (float *)((char *)&s.data[0] + sub * ST::underlying_subtile_bytes + off);
}
}  // namespace detail

#if defined(KITTENS_CDNA4)
template<int R, int C, typename SS, typename L, typename RS>
__device__ inline static void store(st<float, R, C, SS> &dst, const rt<float, R, C, L, RS> &src) {
    using RT = rt<float, R, C, L, RS>;
    using F = detail::hk_frag<RT>;
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int e = 0; e < F::E; e++) {
                const int2 c = detail::hk_frag_coord<RT>(e, lane);
                *detail::hk_st_elem(dst, i * F::R + c.x, j * F::C + c.y) = detail::hk_get(src, i, j, e);
            }
}
template<int R, int C, typename SS, typename L, typename RS>
__device__ inline static void load(rt<float, R, C, L, RS> &dst, const st<float, R, C, SS> &src) {
    using RT = rt<float, R, C, L, RS>;
    using F = detail::hk_frag<RT>;
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int e = 0; e < F::E; e++) {
                const int2 c = detail::hk_frag_coord<RT>(e, lane);
                detail::hk_set(dst, i, j, e,
                               *detail::hk_st_elem(const_cast<st<float, R, C, SS> &>(src),
                                                   i * F::R + c.x, j * F::C + c.y));
            }
}
#endif

#if defined(KITTENS_CDNA4)
namespace detail {
typedef __attribute__((address_space(3))) uint32_t *hk_lds_ptr;
__device__ void hk_buf_ld_lds(hk_i32x4 rsrc, hk_lds_ptr lds, int size, int voffset, int soffset,
                              int offset, int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
}  // namespace detail

/**
 * @brief A group of NW warps copies one LDS tile from a strided_gl with
 * gfx950's DMA, buffer_load_dwordx4 ... lds, and returns with it in flight.
 *
 * Each instruction moves 16 bytes a lane into LDS at M0 + 16 * lane, so the
 * swizzle of the tile's st_shape is applied on the global side, as
 * include/cdna4's group load does: LDS byte p holds the element the swizzled
 * layout puts there. Unlike that load, the descriptor is the tile's own
 * (hk_tile_srsrc): based at the tile and ending at the tensor's last row, so
 * the rows of a last, partial tile read zero instead of past the tensor, and
 * nothing wraps however large it is. The per-lane offsets depend only on the
 * lane, the warp and the row stride, which a K loop does not change, so
 * the compiler computes them once.
 *
 * The copies count on vmcnt only. hk's group_wait_for retires them with the
 * count hk-lds-pipeline derives, and a barrier after it makes every warp's
 * share visible; nothing here waits. `limit` ends the descriptor that many
 * rows from the tile's first as well: a copy with limit 0 fills the tile with
 * zeros, and moves none of its addressing.
 */
template<int NW, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_load_async(ST &dst, const strided_gl<U> &src, const COORD &idx,
                                               int limit = ST::rows) {
    using T = typename ST::dtype;
    static_assert(sizeof(T) == sizeof(U) && sizeof(T) == 2, "16-bit tiles of the global's own dtype");
    static_assert(ST::underlying_subtile_bytes_per_thread == 16, "the 16-byte DMA needs a 16-byte lane run");
    constexpr int PER_WARP = 16 * WARP_THREADS;
    constexpr int PER_CALL = PER_WARP * NW;
    constexpr int BYTES = ST::rows * ST::cols * int(sizeof(T));
    static_assert(BYTES % PER_CALL == 0, "a whole number of copies per thread");
    const auto unit = idx.template unit_coord<2, 3>();
    const U *base = (const U *)&src[unit];
    const int rs = int(src.template stride<2>());
    const detail::hk_i32x4 r = detail::hk_tile_srsrc<ST::rows>(src, unit, base, limit);
    const int lane = laneid();
    const int w = warpid_uniform() % NW;
    const uint32_t lds = __builtin_amdgcn_readfirstlane(
        uint32_t(reinterpret_cast<uintptr_t>(&dst.data[0])) + w * PER_WARP);
    #pragma unroll
    for (int i = 0; i < BYTES / PER_CALL; i++) {
        const int byte = lane * 16 + w * PER_WARP + i * PER_CALL;
        const int sub = byte / ST::underlying_subtile_bytes;
        const int in = byte % ST::underlying_subtile_bytes;
        const uint32_t sw = ST::swizzle({in / ST::underlying_subtile_row_bytes,
                                         (in % ST::underlying_subtile_row_bytes) / int(sizeof(T))});
        const int row = int(sw) / ST::underlying_subtile_row_bytes +
                        (sub / ST::underlying_subtiles_per_row) * ST::underlying_subtile_rows;
        const int col = (int(sw) % ST::underlying_subtile_row_bytes) / int(sizeof(T)) +
                        (sub % ST::underlying_subtiles_per_row) * ST::underlying_subtile_cols;
        detail::hk_buf_ld_lds(r, (detail::hk_lds_ptr)(uintptr_t)(lds + i * PER_CALL), 16,
                              (row * rs + col) * int(sizeof(T)), 0, 0, 0);
    }
}
#endif

#if defined(KITTENS_CDNA3)
/*
 * Register vectors <-> global and shared memory, on include/cdna3.
 *
 * include/cdna3's vector loads are ThunderKittens' wave32 code: they write
 * `dst[o][i]` with o and i computed from the lane, which puts the vector in
 * scratch (gfx942's weighted norms: 72 B a lane; LayerNorm backward's LDS
 * vectors: 648 B; both refused by the gate), and their ortho mapping does
 * not match cdna3's 16-lane base tile. Each lane loads its own entries
 * instead. On a 16x16 base tile an align vector holds, in lane l, the pairs
 * at 16w + 4(l/16) + 2k (the columns, or rows, its tile elements sit on); an
 * ortho vector one entry at 16w + l%16; a naive one 64w + l.
 */
namespace detail {
template<typename RV, typename U> __device__ inline void hk_rv_load(RV &dst, const U *p) {
    using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
    const int lane = laneid();
    #pragma unroll
    for (int w = 0; w < RV::outer_dim; w++) {
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for (int k = 0; k < RV::inner_dim; k++) {
                const int c = 16 * w + 4 * (lane / 16) + 2 * k;
                dst.data[w][k].x = base_types::convertor<T, U>::convert(p[c]);
                dst.data[w][k].y = base_types::convertor<T, U>::convert(p[c + 1]);
            }
        } else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            dst.data[w][0] = base_types::convertor<T, U>::convert(p[16 * w + lane % 16]);
        } else {
            const int i = 64 * w + lane;
            if (i < RV::length) dst.data[w][0] = base_types::convertor<T, U>::convert(p[i]);
        }
    }
}
/// One lane of each group holding an entry writes it.
template<typename RV, typename U> __device__ inline void hk_rv_store(U *p, const RV &src) {
    using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
    const int lane = laneid();
    #pragma unroll
    for (int w = 0; w < RV::outer_dim; w++) {
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            if (lane % 16 == 0) {
                #pragma unroll
                for (int k = 0; k < RV::inner_dim; k++) {
                    const int c = 16 * w + 4 * (lane / 16) + 2 * k;
                    p[c]     = hk_cvt<U, T>(src.data[w][k].x);
                    p[c + 1] = hk_cvt<U, T>(src.data[w][k].y);
                }
            }
        } else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            if (lane < 16) p[16 * w + lane] = hk_cvt<U, T>(src.data[w][0]);
        } else {
            const int i = 64 * w + lane;
            if (i < RV::length) p[i] = hk_cvt<U, T>(src.data[w][0]);
        }
    }
}
}  // namespace detail

template<ducks::rv::all RV, typename U, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const strided_gl<U> &src, const COORD &idx) {
    detail::hk_rv_load(dst, (const U *)&src[(idx.template unit_coord<-1, 3>())]);
}
template<ducks::rv::all RV, typename U, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store(const strided_gl<U> &dst, const RV &src, const COORD &idx) {
    detail::hk_rv_store((U *)&dst[(idx.template unit_coord<-1, 3>())], src);
}
template<ducks::rv::all RV, typename U, size_t N>
__device__ inline static void load(RV &dst, const sv<U, N> &src) {
    static_assert(N == RV::length, "vector lengths differ");
    detail::hk_rv_load(dst, &src.data[0]);
}
template<ducks::rv::all RV, typename U, size_t N>
__device__ inline static void store(sv<U, N> &dst, const RV &src) {
    static_assert(N == RV::length, "vector lengths differ");
    detail::hk_rv_store(&dst.data[0], src);
}
#endif

/* ------------------------------------------------------------------------
 * Scalars, counters and atomics: layout-light, ported from include/rdna3.
 * ---------------------------------------------------------------------- */

/**
 * @brief Store one element of a register vector whose entries are all equal
 * (a folded reduction) to one element of a global. Lane 0 of each warp
 * writes; several warps write the same value to the same address.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store_scalar(const GL &dst, const RV &src, const COORD &idx) {
    using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
    using U = typename GL::dtype;
    if (laneid() != 0) return;
    U *p = (U *)&dst[(idx.template unit_coord<-1, 3>())];
    *p = detail::hk_cvt<U, T>(reinterpret_cast<const T *>(&src.data[0][0])[0]);
}

/** @brief One element of a global, as a wave-uniform integer. */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int load_scalar(const GL &src, const COORD &idx) {
    using U = typename GL::dtype;
    const U *p = (const U *)&src[(idx.template unit_coord<-1, 3>())];
    return __builtin_amdgcn_readfirstlane((int)(*p));
}

/**
 * @brief Count this workgroup's arrival at `cnt[idx]`, device-wide; returns
 * how many arrived before it. The last arrival resets the counter to 0. The
 * fences make every global store issued before the call visible to the last
 * arrival. Contains barriers: every thread of the workgroup must reach it.
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int wg_arrive(const GL &cnt, const COORD &idx, int expected) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "wg_arrive counts in an int32 tensor");
    __shared__ int arrived_before;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int *p = (int *)&cnt[(idx.template unit_coord<-1, 3>())];
        const int old = atomicAdd(p, 1);
        if (old == expected - 1) atomicExch(p, 0);
        arrived_before = old;
    }
    __syncthreads();
    const int arrived = __builtin_amdgcn_readfirstlane(arrived_before);
    __threadfence();
    return arrived;
}

/**
 * @brief Add an fp32 tile into a float global, element by element,
 * atomically. CDNA's register tiles hold each element once (no mirrored
 * halves), so every layout adds each element exactly once.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_add(const GL &dst, const RT &src, const COORD &idx) {
    using F = detail::hk_frag<RT>;
    static_assert(std::is_same_v<typename F::T, float> && std::is_same_v<typename GL::dtype, float>,
                  "atomic_add adds an fp32 tile into a float global");
    float *p = (float *)&dst[(idx.template unit_coord<2, 3>())];
    const int rs = dst.template stride<2>();
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int e = 0; e < F::E; e++) {
                const int2 c = detail::hk_frag_coord<RT>(e, lane);
                atomicAdd(&p[(i * F::R + c.x) * rs + j * F::C + c.y], detail::hk_get(src, i, j, e));
            }
}

/**
 * @brief One atomic on one int32 element, once per warp: `op` 0 adds `v`, 1
 * takes the max. Returns the element's value before it, in every lane.
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int atomic_scalar(const GL &dst, const COORD &idx, int v, int op) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "atomic_scalar works on int32");
    int old = 0;
    if (laneid() == 0) {
        int *p = (int *)&dst[(idx.template unit_coord<-1, 3>())];
        old = op == 0 ? atomicAdd(p, v) : atomicMax(p, v);
    }
    return __builtin_amdgcn_readfirstlane(old);
}

/* ------------------------------------------------------------------------
 * Rows by index: a tile's rows named by an int32 list.
 * ---------------------------------------------------------------------- */

namespace detail {
/// A descriptor over every row of `g` (batch, depth and rows flattened), and
/// the offset a row outside it gets: far enough out that no column offset
/// added to it wraps back in.
template<typename GL>
__device__ inline hk_i32x4 hk_whole_srsrc(const GL &g, uint32_t &oob, int &nrows) {
    nrows = g.batch() * g.depth() * g.rows();
    const uint64_t bytes = uint64_t(nrows) * g.template stride<2>() * sizeof(typename GL::dtype);
    const uint32_t cap = 0xF0000000u;
    oob = bytes < cap ? uint32_t(bytes) : cap;
    return hk_srsrc(g.raw_ptr, oob);
}
/// Base tile row `i`'s index for this lane: idx[at + R*i + lane%R] / div, or
/// -1 at or past `count`.
template<int R, typename IL>
__device__ inline int hk_gather_index(const IL &idx, int at, int count, int i, int div) {
    static_assert(std::is_same_v<typename IL::dtype, int>, "row indices are int32");
    const int pos = at + R * i + laneid() % R;
    if (pos >= count) return -1;
    const int v = idx.raw_ptr[pos];
    return div == 1 ? v : v / div;
}
}  // namespace detail

/**
 * @brief load() with the tile's rows chosen by index: tile row r is row
 * `idx[at + r] / div` of `src` (rows flattened over batch and depth), from
 * column `col`. An index outside [0, rows) or a position at or past `count`
 * reads zeros, with no branch: its offset is past the descriptor. Row layout
 * only, where a lane's elements of a base tile are all in its row.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::gl::all IL>
__device__ inline static void load_gather(RT &dst, const GL &src, const IL &idx,
                                          int at, int count, int col, int div = 1) {
    using F = detail::hk_frag<RT>;
    using T = typename F::T;
    using U = typename GL::dtype;
    static_assert(F::row, "gather fills row-layout tiles");
    uint32_t oob; int nrows;
    const detail::hk_i32x4 r = detail::hk_whole_srsrc(src, oob, nrows);
    const uint32_t row_bytes = uint32_t(src.template stride<2>()) * sizeof(U);
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        const int g = detail::hk_gather_index<F::R>(idx, at, count, i, div);
        const bool in = g >= 0 && g < nrows;
        const uint32_t row_off = in ? uint32_t(g) * row_bytes + uint32_t(col) * sizeof(U) : oob;
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < F::K; k++) {
                const int cy = detail::hk_frag_coord<RT>(k * F::S, lane).y;
                U run[F::S];
                detail::hk_buf_load_bytes<F::S * sizeof(U)>(
                    r, in ? row_off + uint32_t(j * F::C + cy) * sizeof(U) : oob, run);
                #pragma unroll
                for (int s = 0; s < F::S; s++)
                    detail::hk_set(dst, i, j, k * F::S + s, base_types::convertor<T, U>::convert(run[s]));
            }
        }
    }
}

/**
 * @brief store() with the tile's rows chosen by index: tile row r goes to row
 * `idx[at + r]` of `dst`, from column `col`. An index outside [0, rows), or a
 * position at or past `count`, writes nothing. Two tile rows naming one row
 * race; which lands is unspecified.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::gl::all IL>
__device__ inline static void store_scatter(const GL &dst, const RT &src, const IL &idx,
                                            int at, int count, int col) {
    using F = detail::hk_frag<RT>;
    using T = typename F::T;
    using U = typename GL::dtype;
    static_assert(F::row, "scatter stores row-layout tiles");
    uint32_t oob; int nrows;
    const detail::hk_i32x4 r = detail::hk_whole_srsrc(dst, oob, nrows);
    const uint32_t row_bytes = uint32_t(dst.template stride<2>()) * sizeof(U);
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        const int g = detail::hk_gather_index<F::R>(idx, at, count, i, 1);
        const bool in = g >= 0 && g < nrows;
        const uint32_t row_off = in ? uint32_t(g) * row_bytes + uint32_t(col) * sizeof(U) : oob;
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < F::K; k++) {
                const int cy = detail::hk_frag_coord<RT>(k * F::S, lane).y;
                U run[F::S];
                #pragma unroll
                for (int s = 0; s < F::S; s++)
                    run[s] = detail::hk_cvt<U, T>(detail::hk_get(src, i, j, k * F::S + s));
                detail::hk_buf_store_bytes<F::S * sizeof(U)>(
                    r, in ? row_off + uint32_t(j * F::C + cy) * sizeof(U) : oob, run);
            }
        }
    }
}

/* ------------------------------------------------------------------------
 * Random numbers: Triton's Philox, so a kernel and its Triton reference draw
 * the same numbers from the same seed and offsets (include/rdna3 random.cuh).
 * ---------------------------------------------------------------------- */

namespace detail {
__device__ inline uint32_t hk_philox_first(uint32_t c0, uint32_t k0, uint32_t k1, int rounds = 10) {
    uint32_t c1 = 0, c2 = 0, c3 = 0;
    #pragma unroll
    for (int r = 0; r < rounds; r++) {
        const uint32_t a0 = c0, a2 = c2;
        c0 = __umulhi(0xCD9E8D57u, a2) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, a0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * a2;
        c3 = 0xD2511F53u * a0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}
__device__ inline float hk_uniform_from(uint32_t x) {
    int32_t s = int32_t(x);
    s = s < 0 ? -s - 1 : s;
    float u = float(s) * 4.6566127342e-10f;
    // Rounded here, as Triton rounds it; contracted into a later subtraction
    // the product would skip this rounding.
    asm("" : "+v"(u));
    return u;
}
}  // namespace detail

/**
 * @brief Fill an fp32 tile with uniform [0, 1) draws: element (r, c) of the
 * tile draws counter `offset + r * row_stride + c`, as tl.rand would.
 */
template<ducks::rt::all RT>
__device__ inline void philox_uniform(RT &dst, uint32_t seed_lo, uint32_t seed_hi,
                                      int offset, int row_stride) {
    using F = detail::hk_frag<RT>;
    static_assert(std::is_same_v<typename F::T, float>, "draws go into an fp32 tile");
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int e = 0; e < F::E; e++) {
                const int2 c = detail::hk_frag_coord<RT>(e, lane);
                const uint32_t ctr = uint32_t(offset + (i * F::R + c.x) * row_stride + j * F::C + c.y);
                detail::hk_set(dst, i, j, e,
                               detail::hk_uniform_from(detail::hk_philox_first(ctr, seed_lo, seed_hi)));
            }
}

}  // namespace kittens
