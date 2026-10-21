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
/** float -> int8 rounds to nearest-even and saturates; a C cast truncates and wraps.
 * On the device the clamp is one v_med3_f32: written as two comparisons it
 * was two v_cmp, two v_cndmask and the VCC hazard's s_nop an element. */
template<> struct convertor<int8, float> {
    static __host__ __device__ inline int8 convert(const float &u) {
        float r = __builtin_rintf(u);
#if defined(__HIP_DEVICE_COMPILE__)
        r = __builtin_amdgcn_fmed3f(r, -128.f, 127.f);
#else
        r = r < -128.f ? -128.f : (r > 127.f ? 127.f : r);
#endif
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

    __host__ __device__ inline strided_gl(T *data, size_t b, size_t d, size_t r, size_t c,
                                          int row_stride = -1)
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
/** @brief sqrt(x), one v_sqrt_f32, as include/rdna3's base_ops::sqrt: neither
 *  CDNA tree has one, so `hk.math.sqrt` traced for gfx950 and failed in hipcc. */
struct sqrt {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float  sqrt::op<float> (const float &x ) { return __builtin_amdgcn_sqrtf(x); }
template<> __device__ inline float2 sqrt::op<float2>(const float2 &x) {
    return float2{__builtin_amdgcn_sqrtf(x.x), __builtin_amdgcn_sqrtf(x.y)};
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

#if defined(KITTENS_CDNA4)
/** @brief Fill with 1: include/rdna3's and include/cdna3's spelling, which hk's full(1.0) emits; include/cdna4 calls it ones. */
template<ducks::rt::all T> __device__ static inline void one(T &dst) { ones(dst); }
template<ducks::rv::all T> __device__ static inline void one(T &dst) { ones(dst); }
#endif
template<ducks::rt::all T> __device__ static inline void sqrt (T &dst, const T &src) { unary_map<base_ops::sqrt,  T>(dst, src); }
template<ducks::rv::all T> __device__ static inline void sqrt (T &dst, const T &src) { unary_op<base_ops::sqrt,  T>(dst, src); }
template<ducks::rt::all T> __device__ static inline void rsqrt(T &dst, const T &src) { unary_map<base_ops::rsqrt, T>(dst, src); }
template<ducks::rt::all T> __device__ static inline void silu(T &dst, const T &src) { unary_map<base_ops::silu,  T>(dst, src); }
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

namespace detail {
/** Two floats to a bf16 pair, rounding to nearest even: an fptrunc of a
 * 2-vector, which gfx950 selects as one v_cvt_pk_bf16_f32 (two scalar
 * __float2bfloat16 can come out as two converts and the ALU to pack them). */
__device__ inline bf16_2 hk_cvt_pk_bf16(float x, float y) {
    typedef float f2 __attribute__((ext_vector_type(2)));
    typedef __bf16 b2 __attribute__((ext_vector_type(2)));
    const b2 v = __builtin_convertvector(f2{x, y}, b2);
    bf16_2 out;
    __builtin_memcpy(&out, &v, sizeof(out));
    return out;
}

}  // namespace detail

#if defined(KITTENS_CDNA4)
/**
 * @brief fp32 -> bf16 tile copy the compiler can see.
 *
 * include/cdna4's convertor<bf16_2, float2> is an inline-asm
 * v_cvt_pk_bf16_f32. LLVM's hazard recognizer cannot see what inline asm
 * reads, so it does not give the asm the wait state gfx950 needs between a
 * transcendental and a VALU that reads its result: with exp2 scheduled
 * right before the cast (attention's softmax, interleaved with MFMAs), some
 * lanes converted the value from before the exp2 -- measured on an MI355X,
 * a softmax denominator that disagreed with its numerator. __float2bfloat16
 * is an fptrunc, the same v_cvt_pk_bf16_f32 (round to nearest even), and
 * the compiler places it and its wait states itself. More specialized than
 * the library's copy(rt_base<T, L, S>&, const rt_base<U, L, S>&).
 */
template<ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ static inline void copy(rt_base<bf16, layout, shape> &dst, const rt_base<float, layout, shape> &src) {
    #pragma unroll
    for (int k = 0; k < dst.packed_per_thread; k++)
        dst.data[k] = detail::hk_cvt_pk_bf16(src.data[k].x, src.data[k].y);
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

/** The lane index, opaque to LLVM: what is computed from it is computed where
 * it is used, not hoisted out of the loop around it or shared with an
 * earlier use and held live in between. */
__device__ inline int hk_lane_opaque() {
    int lane = laneid();
    asm volatile("" : "+v"(lane));
    return lane;
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
    else if constexpr (N == 1) { const uint8_t x = hk_buf_ld8(r, off, 0, 0); __builtin_memcpy(dst, &x, 1); }
    else static_assert(N == 16, "a run of 1, 2, 4, 8 or a multiple of 16 bytes");
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
    else if constexpr (N == 1) { uint8_t x; __builtin_memcpy(&x, src, 1); hk_buf_st8(x, r, off, 0, 0); }
    else static_assert(N == 16, "a run of 1, 2, 4, 8 or a multiple of 16 bytes");
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
    // Opaque, so the per-lane offsets are computed here and not shared with
    // the load of the same layout at the top of the kernel: shared, LLVM keeps
    // them live across the kernel's main loop (attention's KV walk: a spill).
    const int lane = hk_lane_opaque();
#if defined(KITTENS_CDNA4)
    if constexpr (F::row && F::R == 32 && F::C == 32 && F::S == 4 &&
                  std::is_same_v<T, float> && std::is_same_v<U, bf16>) {
        // A 32x32 accumulator fragment, transposed: lane l holds row l%32,
        // in runs of 4 columns at 4(l/32) + 8k. v_permlane32_swap trades
        // runs between lanes l and l+32, so that each holds runs of 8 --
        // lane l columns 0-7 and 16-23, lane l+32 8-15 and 24-31 -- and the
        // tile goes out in 16-byte stores, half as many as 8-byte ones.
        const int half = lane / 32;
        #pragma unroll
        for (int i = 0; i < RT::height; i++) {
            #pragma unroll
            for (int j = 0; j < RT::width; j++) {
                uint32_t w[4][2];
                #pragma unroll
                for (int k = 0; k < 4; k++)
                    #pragma unroll
                    for (int q = 0; q < 2; q++) {
                        const bf16_2 p = hk_cvt_pk_bf16(src.tiles[i][j].data[2 * k + q].x,
                                                        src.tiles[i][j].data[2 * k + q].y);
                        __builtin_memcpy(&w[k][q], &p, 4);
                    }
                #pragma unroll
                for (int k = 0; k < 4; k += 2)
                    #pragma unroll
                    for (int q = 0; q < 2; q++) {
                        const auto s = __builtin_amdgcn_permlane32_swap(w[k][q], w[k + 1][q], false, true);
                        w[k][q] = s[0];
                        w[k + 1][q] = s[1];
                    }
                const uint32_t row = uint32_t(i * 32 + lane % 32);
                #pragma unroll
                for (int g = 0; g < 2; g++) {
                    uint32_t chunk[4] = {w[2 * g][0], w[2 * g][1], w[2 * g + 1][0], w[2 * g + 1][1]};
                    hk_buf_store_bytes<16>(
                        r, (row * rs + uint32_t(j * 32 + 8 * half + 16 * g)) * sizeof(U), chunk);
                }
            }
        }
        return;
    }
#endif
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            if constexpr (F::row) {
                #pragma unroll
                for (int k = 0; k < F::K; k++) {
                    const int2 c = hk_frag_coord<RT>(k * F::S, lane);
                    U run[F::S];
                    if constexpr (std::is_same_v<U, bf16> && std::is_same_v<T, float>) {
                        // A pair at a time: one packed v_cvt_pk_bf16_f32 for
                        // two elements. Converted one by one, LLVM packed the
                        // halves back with shifts and ors (attention's O
                        // store: 64 converts and ~80 ALU for 32 pairs).
                        #pragma unroll
                        for (int s = 0; s < F::S; s += 2) {
                            const bf16_2 p = hk_cvt_pk_bf16(hk_get(src, i, j, k * F::S + s),
                                                            hk_get(src, i, j, k * F::S + s + 1));
                            __builtin_memcpy(&run[s], &p, sizeof(p));
                        }
                    } else {
                        #pragma unroll
                        for (int s = 0; s < F::S; s++)
                            run[s] = hk_cvt<U, T>(hk_get(src, i, j, k * F::S + s));
                    }
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
 * zeros, and moves none of its addressing. `NT` copies with sc0 nt, past the
 * caches' keeping: for data read once, in runs of whole lines (a tile laid
 * out in half-line runs had each line fetched twice).
 */
namespace detail {
/**
 * The descriptor of one asynchronous copy: based at the tile's first
 * element and ending where this (batch, depth)'s rows of the tensor end, or
 * `limit` rows from the tile's first, whichever comes first. Every input is
 * wave-uniform (the tensor's fields, the tile's coordinates), so it is all
 * scalar arithmetic: a few SALU a copy. An earlier form clamped the row count
 * to the tile's own rows as well, which LLVM folds into a v_med3 and the rest
 * into VALU and readfirstlanes -- ~20 SALU and ~15 VALU a copy, recomputed
 * for every tile of a loop, measured on attention's KV walk. The window does
 * not need the tile's own bound: the copy's lanes address only the tile.
 */
template<typename U>
__device__ inline hk_i32x4 hk_copy_srsrc(const strided_gl<U> &src, int b, int d, int r, int c,
                                         int limit_rows) {
    const uint32_t rs = uint32_t(src.template stride<2>());
    // The slab's base and its rows' end do not depend on the tile: hoisted.
    const U *slab = src.raw_ptr + (int64_t(b) * src.depth() + d) * src.rows() * int64_t(rs);
    // 32-bit row counts, clamped with scalar min/max; the byte counts are
    // 64-bit, and the clamp to the descriptor's 32 bits reads the high word
    // (a 64-bit compare would be a VALU compare and a branch).
    int left = src.rows() - r;
    left = left < limit_rows ? left : limit_rows;
    left = left > 0 ? left : 0;
    const uint64_t first = uint64_t(uint32_t(r)) * rs + uint32_t(c);
    // Whole rows from the tile's base: its lanes address only its own
    // columns, so a row past the end starts past the range. Not ending at
    // the slab's last element keeps the range free of the column, which a K
    // loop moves: the GEMM's range is computed once, not per copy. Past 32
    // bits it saturates, the overflow read off the product's high word
    // (s_mul_hi_u32): a 64-bit compare, however written, is a VALU compare
    // and a VCC-to-SALU hop on gfx9.
    const uint32_t pitch = rs * uint32_t(sizeof(U));
    const uint32_t range = __umulhi(uint32_t(left), pitch) ? 0xFFFFFFFFu : uint32_t(left) * pitch;
    struct { uint64_t ptr; uint32_t range; uint32_t config; } s{
        reinterpret_cast<uint64_t>(slab + first), range, 0x00020000u};
    hk_i32x4 out;
    __builtin_memcpy(&out, &s, sizeof(out));
    return out;
}
}  // namespace detail

template<int NW, bool NT = false, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_load_async_rows(ST &dst, const strided_gl<U> &src, const COORD &idx,
                                                    int limit) {
    using T = typename ST::dtype;
    static_assert(sizeof(T) == sizeof(U) && sizeof(T) == 2, "16-bit tiles of the global's own dtype");
    static_assert(ST::underlying_subtile_bytes_per_thread == 16, "the 16-byte DMA needs a 16-byte lane run");
    constexpr int PER_WARP = 16 * WARP_THREADS;
    constexpr int PER_CALL = PER_WARP * NW;
    constexpr int BYTES = ST::rows * ST::cols * int(sizeof(T));
    static_assert(BYTES % PER_CALL == 0, "a whole number of copies per thread");
    const auto unit = idx.template unit_coord<2, 3>();
    const int rs = int(src.template stride<2>());
    const detail::hk_i32x4 r = detail::hk_copy_srsrc(src, unit.b, unit.d, unit.r, unit.c, limit);
    const int lane = laneid();
    // The warp from threadIdx.x rather than a readfirstlane, which LLVM
    // never moves out of a loop: the per-lane offsets below depend only on
    // the lane, the warp and the row stride, and are computed once.
    const int w = int(threadIdx.x / WARP_THREADS) % NW;
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
                              (row * rs + col) * int(sizeof(T)), 0, 0, NT ? 3 : 0);
    }
}

template<int NW, bool NT = false, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_load_async(ST &dst, const strided_gl<U> &src, const COORD &idx) {
    group_load_async_rows<NW, NT>(dst, src, idx, 0x7FFFFFFF);
}
template<int NW, bool NT = false, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_load_async(ST &dst, const strided_gl<U> &src, const COORD &idx,
                                               int limit) {
    group_load_async_rows<NW, NT>(dst, src, idx, limit);
}

/*
 * Row masks of a col-layout tile that leave nothing loop-invariant per
 * element. include/cdna4's lower_fill and triu compute each element's row as
 * a lane term plus a constant; inside a loop LLVM hoists those sums, one
 * VGPR an element held across the whole loop (attention's KV walk: 16 more
 * VGPRs, and a spill). Here the lane's terms are folded into the one runtime
 * value the mask is compared against, and each element compares it with an
 * immediate: its row offset from the lane's first row.
 *
 * An element (e of base tile i, j) of a col-layout tile sits at row
 * S(l/C) + i R + (e/S) G + e%S and column l%C + j C (hk_frag_coord).
 */
namespace detail {
/* `neg(c, jc)` is negative exactly where element (c, jc) is masked: an int
 * whose sign bit decides. Masking an fp32 tile with -inf is then integer
 * work on that sign bit -- a subtract, a shift and a bitfield insert an
 * element, and no compare: a compare writes VCC, and gfx950 wants two wait
 * states before a v_cndmask reads it (cmp, s_nop 1, cndmask for every
 * element, measured). Other fills keep the select. */
template<typename RT, typename F>
__device__ inline void hk_col_rows_fill(RT &dst, const RT &src, F neg,
                                        typename hk_frag<RT>::T val) {
    using Fr = hk_frag<RT>;
    using T = typename Fr::T;
    const bool neg_inf = std::is_same_v<T, float> && val == -INFINITY;
    if constexpr (std::is_same_v<T, float> && Fr::S % 2 == 0) {
        if (neg_inf) {
            // -inf where masked, +0 elsewhere, added a pair at a time in the
            // pair's own registers: behind a branch, a sum in new registers
            // makes every other path copy the tile into them. (A bit select
            // LLVM turns back into cmp + cndmask; an fminf it canonicalizes.)
            typedef float f2 __attribute__((ext_vector_type(2)));
            auto mask = [&](int c, int jc) {
                const uint32_t mb = uint32_t(__builtin_amdgcn_sbfe(neg(c, jc), 31, 1)) & 0xff800000u;
                float add;
                __builtin_memcpy(&add, &mb, sizeof(add));
                return add;
            };
            #pragma unroll
            for (int i = 0; i < RT::height; i++)
                #pragma unroll
                for (int j = 0; j < RT::width; j++)
                    #pragma unroll
                    for (int e = 0; e < Fr::E; e += 2) {
                        const int c = i * Fr::R + (e / Fr::S) * Fr::G + e % Fr::S;
                        const f2 add = {mask(c, j * Fr::C), mask(c + 1, j * Fr::C)};
                        f2 s = {hk_get(src, i, j, e), hk_get(src, i, j, e + 1)};
                        asm("v_pk_add_f32 %0, %0, %1" : "+v"(s) : "v"(add));
                        hk_set(dst, i, j, e, s.x);
                        hk_set(dst, i, j, e + 1, s.y);
                    }
            return;
        }
    }
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int e = 0; e < Fr::E; e++) {
                const int c = i * Fr::R + (e / Fr::S) * Fr::G + e % Fr::S;
                const int n = neg(c, j * Fr::C);
                if constexpr (std::is_same_v<T, float>) {
                    if (neg_inf) {
                        // -inf where masked, +0 elsewhere, added: a shift,
                        // an and and an add. (A bit select LLVM turns back
                        // into cmp + cndmask; an fminf it canonicalizes.)
                        const uint32_t mb = uint32_t(__builtin_amdgcn_sbfe(n, 31, 1)) & 0xff800000u;
                        float add;
                        __builtin_memcpy(&add, &mb, sizeof(add));
                        hk_set(dst, i, j, e, hk_get(src, i, j, e) + add);
                        continue;
                    }
                }
                hk_set(dst, i, j, e, n < 0 ? val : hk_get(src, i, j, e));
            }
}

template<int N, typename F> __device__ inline void hk_static_for(F &&f) {
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (f(std::integral_constant<int, I>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

/* An element's key from its row offset c (from the lane's first row) and its
 * base tile's first column jc; masked where the lane's t is below it. */
struct hk_key_triu  { static constexpr int key(int c, int jc) { return c - jc; } };
struct hk_key_lower { static constexpr int key(int c, int)    { return c + 1; } };

template<typename RT, typename Key> constexpr bool hk_keys_inline() {
    using Fr = hk_frag<RT>;
    for (int i = 0; i < RT::height; i++)
        for (int j = 0; j < RT::width; j++)
            for (int e = 0; e < Fr::E; e++) {
                const int k = Key::key(i * Fr::R + (e / Fr::S) * Fr::G + e % Fr::S, j * Fr::C);
                if (k < -16 || k > 64) return false;
            }
    return Fr::E % 4 == 0;
}

/* Elements whose key is above the lane's t become -inf, in their own
 * registers: four v_cmp into SGPR pairs, then four v_cndmask, so each select
 * is three instructions after its compare (the wait states a VALU-written
 * lane mask needs). Through VCC, LLVM put an s_nop between every compare and
 * its select, and a mask value plus an add after it: seven issues a pair of
 * elements against four. The keys are inline constants (hk_keys_inline). */
template<typename Key, typename RT>
__device__ inline void hk_ninf_where_key_above(RT &dst, int t) {
    using Fr = hk_frag<RT>;
    const uint32_t ninf_bits = 0xff800000u;
    float ninf;
    __builtin_memcpy(&ninf, &ninf_bits, sizeof(ninf));
    hk_static_for<RT::height>([&](auto I) {
        hk_static_for<RT::width>([&](auto J) {
            hk_static_for<Fr::E / 4>([&](auto Q) {
                constexpr int i = decltype(I)::value, j = decltype(J)::value;
                constexpr int e = 4 * decltype(Q)::value;
                constexpr int k0 = Key::key(i * Fr::R + ((e + 0) / Fr::S) * Fr::G + (e + 0) % Fr::S, j * Fr::C);
                constexpr int k1 = Key::key(i * Fr::R + ((e + 1) / Fr::S) * Fr::G + (e + 1) % Fr::S, j * Fr::C);
                constexpr int k2 = Key::key(i * Fr::R + ((e + 2) / Fr::S) * Fr::G + (e + 2) % Fr::S, j * Fr::C);
                constexpr int k3 = Key::key(i * Fr::R + ((e + 3) / Fr::S) * Fr::G + (e + 3) % Fr::S, j * Fr::C);
                float x0 = hk_get(dst, i, j, e), x1 = hk_get(dst, i, j, e + 1);
                float x2 = hk_get(dst, i, j, e + 2), x3 = hk_get(dst, i, j, e + 3);
                uint64_t m0, m1, m2, m3;
                asm("v_cmp_lt_i32_e64 %4, %8, %10\n\t"
                    "v_cmp_lt_i32_e64 %5, %8, %11\n\t"
                    "v_cmp_lt_i32_e64 %6, %8, %12\n\t"
                    "v_cmp_lt_i32_e64 %7, %8, %13\n\t"
                    "v_cndmask_b32_e64 %0, %0, %9, %4\n\t"
                    "v_cndmask_b32_e64 %1, %1, %9, %5\n\t"
                    "v_cndmask_b32_e64 %2, %2, %9, %6\n\t"
                    "v_cndmask_b32_e64 %3, %3, %9, %7"
                    : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3),
                      "=s"(m0), "=s"(m1), "=s"(m2), "=s"(m3)
                    : "v"(t), "v"(ninf), "n"(k0), "n"(k1), "n"(k2), "n"(k3));
                hk_set(dst, i, j, e, x0);
                hk_set(dst, i, j, e + 1, x1);
                hk_set(dst, i, j, e + 2, x2);
                hk_set(dst, i, j, e + 3, x3);
            });
        });
    });
}
}  // namespace detail

/** @brief Every element becomes -inf, in its own register. Behind a branch
 * (attention's causal mask: a block wholly past the warp's queries) the plain
 * constant fill leaves the branch empty and its moves are placed before it,
 * in registers of their own, and every other path copies the tile into those
 * (16 moves and 16 VGPRs a 32x32 tile, every step, measured). */
template<int R, int C, typename S>
__device__ inline static void neg_infty(rt<float, R, C, ducks::rt_layout::col, S> &dst) {
    using RT = rt<float, R, C, ducks::rt_layout::col, S>;
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int k = 0; k < RT::packed_per_base_tile; k++) {
                float x, y;
                asm volatile("v_mov_b32 %0, 0xff800000" : "=v"(x));
                asm volatile("v_mov_b32 %0, 0xff800000" : "=v"(y));
                dst.tiles[i][j].data[k] = {x, y};
            }
}

/** @brief Rows at or past `row_idx` become `val`. */
template<typename T, int R, int C, typename S>
__device__ inline static void lower_fill(rt<T, R, C, ducks::rt_layout::col, S> &dst,
                                         const rt<T, R, C, ducks::rt_layout::col, S> &src,
                                         const int row_idx, const typename base_types::packing<T>::unpacked_type &val = 0) {
    using RT = rt<T, R, C, ducks::rt_layout::col, S>;
    using Fr = detail::hk_frag<RT>;
    const int t = row_idx - Fr::S * (detail::hk_lane_opaque() / Fr::C);
    // masked where c >= t
    if constexpr (std::is_same_v<T, float> && detail::hk_keys_inline<RT, detail::hk_key_lower>()) {
        if (val == -INFINITY) {
            if (&dst != &src) dst = src;
            detail::hk_ninf_where_key_above<detail::hk_key_lower>(dst, t);
            return;
        }
    }
    detail::hk_col_rows_fill(dst, src, [&](int c, int) { return t - 1 - c; }, val);
}
/** @brief Rows before `row_idx` become `val`. */
template<typename T, int R, int C, typename S>
__device__ inline static void upper_fill(rt<T, R, C, ducks::rt_layout::col, S> &dst,
                                         const rt<T, R, C, ducks::rt_layout::col, S> &src,
                                         const int row_idx, const typename base_types::packing<T>::unpacked_type &val = 0) {
    using Fr = detail::hk_frag<rt<T, R, C, ducks::rt_layout::col, S>>;
    const int t = row_idx - Fr::S * (detail::hk_lane_opaque() / Fr::C);
    // masked where c < t
    detail::hk_col_rows_fill(dst, src, [&](int c, int) { return c - t; }, val);
}
/** @brief Element (r, c) becomes `val` where c < r - row_idx, as include/cdna4's triu. */
template<typename T, int R, int C, typename S>
__device__ inline static void triu(rt<T, R, C, ducks::rt_layout::col, S> &dst,
                                   const rt<T, R, C, ducks::rt_layout::col, S> &src,
                                   const int row_idx, const typename base_types::packing<T>::unpacked_type &val = 0) {
    using RT = rt<T, R, C, ducks::rt_layout::col, S>;
    using Fr = detail::hk_frag<RT>;
    const int lane = detail::hk_lane_opaque();
    const int t = lane % Fr::C - Fr::S * (lane / Fr::C) + row_idx;
    // masked where c - jc > t
    if constexpr (std::is_same_v<T, float> && detail::hk_keys_inline<RT, detail::hk_key_triu>()) {
        if (val == -INFINITY) {
            if (&dst != &src) dst = src;
            detail::hk_ninf_where_key_above<detail::hk_key_triu>(dst, t);
            return;
        }
    }
    detail::hk_col_rows_fill(dst, src, [&](int c, int jc) { return t - (c - jc); }, val);
}
/** @brief Element (r, c) becomes `val` where c > r - row_idx, as include/cdna4's tril. */
template<typename T, int R, int C, typename S>
__device__ inline static void tril(rt<T, R, C, ducks::rt_layout::col, S> &dst,
                                   const rt<T, R, C, ducks::rt_layout::col, S> &src,
                                   const int row_idx, const typename base_types::packing<T>::unpacked_type &val = 0) {
    using Fr = detail::hk_frag<rt<T, R, C, ducks::rt_layout::col, S>>;
    const int lane = detail::hk_lane_opaque();
    const int t = lane % Fr::C - Fr::S * (lane / Fr::C) + row_idx;
    // masked where c - jc < t
    detail::hk_col_rows_fill(dst, src, [&](int c, int jc) { return (c - jc) - t; }, val);
}

/*
 * An MFMA accumulator's registers, read as an operand fragment (hk's
 * as_operand). No instruction: each lane already holds the operand's
 * elements, with the reduction index running down the rows in the
 * accumulator's order.
 *
 * A 32x32 accumulator's lane l holds rows 4(l/32) + 8k + s (k < 4, s < 4)
 * of its column l%32. A 16x32 stride-4 col operand's lane holds rows
 * 4(l/32) + 8k + s (k < 2) of the same column: the top and bottom 16 rows of
 * the accumulator are two such fragments, in register order. A 16x16
 * accumulator's lane holds rows 4(l/16) + s of column l%16, and a 32x16
 * stride-4 col operand rows 4(l/16) + 16k + s: two accumulators stacked.
 */
namespace detail {
template<typename D, typename S> __device__ inline D hk_as_elem(const S &x) { return x; }
template<> __device__ inline bf16_2 hk_as_elem<bf16_2, float2>(const float2 &x) {
    return hk_cvt_pk_bf16(x.x, x.y);
}
}  // namespace detail

/* Packed pair by packed pair; fp32 -> bf16 converts each (to nearest even,
 * the compiler's own v_cvt_pk_bf16_f32: see copy() above for why not the
 * library's inline-asm one). */
template<typename T, typename U, int R, int C>
__device__ inline static void as_operand(rt<T, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_16x32_4> &dst,
                                         const rt<U, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_32x32> &src) {
    using D = rt<T, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_16x32_4>;
    #pragma unroll
    for (int i = 0; i < D::height; i++)
        #pragma unroll
        for (int j = 0; j < D::width; j++)
            #pragma unroll
            for (int k = 0; k < D::packed_per_base_tile; k++)
                dst.tiles[i][j].data[k] = detail::hk_as_elem<typename D::dtype>(
                    src.tiles[i / 2][j].data[(i % 2) * D::packed_per_base_tile + k]);
}
template<typename T, typename U, int R, int C>
__device__ inline static void as_operand(rt<T, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_32x16_4> &dst,
                                         const rt<U, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_16x16> &src) {
    using D = rt<T, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_32x16_4>;
    constexpr int per = D::base_tile_packed_per_stride;
    #pragma unroll
    for (int i = 0; i < D::height; i++)
        #pragma unroll
        for (int j = 0; j < D::width; j++)
            #pragma unroll
            for (int k = 0; k < 2; k++)
                #pragma unroll
                for (int p = 0; p < per; p++)
                    dst.tiles[i][j].data[k * per + p] = detail::hk_as_elem<typename D::dtype>(
                        src.tiles[2 * i + k][j].data[p]);
}

/*
 * 4-bit weight codes -> a dequantized bf16 or half operand tile (hk.load_w4),
 * from hk's layout (hk.ops.w4a16): codes eight to an int32 in K order,
 * nibble t of word w of row n being column 8w + t, the words interleaved
 * [N/16][K/32][16 rows][4 words]; per group of G columns a scale s and the
 * zero point times it, zs, so element (n, k) is q * s - zs, one FMA.
 *
 * A 16x32 operand fragment's lane l holds columns 8(l/16) .. +7 of row l%16:
 * one word of codes. Its 64 lanes read their fragment's 256 contiguous bytes,
 * one dword each, and nothing crosses lanes (include/rdna3's tile holds each
 * row twice, and trades halves with permlanex16).
 */
namespace detail {
/// Eight codes of one word -> four pairs of T, q * s - zs: columns (2t, 2t+1)
/// in out[t]. bf16 rounds to nearest even (one v_cvt_pk_bf16_f32 a pair).
template<typename T>
__device__ inline void hk_w4_word(uint32_t word, float sc, float zsc, uint32_t out[4]) {
    const uint32_t lo = word & 0x0F0F0F0Fu;          // columns 0, 2, 4, 6
    const uint32_t hi = (word >> 4) & 0x0F0F0F0Fu;   // columns 1, 3, 5, 7
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        const float a = float((lo >> (8 * t)) & 0xFFu) * sc - zsc;
        const float b = float((hi >> (8 * t)) & 0xFFu) * sc - zsc;
        if constexpr (std::is_same_v<T, bf16>) {
            const bf16_2 p = hk_cvt_pk_bf16(a, b);
            __builtin_memcpy(&out[t], &p, 4);
        } else {
            const auto p = __builtin_amdgcn_cvt_pkrtz(a, b);
            __builtin_memcpy(&out[t], &p, 4);
        }
    }
}
}  // namespace detail

/**
 * @brief A column reduction of a col-layout fp32 tile in 16-column base
 * tiles (16x16 accumulators): a column's four lanes l, l+16, l+32, l+48
 * combined by v_permlane32_swap and v_permlane16_swap, every lane left with
 * its column's answer. include/cdna4 swaps lanes only for 32-column tiles
 * and shuffles a 16-column one through LDS -- three ds_bpermute, each waited
 * for with lgkmcnt(0), which also waits out the LDS reads in flight (the
 * paged decode's softmax: two of them a page). More constrained than the
 * library's, so such a tile picks it.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
    requires (T::base_tile_cols == 16 && std::is_same_v<typename V::dtype, float>)
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    static_assert(V::outer_dim == T::width, "compatible size");
    static_assert(T::base_tile_threads_per_reduction == 4, "four lanes a column");
    #pragma unroll
    for (int j = 0; j < src.width; j++) {
        float2 acc = src.tiles[0][j].data[0];
        #pragma unroll
        for (int k = 1; k < src.packed_per_base_tile; k++)
            acc = op::template op<float2>(acc, src.tiles[0][j].data[k]);
        #pragma unroll
        for (int i = 1; i < src.height; i++)
            #pragma unroll
            for (int k = 0; k < src.packed_per_base_tile; k++)
                acc = op::template op<float2>(acc, src.tiles[i][j].data[k]);
        float x = op::template op<float>(acc.x, acc.y);
        const auto h = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x),
                                                        false, true);
        x = op::template op<float>(__uint_as_float(h[0]), __uint_as_float(h[1]));
        const auto q = __builtin_amdgcn_permlane16_swap(__float_as_uint(x), __float_as_uint(x),
                                                        false, true);
        x = op::template op<float>(__uint_as_float(q[0]), __uint_as_float(q[1]));
        col_accum[j][0] = reset ? x : op::template op<float>(src_accum[j][0], x);
    }
}

/**
 * @brief mma_ABt of fp16 operands into an fp32 accumulator. include/cdna4's
 * takes fp16 only into an fp16 accumulator, though v_mfma_f32_16x16x32_f16
 * and its 32x32x16 sibling accumulate in fp32 and its mma_ABt_base issues
 * them. More constrained than the library's, so fp16 x fp16 -> fp32 picks it.
 */
template<ducks::rt::col_layout D, ducks::rt::all A, ducks::rt::all B, ducks::rt::col_layout C>
    requires (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, half> &&
              std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, float>)
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows && A::cols == B::cols &&
                  D::rows == C::rows && D::cols == C::cols, "mma_ABt shapes");
    #pragma unroll
    for (int n = 0; n < D::height; n++) {
        #pragma unroll
        for (int m = 0; m < D::width; m++) {
            mma_ABt_base(d.tiles[n][m], a.tiles[n][0], b.tiles[m][0], c.tiles[n][m]);
            #pragma unroll
            for (int k = 1; k < A::width; k++)
                mma_ABt_base(d.tiles[n][m], a.tiles[n][k], b.tiles[m][k], d.tiles[n][m]);
        }
    }
}

template<int G, ducks::rt::all RT, ducks::gl::all QL, ducks::gl::all SL>
__device__ inline static void load_w4(RT &dst, const QL &qw, const SL &s, const SL &zs,
                                      int n0, int k0) {
    using F = detail::hk_frag<RT>;
    using T = typename F::T;
    static_assert(F::row && F::R == 16 && F::C == 32 && F::S == 8 && F::K == 1,
                  "W4 codes load into 16x32 row-layout operand fragments");
    static_assert(std::is_same_v<T, bf16> || std::is_same_v<T, half>,
                  "W4 codes dequantize into bf16 or half tiles");
    static_assert(std::is_same_v<typename QL::dtype, int>, "codes are packed in int32");
    static_assert(std::is_same_v<typename SL::dtype, T>, "scales are the tile's type");
    static_assert(G % 8 == 0, "a lane's eight columns lie in one group");
    constexpr bool one_group = G % (32 * RT::width) == 0;

    const int lane = laneid();
    const int r = lane % 16, wq = lane / 16;      // row, and word of the chunk
    const int chunks = qw.cols() / 4;             // 32-column chunks per row
    const int groups = s.cols();
    const int *qp = reinterpret_cast<const int *>(qw.raw_ptr);
    const T *sp = s.raw_ptr;
    const T *zp = zs.raw_ptr;
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        const int n = n0 + i * 16 + r;
        const int *blk = qp + (n >> 4) * chunks * 64 + r * 4 + wq;
        float sc = 0.f, zsc = 0.f;
        if constexpr (one_group) {
            sc = base_types::convertor<float, T>::convert(sp[n * groups + k0 / G]);
            zsc = base_types::convertor<float, T>::convert(zp[n * groups + k0 / G]);
        }
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            const uint32_t word = uint32_t(blk[((k0 + j * 32) / 32) * 64]);
            if constexpr (!one_group) {
                const int g = (k0 + j * 32 + 8 * wq) / G;
                sc = base_types::convertor<float, T>::convert(sp[n * groups + g]);
                zsc = base_types::convertor<float, T>::convert(zp[n * groups + g]);
            }
            uint32_t out[4];
            detail::hk_w4_word<T>(word, sc, zsc, out);
            #pragma unroll
            for (int t = 0; t < 4; t++)
                __builtin_memcpy(&dst.tiles[i][j].data[t], &out[t], 4);
        }
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
 * @brief load_scalar of an element the kernel never writes: a load from the
 * constant address space at a uniform address, which is an s_load_dword
 * through the scalar cache. A global load's wait counts vmcnt, in one queue
 * with the DMA copies: a page index read beside them waited out every copy
 * issued before it (the paged decode's ring: one page in flight however
 * deep the ring).
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int load_scalar_invariant(const GL &src, const COORD &idx) {
    using U = typename GL::dtype;
    static_assert(sizeof(U) == 4, "a 32-bit element");
    const U *p = (const U *)&src[(idx.template unit_coord<-1, 3>())];
    typedef const __attribute__((address_space(4))) int *hk_const_i32;
    return __builtin_amdgcn_readfirstlane(*(hk_const_i32)(uintptr_t)p);
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

/* ------------------------------------------------------------------------
 * Register strips (kittens::rs) on wave64: include/rdna3's rs, its warp ops
 * and its memory ops, with the two facts that differ changed.
 *
 *  - A chunk is 64 lanes x RUN: lane l owns, of every row, the runs
 *    [j*64*RUN + l*RUN, +RUN). A bf16 run of 8 is one buffer_load_dwordx4.
 *  - The row reduction's butterfly crosses 64 lanes: DPP quad_perm (xor 1,
 *    xor 2), row_half_mirror and row_mirror inside each 16-lane row, then
 *    v_permlane16_swap and v_permlane32_swap across rows (gfx950), or
 *    ds_bpermute (__shfl_xor) on gfx942, which has no swap. Each stage
 *    combines a value with a partner that combines the same pair, so every
 *    lane ends with the same bits, as on gfx11.
 *
 * Stores narrow through hk_cvt (bf16 rounds to nearest even; the CDNA
 * trees' convertor truncates), and the bounded forms go through hk_srsrc.
 * ---------------------------------------------------------------------- */

namespace ducks {
namespace rs {
struct identifier {};
struct cv_identifier {};
}  // namespace rs
}  // namespace ducks

/** @brief One value per row of a strip, held by every lane of the warp. */
template<typename _T, int _length>
struct rs_cv {
    using identifier = ducks::rs::cv_identifier;
    static_assert(_length > 0, "a strip has at least one row");
    using T = typename base_types::packing<_T>::unpacked_type;
    using dtype = T;
    static constexpr int length    = _length;
    static constexpr int outer_dim = _length;
    static constexpr int inner_dim = 1;
    T data[outer_dim][inner_dim];
    __device__ inline       T* operator[](size_t idx)       { return &data[idx][0]; }
    __device__ inline const T* operator[](size_t idx) const { return &data[idx][0]; }
};

/** @brief ROWS rows of COLS columns, RUN contiguous elements of a row per
 * lane per 64*RUN-column chunk. */
template<typename _T, int _rows, int _cols, int _run = 8>
struct rs {
    using identifier = ducks::rs::identifier;
    using T = typename base_types::packing<_T>::unpacked_type;
    using dtype = T;
    static constexpr int rows   = _rows;
    static constexpr int cols   = _cols;
    static constexpr int run    = _run;
    static_assert(rows > 0, "a strip has at least one row");
    static_assert(run == 1 || run == 2 || run == 4 || run == 8 || run == 16,
                  "a run is a power of two of at most 16 elements");
    static constexpr int span   = run * WARP_THREADS;
    static_assert(cols % span == 0, "a strip is a whole number of chunks: COLS % (64*RUN) == 0");
    static constexpr int chunks = cols / span;
    static constexpr int elements_per_thread = rows * chunks * run;
    T data[rows][chunks][run];
    using col_vec = rs_cv<T, rows>;
    using row_vec = rs<T, 1, cols, run>;
};

namespace ducks {
namespace rs {
template<typename T>
concept all = requires { typename T::identifier; } &&
              std::is_same_v<typename T::identifier, identifier>;
template<typename T>
concept cv = requires { typename T::identifier; } &&
             std::is_same_v<typename T::identifier, cv_identifier>;
}  // namespace rs
}  // namespace ducks

namespace detail {

template<typename T> __device__ inline uint32_t hk_bits(const T &v) {
    if constexpr (sizeof(T) == 4) return __builtin_bit_cast(uint32_t, v);
    else { static_assert(sizeof(T) == 2, "a strip holds 16- and 32-bit elements");
           return uint32_t(__builtin_bit_cast(uint16_t, v)); }
}
template<typename T> __device__ inline T hk_from_bits(uint32_t u) {
    if constexpr (sizeof(T) == 4) return __builtin_bit_cast(T, u);
    else return __builtin_bit_cast(T, uint16_t(u));
}

/// v from the lane DPP control CTRL names (gfx9's encoding).
template<int CTRL, typename T> __device__ inline T hk_dpp(const T &v) {
    return hk_from_bits<T>(uint32_t(__builtin_amdgcn_mov_dpp(int(hk_bits(v)), CTRL, 0xF, 0xF, true)));
}

/// op of the value in this lane's 16-lane row and the one in the row XOR 1
/// (gfx950: v_permlane16_swap), or the half XOR 1 (v_permlane32_swap), in
/// that order in every lane.
template<typename op, int ROWS_APART, typename T> __device__ inline T hk_swap_op(const T &v) {
#if defined(KITTENS_CDNA4)
    const uint32_t u = hk_bits(v);
    if constexpr (ROWS_APART == 1) {
        const auto r = __builtin_amdgcn_permlane16_swap(u, u, false, true);
        return op::template op<T>(hk_from_bits<T>(r[0]), hk_from_bits<T>(r[1]));
    } else {
        const auto r = __builtin_amdgcn_permlane32_swap(u, u, false, true);
        return op::template op<T>(hk_from_bits<T>(r[0]), hk_from_bits<T>(r[1]));
    }
#else
    return op::template op<T>(v, hk_from_bits<T>(uint32_t(
        __shfl_xor(int(hk_bits(v)), 16 * ROWS_APART, WARP_THREADS))));
#endif
}

/// Fold N values with `op`, as a balanced tree.
template<typename op, typename T, int N>
__device__ inline T strip_tree(T (&a)[N]) {
    #pragma unroll
    for (int n = N; n > 1; n = (n + 1) / 2) {
        const int h = (n + 1) / 2;
        #pragma unroll
        for (int i = 0; i < n - h; i++) a[i] = op::template op<T>(a[i], a[i + h]);
    }
    return a[0];
}

/// Combine one value across all 64 lanes; every lane ends with the answer.
template<typename op, typename T>
__device__ inline T strip_butterfly(T v) {
    v = op::template op<T>(v, hk_dpp<0xB1>(v));    // quad_perm [1,0,3,2]: lane ^ 1
    v = op::template op<T>(v, hk_dpp<0x4E>(v));    // quad_perm [2,3,0,1]: lane ^ 2
    v = op::template op<T>(v, hk_dpp<0x141>(v));   // row_half_mirror: the other quad of 8
    v = op::template op<T>(v, hk_dpp<0x140>(v));   // row_mirror: the other 8 of 16
    v = hk_swap_op<op, 1>(v);
    return hk_swap_op<op, 2>(v);
}

template<typename RS, typename F>
__device__ inline void strip_each(F &&f) {
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            #pragma unroll
            for (int e = 0; e < RS::run; e++) f(r, j, e);
        }
    }
}

template<typename RS, typename U>
__device__ inline typename RS::T strip_scalar(const U &u) {
    using T = typename RS::T;
    if constexpr (std::is_same_v<U, T>) return u;
    else return hk_cvt<T, float>(float(u));
}

}  // namespace detail

/* ----------  maps  ---------- */

template<typename op, ducks::rs::all RS>
__device__ static inline void unary_map(RS &dst, const RS &src) {
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e]);
    });
}
template<typename op, ducks::rs::cv CV>
__device__ static inline void unary_map(CV &dst, const CV &src) {
    #pragma unroll
    for (int r = 0; r < CV::length; r++)
        dst.data[r][0] = op::template op<typename CV::T>(src.data[r][0]);
}

template<typename op, ducks::rs::all RS, typename U>
__device__ static inline void bin_map(RS &dst, const RS &lhs, const U &rhs) {
    using T = typename RS::T;
    if constexpr (std::is_same_v<U, RS>) {
        detail::strip_each<RS>([&](int r, int j, int e) {
            dst.data[r][j][e] = op::template op<T>(lhs.data[r][j][e], rhs.data[r][j][e]);
        });
    } else {
        const T s = detail::strip_scalar<RS>(rhs);
        detail::strip_each<RS>([&](int r, int j, int e) {
            dst.data[r][j][e] = op::template op<T>(lhs.data[r][j][e], s);
        });
    }
}
template<typename op, ducks::rs::cv CV, typename U>
__device__ static inline void bin_map(CV &dst, const CV &lhs, const U &rhs) {
    using T = typename CV::T;
    #pragma unroll
    for (int r = 0; r < CV::length; r++) {
        if constexpr (std::is_same_v<U, CV>)
            dst.data[r][0] = op::template op<T>(lhs.data[r][0], rhs.data[r][0]);
        else
            dst.data[r][0] = op::template op<T>(lhs.data[r][0], detail::strip_scalar<CV>(rhs));
    }
}

#define HK_STRIP_UNARY(name, opname)                                               \
    template<typename S> requires (ducks::rs::all<S> || ducks::rs::cv<S>)          \
    __device__ static inline void name(S &dst, const S &src) {                     \
        unary_map<base_ops::opname, S>(dst, src);                                   \
    }
HK_STRIP_UNARY(exp,   exp)
HK_STRIP_UNARY(exp2,  exp2)
HK_STRIP_UNARY(log,   log)
HK_STRIP_UNARY(log2,  log2)
HK_STRIP_UNARY(abs,   abs)
HK_STRIP_UNARY(relu,  relu)
HK_STRIP_UNARY(gelu,  gelu)
HK_STRIP_UNARY(dgelu, dgelu)
HK_STRIP_UNARY(sqrt,  sqrt)
HK_STRIP_UNARY(rsqrt, rsqrt)
HK_STRIP_UNARY(silu,  silu)
HK_STRIP_UNARY(step,  step)
#undef HK_STRIP_UNARY

#define HK_STRIP_FILL(name, opname)                                                \
    template<typename S> requires (ducks::rs::all<S> || ducks::rs::cv<S>)          \
    __device__ static inline void name(S &dst) {                                   \
        unary_map<base_ops::opname, S>(dst, dst);                                   \
    }
HK_STRIP_FILL(zero,      zero)
HK_STRIP_FILL(pos_infty, pos_infty)
HK_STRIP_FILL(neg_infty, neg_infty)
#undef HK_STRIP_FILL
/** @brief Fill with 1 (include/cdna4's base op is `one` too). */
template<typename S> requires (ducks::rs::all<S> || ducks::rs::cv<S>)
__device__ static inline void one(S &dst) {
    if constexpr (ducks::rs::all<S>)
        detail::strip_each<S>([&](int r, int j, int e) { dst.data[r][j][e] = typename S::T(1.f); });
    else
        for (int r = 0; r < S::length; r++) dst.data[r][0] = typename S::T(1.f);
}

#define HK_STRIP_BINARY(name, opname)                                              \
    template<typename S, typename U> requires (ducks::rs::all<S> || ducks::rs::cv<S>) \
    __device__ static inline void name(S &dst, const S &lhs, const U &rhs) {       \
        bin_map<base_ops::opname, S>(dst, lhs, rhs);                                \
    }
HK_STRIP_BINARY(add, sum)
HK_STRIP_BINARY(sub, sub)
HK_STRIP_BINARY(mul, mul)
HK_STRIP_BINARY(div, div)
HK_STRIP_BINARY(max, max)
HK_STRIP_BINARY(min, min)
#undef HK_STRIP_BINARY

/** @brief Copy a strip or a per-row vector, converting the element type. */
template<ducks::rs::all D, ducks::rs::all S>
__device__ static inline void copy(D &dst, const S &src) {
    static_assert(D::rows == S::rows && D::cols == S::cols && D::run == S::run,
                  "a strip copy keeps the shape and the run");
    detail::strip_each<D>([&](int r, int j, int e) {
        dst.data[r][j][e] = detail::hk_cvt<typename D::T, typename S::T>(src.data[r][j][e]);
    });
}
template<ducks::rs::cv D, ducks::rs::cv S>
__device__ static inline void copy(D &dst, const S &src) {
    static_assert(D::length == S::length, "a vector copy keeps the length");
    #pragma unroll
    for (int r = 0; r < D::length; r++)
        dst.data[r][0] = detail::hk_cvt<typename D::T, typename S::T>(src.data[r][0]);
}

/**
 * @brief Each row's two halves exchanged: dst(r, c) = src(r, (c + COLS/2) %
 * COLS). A row of an even number of chunks swaps registers; a row of one
 * chunk exchanges lane l's run with lane l ^ 32's, one v_permlane32_swap a
 * dword on gfx950 (ds_bpermute on gfx942).
 */
template<ducks::rs::all RS>
__device__ static inline void swap_halves(RS &dst, const RS &src) {
    using T = typename RS::T;
    if constexpr (RS::chunks % 2 == 0) {
        RS tmp;
        detail::strip_each<RS>([&](int r, int j, int e) {
            tmp.data[r][j][e] = src.data[r][(j + RS::chunks / 2) % RS::chunks][e];
        });
        dst = tmp;
    } else {
        static_assert(RS::chunks == 1, "swap_halves takes a row of one chunk or an even number");
        detail::strip_each<RS>([&](int r, int j, int e) {
            const uint32_t u = detail::hk_bits(src.data[r][j][e]);
#if defined(KITTENS_CDNA4)
            // vdst's upper half takes vsrc's lower and vsrc's lower vdst's
            // upper: from (u, u), lane l < 32 finds lane l + 32's value in
            // the second result and lane l + 32 finds lane l's in the first.
            const auto p = __builtin_amdgcn_permlane32_swap(u, u, false, true);
            dst.data[r][j][e] = detail::hk_from_bits<T>(laneid() < 32 ? p[1] : p[0]);
#else
            dst.data[r][j][e] = detail::hk_from_bits<T>(uint32_t(__shfl_xor(int(u), 32, WARP_THREADS)));
#endif
        });
    }
}

/**
 * @brief NeoX's rotate_half on each SPAN-column segment of every row:
 * dst(r, c) = -src(r, c + SPAN/2) in a segment's first half and
 * src(r, c - SPAN/2) in its second. Halves whole chunks apart swap
 * registers; halves 8, 16 or 32 lanes apart exchange lanes -- DPP row_ror:8
 * inside a 16-lane row, v_permlane16_swap, v_permlane32_swap (gfx950) -- and
 * the first half's sign is one XOR a value.
 */
template<int SPAN, ducks::rs::all RS>
__device__ static inline void rotate_half(RS &dst, const RS &src) {
    using T = typename RS::T;
    static_assert(RS::cols % SPAN == 0, "segments tile the row");
    constexpr uint32_t SIGN = sizeof(T) == 4 ? 0x80000000u : 0x8000u;
    if constexpr (SPAN % (2 * RS::span) == 0) {
        constexpr int SEG = SPAN / RS::span, H = SEG / 2;
        RS tmp;
        detail::strip_each<RS>([&](int r, int j, int e) {
            const int s = j % SEG;
            tmp.data[r][j][e] = s < H
                ? detail::hk_from_bits<T>(detail::hk_bits(src.data[r][j + H][e]) ^ SIGN)
                : src.data[r][j - H][e];
        });
        dst = tmp;
    } else {
        constexpr int LANES = SPAN / (2 * RS::run);
        static_assert(SPAN % (2 * RS::run) == 0 && (LANES == 8 || LANES == 16 || LANES == 32),
                      "a segment's halves 8, 16 or 32 lanes apart, or whole chunks");
        const int l = laneid();
        const uint32_t sign = (l % (2 * LANES)) < LANES ? SIGN : 0u;
        detail::strip_each<RS>([&](int r, int j, int e) {
            const uint32_t u = detail::hk_bits(src.data[r][j][e]);
            uint32_t p;
#if defined(KITTENS_CDNA4)
            if constexpr (LANES == 8) {
                p = uint32_t(__builtin_amdgcn_mov_dpp(int(u), 0x128, 0xF, 0xF, true));
            } else if constexpr (LANES == 16) {
                // As swap_halves: the even row of each pair finds its
                // partner's value in the second result, the odd row in the first.
                const auto q = __builtin_amdgcn_permlane16_swap(u, u, false, true);
                p = (l % 32) < 16 ? q[1] : q[0];
            } else {
                const auto q = __builtin_amdgcn_permlane32_swap(u, u, false, true);
                p = l < 32 ? q[1] : q[0];
            }
#else
            p = uint32_t(__shfl_xor(int(u), LANES, WARP_THREADS));
#endif
            dst.data[r][j][e] = detail::hk_from_bits<T>(p ^ sign);
        });
    }
}

/* ----------  reductions  ---------- */

namespace detail {
template<typename op, bool reset, ducks::rs::cv CV, ducks::rs::all RS>
__device__ inline void strip_row_reduce(CV &dst, const RS &src, const CV &acc) {
    static_assert(CV::length == RS::rows, "one value per row of the strip");
    using T = typename RS::T;
    constexpr int N = RS::chunks * RS::run;
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        T a[N];
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            #pragma unroll
            for (int e = 0; e < RS::run; e++) a[j*RS::run + e] = src.data[r][j][e];
        }
        const T v = strip_butterfly<op>(strip_tree<op>(a));
        if constexpr (reset) dst.data[r][0] = v;
        else                 dst.data[r][0] = op::template op<T>(acc.data[r][0], v);
    }
}

template<typename op, bool reset, ducks::rs::all RV, ducks::rs::all RS>
__device__ inline void strip_col_reduce(RV &dst, const RS &src, const RV &acc) {
    static_assert(RV::rows == 1 && RV::cols == RS::cols && RV::run == RS::run,
                  "a column reduction gives the strip's row_vec");
    using T = typename RS::T;
    #pragma unroll
    for (int j = 0; j < RS::chunks; j++) {
        #pragma unroll
        for (int e = 0; e < RS::run; e++) {
            T v = src.data[0][j][e];
            #pragma unroll
            for (int r = 1; r < RS::rows; r++) v = op::template op<T>(v, src.data[r][j][e]);
            if constexpr (reset) dst.data[0][j][e] = v;
            else                 dst.data[0][j][e] = op::template op<T>(acc.data[0][j][e], v);
        }
    }
}
}  // namespace detail

#define HK_STRIP_REDUCE(name, opname, kind, VEC)                                   \
    template<ducks::rs::VEC V, ducks::rs::all RS>                                   \
    __device__ static inline void name(V &dst, const RS &src) {                    \
        detail::strip_##kind##_reduce<base_ops::opname, true>(dst, src, dst);      \
    }                                                                              \
    template<ducks::rs::VEC V, ducks::rs::all RS>                                   \
    __device__ static inline void name(V &dst, const RS &src, const V &acc) {      \
        detail::strip_##kind##_reduce<base_ops::opname, false>(dst, src, acc);     \
    }
HK_STRIP_REDUCE(row_sum,  sum, row, cv)
HK_STRIP_REDUCE(row_max,  max, row, cv)
HK_STRIP_REDUCE(row_min,  min, row, cv)
HK_STRIP_REDUCE(row_prod, mul, row, cv)
HK_STRIP_REDUCE(col_sum,  sum, col, all)
HK_STRIP_REDUCE(col_max,  max, col, all)
HK_STRIP_REDUCE(col_min,  min, col, all)
HK_STRIP_REDUCE(col_prod, mul, col, all)
#undef HK_STRIP_REDUCE

/* ----------  broadcasts  ---------- */

template<typename op, ducks::rs::all RS, ducks::rs::cv CV>
__device__ static inline void row_map(RS &dst, const RS &src, const CV &v) {
    static_assert(CV::length == RS::rows, "one value per row of the strip");
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e], v.data[r][0]);
    });
}
template<typename op, ducks::rs::all RS>
__device__ static inline void col_map(RS &dst, const RS &src, const typename RS::row_vec &v) {
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e], v.data[0][j][e]);
    });
}

#define HK_STRIP_BROADCAST(name, opname)                                           \
    template<ducks::rs::all RS, ducks::rs::cv CV>                                   \
    __device__ static inline void name##_row(RS &dst, const RS &src, const CV &v) { \
        row_map<base_ops::opname>(dst, src, v);                                     \
    }                                                                              \
    template<ducks::rs::all RS>                                                     \
    __device__ static inline void name##_col(RS &dst, const RS &src,               \
                                             const typename RS::row_vec &v) {      \
        col_map<base_ops::opname, RS>(dst, src, v);                                 \
    }
HK_STRIP_BROADCAST(add, sum)
HK_STRIP_BROADCAST(sub, sub)
HK_STRIP_BROADCAST(mul, mul)
HK_STRIP_BROADCAST(div, div)
#undef HK_STRIP_BROADCAST

/* ----------  fills  ---------- */

namespace detail {
template<bool left, ducks::rs::all RS>
__device__ inline void strip_axis_fill(RS &dst, const RS &src, int col_idx,
                                       const typename RS::T &val) {
    const int lane0 = laneid() * RS::run;
    detail::strip_each<RS>([&](int r, int j, int e) {
        const int c = j*RS::span + lane0 + e;
        const bool hit = left ? c < col_idx : c >= col_idx;
        dst.data[r][j][e] = hit ? val : src.data[r][j][e];
    });
}
}  // namespace detail

/// Columns col_idx and past it become `val`; the rest are copied from `src`.
template<ducks::rs::all RS>
__device__ static inline void right_fill(RS &dst, const RS &src, const int col_idx,
                                         const typename RS::T &val = 0) {
    detail::strip_axis_fill<false>(dst, src, col_idx, val);
}
/// Columns before col_idx become `val`; the rest are copied from `src`.
template<ducks::rs::all RS>
__device__ static inline void left_fill(RS &dst, const RS &src, const int col_idx,
                                        const typename RS::T &val = 0) {
    detail::strip_axis_fill<true>(dst, src, col_idx, val);
}

/* ----------  memory  ---------- */

namespace detail {

using hk_u32x4 = unsigned __attribute__((ext_vector_type(4)));
using hk_u32x2 = unsigned __attribute__((ext_vector_type(2)));

/// RUN elements at `p`, in the widest accesses the run's bytes allow.
template<typename U, int RUN>
__device__ inline void strip_read(U (&run)[RUN], const U *p) {
    constexpr int BYTES = RUN * sizeof(U);
    if constexpr (BYTES % 16 == 0) {
        #pragma unroll
        for (int k = 0; k < BYTES / 16; k++) {
            const hk_u32x4 v = *(reinterpret_cast<const hk_u32x4*>(p) + k);
            __builtin_memcpy((void*)&run[k * 16 / sizeof(U)], &v, 16);
        }
    } else if constexpr (BYTES == 8) {
        const hk_u32x2 v = *reinterpret_cast<const hk_u32x2*>(p);
        __builtin_memcpy((void*)&run[0], &v, 8);
    } else if constexpr (BYTES == 4) {
        const uint32_t v = *reinterpret_cast<const uint32_t*>(p);
        __builtin_memcpy((void*)&run[0], &v, 4);
    } else if constexpr (BYTES == 2) {
        const uint16_t v = *reinterpret_cast<const uint16_t*>(p);
        __builtin_memcpy((void*)&run[0], &v, 2);
    } else {
        static_assert(BYTES == 1, "a run is 1, 2, 4, 8 or a multiple of 16 bytes");
        __builtin_memcpy((void*)&run[0], (const void*)p, 1);
    }
}

template<typename U, int RUN>
__device__ inline void strip_write(U *p, const U (&run)[RUN]) {
    constexpr int BYTES = RUN * sizeof(U);
    if constexpr (BYTES % 16 == 0) {
        #pragma unroll
        for (int k = 0; k < BYTES / 16; k++) {
            hk_u32x4 v;
            __builtin_memcpy(&v, (const void*)&run[k * 16 / sizeof(U)], 16);
            *(reinterpret_cast<hk_u32x4*>(p) + k) = v;
        }
    } else if constexpr (BYTES == 8) {
        hk_u32x2 v; __builtin_memcpy(&v, (const void*)&run[0], 8);
        *reinterpret_cast<hk_u32x2*>(p) = v;
    } else if constexpr (BYTES == 4) {
        uint32_t v; __builtin_memcpy(&v, (const void*)&run[0], 4);
        *reinterpret_cast<uint32_t*>(p) = v;
    } else if constexpr (BYTES == 2) {
        uint16_t v; __builtin_memcpy(&v, (const void*)&run[0], 2);
        *reinterpret_cast<uint16_t*>(p) = v;
    } else {
        static_assert(BYTES == 1, "a run is 1, 2, 4, 8 or a multiple of 16 bytes");
        __builtin_memcpy((void*)p, (const void*)&run[0], 1);
    }
}

/// Past any buffer range this library builds: reads zero, writes nothing.
static constexpr uint32_t HK_STRIP_OOB = 0x80000000u;

/// The descriptor over a strip's rows that are in bounds, from column 0.
template<int ROWS, typename GL>
__device__ inline hk_i32x4 strip_srsrc(const GL &g, const coord<> &idx) {
    using U = typename GL::dtype;
    coord<> row0 = idx;
    row0.c = 0;
    const int left = int(g.rows()) - idx.r;
    const int valid = left <= 0 ? 0 : (left < ROWS ? left : ROWS);
    return hk_srsrc(&g[row0], uint64_t(valid) * uint64_t(g.row_stride()) * sizeof(U));
}

}  // namespace detail

/** @brief Global -> strip. Row r of the strip is row idx.r + r of `src`, from column idx.c. */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void load(RS &dst, const GL &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const U *base = &src[idx] + laneid() * RS::run;
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            U run[RS::run];
            detail::strip_read(run, base + r * stride + j * RS::span);
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                dst.data[r][j][e] = base_types::convertor<T, U>::convert(run[e]);
        }
    }
}

/** @brief load() of PERIOD columns repeated: strip column c is column
 * idx.c + c % PERIOD of the row. Unbounded. */
template<int PERIOD, ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void load_period(RS &dst, const GL &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    static_assert(PERIOD % RS::run == 0, "a period of whole runs");
    const U *base = &src[idx];
    const int lane0 = laneid() * RS::run;
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            U run[RS::run];
            detail::strip_read(run, base + r * stride + (j * RS::span + lane0) % PERIOD);
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                dst.data[r][j][e] = base_types::convertor<T, U>::convert(run[e]);
        }
    }
}

/** @brief Strip -> global, the inverse of load(). */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void store(const GL &dst, const RS &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    U *base = &dst[idx] + laneid() * RS::run;
    const int stride = dst.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            U run[RS::run];
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                run[e] = detail::hk_cvt<U, T>(src.data[r][j][e]);
            detail::strip_write(base + r * stride + j * RS::span, run);
        }
    }
}

/** @brief load(), predicated on the global's extent: rows past its last row
 * and runs at or past its last column read as zero and touch no memory. */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void load_bounded(RS &dst, const GL &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const detail::hk_i32x4 srsrc = detail::strip_srsrc<RS::rows>(src, idx);
    const int lane0 = laneid() * RS::run;
    const int cols_left = int(src.cols()) - idx.c;
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            const int c = j * RS::span + lane0;
            const uint32_t voff = c < cols_left ? uint32_t(r * stride + idx.c + c) * sizeof(U)
                                                : detail::HK_STRIP_OOB;
            U run[RS::run];
            detail::hk_buf_load_bytes<RS::run * sizeof(U)>(srsrc, voff, run);
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                dst.data[r][j][e] = base_types::convertor<T, U>::convert(run[e]);
        }
    }
}

/** @brief store(), predicated as load_bounded(). */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void store_bounded(const GL &dst, const RS &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const detail::hk_i32x4 srsrc = detail::strip_srsrc<RS::rows>(dst, idx);
    const int lane0 = laneid() * RS::run;
    const int cols_left = int(dst.cols()) - idx.c;
    const int stride = dst.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            const int c = j * RS::span + lane0;
            const uint32_t voff = c < cols_left ? uint32_t(r * stride + idx.c + c) * sizeof(U)
                                                : detail::HK_STRIP_OOB;
            U run[RS::run];
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                run[e] = detail::hk_cvt<U, T>(src.data[r][j][e]);
            detail::hk_buf_store_bytes<RS::run * sizeof(U)>(srsrc, voff, run);
        }
    }
}

/** @brief Global -> one value per row: entry r is element (idx.r + r, idx.c). */
template<ducks::rs::cv CV, ducks::gl::all GL>
__device__ inline static void load(CV &dst, const GL &src, const coord<> &idx) {
    using U = typename GL::dtype;
    const U *base = &src[idx];
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < CV::length; r++)
        dst.data[r][0] = base_types::convertor<typename CV::T, U>::convert(base[r * stride]);
}

/** @brief Entry 0 of a uniform per-row vector -> one element, from lane 0. */
template<ducks::gl::all GL, ducks::rs::cv CV>
__device__ inline static void store_scalar(const GL &dst, const CV &src, const coord<> &idx) {
    using U = typename GL::dtype;
    if (laneid() != 0) return;
    *(U*)&dst[idx] = detail::hk_cvt<U, typename CV::T>(src.data[0][0]);
}

namespace detail {
/// An LDS vector's entries through an LDS pointer: through the generic one
/// the vector is reached by, they are flat_load/flat_store, which wait on
/// both counters and take the flat path.
template<typename U> __device__ inline __attribute__((address_space(3))) U *hk_lds(const U *p) {
    return (__attribute__((address_space(3))) U *)(uint32_t)(uintptr_t)p;
}
}  // namespace detail

/** @brief A per-row vector -> the first CV::length entries of an LDS vector,
 * from lane 0 (every lane holds the same values). */
template<ducks::sv::all SV, ducks::rs::cv CV>
__device__ inline static void store(SV &dst, const CV &src) {
    static_assert(SV::length >= CV::length, "the LDS vector holds every row's value");
    using U = typename SV::dtype;
    if (laneid() == 0) {
        auto *p = detail::hk_lds(&dst.data[0]);
        #pragma unroll
        for (int r = 0; r < CV::length; r++)
            p[r] = detail::hk_cvt<U, typename CV::T>(src.data[r][0]);
    }
}

/** @brief The first CV::length entries of an LDS vector -> a per-row vector, in every lane. */
template<ducks::rs::cv CV, ducks::sv::all SV>
__device__ inline static void load(CV &dst, const SV &src) {
    static_assert(SV::length >= CV::length, "the LDS vector holds every row's value");
    using U = typename SV::dtype;
    const auto *p = detail::hk_lds(&src.data[0]);
    #pragma unroll
    for (int r = 0; r < CV::length; r++) {
        const U u = p[r];
        dst.data[r][0] = base_types::convertor<typename CV::T, U>::convert(u);
    }
}

}  // namespace kittens
