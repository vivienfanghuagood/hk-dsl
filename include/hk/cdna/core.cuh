/**
 * @file
 * @brief include/hk/cdna.cuh, part core.
 *
 * The base every other part builds on: int8 as a storage type, strided_gl and
 * make_gl, the waits and binds, and the detail plumbing (buffer descriptors and
 * intrinsics, fragment coordinates, conversions, the register-tile load / store bodies,
 * scoped and LDS helpers).
 *
 * Included by hk/cdna.cuh in the order core, lib, formats, special, primitives, escape;
 * the kind of every function is in python/hk/dsl/surface.py (CPP).
 */

#pragma once

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
#ifdef HK_SRSRC_DIRECT
    // the four words built straight from uniform scalars (no struct round trip), so the backend
    // keeps the descriptor in SGPRs across a loop instead of a VGPR copy it readfirstlanes per use
    const uint64_t p = reinterpret_cast<uint64_t>(base);
    const uint32_t range = bytes > 0xFFFFFFFFull ? 0xFFFFFFFFu : uint32_t(bytes);
    return hk_i32x4{int(__builtin_amdgcn_readfirstlane(uint32_t(p))),
                    int(__builtin_amdgcn_readfirstlane(uint32_t(p >> 32))),
                    int(__builtin_amdgcn_readfirstlane(range)), 0x00020000};
#endif
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
    static_assert(base_types::packing<typename RT::dtype>::num() == 2 ||
                  (base_types::packing<typename RT::dtype>::num() == 4 && sizeof(T) == 1),
                  "hk's CDNA memory ops take bf16, fp16, fp32 and fp8 register tiles");
    static constexpr bool row = std::is_same_v<typename RT::layout, ducks::rt_layout::row>;
#if defined(KITTENS_CDNA4)
    static constexpr int R = RT::base_tile_rows, C = RT::base_tile_cols;
    // fp8 (16x128) is include/cdna4's rt_16x128: lane l holds row l%16 and
    // two runs of 16 along K, at 16(l/16) and 64 + 16(l/16) -- the operand
    // layout of v_mfma_scale_f32_16x16x128_f8f6f4, which applies its block
    // scales by K position, not by which lane holds the data (measured: a
    // lane holding one contiguous 32-wide block scaled the wrong blocks).
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
    if constexpr (std::is_same_v<U, T>) return v;
    else if constexpr (std::is_integral_v<U> && std::is_integral_v<T>) return static_cast<U>(v);
    else if constexpr (std::is_same_v<U, bf16> && std::is_same_v<T, float>) return __float2bfloat16(v);
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

/** `t`, its values fixed here: an empty asm takes every element, so what computed them (an
 * MFMA, the VALU after it) stays above this point. For a primitive that consumes a tile only
 * inside a lane-divergent `if`: roc-7.2.3's MachineSink moves V_MFMA_SCALE (not the plain
 * MFMAs) into the one block that uses its result, and the copies that assemble its operands
 * (an FP4 fragment out of AGPRs) go with it. They then run under the `if`'s partial EXEC and
 * leave the masked lanes' operand registers stale; the MFMA reads all 64 lanes, so even the
 * valid rows come out wrong (K1's scatter-add stage 2 was 0.28 off). No instruction of its
 * own; the tile is live before the `if`, as the stores need it. */
template<typename RT> __device__ inline RT hk_pinned(const RT &t) {
    RT p = t;
#ifndef HK_UNPINNED     // tests/hk/codegen/test_mfma_scale_sink.py builds without the pin
    constexpr int N = sizeof(p.tiles[0][0].data) / sizeof(p.tiles[0][0].data[0]);
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int k = 0; k < N; k++)
                asm volatile("" : "+v"(p.tiles[i][j].data[k].x), "+v"(p.tiles[i][j].data[k].y));
#endif
    return p;
}

template<typename RT> __device__ inline auto hk_get(const RT &t, int i, int j, int e) {
    return (e & 1) ? t.tiles[i][j].data[e >> 1].y : t.tiles[i][j].data[e >> 1].x;
}
template<typename RT, typename V> __device__ inline void hk_set(RT &t, int i, int j, int e, V v) {
    if (e & 1) t.tiles[i][j].data[e >> 1].y = v;
    else       t.tiles[i][j].data[e >> 1].x = v;
}

// AUX: the access's cache-policy bits (gfx950: SC0 1, NT 2, SC1 16; {SC1,SC0} = its scope) -- 0 but for
// the scoped accesses of hk.comm (kittens::load_scoped / store_scoped).
template<typename U, int AUX = 0> __device__ inline U hk_buf_load_elem(hk_i32x4 r, uint32_t off) {
    U v;
    if constexpr (sizeof(U) == 1) { const uint8_t  x = hk_buf_ld8 (r, off, 0, AUX); __builtin_memcpy(&v, &x, 1); }
    else if constexpr (sizeof(U) == 2) { const uint16_t x = hk_buf_ld16(r, off, 0, AUX); __builtin_memcpy(&v, &x, 2); }
    else { static_assert(sizeof(U) == 4, "8-, 16- or 32-bit elements"); const uint32_t x = hk_buf_ld32(r, off, 0, AUX); __builtin_memcpy(&v, &x, 4); }
    return v;
}
template<typename U, int AUX = 0> __device__ inline void hk_buf_store_elem(hk_i32x4 r, uint32_t off, U v) {
    if constexpr (sizeof(U) == 1) { uint8_t  x; __builtin_memcpy(&x, &v, 1); hk_buf_st8 (x, r, off, 0, AUX); }
    else if constexpr (sizeof(U) == 2) { uint16_t x; __builtin_memcpy(&x, &v, 2); hk_buf_st16(x, r, off, 0, AUX); }
    else { static_assert(sizeof(U) == 4, "8-, 16- or 32-bit elements"); uint32_t x; __builtin_memcpy(&x, &v, 4); hk_buf_st32(x, r, off, 0, AUX); }
}

/** `N` contiguous bytes at `off`, in the widest accesses that tile them. */
template<int N, int AUX = 0> __device__ inline void hk_buf_load_bytes(hk_i32x4 r, uint32_t off, void *dst) {
    if constexpr (N % 16 == 0) {
        #pragma unroll
        for (int q = 0; q < N / 16; q++) {
            // a masked run (off ~0) stays past the range: off + 16 q would wrap to the tensor's start
            const __uint128_t x = hk_buf_ld128(r, off == ~0u ? ~0u : off + 16 * q, 0, AUX);
            __builtin_memcpy((char *)dst + 16 * q, &x, 16);
        }
    }
    else if constexpr (N == 8) { const uint64_t x = hk_buf_ld64(r, off, 0, AUX); __builtin_memcpy(dst, &x, 8); }
    else if constexpr (N == 4) { const uint32_t x = hk_buf_ld32(r, off, 0, AUX); __builtin_memcpy(dst, &x, 4); }
    else if constexpr (N == 2) { const uint16_t x = hk_buf_ld16(r, off, 0, AUX); __builtin_memcpy(dst, &x, 2); }
    else if constexpr (N == 1) { const uint8_t x = hk_buf_ld8(r, off, 0, AUX); __builtin_memcpy(dst, &x, 1); }
    else static_assert(N == 16, "a run of 1, 2, 4, 8 or a multiple of 16 bytes");
}
template<int N, int AUX = 0> __device__ inline void hk_buf_store_bytes(hk_i32x4 r, uint32_t off, const void *src) {
    if constexpr (N % 16 == 0) {
        #pragma unroll
        for (int q = 0; q < N / 16; q++) {
            __uint128_t x; __builtin_memcpy(&x, (const char *)src + 16 * q, 16);
            hk_buf_st128(x, r, off == ~0u ? ~0u : off + 16 * q, 0, AUX);   // (as the load)
        }
    }
    else if constexpr (N == 8) { uint64_t x; __builtin_memcpy(&x, src, 8); hk_buf_st64(x, r, off, 0, AUX); }
    else if constexpr (N == 4) { uint32_t x; __builtin_memcpy(&x, src, 4); hk_buf_st32(x, r, off, 0, AUX); }
    else if constexpr (N == 2) { uint16_t x; __builtin_memcpy(&x, src, 2); hk_buf_st16(x, r, off, 0, AUX); }
    else if constexpr (N == 1) { uint8_t x; __builtin_memcpy(&x, src, 1); hk_buf_st8(x, r, off, 0, AUX); }
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

template<typename RT, typename GL, typename COORD, int AUX = 0>
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
                    hk_buf_load_bytes<F::S * sizeof(U), AUX>(
                        r, ((i * F::R + c.x) * rs + j * F::C + c.y) * sizeof(U), run);
                    if constexpr (sizeof(T) == 1) {
                        // fp8: a lane's run is S consecutive bytes of its
                        // packed registers, in element order.
                        static_assert(std::is_same_v<T, U>, "an fp8 tile loads from the same fp8 type");
                        __builtin_memcpy((char *)&dst.tiles[i][j].data[0] + k * F::S, run, F::S);
                    } else {
                        #pragma unroll
                        for (int s = 0; s < F::S; s++)
                            hk_set(dst, i, j, k * F::S + s, base_types::convertor<T, U>::convert(run[s]));
                    }
                }
            } else {
                static_assert(sizeof(T) != 1, "an fp8 register tile loads in the row layout");
                // A run goes down a column: one element per row.
                #pragma unroll
                for (int e = 0; e < F::E; e++) {
                    const int2 c = hk_frag_coord<RT>(e, lane);
                    hk_set(dst, i, j, e, base_types::convertor<T, U>::convert(hk_buf_load_elem<U, AUX>(
                        r, ((i * F::R + c.x) * rs + j * F::C + c.y) * sizeof(U))));
                }
            }
        }
    }
}

template<typename RT, typename GL, typename COORD, int AUX = 0>
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
                    hk_buf_store_bytes<16, AUX>(
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
                    hk_buf_store_bytes<F::S * sizeof(U), AUX>(
                        r, ((i * F::R + c.x) * rs + j * F::C + c.y) * sizeof(U), run);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < F::E; e++) {
                    const int2 c = hk_frag_coord<RT>(e, lane);
                    hk_buf_store_elem<U, AUX>(r, ((i * F::R + c.x) * rs + j * F::C + c.y) * sizeof(U),
                                         hk_cvt<U, T>(hk_get(src, i, j, e)));
                }
            }
        }
    }
}

}  // namespace detail

namespace detail {
/// hk.atomic_scalar's ops by code: 0 add, 1 max, 2 min, 3 or, 4 and, 5 xor, 6 exch (relaxed, at SCOPE).
template<int SCOPE> __device__ inline int hk_atomic_op(int *p, int v, int op) {
    switch (op) {
    case 1: return __hip_atomic_fetch_max(p, v, __ATOMIC_RELAXED, SCOPE);
    case 2: return __hip_atomic_fetch_min(p, v, __ATOMIC_RELAXED, SCOPE);
    case 3: return __hip_atomic_fetch_or(p, v, __ATOMIC_RELAXED, SCOPE);
    case 4: return __hip_atomic_fetch_and(p, v, __ATOMIC_RELAXED, SCOPE);
    case 5: return __hip_atomic_fetch_xor(p, v, __ATOMIC_RELAXED, SCOPE);
    case 6: return __hip_atomic_exchange(p, v, __ATOMIC_RELAXED, SCOPE);
    default: return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, SCOPE);
    }
}
}  // namespace detail

namespace detail {
/// An LDS pointer, not a generic one. Through a generic pointer (the allocator's int * hides the
/// address space from the compiler) every access was a FLAT load or store: slower to issue than
/// ds_read / ds_write, and counted in vmcnt as well as lgkmcnt, so each wait for an LDS value also
/// waited for every global load in flight.
typedef __attribute__((address_space(3))) float hk_lds_float;
template<typename ST> __device__ inline hk_lds_float *hk_st_elem(ST &s, int r, int c) {
    const int sub = (r / ST::underlying_subtile_rows) * ST::underlying_subtiles_per_row +
                    c / ST::underlying_subtile_cols;
    const uint32_t off = ST::swizzle({r % ST::underlying_subtile_rows, c % ST::underlying_subtile_cols});
    typedef __attribute__((address_space(3))) char lds_char;
    return (hk_lds_float *)((lds_char *)&s.data[0] + sub * ST::underlying_subtile_bytes + off);
}
}  // namespace detail

#if defined(KITTENS_CDNA4)
namespace detail {
typedef __attribute__((address_space(3))) uint32_t *hk_lds_ptr;
__device__ void hk_buf_ld_lds(hk_i32x4 rsrc, hk_lds_ptr lds, int size, int voffset, int soffset,
                              int offset, int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
}  // namespace detail
#endif

#if defined(KITTENS_CDNA4)
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
#endif

namespace detail {
/// An element access with the cache-policy scope AUX (16 device, 17 system): a relaxed atomic of that
/// scope, which LLVM's memory model lowers to the SC1 / SC0|SC1 load or store (32-bit elements).
template<int AUX, typename U> __device__ inline U hk_ld_aux(const U *p) {
    if constexpr (AUX == 0) return *p;
    else {
        static_assert(sizeof(U) == 4, "a scoped vector access moves 32-bit elements");
        return __hip_atomic_load(p, __ATOMIC_RELAXED,
                                 AUX == 17 ? __HIP_MEMORY_SCOPE_SYSTEM : __HIP_MEMORY_SCOPE_AGENT);
    }
}
template<int AUX, typename U> __device__ inline void hk_st_aux(U *p, U v) {
    if constexpr (AUX == 0) *p = v;
    else {
        static_assert(sizeof(U) == 4, "a scoped vector access moves 32-bit elements");
        __hip_atomic_store(p, v, __ATOMIC_RELAXED,
                           AUX == 17 ? __HIP_MEMORY_SCOPE_SYSTEM : __HIP_MEMORY_SCOPE_AGENT);
    }
}
}  // namespace detail

}  // namespace kittens
