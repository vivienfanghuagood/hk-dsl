/**
 * @file
 * @brief include/hk/cdna.cuh, part lib.
 *
 * Library fixes and gap fills: what include/cdna3 and include/cdna4 lack, or define
 * wrongly for hk -- base ops, bf16 round-to-nearest-even copies, fp32 tiles and vectors to LDS
 * through an LDS pointer, the bounded group copies, fp16 x fp16 -> fp32 mma_ABt, gfx942's
 * register vectors. Registry kind `internal`; candidates to upstream to the HipKittens fork.
 *
 * Included by hk/cdna.cuh in the order core, lib, formats, special, primitives, escape;
 * the kind of every function is in python/hk/dsl/surface.py (CPP).
 */

#pragma once

namespace kittens {

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
/*
 * fp32 register vectors <-> LDS vectors (hk.SharedVec: hk.collective.cross_warp, a vector read back
 * in another tile's layout). include/cdna4's load(rv, sv) / store(sv, rv) index sv::data through a
 * generic pointer, so they compiled to FLAT, as the fp32 tiles above did: each wait for the vector
 * also waited for every global load in flight -- in an attention kernel, the next KV block's DMA.
 * The same element order through an LDS pointer. More specialized than the library's templates.
 */
template<size_t N, size_t L, size_t TL, typename SH, typename LAY>
__device__ inline static void store(sv<float, N> &dst, const rv<float, L, TL, SH, LAY> &src) {
    using RV = rv<float, L, TL, SH, LAY>;
    static_assert(N == RV::length);
    typedef __attribute__((address_space(3))) char lds_char;
    typedef __attribute__((address_space(3))) float lds_f;
    lds_f *d = (lds_f *)((lds_char *)&dst.data[0]);
    const int laneid = ::kittens::laneid();
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for (int w = 0; w < RV::outer_dim; w++) {
            const int idx = w * RV::reductions + RV::stride * (laneid / RV::aligned_threads);
            #pragma unroll
            for (int i = 0; i < RV::strides_per_tile; i++)
                #pragma unroll
                for (int j = 0; j < RV::packed_per_stride; j++)
                {
                    // two floats, not a float2: HIP's vector types are classes, and a class
                    // object in address space 3 has no assignment operator
                    const float2 v = src[w][i * RV::packed_per_stride + j];
                    lds_f *p = d + idx + i * RV::elements_per_stride_group + j * RV::packing;
                    p[0] = v.x;
                    p[1] = v.y;
                }
        }
    } else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for (int w = 0; w < RV::outer_dim; w++)
            d[w * RV::reductions + (laneid % RV::reductions)] = src[w][0];
    } else {
        const int offset = laneid * RV::inner_dim;
        if (offset >= RV::length) return;
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; i++) d[offset + i] = src[0][i];
    }
}
/*
 * A 16-bit col-layout register tile -> LDS (an attention kernel's P exchanged between warps).
 * include/cdna4's store(st, rt<col>) writes element by element through `U *` into st::data: a generic
 * pointer, so FLAT stores, counted in vmcnt. The same addresses through an LDS pointer.
 */
template<ducks::st::all ST, int R, int C, typename RS>
    requires (std::is_same_v<typename ST::dtype, bf16> || std::is_same_v<typename ST::dtype, half>)
__device__ inline static void store(ST &dst, const rt<typename ST::dtype, R, C, ducks::rt_layout::col, RS> &src) {
    using RT = rt<typename ST::dtype, R, C, ducks::rt_layout::col, RS>;
    using U = typename ST::dtype;
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    static_assert(RT::rows == ST::rows && RT::cols == ST::cols, "register tile and shared tile must match");
    typedef __attribute__((address_space(3))) char lds_char;
    typedef __attribute__((address_space(3))) unsigned short lds_U;  // 16-bit bits: bf16 is a class
    lds_U *base = (lds_U *)((lds_char *)&dst.data[0]);
    static_assert(sizeof(U) == 2);
    const int laneid = kittens::laneid();
    const int row_offset = src.base_tile_stride * (laneid / src.base_tile_cols);
    const int col_offset = laneid % src.base_tile_cols;
    if constexpr (ST::underlying_subtile_rows >= RT::base_tile_rows && ST::underlying_subtile_cols >= RT::base_tile_cols) {
        constexpr int per_row = ST::underlying_subtile_cols / RT::base_tile_cols;
        constexpr int per_col = ST::underlying_subtile_rows / RT::base_tile_rows;
        #pragma unroll
        for (int k = 0; k < RT::base_tile_num_strides; k++)
            #pragma unroll
            for (int l = 0; l < src.base_tile_stride / packing; l++)
                #pragma unroll
                for (int i = 0; i < per_col; i++)
                    #pragma unroll
                    for (int j = 0; j < per_row; j++) {
                        const int row = i * RT::base_tile_rows + row_offset + k * RT::base_tile_elements_per_stride_group + l * 2;
                        const int col = j * RT::base_tile_cols + col_offset;
                        lds_U *addr = base + dst.swizzle({row, col}) / sizeof(U);
                        lds_U *next = base + dst.swizzle({row + 1, col}) / sizeof(U);
                        const int idx = l + k * RT::base_tile_stride / packing;
                        #pragma unroll
                        for (int ii = 0; ii < ST::subtiles_per_col; ii++)
                            #pragma unroll
                            for (int jj = 0; jj < ST::subtiles_per_row; jj++) {
                                const int off = ((ii * ST::underlying_subtiles_per_row + jj) * ST::underlying_subtile_bytes) / sizeof(U);
                                const int rr = ii * per_col + i, rc = jj * per_row + j;
                                addr[off] = __builtin_bit_cast(unsigned short, base_types::convertor<U, T>::convert(src.tiles[rr][rc].data[idx].x));
                                next[off] = __builtin_bit_cast(unsigned short, base_types::convertor<U, T>::convert(src.tiles[rr][rc].data[idx].y));
                            }
                    }
    } else if constexpr (ST::underlying_subtile_rows <= RT::base_tile_rows && ST::underlying_subtile_cols <= RT::base_tile_cols) {
        constexpr int sub_per_row = RT::base_tile_cols / ST::underlying_subtile_cols;
        const int col = col_offset % ST::underlying_subtile_cols;
        const int base_col = col_offset / ST::underlying_subtile_cols;
        #pragma unroll
        for (int k = 0; k < RT::base_tile_num_strides; k++)
            #pragma unroll
            for (int l = 0; l < src.base_tile_stride / packing; l++) {
                const int row = (row_offset + k * RT::base_tile_elements_per_stride_group) % ST::underlying_subtile_rows + l * 2;
                const int base_row = (row_offset + k * RT::base_tile_elements_per_stride_group) / ST::underlying_subtile_rows;
                const int base_off = (base_row * sub_per_row + base_col) * ST::underlying_subtile_bytes;
                lds_U *addr = base + (dst.swizzle({row, col}) + base_off) / sizeof(U);
                lds_U *next = base + (dst.swizzle({row + 1, col}) + base_off) / sizeof(U);
                const int idx = l + k * RT::base_tile_stride / packing;
                #pragma unroll
                for (int i = 0; i < RT::height; i++)
                    #pragma unroll
                    for (int j = 0; j < RT::width; j++) {
                        const int sid = (i * (RT::base_tile_rows / ST::underlying_subtile_rows)) * ST::underlying_subtiles_per_row +
                                        j * sub_per_row;
                        const int off = (sid * ST::underlying_subtile_bytes) / sizeof(U);
                        addr[off] = __builtin_bit_cast(unsigned short, base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].x));
                        next[off] = __builtin_bit_cast(unsigned short, base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].y));
                    }
            }
    } else {
        static_assert(false, "Unsupported subtile sizes");
    }
}
template<size_t N, size_t L, size_t TL, typename SH, typename LAY>
__device__ inline static void load(rv<float, L, TL, SH, LAY> &dst, const sv<float, N> &src) {
    using RV = rv<float, L, TL, SH, LAY>;
    static_assert(N == RV::length);
    typedef __attribute__((address_space(3))) char lds_char;
    typedef __attribute__((address_space(3))) float lds_f;
    const lds_f *s = (const lds_f *)((const lds_char *)&src.data[0]);
    const int laneid = ::kittens::laneid();
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for (int w = 0; w < RV::outer_dim; w++) {
            const int idx = w * RV::reductions + RV::stride * (laneid / RV::aligned_threads);
            #pragma unroll
            for (int i = 0; i < RV::strides_per_tile; i++)
                #pragma unroll
                for (int j = 0; j < RV::packed_per_stride; j++)
                {
                    const lds_f *p = s + idx + i * RV::elements_per_stride_group + j * RV::packing;
                    dst[w][i * RV::packed_per_stride + j] = float2{p[0], p[1]};
                }
        }
    } else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for (int w = 0; w < RV::outer_dim; w++)
            dst[w][0] = s[w * RV::reductions + (laneid % RV::reductions)];
    } else {
        const int offset = laneid * RV::inner_dim;
        if (offset >= RV::length) return;
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; i++) dst[0][i] = s[offset + i];
    }
}
#endif

#if defined(KITTENS_CDNA4)
/**
 * @brief The group store (hk's `g.store(dst, src)`), predicated on the
 * tensor's rows: include/cdna4's store writes every row of the tile, so a
 * last, partial tile wrote past the tensor -- into the neighbour's memory,
 * silently. Rows past the tensor's last are not written; the rest is that
 * store's code.
 */
template<int NW, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_store_bounded(const strided_gl<U> &dst, const ST &src, const COORD &idx) {
    using T = typename ST::dtype;
    static_assert(std::is_same_v<T, U>, "T and U must be the same type");
    constexpr int N_THREADS = NW * WARP_THREADS;
    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);
    constexpr int bytes_per_warp = bytes_per_thread * WARP_THREADS;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    constexpr int num_warps = NW;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;
    const int row_stride = dst.template stride<2>();
    coord<> unit_coord = idx.template unit_coord<2, 3>();
    const int valid = dst.rows() - unit_coord.r;
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(&dst[unit_coord]);
    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    auto copy = [&](int lane_byte_offset) {
        const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
        const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
        const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
        const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;
        const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
        const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
        const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});
        const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) +
                                        subtile_row * ST::underlying_subtile_rows;
        const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) +
                                        subtile_col * ST::underlying_subtile_cols;
        if (swizzled_global_row >= valid)
            return;
        U *dst_elem_ptr = (U *)(dst_ptr + (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T));
        T *src_elem_ptr = (T *)(src_ptr + lane_byte_offset);
        #pragma unroll
        for (int j = 0; j < elems_per_thread; j++)
            dst_elem_ptr[j] = src_elem_ptr[j];
    };
    if constexpr (memcpy_per_tile > 0) {
        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++)
            copy((laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp));
    }
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {
        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_warps = leftover_bytes / bytes_per_thread / WARP_THREADS;
        if (warpid < leftover_warps)
            copy((laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp));
    }
}
#endif

#if defined(KITTENS_CDNA4)
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
#endif

#if defined(KITTENS_CDNA3)
/*
 * The group copies on gfx942: include/cdna3's, unbounded -- a last, partial
 * tile reads and writes past the tensor there (gfx950's are bounded above).
 * Not changed without a gfx942 to test it on.
 */
template<int NW, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_load_sync(ST &dst, const strided_gl<U> &src, const COORD &idx) {
    group<NW>::load(dst, src, idx);
}
template<int NW, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_store_bounded(const strided_gl<U> &dst, const ST &src, const COORD &idx) {
    group<NW>::store(dst, src, idx);
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
template<typename RV, typename U, int AUX = 0> __device__ inline void hk_rv_load(RV &dst, const U *p) {
    using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
    const int lane = laneid();
    #pragma unroll
    for (int w = 0; w < RV::outer_dim; w++) {
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            #pragma unroll
            for (int k = 0; k < RV::inner_dim; k++) {
                const int c = 16 * w + 4 * (lane / 16) + 2 * k;
                dst.data[w][k].x = base_types::convertor<T, U>::convert(hk_ld_aux<AUX>(p + c));
                dst.data[w][k].y = base_types::convertor<T, U>::convert(hk_ld_aux<AUX>(p + c + 1));
            }
        } else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            dst.data[w][0] = base_types::convertor<T, U>::convert(hk_ld_aux<AUX>(p + 16 * w + lane % 16));
        } else {
            const int i = 64 * w + lane;
            if (i < RV::length) dst.data[w][0] = base_types::convertor<T, U>::convert(hk_ld_aux<AUX>(p + i));
        }
    }
}
/// One lane of each group holding an entry writes it.
template<typename RV, typename U, int AUX = 0> __device__ inline void hk_rv_store(U *p, const RV &src) {
    using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
    const int lane = laneid();
    #pragma unroll
    for (int w = 0; w < RV::outer_dim; w++) {
        if constexpr (std::is_same_v<typename RV::layout, align_l>) {
            if (lane % 16 == 0) {
                #pragma unroll
                for (int k = 0; k < RV::inner_dim; k++) {
                    const int c = 16 * w + 4 * (lane / 16) + 2 * k;
                    hk_st_aux<AUX>(p + c, hk_cvt<U, T>(src.data[w][k].x));
                    hk_st_aux<AUX>(p + c + 1, hk_cvt<U, T>(src.data[w][k].y));
                }
            }
        } else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
            if (lane < 16) hk_st_aux<AUX>(p + 16 * w + lane, hk_cvt<U, T>(src.data[w][0]));
        } else {
            const int i = 64 * w + lane;
            if (i < RV::length) hk_st_aux<AUX>(p + i, hk_cvt<U, T>(src.data[w][0]));
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
/** @brief load() / store() of a register vector with the access scope AUX (hk.comm; 32-bit elements). */
template<int AUX, ducks::rv::all RV, typename U, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load_scoped(RV &dst, const strided_gl<U> &src, const COORD &idx) {
    detail::hk_rv_load<RV, U, AUX>(dst, (const U *)&src[(idx.template unit_coord<-1, 3>())]);
}
template<int AUX, ducks::rv::all RV, typename U, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store_scoped(const strided_gl<U> &dst, const RV &src, const COORD &idx) {
    detail::hk_rv_store<RV, U, AUX>((U *)&dst[(idx.template unit_coord<-1, 3>())], src);
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

}  // namespace kittens
