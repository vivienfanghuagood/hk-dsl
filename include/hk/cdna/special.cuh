/**
 * @file
 * @brief include/hk/cdna.cuh, part special.
 *
 * Type-selected specializations: the col-layout masks and fills with -inf and the
 * 16-column column reduction. Chosen by type alone, bitwise the generic path's
 * (tests/hk/gpu/test_specializations.py). Registry kind `internal`.
 *
 * Included by hk/cdna.cuh in the order core, lib, formats, special, primitives, escape;
 * the kind of every function is in python/hk/dsl/surface.py (CPP).
 */

#pragma once

namespace kittens {

#if defined(KITTENS_CDNA4)
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
#endif

#if defined(KITTENS_CDNA4)
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
#endif

}  // namespace kittens
