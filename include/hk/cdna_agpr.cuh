/**
 * @file
 * @brief An MMA accumulator held in AGPRs (gfx950/gfx942, hk.arch.agpr).
 *
 * LLVM picks the VGPR form of every MFMA unless register pressure forces
 * otherwise, so a tile hk wants in AGPRs is kept out of LLVM's hands: its
 * storage is one vector a base tile, every MFMA into it is inline asm with
 * "+a" C/D, and every other access goes through an explicit copy (an empty
 * asm that ties the value into the other register file; LLVM emits the
 * v_accvgpr_read / write). The MFMAs are invisible to LLVM's hazard
 * recognizer, so the waits are here:
 *   - each MFMA is issued after `s_nop 1` (a VALU or accvgpr write of an
 *     operand needs 2 wait states before an MFMA reads it);
 *   - `drain()` (24 wait states) goes before any read of the accumulator:
 *     a 16-pass XDL op needs 20 before a VALU reads its result on gfx950.
 * hk's C++ emitter calls these; a kernel never does. Values are bitwise the
 * VGPR form's: the same MFMAs on the same operands in the same order.
 */

#pragma once

#if !defined(KITTENS_CDNA3) && !defined(KITTENS_CDNA4)
#error "hk/cdna_agpr.cuh is for KITTENS_CDNA3 / KITTENS_CDNA4 builds"
#endif

#include <type_traits>

namespace hkagpr {

template <typename RT>
struct base_of {
    using type = std::remove_reference_t<decltype(std::declval<RT &>().tiles[0][0])>;
    static constexpr int floats = sizeof(type::data) / sizeof(float);
};

/// The accumulator: RT's tiles, one float vector of the base tile's
/// per-lane elements each, in the same lane order as RT's data.
template <typename RT>
struct acc {
    static constexpr int height = RT::height, width = RT::width;
    static constexpr int floats = base_of<RT>::floats;
    typedef float vec_t __attribute__((ext_vector_type(floats)));
    vec_t v[height][width];
};

__device__ __forceinline__ void drain() {
    asm volatile("s_nop 7\n\ts_nop 7\n\ts_nop 7");
}

template <typename RT>
__device__ __forceinline__ void zero(acc<RT> &t) {
#pragma unroll
    for (int i = 0; i < RT::height; i++)
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            typename acc<RT>::vec_t z = {};
            asm volatile("" : "=a"(t.v[i][j]) : "0"(z));
        }
}

/// out = t, in VGPRs. Call drain() first.
template <typename RT>
__device__ __forceinline__ void read(RT &out, const acc<RT> &t) {
#pragma unroll
    for (int i = 0; i < RT::height; i++)
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            typename acc<RT>::vec_t x;
            asm volatile("" : "=v"(x) : "0"(t.v[i][j]));
            __builtin_memcpy(&out.tiles[i][j].data[0], &x, sizeof(x));
        }
}

/// t = in.
template <typename RT>
__device__ __forceinline__ void write(acc<RT> &t, const RT &in) {
#pragma unroll
    for (int i = 0; i < RT::height; i++)
#pragma unroll
        for (int j = 0; j < RT::width; j++) {
            typename acc<RT>::vec_t x;
            __builtin_memcpy(&x, &in.tiles[i][j].data[0], sizeof(x));
            asm volatile("" : "=a"(t.v[i][j]) : "0"(x));
        }
}

/// t[:, j] (op)= v[j] on a col-layout tile (kittens::col_map's indexing),
/// a base tile at a time through VGPRs. Call drain() first.
template <typename RT, typename V, bool Div>
__device__ __forceinline__ void col_scale(acc<RT> &t, const V &vec) {
    static_assert(std::is_same_v<typename RT::layout, kittens::ducks::rt_layout::col>,
                  "hkagpr: column ops on a col-layout accumulator");
#pragma unroll
    for (int j = 0; j < RT::width; j++) {
        const float s = vec[j][0];
#pragma unroll
        for (int i = 0; i < RT::height; i++) {
            typename acc<RT>::vec_t x;
            asm volatile("" : "=v"(x) : "0"(t.v[i][j]));
            if constexpr (Div) x = x / s; else x = x * s;
            asm volatile("" : "=a"(t.v[i][j]) : "0"(x));
        }
    }
}
template <typename RT, typename V>
__device__ __forceinline__ void mul_col(acc<RT> &t, const V &vec) { col_scale<RT, V, false>(t, vec); }
template <typename RT, typename V>
__device__ __forceinline__ void div_col(acc<RT> &t, const V &vec) { col_scale<RT, V, true>(t, vec); }

typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 fp16x8_t;

/// One MFMA: c += a * b on 8-element operands, the base tile's shape by
/// its per-lane float count (16: 32x32x16, 4: 16x16x32).
template <int F, typename V, typename OA, typename OB>
__device__ __forceinline__ void mfma(V &c, const OA &a, const OB &b) {
    using E = std::remove_cv_t<std::remove_reference_t<decltype(a.data[0])>>;
    static_assert(sizeof(a.data) == 16 && sizeof(b.data) == 16, "hkagpr: 8-element bf16/fp16 operands");
    if constexpr (std::is_same_v<E, kittens::bf16_2>) {
        if constexpr (F == 16)
            asm volatile("s_nop 1\n\tv_mfma_f32_32x32x16_bf16 %0, %1, %2, %0"
                         : "+a"(c) : "v"(*(const bf16x8_t *)a.data), "v"(*(const bf16x8_t *)b.data));
        else {
            static_assert(F == 4, "hkagpr: a 32x32 or 16x16 accumulator");
            asm volatile("s_nop 1\n\tv_mfma_f32_16x16x32_bf16 %0, %1, %2, %0"
                         : "+a"(c) : "v"(*(const bf16x8_t *)a.data), "v"(*(const bf16x8_t *)b.data));
        }
    } else {
        static_assert(std::is_same_v<E, kittens::half_2>, "hkagpr: bf16 or fp16 operands");
        if constexpr (F == 16)
            asm volatile("s_nop 1\n\tv_mfma_f32_32x32x16_f16 %0, %1, %2, %0"
                         : "+a"(c) : "v"(*(const fp16x8_t *)a.data), "v"(*(const fp16x8_t *)b.data));
        else {
            static_assert(F == 4, "hkagpr: a 32x32 or 16x16 accumulator");
            asm volatile("s_nop 1\n\tv_mfma_f32_16x16x32_f16 %0, %1, %2, %0"
                         : "+a"(c) : "v"(*(const fp16x8_t *)a.data), "v"(*(const fp16x8_t *)b.data));
        }
    }
}

// d += op(a) op(b), the base tiles in kittens::mma_*'s order.
template <typename RT, typename A, typename B>
__device__ __forceinline__ void mma_AB(acc<RT> &d, const A &a, const B &b) {
#pragma unroll
    for (int n = 0; n < RT::height; n++)
#pragma unroll
        for (int m = 0; m < RT::width; m++)
#pragma unroll
            for (int k = 0; k < A::width; k++)
                mfma<acc<RT>::floats>(d.v[n][m], a.tiles[n][k], b.tiles[k][m]);
}
template <typename RT, typename A, typename B>
__device__ __forceinline__ void mma_ABt(acc<RT> &d, const A &a, const B &b) {
#pragma unroll
    for (int n = 0; n < RT::height; n++)
#pragma unroll
        for (int m = 0; m < RT::width; m++)
#pragma unroll
            for (int k = 0; k < A::width; k++)
                mfma<acc<RT>::floats>(d.v[n][m], a.tiles[n][k], b.tiles[m][k]);
}
template <typename RT, typename A, typename B>
__device__ __forceinline__ void mma_AtB(acc<RT> &d, const A &a, const B &b) {
#pragma unroll
    for (int n = 0; n < RT::height; n++)
#pragma unroll
        for (int m = 0; m < RT::width; m++)
#pragma unroll
            for (int k = 0; k < A::height; k++)
                mfma<acc<RT>::floats>(d.v[n][m], a.tiles[k][n], b.tiles[k][m]);
}
template <typename RT, typename A, typename B>
__device__ __forceinline__ void mma_AtBt(acc<RT> &d, const A &a, const B &b) {
#pragma unroll
    for (int n = 0; n < RT::height; n++)
#pragma unroll
        for (int m = 0; m < RT::width; m++)
#pragma unroll
            for (int k = 0; k < A::height; k++)
                mfma<acc<RT>::floats>(d.v[n][m], a.tiles[k][n], b.tiles[m][k]);
}

}  // namespace hkagpr
