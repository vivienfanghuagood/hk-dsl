/**
 * @file
 * @brief hk.load_w4's symmetric form on RDNA (include/rdna3, include/rdna4): 4-bit weight codes
 * with one zero point for every group, dequantized as (q - ZERO) * s -- no zs read.
 *
 * The trees' load_w4(dst, qw, s, zs, n0, k0) reads a scale s and a zero-times-scale zs per row and
 * group: at GPTQ's 32-column groups that is 4 bytes of metadata per 16 bytes of codes a row, +25%
 * over the codes. A symmetric checkpoint (GPTQ's uint4b8: every zero point 8) need not store zs;
 * this form reads no zs (12.5% fewer bytes). A half tile decodes (q - ZERO) * s in packed fp16 --
 * q - ZERO exact, the product rounded once to nearest even (w4_word_sym_f16), half the VALU of the
 * fp32 path, which rounds toward zero; a bf16 tile keeps the trees' fp32 decode with zs = ZERO * s
 * (bit for bit the asymmetric form's on such a checkpoint). Otherwise the trees' load_w4 line for
 * line: the same layout, loads and (on gfx11) half exchange. Included by a kernel that uses it
 * (scaffold.py).
 */

#pragma once

namespace kittens {

namespace detail {
/// Eight codes of one word -> four half pairs (q - ZERO) * s, columns (2t, 2t+1) in out[t]:
/// half(0x6400 | q) is 1024 + q exactly, so subtracting 1024 + ZERO gives q - ZERO exactly, and
/// one v_pk_mul_f16 by s rounds the product once, to nearest even. A pair is a v_perm, an or,
/// a v_pk_add and a v_pk_mul: half the work of the fp32 path (two conversions, two FMAs and a
/// pack, rounding toward zero).
template<int ZERO>
__device__ inline void w4_word_sym_f16(uint32_t word, uint32_t s2, uint32_t out[4]) {
    typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
    const uint32_t lo = word & 0x0F0F0F0Fu;          // columns 0, 2, 4, 6
    const uint32_t hi = (word >> 4) & 0x0F0F0F0Fu;   // columns 1, 3, 5, 7
    h2_t sc;
    __builtin_memcpy(&sc, &s2, 4);
    const h2_t bias = {_Float16(-(1024 + ZERO)), _Float16(-(1024 + ZERO))};
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        // bytes: lo's byte t, 0, hi's byte t, 0 -- then the exponent bits of 1024 in each half
        const uint32_t u = __builtin_amdgcn_perm(hi, lo, 0x0C040C00u + 0x00010001u * t) | 0x64006400u;
        h2_t h;
        __builtin_memcpy(&h, &u, 4);
        const h2_t v = (h + bias) * sc;
        __builtin_memcpy(&out[t], &v, 4);
    }
}
}  // namespace detail

/**
 * @brief Tile rows n0.. and columns k0.. of W from its 4-bit codes, element (n, k) being
 * (q - ZERO) * s[n][k / G], dequantized into a bf16 or half row-layout tile. The contract of the
 * trees' load_w4 otherwise (a whole number of 32-column chunks, k0 a multiple of 32). SBLK: s is
 * [N/16][K/G * 16], a 16-row block's scales of a group 16 consecutive values (32 contiguous bytes
 * for the 16 rows a k-step reads, where [N][K/G] puts them on 16 lines).
 */
template<int G, int ZERO, bool SBLK = false, ducks::rt::all RT, ducks::gl::all QL, ducks::gl::all SL>
__device__ inline static void load_w4(RT &dst, const QL &qw, const SL &s, int n0, int k0) {
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using L = typename RT::layout;
    using base = rt_base<T, L>;
    static_assert(std::is_same_v<L, ducks::rt_layout::row>, "W4 codes load into row-layout tiles");
    static_assert(std::is_same_v<T, bf16> || std::is_same_v<T, half>,
                  "W4 codes dequantize into bf16 or half tiles");
    static_assert(base::element_stride == 1 &&
                  (base::elements_per_thread == 16 || base::elements_per_thread == 8),
                  "a lane holds a base tile's 16 columns of its row (gfx11) or 8 of them (gfx12)");
    static_assert(std::is_same_v<typename QL::dtype, int>, "codes are packed in int32");
    static_assert(std::is_same_v<typename SL::dtype, T>, "scales are the tile's type");
    static_assert(G % 16 == 0, "a group spans whole base tiles");
    static_assert(RT::width % 2 == 0, "the tile is a whole number of 32-column chunks");
    static_assert(ZERO >= 0 && ZERO < 16, "a 4-bit zero point");
    constexpr bool one_group = G % (16 * RT::width) == 0;
    constexpr bool halves = base::elements_per_thread == 16;   // gfx11: both halves hold the row

    const int r = laneid() & 15;
    const bool upper = laneid() >= 16;           // this half's columns: 8..15
    const int chunks = qw.cols() / 4;            // 32-column chunks per row
    const int groups = SBLK ? s.cols() / 16 : s.cols();
    const int *qp = reinterpret_cast<const int *>(qw.raw_ptr);
    const T *sp = s.raw_ptr;
    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        const int n = n0 + i * dst.tile_size_row + r;
        const int *blk = qp + (n >> 4) * chunks * 64 + r * 4;
        // a group's scale of row n: s[n][g], or with SBLK s[n / 16][16 g + n % 16]
        auto s_at = [&](int g) { return SBLK ? (n >> 4) * groups * 16 + g * 16 + (n & 15) : n * groups + g; };
        float sc = 0.f, zsc = 0.f;
        uint32_t s2 = 0;                         // half tiles: the scale as a half pair
        auto scale = [&](int g) {
            if constexpr (std::is_same_v<T, half>) {
                uint16_t b;
                __builtin_memcpy(&b, &sp[s_at(g)], 2);
                s2 = uint32_t(b) * 0x00010001u;
            } else {
                sc = base_types::convertor<float, T>::convert(sp[s_at(g)]);
                zsc = sc * float(ZERO);
            }
        };
        if constexpr (one_group)
            scale(k0 / G);
        #pragma unroll
        for (int j = 0; j < dst.width; j += 2) {
            const int4 w = *reinterpret_cast<const int4 *>(
                blk + ((k0 + j * dst.tile_size_col) / 32) * 64);
            #pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                if constexpr (!one_group)
                    scale((k0 + (j + jj) * dst.tile_size_col) / G);
                // Word 0 of a base tile is its columns 0..7, word 1 8..15.
                const uint32_t mine = uint32_t(jj ? (upper ? w.w : w.z) : (upper ? w.y : w.x));
                uint32_t own[4];
                if constexpr (std::is_same_v<T, half>)
                    detail::w4_word_sym_f16<ZERO>(mine, s2, own);
                else
                    detail::w4_word<T>(mine, sc, zsc, own);
                if constexpr (halves) {
                    uint32_t other[4];
                    #pragma unroll
                    for (int t = 0; t < 4; t++) {
                        other[t] = uint32_t(__builtin_amdgcn_permlanex16(
                            int(own[t]), int(own[t]), 0x76543210u, 0xfedcba98u, false, false));
                    }
                    #pragma unroll
                    for (int t = 0; t < 4; t++) {
                        const uint32_t c0 = upper ? other[t] : own[t];
                        const uint32_t c1 = upper ? own[t] : other[t];
                        __builtin_memcpy(&dst.tiles[i][j + jj].data[t], &c0, 4);
                        __builtin_memcpy(&dst.tiles[i][j + jj].data[4 + t], &c1, 4);
                    }
                } else {
                    #pragma unroll
                    for (int t = 0; t < 4; t++)
                        __builtin_memcpy(&dst.tiles[i][j + jj].data[t], &own[t], 4);
                }
            }
        }
    }
}

}  // namespace kittens
