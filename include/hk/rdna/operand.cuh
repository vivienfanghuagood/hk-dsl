/**
 * @file
 * @brief RDNA3's paired operand: hk's Tile.as_operand(fp16, k_order="even_odd") (gfx1100).
 *
 * P^T for a PV WMMA, two S^T fragments at a time. src: rt<float, 32, 16, col>, the fp32
 * accumulators of kv rows [0, 16) and [16, 32) of a 32-row pair: lane l, element e holds kv row
 * 2e + h of its fragment (h = l >> 4), query l & 15. dst: rt<half, 32, 16, col>; tile 0 is the
 * operand whose K axis is the pair's 16 even rows (0, 2, .., 30), tile 1 the odd rows. A lane's
 * own values are then already one of the two operands, in order; the other comes from the mirror
 * lane: per packed dword one conversion pair (to nearest even), one permlanex16 and two selects,
 * against a per-element exchange for the natural order. (attn_k3's kda::pair_p, moved here
 * unchanged: DSL-surface S11.) Included by a kernel that uses it (scaffold.py).
 */

#pragma once

namespace kittens {

template<typename DT, typename ST>
__device__ static inline void as_operand_even_odd(DT &dst, const ST &src) {
    typedef float f2_t __attribute__((ext_vector_type(2)));
    typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
    static_assert(ST::height == 2 && ST::width == 1 && DT::height == 2 && DT::width == 1, "a pair");
    static_assert(sizeof(src.tiles[0][0].data) == 32 && sizeof(dst.tiles[0][0].data) == 32, "16x16");
    const bool h = (__lane_id() >> 4) != 0;
    #pragma unroll
    for (int f = 0; f < 2; f++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            f2_t x;
            __builtin_memcpy(&x, (const char *)&src.tiles[f][0].data[0] + 8 * q, 8);
            const h2_t hh = __builtin_convertvector(x, h2_t);     // v_cvt_f16_f32: nearest even
            int own;
            __builtin_memcpy(&own, &hh, 4);
            const int oth = __builtin_amdgcn_permlanex16(own, own, 0x76543210, 0xfedcba98, false, false);
            const int ev = h ? oth : own, od = h ? own : oth;
            __builtin_memcpy((char *)&dst.tiles[0][0].data[0] + 4 * (4 * f + q), &ev, 4);
            __builtin_memcpy((char *)&dst.tiles[1][0].data[0] + 4 * (4 * f + q), &od, 4);
        }
    }
}

}  // namespace kittens
