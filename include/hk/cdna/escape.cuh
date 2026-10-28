/**
 * @file
 * @brief include/hk/cdna.cuh, part escape.
 *
 * Escape hatches: map_cpp (a C++ expression over strips) and mfma_strip (one raw MFMA on
 * strip registers). Allowed only within their registered boundaries.
 *
 * Included by hk/cdna.cuh in the order core, lib, formats, special, primitives, escape;
 * the kind of every function is in python/hk/dsl/surface.py (CPP).
 */

#pragma once

namespace kittens {

/** @brief hk mfma_strip f32_16x16x4: acc (4 a lane) += A B^T, lane l holding A[l%16][l/16], B[l%16][l/16]. */
template<ducks::rs::all D, ducks::rs::all A, ducks::rs::all B>
__device__ static inline void mfma_strip_f32_16x16x4(D &dst, const A &a, const B &b, const D &acc) {
    typedef float v4f __attribute__((ext_vector_type(4)));
    v4f c = {acc.data[0][0][0], acc.data[0][0][1], acc.data[0][0][2], acc.data[0][0][3]};
    c = __builtin_amdgcn_mfma_f32_16x16x4f32(a.data[0][0][0], b.data[0][0][0], c, 0, 0, 0);
    dst.data[0][0][0] = c[0]; dst.data[0][0][1] = c[1]; dst.data[0][0][2] = c[2]; dst.data[0][0][3] = c[3];
}

namespace detail {
template<typename A> __device__ inline auto hk_lane_arg(const A &a, int r, int j, int e) {
    if constexpr (ducks::rs::all<A>) return a.data[r][j][e];
    else return a;
}
}  // namespace detail
/** @brief hk.map_cpp: dst's element (r, c) = f(each strip argument's element (r, c), or the scalar). */
template<ducks::rs::all D, typename F, typename... A>
__device__ static inline void map_cpp(D &dst, F &&f, const A &... a) {
    detail::strip_each<D>([&](int r, int j, int e) {
        dst.data[r][j][e] = static_cast<typename D::T>(f(detail::hk_lane_arg(a, r, j, e)...));
    });
}

}  // namespace kittens
