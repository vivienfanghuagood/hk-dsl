/**
 * @file
 * @brief hk.map on RDNA strips (include/rdna3, include/rdna4: wave32).
 *
 * The lowering of the hk.map op -- a region of scalar ops, or hk.map_cpp's C++ body -- as
 * include/hk/cdna/escape.cuh has it for gfx950: dst's element (r, c) is f of each strip
 * argument's element (r, c), or of a scalar argument itself. The trees' own detail::strip_each
 * walks a strip's elements. Included by a kernel that has an hk.map (scaffold.py).
 *
 * Mask strips (b1) do not exist here: the trees' rs takes no bool (base_types::T1), and the
 * hk-map-fuse pass refuses a mask that would leave its map on RDNA.
 */

#pragma once

namespace kittens {

namespace detail {
template<typename A> __device__ inline auto hk_lane_arg(const A &a, int r, int j, int e) {
    if constexpr (ducks::rs::all<A>) return a.data[r][j][e];
    else return a;
}
}  // namespace detail

/** @brief hk.map: dst's element (r, c) = f(each strip argument's element (r, c), or the scalar). */
template<ducks::rs::all D, typename F, typename... A>
__device__ static inline void map_cpp(D &dst, F &&f, const A &... a) {
    detail::strip_each<D>([&](int r, int j, int e) {
        dst.data[r][j][e] = static_cast<typename D::T>(f(detail::hk_lane_arg(a, r, j, e)...));
    });
}

}  // namespace kittens
