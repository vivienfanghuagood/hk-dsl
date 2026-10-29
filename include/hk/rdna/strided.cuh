/**
 * @file
 * @brief Strided globals on RDNA: `!hk.gl<T, strided>`, `kittens::gl_strided<T>`.
 *
 * A gl is a base pointer, four extents and one row stride, with the outer axes
 * compact over the rows. That is every tensor a kernel allocates for itself,
 * and not the layouts engines hand over. vLLM's paged KV caches are the case in
 * point: ROCM_ATTN keeps K as (blocks, kv_heads, head_dim / x, block, x) with
 * x = 16 bytes of elements, so one token's row is head_dim / x granules of 16
 * bytes, `block * 16` bytes apart; the flash layout (blocks, block, kv_heads,
 * head_dim) puts a head's rows kv_heads * head_dim apart and its pages
 * block * kv_heads * head_dim apart. Neither has a gl.
 *
 * gl_strided carries every stride, in elements: batch, depth and row, and a
 * *granule* stride along the columns -- from one 16-byte granule of a row to
 * the next (16 / sizeof(T) for a row-contiguous tensor). Element (b, d, r, c)
 * is at
 *
 *     b * batch_stride + d * depth_stride + r * row_stride
 *       + (c / G) * granule_stride + c % G,          G = 16 / sizeof(T).
 *
 * The launchers fill it from the torch tensor's strides (runtime/codeobj.py,
 * tensor_args): a 4-D view whose last axis is contiguous, or a 5-D view
 * (b, d, r, granules, G) whose last axis is one 16-byte granule. Field names and
 * the first six fields' order are gl's, so the layout table the launchers pack
 * by (codegen/scaffold.py, GL_FIELDS) addresses both; the three strides past
 * them have a table of their own (STRIDED_FIELDS).
 *
 * The ops that read one: g.stage (stage_load_strided below). The translation
 * refuses any other use, so a kernel never compiles against an op that would
 * address it as a gl.
 */

#pragma once

namespace kittens {

template<typename _T> struct gl_strided {
    using T     = typename base_types::packing<_T>::unpacked_type;
    using dtype = T;
    static constexpr int G = 16 / int(sizeof(T));   // elements in a 16-byte granule

    T* raw_ptr;
    ducks::gl::runtime_dim batch_internal, depth_internal, rows_internal, cols_internal;
    int row_stride_internal;
    long long depth_stride_internal, batch_stride_internal, granule_stride_internal;

    __host__ __device__ inline int batch() const { return int(batch_internal); }
    __host__ __device__ inline int depth() const { return int(depth_internal); }
    __host__ __device__ inline int rows() const { return int(rows_internal); }
    __host__ __device__ inline int cols() const { return int(cols_internal); }
    __host__ __device__ inline int row_stride() const { return row_stride_internal; }
    __host__ __device__ inline long long depth_stride() const { return depth_stride_internal; }
    __host__ __device__ inline long long batch_stride() const { return batch_stride_internal; }
    __host__ __device__ inline long long granule_stride() const { return granule_stride_internal; }

    /// Element offset of (b, d, r, c), 64-bit: a cache is easily more than 2^31 elements.
    __host__ __device__ inline long long offset(int b, int d, int r, int c) const {
        return (long long)b * batch_stride_internal + (long long)d * depth_stride_internal
             + (long long)r * row_stride_internal
             + (long long)(c / G) * granule_stride_internal + (c % G);
    }
    __device__ inline T& operator[](const coord<ducks::default_type> &i) const {
        return raw_ptr[offset(i.b, i.d, i.r, i.c)];
    }
};

/**
 * @brief g.stage from a strided global: load_global_to_register_buffer's
 * chunk -> lane mapping (slot c holds chunk c * N_THREADS + lane, a chunk is
 * one 16-byte granule, row-major over the tile), each chunk addressed through
 * the global's row and granule strides. store_register_buffer_to_shared then
 * writes the tile unchanged.
 *
 * One raw buffer spans the tile's first element to its last: rows past the
 * tensor's last row read zero by predicate (with a granule stride larger than
 * the row stride, as in an x-split cache, a row past the end can sit inside the
 * span, so the range alone would not zero it). The descriptor's base is the
 * tile's first element, computed in 64 bits; offsets inside a tile are 32-bit.
 */
template<int N_THREADS, ducks::st::all ST, typename U, typename COORD>
__device__ inline void stage_load_strided(float4* reg_buffer, const int buffer_size,
                                          const gl_strided<U>& src, const COORD& idx,
                                          const ST& dst_template) {
    using T = typename ST::dtype;
    static_assert(sizeof(T) == sizeof(typename gl_strided<U>::T), "a stage keeps the element size");
    constexpr int G = 16 / int(sizeof(T));
    constexpr int granules_per_row = ST::cols / G;
    constexpr int total_chunks = (ST::rows * ST::cols) / G;
    constexpr int total_calls = stage_calls<ST, N_THREADS>;
    static_assert(ST::cols % G == 0, "a tile row is whole granules");

    coord<> u = idx.template unit_coord<2, 3>();
    const T* base = reinterpret_cast<const T*>(src.raw_ptr) + src.offset(u.b, u.d, u.r, u.c);
    const int left = src.rows() - u.r;
    const int valid_rows = left <= 0 ? 0 : (left < ST::rows ? left : ST::rows);
    const long long rs = src.row_stride(), gs = src.granule_stride();
    const long long span = valid_rows == 0 ? 0
        : ((valid_rows - 1) * rs + (granules_per_row - 1) * gs + G) * (long long)sizeof(T);
    i32x4 srsrc = make_srsrc(base, uint32_t(span));
    const int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for (int c = 0; c < total_calls; ++c) {
        const int chunk_idx = c * N_THREADS + laneid;
        if (c < buffer_size && chunk_idx < total_chunks) {
            const int row = chunk_idx / granules_per_row;
            const int g = chunk_idx % granules_per_row;
            uint32_t byte_offset = uint32_t((row * rs + g * gs) * (long long)sizeof(T));
            if (row >= valid_rows) byte_offset = 0x80000000u;   // past the range: reads 0
            __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
            reg_buffer[c] = *reinterpret_cast<float4*>(&raw);
        }
    }
}

} // namespace kittens
