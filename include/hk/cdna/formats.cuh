/**
 * @file
 * @brief include/hk/cdna.cuh, part formats.
 *
 * Data formats: hk's int4 weights (load_w4) and the MX block-scaled operands (E8M0
 * scales in the scaled MFMA's lanes, AITER's shuffled MXFP4 weights, the MX quantizing
 * epilogue, mma_ABt_mx). Registry kind `format` (or `merge` / `primitive` as listed).
 *
 * Included by hk/cdna.cuh in the order core, lib, formats, special, primitives, escape;
 * the kind of every function is in python/hk/dsl/surface.py (CPP).
 */

#pragma once

namespace kittens {

#if defined(KITTENS_CDNA4)
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
#endif

#if defined(KITTENS_CDNA4)
#if defined(KITTENS_CDNA4)
/**
 * E8M0 block scales of an fp8 operand tile of R rows and KB 32-wide blocks,
 * in the lanes v_mfma_scale_f32_16x16x128_f8f6f4 reads them from: lane l
 * holds, for 16-row base tile i and 128-wide K step s, the scale of row
 * i*16 + l%16 and block s*4 + l/16, as byte i%4 of reg[i/4][s] -- the byte an
 * MFMA of base tile i selects with opsel = i%4.
 */
template<int R, int KB, int F = 0> struct hk_mxs {
    static_assert(R % 16 == 0 && KB % 4 == 0, "16-row base tiles, 128-wide K steps");
    static_assert(F >= 0 && F <= 2 && (F != 2 || R % 32 == 0), "fmt 0, 1, 2 (2: 32-row blocks)");
    static constexpr int tiles = R / 16, steps = KB / 4, regs = (tiles + 3) / 4;
    // F 0 packed: base tile i, step s -> byte i % 4 of reg[i / 4][s]
    // F 1 loose: reg[i][s], byte 0          F 2 AITER raw: reg[i / 2][s / 2], byte (s % 2) * 2 + i % 2
    static constexpr int D0 = F == 0 ? regs : F == 1 ? tiles : tiles / 2;
    static constexpr int D1 = F == 2 ? (steps + 1) / 2 : steps;
    uint32_t reg[D0][D1];
    __device__ __forceinline__ uint32_t word(int i, int s) const {
        if constexpr (F == 0) return reg[i / 4][s];
        else if constexpr (F == 1) return reg[i][s];
        else return reg[i / 2][s / 2];
    }
    // F 2 with one K step: a 16-bit load of that step's (gate, up) bytes -- byte i % 2 -- so the
    // step's parity is an address, not a select; with more steps the tile starts on an even step
    static constexpr int sel(int i, int s) {
        return F == 0 ? i % 4 : F == 1 ? 0 : (steps == 1 ? i % 2 : (s % 2) * 2 + i % 2);
    }
};

/**
 * Scales of rows row0.. and blocks kb0.. of an E8M0 tensor: [rows, K/32] uint8
 * (one byte each, the OCP MX layout), or the packed layout (int32): row g of
 * [rows/64, K/128 * 64] holds, at column s*64 + l, the dword lane l takes for
 * the 64 rows g*64.. and the 128-wide K step s -- byte i the scale of row
 * g*64 + i*16 + l%16 and block s*4 + l/16 -- so a lane reads one dword per four
 * base tiles and K step, coalesced (row0 % 64 == 0, kb0 % 4 == 0).
 */
template<int R, int KB, int F, ducks::gl::all GL>
__device__ inline static void load_mx_scales(hk_mxs<R, KB, F> &dst, const GL &src, int row0, int kb0) {
    static_assert(sizeof(typename GL::dtype) == 1 || sizeof(typename GL::dtype) == 4,
                  "E8M0 scales: one byte each, or packed dwords");
    if constexpr (F == 1) {
        // loose: the byte load lands in its own register as it is (zero-extended, byte 0)
        static_assert(sizeof(typename GL::dtype) == 1, "loose scales: the byte layout");
        const int lane = laneid();
        const uint32_t rs = uint32_t(src.template stride<2>());
        const uint8_t *base = reinterpret_cast<const uint8_t *>(src.raw_ptr);
        const detail::hk_i32x4 r = detail::hk_srsrc(base, uint64_t(src.template shape<2>()) * rs);
        #pragma unroll
        for (int i = 0; i < R / 16; i++)
            #pragma unroll
            for (int s = 0; s < KB / 4; s++)
                dst.reg[i][s] = detail::hk_buf_ld8(r, uint32_t(row0 + i * 16 + lane % 16) * rs +
                                                          uint32_t(kb0 + s * 4 + lane / 16), 0, 0);
        return;
    } else {
    static_assert(F == 0, "packed scales are fmt 0");
    if constexpr (sizeof(typename GL::dtype) == 4) {
        using M = hk_mxs<R, KB>;
        static_assert(R % 64 == 0, "packed scales: whole 64-row groups");
        const int lane = laneid();
        const uint32_t rs = uint32_t(src.template stride<2>()) * 4u;
        const uint8_t *base = reinterpret_cast<const uint8_t *>(src.raw_ptr);
        const detail::hk_i32x4 r = detail::hk_srsrc(base, uint64_t(src.template shape<2>()) * rs);
        const uint32_t off0 = uint32_t(row0 / 64) * rs + uint32_t(kb0 / 4) * 256u + uint32_t(lane) * 4u;
        #pragma unroll
        for (int g = 0; g < M::regs; g++)
            #pragma unroll
            for (int s = 0; s < M::steps; s++)
                dst.reg[g][s] = detail::hk_buf_ld32(r, off0 + uint32_t(g) * rs + uint32_t(s) * 256u, 0, 0);
        return;
    }
    using M = hk_mxs<R, KB>;
    const int lane = laneid();
    const uint32_t rs = uint32_t(src.template stride<2>());
    const uint8_t *base = reinterpret_cast<const uint8_t *>(src.raw_ptr);
    const uint64_t bytes = uint64_t(src.template shape<2>()) * rs;
    const detail::hk_i32x4 r = detail::hk_srsrc(base, bytes);
    #pragma unroll
    for (int g = 0; g < M::regs; g++)
        #pragma unroll
        for (int s = 0; s < M::steps; s++)
            dst.reg[g][s] = 0x7F7F7F7Fu;              // unit scales past the last base tile
    #pragma unroll
    for (int i = 0; i < M::tiles; i++)
        #pragma unroll
        for (int s = 0; s < M::steps; s++) {
            const uint32_t off = uint32_t(row0 + i * 16 + lane % 16) * rs +
                                 uint32_t(kb0 + s * 4 + lane / 16);
            const uint32_t b = detail::hk_buf_ld8(r, off, 0, 0);
            const int sh = 8 * (i % 4);
            dst.reg[i / 4][s] = (dst.reg[i / 4][s] & ~(0xFFu << sh)) | (b << sh);
        }
    }
}

/**
 * AITER's shuffled MoE weight scales (gfx950, shuffle_scale(is_guinterleave=True)), uint8
 * [E * GR, K/32]: per expert, dwords at ((n1 * K/256 + k1) * 64 + lane) * 4 whose bytes are
 * (kpack, npack) = (K step parity, 16-row half): for LAYOUT 1 (gate|up weights) npack is gate or
 * up of 16-row block n1, for LAYOUT 2 the half of 32-row block n1. Tile base tile i (rows row0 +
 * 16 i, or with LAYOUT 1 gate/up of block row0/16 + i/2) and step s go to byte i % 4 of reg[i/4][s].
 * One dword load per base tile and step (the vmcnt count LdsPipeline assumes for byte scales).
 */
template<int LAYOUT, int GR, int R, int KB, int F, ducks::gl::all GL>
__device__ inline static void load_mx_scales_aiter(hk_mxs<R, KB, F> &dst, const GL &src, int row0, int kb0) {
    static_assert(sizeof(typename GL::dtype) == 1, "AITER scales: one byte each");
    static_assert(LAYOUT == 1 || LAYOUT == 2, "AITER layouts 1 (gate|up) and 2");
    using M = hk_mxs<R, KB, F>;
    const int lane = laneid();
    const uint32_t kb = uint32_t(src.template stride<2>());          // K/32 bytes a row
    const uint8_t *base = reinterpret_cast<const uint8_t *>(src.raw_ptr);
    const detail::hk_i32x4 r = detail::hk_srsrc(base, uint64_t(src.template shape<2>()) * kb);
    const int e = row0 / GR, lr = row0 % GR;
    const uint32_t k1n = kb / 8u;
    const uint32_t ebase = uint32_t(e) * uint32_t(GR) * kb;
    if constexpr (F == 0) {
        #pragma unroll
        for (int g = 0; g < M::regs; g++)
            #pragma unroll
            for (int s = 0; s < M::steps; s++)
                dst.reg[g][s] = 0x7F7F7F7Fu;
    }
    // One dword a lane holds a 32-row block's two 16-row halves for two consecutive K steps
    // (an even step and the odd one after it). Each is loaded once: per 32-row block (row0 32-row
    // aligned, base tiles in pairs) and K-step pair -- (R/32) * ceil(steps/2) loads, the count
    // hk-lds-pipeline assumes. A tile of two or more steps must start on an even step (kb0 % 8).
    static_assert(R % 32 == 0, "AITER scales: whole 32-row blocks");
    constexpr int G = (M::steps + 1) / 2;
    #pragma unroll
    for (int blk = 0; blk < M::tiles / 2; blk++) {
        int n1;
        if constexpr (LAYOUT == 1) n1 = lr / 16 + blk;
        else n1 = lr / 32 + blk;
        if constexpr (F == 2 && M::steps == 1) {
            // raw, one step: the step's (gate, up) bytes by a 16-bit load (sel = i % 2)
            const int st = kb0 / 4, kpack = st % 2;
            const uint32_t off = ebase + ((uint32_t(n1) * k1n + uint32_t(st / 2)) * 64u +
                                          uint32_t(lane)) * 4u + uint32_t(kpack) * 2u;
            dst.reg[blk][0] = detail::hk_buf_ld16(r, off, 0, 0);
            continue;
        }
        uint32_t d[G];
        #pragma unroll
        for (int g = 0; g < G; g++) {
            const uint32_t k1 = uint32_t(kb0 / 8 + g);
            const uint32_t off = ebase + ((uint32_t(n1) * k1n + k1) * 64u + uint32_t(lane)) * 4u;
            d[g] = detail::hk_buf_ld32(r, off, 0, 0);
        }
        if constexpr (F == 2) {
            // raw, several steps (the tile starts on an even step): the dwords as loaded
            #pragma unroll
            for (int g = 0; g < G; g++)
                dst.reg[blk][g] = d[g];
            continue;
        }
        if constexpr (F == 0) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const int i = 2 * blk + half;
                #pragma unroll
                for (int s = 0; s < M::steps; s++) {
                    const int kpack = (kb0 / 4 + s) % 2;
                    const uint32_t b = (d[s / 2] >> (8 * (kpack * 2 + half))) & 0xFFu;
                    const int sh = 8 * (i % 4);
                    dst.reg[i / 4][s] = (dst.reg[i / 4][s] & ~(0xFFu << sh)) | (b << sh);
                }
            }
        }
    }
}

/**
 * MX-quantize an LDS tile's rows (bf16 or fp32) into FP8 e4m3 + E8M0, AITER's per_1x32 rule:
 * e = ceil(log2(amax / 448)), from amax's bits (floor(log2 amax) - 8, plus one when the mantissa
 * is past 1.75), so the scale and every value match hk.mx.quant bit for bit. Every
 * thread of the workgroup takes (row, 32-column group) pairs: 32 plain LDS reads in 16-byte runs,
 * one 32-byte store of the FP8 row piece, one byte of scale at its place in AITER's 32-row layout
 * (hk.mx.aiter_scales: dword (n1 * K1 + k1) * 64 + kb4 * 16 + r16, byte kpack * 2 + h) -- or, when
 * `s` is an int32 tensor, in the packed layout load_mx_scales reads (hk.mx.pack_scales: row R/64,
 * dword st * 64 + (b % 4) * 16 + R % 16, byte (R % 64) / 16, for block b = st * 4 + b % 4), where
 * rows past q's last get the unit scale (127), as pack_scales pads them. The caller has written
 * src and passed a barrier.
 */
template<ducks::st::all ST, ducks::gl::all GLQ, ducks::gl::all GLS>
__device__ inline static void mx_quant_store(const GLQ &q, const GLS &s, const ST &src, int row0, int col0) {
    using T = typename ST::dtype;
    static_assert(std::is_same_v<T, bf16> || std::is_same_v<T, float>, "a bf16 or fp32 LDS tile");
    static_assert(ST::cols % 32 == 0, "32-column groups");
    // elements a contiguous LDS run: the swizzle keeps bytes_per_thread together
    constexpr int G = ST::cols / 32, RB = ST::underlying_subtile_bytes_per_thread, RUN = RB / sizeof(T);
    static_assert(RUN >= 1 && 32 % RUN == 0 && ST::underlying_subtile_cols % RUN == 0,
                  "whole runs inside a swizzle row");
    const char *base = reinterpret_cast<const char *>(&src.data[0]);
    const int rows = int(q.template shape<2>()), qcols = int(q.template shape<3>());
    const uint64_t qstride = uint64_t(q.template stride<2>());
    const int k1n = ((qcols / 32 + 7) / 8);                        // 256-wide K pairs a scale row
    uint8_t *qp = reinterpret_cast<uint8_t *>(q.raw_ptr);
    uint8_t *sp = reinterpret_cast<uint8_t *>(s.raw_ptr);
    const int nthreads = int(blockDim.x), tid = int(threadIdx.x);
    constexpr bool PACKED = sizeof(typename GLS::dtype) == 4;
    const uint64_t sdw = PACKED ? uint64_t(s.template stride<2>()) : 0;   // dwords a 64-row group
    for (int p = tid; p < ST::rows * G; p += nthreads) {
        const int r = p / G, g = p % G, R = row0 + r;
        if (R >= rows) {
            if constexpr (PACKED) {
                const int b = col0 / 32 + g;
                sp[((uint64_t(R / 64) * sdw + (b / 4) * 64 + (b % 4) * 16 + R % 16) * 4) + (R % 64) / 16] = 127;
            }
            continue;
        }
        float v[32];
        #pragma unroll
        for (int u = 0; u < 32 / RUN; u++) {
            const int c = g * 32 + u * RUN;
            const int sub = (r / ST::underlying_subtile_rows) * ST::underlying_subtiles_per_row +
                            c / ST::underlying_subtile_cols;
            const uint32_t off = uint32_t(sub) * ST::underlying_subtile_bytes +
                                 ST::swizzle({r % ST::underlying_subtile_rows, c % ST::underlying_subtile_cols});
            T run[RUN];
            __builtin_memcpy(run, base + off, RB);
            #pragma unroll
            for (int e = 0; e < RUN; e++) v[u * RUN + e] = float(run[e]);
        }
        float amax = 0.f;
        #pragma unroll
        for (int e = 0; e < 32; e++) amax = fmaxf(amax, fabsf(v[e]));
        amax = fmaxf(amax, 1.1754943508222875e-38f);                 // clamp_min(2^-126)
        const uint32_t ab = __float_as_uint(amax);
        int byte = int(ab >> 23) - 8 + ((ab & 0x7FFFFFu) > 0x600000u ? 1 : 0);   // e + 127
        byte = byte < 0 ? 0 : (byte > 254 ? 254 : byte);
        const float inv = byte < 254 ? __uint_as_float(uint32_t(254 - byte) << 23)   // 2^-e
                                     : ldexpf(1.f, 127 - byte);
        uint32_t packed[8];
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            float a[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) a[e] = fminf(fmaxf(v[d * 4 + e] * inv, -448.f), 448.f);
            int w = __builtin_amdgcn_cvt_pk_fp8_f32(a[0], a[1], 0, false);
            w = __builtin_amdgcn_cvt_pk_fp8_f32(a[2], a[3], w, true);
            packed[d] = uint32_t(w);
        }
        uint8_t *qrow = qp + uint64_t(R) * qstride + uint64_t(col0 + g * 32);
        reinterpret_cast<__uint128_t *>(qrow)[0] = *reinterpret_cast<const __uint128_t *>(&packed[0]);
        reinterpret_cast<__uint128_t *>(qrow)[1] = *reinterpret_cast<const __uint128_t *>(&packed[4]);
        const int b = col0 / 32 + g;
        if constexpr (PACKED) {
            sp[((uint64_t(R / 64) * sdw + (b / 4) * 64 + (b % 4) * 16 + R % 16) * 4) + (R % 64) / 16] =
                uint8_t(byte);
        } else {
            const int n1 = R / 32, h = (R % 32) / 16, r16 = R % 16, k1 = b / 8, kpack = (b % 8) / 4, kb4 = b % 4;
            sp[((((uint64_t(n1) * k1n + k1) * 4 + kb4) * 16 + r16) * 2 + kpack) * 2 + h] = uint8_t(byte);
        }
    }
}

/**
 * AITER's shuffled MXFP4 MoE weights (gfx950, shuffle_weight(is_guinterleave=True)), uint8
 * [E * GR, K/2]: per expert, 1 KB chunks [16-row block][(gate|up)][K/128 step], each one 16x128
 * FP4 operand in lane order (16 bytes a lane at lane * 16). Fragment (i, s) of the fp8-typed
 * container gets the chunk in its first four dwords (the upper four are never written: an FP4
 * operand reads four registers). GU: base tiles alternate gate and up of block row0/16 + i/2.
 */
template<int GU, int GR, ducks::rt::row_layout RT, ducks::gl::all GL>
__device__ inline static void load_fp4_frags(RT &dst, const GL &src, int row0, int kc0) {
    static_assert(std::is_same_v<typename RT::T, fp8e4m3> && RT::base_tile_rows == 16 &&
                  RT::base_tile_cols == 128, "fp8 e4m3 16x128 fragments (the FP4 container)");
    static_assert(sizeof(typename GL::dtype) == 1, "packed FP4 bytes");
    const int lane = laneid();
    const uint32_t kbytes = uint32_t(src.template stride<2>());      // K/2 bytes a row
    const int e = row0 / GR, lr = row0 % GR;
    // A resource per expert, based in 64 bits: all experts' weights pass 4 GB (DeepSeek-V4.1's
    // 384 x 4608 x 2560 B gate/up at TP1 is 4.5 GB), past a 32-bit offset and num_records.
    // An expert past the tensor (a padded unit's id) gets an empty range: its loads read zero,
    // as the one whole-tensor resource's bounds check made them before.
    // 32-bit, branch-free (a select): the 64-bit compare of (e + 1) * GR with the row count became
    // two scalar branches at every call, i.e. every K step (K1 k1-prefill c38)
    const uint32_t ne = uint32_t(src.template shape<2>()) / uint32_t(GR);
    const bool in = uint32_t(e) < ne;
    const uint64_t eoff = uint64_t(in ? uint32_t(e) : 0u) * (uint64_t(GR) * kbytes);
    const uint8_t *base = reinterpret_cast<const uint8_t *>(src.raw_ptr) + eoff;
    const detail::hk_i32x4 r = detail::hk_srsrc(base, in ? uint64_t(GR) * kbytes : 0);
    const uint32_t k0n = kbytes / 64u;
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int s = 0; s < RT::width; s++) {
            uint32_t chunk;
            if constexpr (GU) chunk = (uint32_t(lr / 16 + i / 2) * 2u + uint32_t(i % 2)) * k0n + uint32_t(kc0 + s);
            else chunk = uint32_t(lr / 16 + i) * k0n + uint32_t(kc0 + s);
            const uint32_t off = chunk * 1024u + uint32_t(lane) * 16u;
            const __uint128_t v = detail::hk_buf_ld128(r, off, 0, 0);
            __builtin_memcpy(&dst.tiles[i][s].data[0], &v, 16);
        }
}

/**
 * load_fp4_frags from LDS: src an fp8-typed shared tile of raw AITER weight bytes, a row a 16-row
 * block (gate|up interleaved) and 1 KB a 128-wide K step. Fragment (i, s) <- row row0 + i, chunk
 * chunk0 + s, lane * 16: a 16-byte run, which the tile's swizzle keeps whole, read at the place
 * the swizzle (an involution) puts logical (row, col). Plain loads; the compiler waits for them.
 */
template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ inline static void load_fp4_frags_lds(RT &dst, const ST &src, int row0, int chunk0) {
    static_assert(std::is_same_v<typename RT::T, fp8e4m3> && RT::base_tile_rows == 16 &&
                  RT::base_tile_cols == 128, "fp8 e4m3 16x128 fragments (the FP4 container)");
    static_assert(sizeof(typename ST::dtype) == 1 && ST::underlying_subtile_bytes_per_thread == 16,
                  "a byte tile whose swizzle keeps 16-byte runs");
    const int lane = laneid();
    const char *base = reinterpret_cast<const char *>(&src.data[0]);
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int s = 0; s < RT::width; s++) {
            const int r = row0 + i, c = (chunk0 + s) * 1024 + lane * 16;
            const int sub = (r / ST::underlying_subtile_rows) * ST::underlying_subtiles_per_row +
                            c / ST::underlying_subtile_cols;
            const uint32_t off = uint32_t(sub) * ST::underlying_subtile_bytes +
                                 ST::swizzle({r % ST::underlying_subtile_rows, c % ST::underlying_subtile_cols});
            const __uint128_t v = *reinterpret_cast<const __uint128_t *>(base + off);
            __builtin_memcpy(&dst.tiles[i][s].data[0], &v, 16);
        }
}

namespace detail {
template<int F>
__device__ inline auto hk_mx_operand(const uint32_t (&r)[8]) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(int)))) int intx4_t;
    if constexpr (F == 4) {
        // FP4: four registers; the operand's upper half is left undefined, not zeroed
        const intx4_t lo = *(const intx4_t *)r;
        return intx8_t(__builtin_shufflevector(lo, lo, 0, 1, 2, 3, -1, -1, -1, -1));
    } else {
        return *(const intx8_t *)r;
    }
}
template<int OA, int OB, int AF = 0, int BF = 0>
__device__ inline void hk_mfma_mx(float (&d)[4], const uint32_t (&a)[8], const uint32_t (&b)[8],
                                  uint32_t sa, uint32_t sb) {
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
    *(floatx4_t *)d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        hk_mx_operand<AF>(a), hk_mx_operand<BF>(b), *(floatx4_t *)d, AF, BF, OA, sa, OB, sb);
}
template<int OA, int AF = 0, int BF = 0>
__device__ inline void hk_mfma_mx_b(int ob, float (&d)[4], const uint32_t (&a)[8],
                                    const uint32_t (&b)[8], uint32_t sa, uint32_t sb) {
    switch (ob) {
        case 0: hk_mfma_mx<OA, 0, AF, BF>(d, a, b, sa, sb); break;
        case 1: hk_mfma_mx<OA, 1, AF, BF>(d, a, b, sa, sb); break;
        case 2: hk_mfma_mx<OA, 2, AF, BF>(d, a, b, sa, sb); break;
        default: hk_mfma_mx<OA, 3, AF, BF>(d, a, b, sa, sb); break;
    }
}
}  // namespace detail

/**
 * d = c + (sa . A)(sb . B)^T: A and B fp8 e4m3 row tiles of 16x128 fragments,
 * C/D fp32 16x16 col accumulators, one scaled MFMA per base tile and K step.
 * The opsel immediates are each base tile's index mod 4; the loops unroll, so
 * the switches fold.
 */
template<int AF = 0, int BF = 0, int ATO = 0, int ASO = 0, int BTO = 0, int BSO = 0,
         ducks::rt::col_layout D, ducks::rt::all A, ducks::rt::all B, ducks::rt::col_layout C,
         int RA, int RB, int KBA, int KBB, int FA, int FB>
__device__ static inline void mma_ABt_mx(D &d, const A &a, const B &b, const C &c,
                                         const hk_mxs<RA, KBA, FA> &sa, const hk_mxs<RB, KBB, FB> &sb) {
    // ATO / ASO (BTO / BSO): A's base tile i and K step s use sa's tile i + ATO and step s + ASO
    constexpr bool exact_a = RA == A::rows && KBA * 32 == A::cols && !ATO && !ASO,
                   exact_b = RB == B::rows && KBB * 32 == B::cols && !BTO && !BSO;
    static_assert(std::is_same_v<typename A::T, fp8e4m3> && std::is_same_v<typename B::T, fp8e4m3>,
                  "fp8 e4m3 operands");
    static_assert(std::is_same_v<typename D::T, float> && std::is_same_v<typename C::T, float>,
                  "fp32 accumulator");
    static_assert(A::base_tile_rows == 16 && A::base_tile_cols == 128 &&
                  B::base_tile_rows == 16 && B::base_tile_cols == 128, "16x128 operand fragments");
    static_assert(D::rows == A::rows && D::cols == B::rows && A::cols == B::cols &&
                  (exact_a ? A::rows == RA && A::cols == KBA * 32
                           : RA >= A::rows + 16 * ATO && KBA * 32 >= A::cols + 128 * ASO) &&
                  (exact_b ? B::rows == RB && B::cols == KBB * 32
                           : RB >= B::rows + 16 * BTO && KBB * 32 >= B::cols + 128 * BSO),
                  "mma_ABt_mx shapes");
    #pragma unroll
    for (int n = 0; n < D::height; n++) {
        #pragma unroll
        for (int m = 0; m < D::width; m++) {
            float acc[4];
            __builtin_memcpy(acc, &c.tiles[n][m].data[0], sizeof(acc));
            #pragma unroll
            for (int k = 0; k < A::width; k++) {
                const uint32_t(&ra)[8] = *reinterpret_cast<const uint32_t(*)[8]>(&a.tiles[n][k].data[0]);
                const uint32_t(&rb)[8] = *reinterpret_cast<const uint32_t(*)[8]>(&b.tiles[m][k].data[0]);
                const uint32_t s_a = sa.word(n + ATO, k + ASO), s_b = sb.word(m + BTO, k + BSO);
                const int ob = hk_mxs<RB, KBB, FB>::sel(m + BTO, k + BSO);
                switch (hk_mxs<RA, KBA, FA>::sel(n + ATO, k + ASO)) {
                    case 0: detail::hk_mfma_mx_b<0, AF, BF>(ob, acc, ra, rb, s_a, s_b); break;
                    case 1: detail::hk_mfma_mx_b<1, AF, BF>(ob, acc, ra, rb, s_a, s_b); break;
                    case 2: detail::hk_mfma_mx_b<2, AF, BF>(ob, acc, ra, rb, s_a, s_b); break;
                    default: detail::hk_mfma_mx_b<3, AF, BF>(ob, acc, ra, rb, s_a, s_b); break;
                }
            }
            __builtin_memcpy(&d.tiles[n][m].data[0], acc, sizeof(acc));
        }
    }
}
#endif
#endif

#if defined(KITTENS_CDNA4)

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

}  // namespace kittens
