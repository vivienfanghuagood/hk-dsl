/**
 * @file
 * @brief include/hk/cdna.cuh, part primitives.
 *
 * The general primitives hk's ops lower to: register tiles to and from memory, scoped
 * accesses and hand-offs, the LDS DMA, as_operand, scalars and atomics, rows by index,
 * Philox, and the wave64 strips (maps, element access, lane exchanges, reductions,
 * broadcasts, fills, memory). Registry kind `primitive`.
 *
 * Included by hk/cdna.cuh in the order core, lib, formats, special, primitives, escape;
 * the kind of every function is in python/hk/dsl/surface.py (CPP).
 */

#pragma once

namespace kittens {

/**
 * @brief load() of a register tile from a strided_gl. Rows of the tile past
 * the tensor's last row read as zero (as include/rdna3's load does), and the
 * rows are `row_stride` apart.
 *
 * More specialized than include/cdna{3,4}'s load(RT&, const GL&, ...), so it
 * is the one a strided_gl picks. Same accesses as the library's on its fast
 * path: one buffer_load of a lane's run (dwordx4 for an rt_16x32 bf16 run).
 */
template<ducks::rt::all RT, typename U, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const strided_gl<U> &src, const COORD &idx) {
    detail::hk_load_tile(dst, src, idx, RT::rows);
}
/** @brief store() of a register tile to a strided_gl; rows past the tensor's last are dropped. */
template<ducks::rt::all RT, typename U, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const strided_gl<U> &dst, const RT &src, const COORD &idx) {
    detail::hk_store_tile(dst, src, idx, RT::rows);
}

/**
 * @brief load(), predicated on the tensor's rows and on `limit` rows from
 * the tile's first: rows past either read as zero and their memory is never
 * touched. No branch and no masked lane: the descriptor ends there.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_bounded(RT &dst, const GL &src, const COORD &idx,
                                           int limit = RT::rows) {
    detail::hk_load_tile(dst, src, idx, limit);
}
/** @brief store(), predicated the same way: rows past either are not written. */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_bounded(const GL &dst, const RT &src, const COORD &idx,
                                            int limit = RT::rows) {
    detail::hk_store_tile(dst, src, idx, limit);
}

/* ------------------------------------------------------------------------
 * Scoped accesses and signals (hk.comm): hand-offs between XCDs and GPUs.
 *
 * AUX is the cache-policy field of every access: 16 (SC1, device scope) or
 * 17 (SC0|SC1, system scope). What is correct on gfx950 was measured by the
 * litmus suite (tools/cdna-probes/litmus_comm.hip):
 *   writer: data stored with AUX, s_waitcnt vmcnt(0), flag stored with AUX;
 *   reader: flag polled with AUX, data loaded with AUX.
 * Plain data stores stay in the writer's L2 and plain loads re-read stale
 * L1/L2 lines (also on uncached memory), so every access to memory another
 * agent touches carries AUX -- which is why hk.comm puts the scope on the
 * tensor and not on the access.
 * ---------------------------------------------------------------------- */

/** @brief store() of a register tile with the access scope AUX (rows past the tensor's last dropped). */
template<int AUX, ducks::rt::all RT, typename GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_scoped(const GL &dst, const RT &src, const COORD &idx,
                                           int limit = RT::rows) {
    detail::hk_store_tile<RT, GL, COORD, AUX>(dst, src, idx, limit);
}
/** @brief load() of a register tile with the access scope AUX (rows past the tensor's last read zero). */
template<int AUX, ducks::rt::all RT, typename GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_scoped(RT &dst, const GL &src, const COORD &idx,
                                          int limit = RT::rows) {
    detail::hk_load_tile<RT, GL, COORD, AUX>(dst, src, idx, limit);
}

namespace detail {
template<int AUX> __device__ inline uint32_t hk_scoped_ld32(const int *p) {
    return hk_buf_ld32(hk_srsrc(p, 4), 0, 0, AUX);
}
template<int AUX> __device__ inline void hk_scoped_st32(int *p, uint32_t v) {
    hk_buf_st32(v, hk_srsrc(p, 4), 0, 0, AUX);
}
}  // namespace detail

/**
 * @brief Publish: every store this workgroup issued before it is complete
 * (each wave waits for its stores' acknowledgements: vmcnt(0)), then thread 0
 * stores `v` into the int32 element at `idx` with scope AUX. The data a
 * reader expects after the signal must have been stored with the same (or a
 * wider) scope -- stores to a scoped tensor are. Contains a barrier.
 */
template<int AUX, ducks::gl::all GL, typename COORD>
__device__ inline static void signal(const GL &flag, const COORD &idx, int v) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a signal is an int32 element");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");   // this wave's stores are acknowledged
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
    if (threadIdx.x == 0)
        detail::hk_scoped_st32<AUX>((int *)&flag[(idx.template unit_coord<-1, 3>())], uint32_t(v));
}

/**
 * @brief Arrive: as signal(), but thread 0 adds `v` to the int32 element with
 * an atomic of scope AUX; returns the element's value before the add in every
 * lane. Contains a barrier.
 */
template<int AUX, ducks::gl::all GL, typename COORD>
__device__ inline static int arrive(const GL &cnt, const COORD &idx, int v) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a counter is an int32 element");
    __shared__ int hk_arrive_old;
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");   // this wave's stores are acknowledged
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
    if (threadIdx.x == 0) {
        int *p = (int *)&cnt[(idx.template unit_coord<-1, 3>())];
        hk_arrive_old = AUX == 17 ? __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                                  : __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
    return __builtin_amdgcn_readfirstlane(hk_arrive_old);
}

/**
 * @brief Claim: thread 0 adds `v` to the int32 element with an atomic of scope AUX -- no drain, the
 * workgroup's own stores are not ordered before it -- and every lane gets the value before the add.
 * A work queue's take: `t = claim(next, 1)` hands each workgroup a distinct task. Contains a barrier.
 */
template<int AUX, ducks::gl::all GL, typename COORD>
__device__ inline static int claim(const GL &cnt, const COORD &idx, int v) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a counter is an int32 element");
    __shared__ int hk_claim_old;
    if (threadIdx.x == 0) {
        int *p = (int *)&cnt[(idx.template unit_coord<-1, 3>())];
        hk_claim_old = AUX == 17 ? __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                                 : __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
    const int old = __builtin_amdgcn_readfirstlane(hk_claim_old);
    __builtin_amdgcn_s_barrier();      // the slot is reused by the next claim
    return old;
}

/** @brief The XCD (accelerator complex die) this wave runs on: HW_REG_XCC_ID[3:0] (gfx940+). */
__device__ inline static int xcd_id() {
    return __builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20) & 0xF;
}

/**
 * @brief Wait until the int32 element at `idx` compares with `v` (CMP 0:
 * >= as int32, 1: ==), polling with scope AUX and s_sleep between polls;
 * returns the value seen. Every wave polls (the value is uniform). After
 * `timeout` ticks of the 100 MHz s_memrealtime clock (0: never) it stops:
 * with `err` set, lane 0 of the wave ORs 1 into *err (system scope) and the
 * value seen is returned; without, the kernel traps -- a hang becomes an error.
 */
template<int AUX, int CMP, ducks::gl::all GL, typename COORD>
__device__ inline static int wait(const GL &flag, const COORD &idx, int v, long long timeout,
                                  int *err = nullptr) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a signal is an int32 element");
    const int *p = (const int *)&flag[(idx.template unit_coord<-1, 3>())];
    const long long t0 = __builtin_amdgcn_s_memrealtime();
    int seen;
    while (true) {
        seen = __builtin_amdgcn_readfirstlane(int(detail::hk_scoped_ld32<AUX>(p)));
        if (CMP == 0 ? seen >= v : seen == v) break;
        if (timeout && __builtin_amdgcn_s_memrealtime() - t0 > timeout) {
            if (err) {
                if (laneid() == 0) __hip_atomic_fetch_or(err, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                break;
            }
            __builtin_trap();
        }
        __builtin_amdgcn_s_sleep(1);
        asm volatile("" ::: "memory");   // the poll is re-issued: never hoisted out of the loop
    }
    return seen;
}

namespace detail {
/** The 8-byte slot of a mailbox (int32 elements idx, idx + 1); a slot off 8-byte alignment could tear: trap. */
template<ducks::gl::all GL, typename COORD>
__device__ inline int *hk_mailbox_slot(const GL &box, const COORD &idx) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a mailbox is int32 elements");
    int *p = (int *)&box[(idx.template unit_coord<-1, 3>())];
    if (reinterpret_cast<uintptr_t>(p) & 7) __builtin_trap();
    return p;
}
}  // namespace detail

/**
 * @brief Post a tagged message: thread 0 stores (`v`, `tag`) into the 8-byte slot at the int32
 * element `idx` (elements idx and idx + 1, 8-byte aligned) as ONE store of scope AUX. The value
 * travels with its tag, so the writer needs no drain and the reader no separate flag and data
 * read (8 B stores never tore over xGMI in the litmus suite). It orders
 * nothing else: data the reader needs besides the value still goes through signal(). No barrier.
 */
template<int AUX, ducks::gl::all GL, typename COORD>
__device__ inline static void post(const GL &box, const COORD &idx, int v, int tag) {
    if (threadIdx.x == 0) {
        int *p = detail::hk_mailbox_slot(box, idx);
        detail::hk_buf_st64(uint64_t(uint32_t(v)) | (uint64_t(uint32_t(tag)) << 32), detail::hk_srsrc(p, 8), 0, 0, AUX);
    }
}

/**
 * @brief Take a tagged message: poll the 8-byte slot at `idx` with scope AUX (one 8-byte load a
 * poll, s_sleep between) until its tag is `tag`; returns its value (wave-uniform). Every wave
 * polls. Timeout and `err` as wait().
 */
template<int AUX, ducks::gl::all GL, typename COORD>
__device__ inline static int take(const GL &box, const COORD &idx, int tag, long long timeout,
                                  int *err = nullptr) {
    const int *p = detail::hk_mailbox_slot(box, idx);
    const long long t0 = __builtin_amdgcn_s_memrealtime();
    uint64_t m;
    while (true) {
        m = detail::hk_buf_ld64(detail::hk_srsrc(p, 8), 0, 0, AUX);
        if (__builtin_amdgcn_readfirstlane(int(uint32_t(m >> 32))) == tag) break;
        if (timeout && __builtin_amdgcn_s_memrealtime() - t0 > timeout) {
            if (err) {
                if (laneid() == 0) __hip_atomic_fetch_or(err, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                break;
            }
            __builtin_trap();
        }
        __builtin_amdgcn_s_sleep(1);
        asm volatile("" ::: "memory");   // the poll is re-issued: never hoisted out of the loop
    }
    return __builtin_amdgcn_readfirstlane(int(uint32_t(m)));
}

/** @brief atomic_scalar at scope AUX (16: device, 17: system): one per warp, the value before it. */
template<int AUX, ducks::gl::all GL, typename COORD>
__device__ inline static int atomic_scalar_scoped(const GL &dst, const COORD &idx, int v, int op) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "atomic_scalar works on int32");
    int old = 0;
    if (laneid() == 0) {
        int *p = (int *)&dst[(idx.template unit_coord<-1, 3>())];
        old = AUX == 17 ? detail::hk_atomic_op<__HIP_MEMORY_SCOPE_SYSTEM>(p, v, op)
                        : detail::hk_atomic_op<__HIP_MEMORY_SCOPE_AGENT>(p, v, op);
    }
    return __builtin_amdgcn_readfirstlane(old);
}

#if defined(KITTENS_CDNA4)
/**
 * @brief A group of NW warps copies one LDS tile from a strided_gl with
 * gfx950's DMA, buffer_load_dwordx4 ... lds, and returns with it in flight.
 *
 * Each instruction moves 16 bytes a lane into LDS at M0 + 16 * lane, so the
 * swizzle of the tile's st_shape is applied on the global side, as
 * include/cdna4's group load does: LDS byte p holds the element the swizzled
 * layout puts there. Unlike that load, the descriptor is the tile's own
 * (hk_tile_srsrc): based at the tile and ending at the tensor's last row, so
 * the rows of a last, partial tile read zero instead of past the tensor, and
 * nothing wraps however large it is. The per-lane offsets depend only on the
 * lane, the warp and the row stride, which a K loop does not change, so
 * the compiler computes them once.
 *
 * The copies count on vmcnt only. hk's group_wait_for retires them with the
 * count hk-lds-pipeline derives, and a barrier after it makes every warp's
 * share visible; nothing here waits. `limit` ends the descriptor that many
 * rows from the tile's first as well: a copy with limit 0 fills the tile with
 * zeros, and moves none of its addressing. `NT` copies with sc0 nt, past the
 * caches' keeping: for data read once, in runs of whole lines (a tile laid
 * out in half-line runs had each line fetched twice).
 */

template<int NW, bool NT = false, bool UNIFORM = false, ducks::st::all ST, typename U,
         ducks::coord::tile COORD>
__device__ inline static void group_load_async_rows(ST &dst, const strided_gl<U> &src, const COORD &idx,
                                                    int limit) {
    using T = typename ST::dtype;
    static_assert(sizeof(T) == sizeof(U) && (sizeof(T) == 2 || sizeof(T) == 1),
                  "16-bit or fp8 tiles of the global's own dtype");
    static_assert(ST::underlying_subtile_bytes_per_thread == 16, "the 16-byte DMA needs a 16-byte lane run");
    constexpr int PER_WARP = 16 * WARP_THREADS;
    constexpr int PER_CALL = PER_WARP * NW;
    constexpr int BYTES = ST::rows * ST::cols * int(sizeof(T));
    // Whole copies a warp: every warp issues FULL, and the first PART warps
    // one more -- a 96-row half of a 192-wide block is 1.5 copies a thread
    // for eight warps. hk-lds-pipeline counts FULL for every warp
    // (GroupLoadOp::dmaCalls), so a wait behind such a copy is one copy
    // stricter than it need be in the warps that issued the extra one, and
    // exact in the others.
    static_assert(BYTES % PER_WARP == 0, "a whole number of copies per warp");
    constexpr int FULL = BYTES / PER_CALL;
    constexpr int PART = (BYTES % PER_CALL) / PER_WARP;
    const auto unit = idx.template unit_coord<2, 3>();
    const int rs = int(src.template stride<2>());
    detail::hk_i32x4 r = detail::hk_copy_srsrc(src, unit.b, unit.d, unit.r, unit.c, limit);
    // UNIFORM (the synchronous g.load's form): the descriptor's words read from lane 0. Its inputs
    // are wave-uniform, but where LLVM cannot prove it (mxfp8.gemm) a descriptor in VGPRs is a
    // waterfall loop around every copy (8 loops, +102 instructions there). Not by default: the
    // readfirstlanes do not fold away, and the asynchronous kernels' code was measured as it is.
    if constexpr (UNIFORM)
        r = detail::hk_i32x4{__builtin_amdgcn_readfirstlane(r.x), __builtin_amdgcn_readfirstlane(r.y),
                             __builtin_amdgcn_readfirstlane(r.z), __builtin_amdgcn_readfirstlane(r.w)};
    const int lane = laneid();
    // The warp from threadIdx.x rather than a readfirstlane, which LLVM
    // never moves out of a loop: the per-lane offsets below depend only on
    // the lane, the warp and the row stride, and are computed once.
    const int w = int(threadIdx.x / WARP_THREADS) % NW;
    const uint32_t lds = __builtin_amdgcn_readfirstlane(
        uint32_t(reinterpret_cast<uintptr_t>(&dst.data[0])) + w * PER_WARP);
    auto copy = [&](int i) {
        const int byte = lane * 16 + w * PER_WARP + i * PER_CALL;
        const int sub = byte / ST::underlying_subtile_bytes;
        const int in = byte % ST::underlying_subtile_bytes;
        const uint32_t sw = ST::swizzle({in / ST::underlying_subtile_row_bytes,
                                         (in % ST::underlying_subtile_row_bytes) / int(sizeof(T))});
        const int row = int(sw) / ST::underlying_subtile_row_bytes +
                        (sub / ST::underlying_subtiles_per_row) * ST::underlying_subtile_rows;
        const int col = (int(sw) % ST::underlying_subtile_row_bytes) / int(sizeof(T)) +
                        (sub % ST::underlying_subtiles_per_row) * ST::underlying_subtile_cols;
        detail::hk_buf_ld_lds(r, (detail::hk_lds_ptr)(uintptr_t)(lds + i * PER_CALL), 16,
                              (row * rs + col) * int(sizeof(T)), 0, 0, NT ? 3 : 0);
    };
    #pragma unroll
    for (int i = 0; i < FULL; i++)
        copy(i);
    if constexpr (PART > 0) {
        // Wave-uniform: a scalar branch around the last copy.
        if (__builtin_amdgcn_readfirstlane(w) < PART)
            copy(FULL);
    }
}

/**
 * group_load_async with the tile's rows by index: row r of the tile is the source's row
 * ids[at + r] & 0xFFFFFF (AITER's MoE sort: token | choice << 24, padding = the token count, a
 * row past the tensor that reads zero); idx gives the column. Same copies, offsets and lanes
 * as group_load_async -- only a lane's row changes, so hk-lds-pipeline counts it the same.
 */
namespace detail {
template<int N> __device__ inline int *hk_gather_rows_lds() {
    __shared__ int rows[N];
    return rows;
}
template<int N> __device__ inline int *hk_scatter_ids_lds() {
    __shared__ int ids[N];
    return ids;
}
}  // namespace detail

/**
 * Stage the source rows of a gathered copy in LDS: rows[r] = ids[at + r] & 0xFFFFFF for r < N,
 * read once by the whole workgroup, then a barrier. A gathered copy reading its rows from
 * there (group_load_async_gather<NW, N>) waits on lgkmcnt for them, not on vmcnt -- a global
 * load of a lane's row index would be a vmcnt wait, which retires every load in flight before
 * it (in order on gfx950): the whole prefetch.
 */
template<int N, int RAW = 0, ducks::gl::all IL>
__device__ inline static void gather_prep(const IL &ids, int at) {
    // RAW: the ids themselves, for a scatter's rows (store_scatter<.., STAGED = N>), in their
    // own array: ids[at + r] at [r]; `at` a multiple of N (the scatter indexes it pos % N)
    int *rows = RAW ? detail::hk_scatter_ids_lds<N>() : detail::hk_gather_rows_lds<N>();
    const int *idp = reinterpret_cast<const int *>(ids.raw_ptr);
    // positions past the ids tensor are not read: their rows are 0xFFFFFF (a row past any tensor)
    const int n_ids = int(ids.batch() * ids.depth() * ids.rows() * ids.cols());
    for (int r = int(threadIdx.x); r < N; r += int(blockDim.x))
        rows[r] = at + r < n_ids ? (RAW ? idp[at + r] : idp[at + r] & 0xFFFFFF) : (RAW ? -1 : 0xFFFFFF);
    __syncthreads();
}

template<int NW, int N = 0, ducks::st::all ST, typename U, ducks::coord::tile COORD, ducks::gl::all IL>
__device__ inline static void group_load_async_gather(ST &dst, const strided_gl<U> &src,
                                                      const COORD &idx, const IL &ids, int at,
                                                      int limit = 0x7FFFFFFF) {
    // limit: tile rows from `limit` on read zero and their ids are not read (a list that ends
    // inside the tile -- the last query of a CSR -- is never read past its end)
    using T = typename ST::dtype;
    static_assert(sizeof(T) == sizeof(U) && (sizeof(T) == 2 || sizeof(T) == 1),
                  "16-bit or fp8 tiles of the global's own dtype");
    static_assert(ST::underlying_subtile_bytes_per_thread == 16, "the 16-byte DMA needs a 16-byte lane run");
    static_assert(sizeof(typename IL::dtype) == 4, "int32 row indices");
    constexpr int PER_WARP = 16 * WARP_THREADS;
    constexpr int PER_CALL = PER_WARP * NW;
    constexpr int BYTES = ST::rows * ST::cols * int(sizeof(T));
    static_assert(BYTES % PER_WARP == 0, "a whole number of copies per warp");
    constexpr int FULL = BYTES / PER_CALL;
    constexpr int PART = (BYTES % PER_CALL) / PER_WARP;
    const auto unit = idx.template unit_coord<2, 3>();
    const int rs = int(src.template stride<2>());
    // the whole slab from row 0 and the tile's column: an index past the last row reads zero
    const detail::hk_i32x4 r = detail::hk_copy_srsrc(src, unit.b, unit.d, 0, unit.c, 0x7FFFFFFF);
    const int *ids0 = reinterpret_cast<const int *>(ids.raw_ptr);
    const int n_ids = int(ids.batch() * ids.depth() * ids.rows() * ids.cols());   // a copy reads no id past it
    const int lane = laneid();
    const int w = int(threadIdx.x / WARP_THREADS) % NW;
    const uint32_t lds = __builtin_amdgcn_readfirstlane(
        uint32_t(reinterpret_cast<uintptr_t>(&dst.data[0])) + w * PER_WARP);
    // where copy i's lane lands in the tile
    auto place = [&](int i, int &row, int &col) {
        const int byte = lane * 16 + w * PER_WARP + i * PER_CALL;
        const int sub = byte / ST::underlying_subtile_bytes;
        const int in = byte % ST::underlying_subtile_bytes;
        const uint32_t sw = ST::swizzle({in / ST::underlying_subtile_row_bytes,
                                         (in % ST::underlying_subtile_row_bytes) / int(sizeof(T))});
        row = int(sw) / ST::underlying_subtile_row_bytes +
              (sub / ST::underlying_subtiles_per_row) * ST::underlying_subtile_rows;
        col = (int(sw) % ST::underlying_subtile_row_bytes) / int(sizeof(T)) +
              (sub % ST::underlying_subtiles_per_row) * ST::underlying_subtile_cols;
    };
    // A lane past the limit reads the last live id (its position clamped: ints only) and adds 2^31
    // to its offset, past the range, so it reads zero. The biases go through empty asms, all before
    // the first index load: LLVM would otherwise fold a bias into a select of the loaded id and lower
    // that to a branch on the lane being live, and an asm between the loads keeps them from issuing
    // together -- either is a vmcnt wait per copy (sparse MLA 711 -> 800-1000 us).
    constexpr int CALLS = FULL + (PART > 0 ? 1 : 0);
    uint32_t past[CALLS > 0 ? CALLS : 1];
    #pragma unroll
    for (int i = 0; i < CALLS; i++) {
        int row, col;
        place(i, row, col);
        past[i] = row < limit && at + row < n_ids ? 0u : 0x80000000u;
        asm volatile("" : "+v"(past[i]));
    }
    auto copy = [&](int i) {
        int row, col;
        place(i, row, col);
        // the position read: this lane's, or for a dead lane the list's last live one -- in the
        // ids tensor either way (a dead lane's row is discarded by its bias)
        int k = row < limit ? at + row : at + limit - 1;
        k = k < n_ids ? k : n_ids - 1;
        k = k > 0 ? k : 0;
        int src_row;
        if constexpr (N > 0) src_row = detail::hk_gather_rows_lds<N>()[k - at > 0 ? k - at : 0];   // gather_prep's
        else src_row = ids0[k] & 0xFFFFFF;
        detail::hk_buf_ld_lds(r, (detail::hk_lds_ptr)(uintptr_t)(lds + i * PER_CALL), 16,
                              uint32_t((src_row * rs + col) * int(sizeof(T))) + past[i], 0, 0, 0);
    };
    #pragma unroll
    for (int i = 0; i < FULL; i++)
        copy(i);
    if constexpr (PART > 0) {
        if (__builtin_amdgcn_readfirstlane(w) < PART)
            copy(FULL);
    }
}

template<int NW, bool NT = false, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_load_async(ST &dst, const strided_gl<U> &src, const COORD &idx) {
    group_load_async_rows<NW, NT>(dst, src, idx, 0x7FFFFFFF);
}
template<int NW, bool NT = false, ducks::st::all ST, typename U, ducks::coord::tile COORD>
__device__ inline static void group_load_async(ST &dst, const strided_gl<U> &src, const COORD &idx,
                                               int limit) {
    group_load_async_rows<NW, NT>(dst, src, idx, limit);
}
#endif

#if defined(KITTENS_CDNA4)
/*
 * An MFMA accumulator's registers, read as an operand fragment (hk's
 * as_operand). No instruction: each lane already holds the operand's
 * elements, with the reduction index running down the rows in the
 * accumulator's order.
 *
 * A 32x32 accumulator's lane l holds rows 4(l/32) + 8k + s (k < 4, s < 4)
 * of its column l%32. A 16x32 stride-4 col operand's lane holds rows
 * 4(l/32) + 8k + s (k < 2) of the same column: the top and bottom 16 rows of
 * the accumulator are two such fragments, in register order. A 16x16
 * accumulator's lane holds rows 4(l/16) + s of column l%16, and a 32x16
 * stride-4 col operand rows 4(l/16) + 16k + s: two accumulators stacked.
 */
namespace detail {
template<typename D, typename S> __device__ inline D hk_as_elem(const S &x) { return x; }
template<> __device__ inline bf16_2 hk_as_elem<bf16_2, float2>(const float2 &x) {
    return hk_cvt_pk_bf16(x.x, x.y);
}
}  // namespace detail

/* Packed pair by packed pair; fp32 -> bf16 converts each (to nearest even,
 * the compiler's own v_cvt_pk_bf16_f32: see copy() above for why not the
 * library's inline-asm one). */
template<typename T, typename U, int R, int C>
__device__ inline static void as_operand(rt<T, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_16x32_4> &dst,
                                         const rt<U, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_32x32> &src) {
    using D = rt<T, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_16x32_4>;
    #pragma unroll
    for (int i = 0; i < D::height; i++)
        #pragma unroll
        for (int j = 0; j < D::width; j++)
            #pragma unroll
            for (int k = 0; k < D::packed_per_base_tile; k++)
                dst.tiles[i][j].data[k] = detail::hk_as_elem<typename D::dtype>(
                    src.tiles[i / 2][j].data[(i % 2) * D::packed_per_base_tile + k]);
}
template<typename T, typename U, int R, int C>
__device__ inline static void as_operand(rt<T, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_32x16_4> &dst,
                                         const rt<U, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_16x16> &src) {
    using D = rt<T, R, C, ducks::rt_layout::col, ducks::rt_shape::rt_32x16_4>;
    constexpr int per = D::base_tile_packed_per_stride;
    #pragma unroll
    for (int i = 0; i < D::height; i++)
        #pragma unroll
        for (int j = 0; j < D::width; j++)
            #pragma unroll
            for (int k = 0; k < 2; k++)
                #pragma unroll
                for (int p = 0; p < per; p++)
                    dst.tiles[i][j].data[k * per + p] = detail::hk_as_elem<typename D::dtype>(
                        src.tiles[2 * i + k][j].data[p]);
}
#endif

#if defined(KITTENS_CDNA4)
/**
 * @brief load() / store() of a register vector with the access scope AUX (hk.comm; 32-bit elements),
 * on HipKittens' CDNA4 vector layouts: ortho (a row tile's col_vec: lane l holds entry l % reductions of
 * each block of `reductions`) -- what a per-row scale is. The addresses are HipKittens' own load/store.
 */
template<int AUX, ducks::rv::all RV, typename GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load_scoped(RV &dst, const GL &src, const COORD &idx) {
    using U = typename GL::dtype;
    using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
    static_assert(std::is_same_v<typename RV::layout, ortho_l>, "a scoped vector is a col_vec of a row tile (ortho)");
    const U *p = (const U *)&src[(idx.template unit_coord<-1, 3>())];
    const int lane = laneid();
    #pragma unroll
    for (int w = 0; w < RV::outer_dim; w++)
        dst[w][0] = base_types::convertor<T, U>::convert(detail::hk_ld_aux<AUX>(p + w * RV::reductions + lane % RV::reductions));
}
template<int AUX, ducks::rv::all RV, typename GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store_scoped(const GL &dst, const RV &src, const COORD &idx) {
    using U = typename GL::dtype;
    using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
    static_assert(std::is_same_v<typename RV::layout, ortho_l>, "a scoped vector is a col_vec of a row tile (ortho)");
    U *p = (U *)&dst[(idx.template unit_coord<-1, 3>())];
    const int lane = laneid();
    #pragma unroll
    for (int w = 0; w < RV::outer_dim; w++)
        if (lane < RV::reductions)   // the entries are replicated over the lanes; one writer each
            detail::hk_st_aux<AUX>(p + w * RV::reductions + lane, detail::hk_cvt<U, T>(src[w][0]));
}
#endif

/* ------------------------------------------------------------------------
 * Scalars, counters and atomics: layout-light, ported from include/rdna3.
 * ---------------------------------------------------------------------- */

/**
 * @brief Store one element of a register vector whose entries are all equal
 * (a folded reduction) to one element of a global. Lane 0 of each warp
 * writes; several warps write the same value to the same address.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store_scalar(const GL &dst, const RV &src, const COORD &idx) {
    using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
    using U = typename GL::dtype;
    if (laneid() != 0) return;
    U *p = (U *)&dst[(idx.template unit_coord<-1, 3>())];
    *p = detail::hk_cvt<U, T>(reinterpret_cast<const T *>(&src.data[0][0])[0]);
}

/** @brief One element of a global, as a wave-uniform integer. */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int load_scalar(const GL &src, const COORD &idx) {
    using U = typename GL::dtype;
    const U *p = (const U *)&src[(idx.template unit_coord<-1, 3>())];
    return __builtin_amdgcn_readfirstlane((int)(*p));
}

/**
 * @brief load_scalar of an element the kernel never writes: a load from the
 * constant address space at a uniform address, which is an s_load_dword
 * through the scalar cache. A global load's wait counts vmcnt, in one queue
 * with the DMA copies: a page index read beside them waited out every copy
 * issued before it (the paged decode's ring: one page in flight however
 * deep the ring).
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int load_scalar_invariant(const GL &src, const COORD &idx) {
    using U = typename GL::dtype;
    static_assert(sizeof(U) == 4, "a 32-bit element");
    const U *p = (const U *)&src[(idx.template unit_coord<-1, 3>())];
    typedef const __attribute__((address_space(4))) int *hk_const_i32;
    return __builtin_amdgcn_readfirstlane(*(hk_const_i32)(uintptr_t)p);
}

/**
 * @brief Count this workgroup's arrival at `cnt[idx]`, device-wide; returns
 * how many arrived before it. The last arrival resets the counter to 0. The
 * fences make every global store issued before the call visible to the last
 * arrival. Contains barriers: every thread of the workgroup must reach it.
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int wg_arrive(const GL &cnt, const COORD &idx, int expected) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "wg_arrive counts in an int32 tensor");
    __shared__ int arrived_before;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int *p = (int *)&cnt[(idx.template unit_coord<-1, 3>())];
        const int old = atomicAdd(p, 1);
        if (old == expected - 1) atomicExch(p, 0);
        arrived_before = old;
    }
    __syncthreads();
    const int arrived = __builtin_amdgcn_readfirstlane(arrived_before);
    __threadfence();
    return arrived;
}

/**
 * @brief wg_arrive for workgroups that share an L2: every workgroup that
 * arrives at `cnt[idx]` runs on the same XCD of a gfx950. The global stores
 * issued before the call are retired (the L1 writes through, so they are in
 * the L2) and the last arrival invalidates its L1 before it reads them --
 * no L2 writeback or invalidate, which a device-wide __threadfence costs
 * every workgroup on the XCD (a stream-K GEMM whose every workgroup arrives
 * once ran 5-50x slower with it). The counter's atomics are device-wide, so
 * counters of different XCDs may share a cache line.
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int wg_arrive_xcd(const GL &cnt, const COORD &idx, int expected) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "wg_arrive counts in an int32 tensor");
    __shared__ int arrived_before;
    __builtin_amdgcn_s_waitcnt(0x0F70);             // vmcnt(0): this wave's stores are in the L2
    __syncthreads();
    if (threadIdx.x == 0) {
        int *p = (int *)&cnt[(idx.template unit_coord<-1, 3>())];
        const int old = __hip_atomic_fetch_add(p, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (old == expected - 1)
            __hip_atomic_store(p, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        arrived_before = old;
    }
    __syncthreads();
    const int arrived = __builtin_amdgcn_readfirstlane(arrived_before);
    if (arrived == expected - 1)
        asm volatile("buffer_inv sc0" ::: "memory");  // this CU's L1: read the shares from the L2
    return arrived;
}

/**
 * @brief Add an fp32 tile into a float global, element by element,
 * atomically. CDNA's register tiles hold each element once (no mirrored
 * halves), so every layout adds each element exactly once.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_add(const GL &dst, const RT &src, const COORD &idx) {
    using F = detail::hk_frag<RT>;
    static_assert(std::is_same_v<typename F::T, float> && std::is_same_v<typename GL::dtype, float>,
                  "atomic_add adds an fp32 tile into a float global");
    float *p = (float *)&dst[(idx.template unit_coord<2, 3>())];
    const int rs = dst.template stride<2>();
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int e = 0; e < F::E; e++) {
                const int2 c = detail::hk_frag_coord<RT>(e, lane);
                atomicAdd(&p[(i * F::R + c.x) * rs + j * F::C + c.y], detail::hk_get(src, i, j, e));
            }
}

/**
 * @brief One atomic on one int32 element, once per warp: `op` 0 adds `v`, 1
 * takes the max. Returns the element's value before it, in every lane.
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int atomic_scalar(const GL &dst, const COORD &idx, int v, int op) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "atomic_scalar works on int32");
    int old = 0;
    if (laneid() == 0) {
        int *p = (int *)&dst[(idx.template unit_coord<-1, 3>())];
        old = detail::hk_atomic_op<__HIP_MEMORY_SCOPE_AGENT>(p, v, op);
    }
    return __builtin_amdgcn_readfirstlane(old);
}

/* ------------------------------------------------------------------------
 * Rows by index: a tile's rows named by an int32 list.
 * ---------------------------------------------------------------------- */

namespace detail {
/// A descriptor over every row of `g` (batch, depth and rows flattened), and
/// the offset a row outside it gets: far enough out that no column offset
/// added to it wraps back in.
template<typename GL>
__device__ inline hk_i32x4 hk_whole_srsrc(const GL &g, uint32_t &oob, int &nrows) {
    nrows = g.batch() * g.depth() * g.rows();
    const uint64_t bytes = uint64_t(nrows) * g.template stride<2>() * sizeof(typename GL::dtype);
    const uint32_t cap = 0xF0000000u;
    oob = bytes < cap ? uint32_t(bytes) : cap;
    return hk_srsrc(g.raw_ptr, oob);
}
/// Base tile row `i`'s index for this lane: idx[at + R*i + lane%R] / div, or
/// -1 at or past `count`.
template<int R, typename IL>
__device__ inline int hk_gather_index(const IL &idx, int at, int count, int i, int div) {
    static_assert(std::is_same_v<typename IL::dtype, int>, "row indices are int32");
    const int pos = at + R * i + laneid() % R;
    if (pos >= count) return -1;
    const int v = idx.raw_ptr[pos];
    return div == 1 ? v : v / div;
}
/// A scatter's index: from the LDS copy gather_prep<STAGED, 1> made (an lgkmcnt wait), or global
/// (a vmcnt wait, which on gfx950 retires every vector-memory op issued before it, in order).
template<int R, int STAGED, typename IL>
__device__ inline int hk_scatter_index(const IL &idx, int at, int count, int i) {
    if constexpr (STAGED > 0) {
        const int pos = at + R * i + laneid() % R;
        return pos >= count ? -1 : hk_scatter_ids_lds<STAGED>()[pos % STAGED];
    } else {
        return hk_gather_index<R>(idx, at, count, i, 1);
    }
}
}  // namespace detail

/**
 * @brief load() with the tile's rows chosen by index: tile row r is row
 * `idx[at + r] / div` of `src` (rows flattened over batch and depth), from
 * column `col`. An index outside [0, rows) or a position at or past `count`
 * reads zeros, with no branch: its offset is past the descriptor. Row layout
 * only, where a lane's elements of a base tile are all in its row.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::gl::all IL>
__device__ inline static void load_gather(RT &dst, const GL &src, const IL &idx,
                                          int at, int count, int col, int div = 1) {
    using F = detail::hk_frag<RT>;
    using T = typename F::T;
    using U = typename GL::dtype;
    static_assert(F::row, "gather fills row-layout tiles");
    uint32_t oob; int nrows;
    const detail::hk_i32x4 r = detail::hk_whole_srsrc(src, oob, nrows);
    const uint32_t row_bytes = uint32_t(src.template stride<2>()) * sizeof(U);
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        const int g = detail::hk_gather_index<F::R>(idx, at, count, i, div);
        const bool in = g >= 0 && g < nrows;
        const uint32_t row_off = in ? uint32_t(g) * row_bytes + uint32_t(col) * sizeof(U) : oob;
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < F::K; k++) {
                const int cy = detail::hk_frag_coord<RT>(k * F::S, lane).y;
                U run[F::S];
                detail::hk_buf_load_bytes<F::S * sizeof(U)>(
                    r, in ? row_off + uint32_t(j * F::C + cy) * sizeof(U) : oob, run);
                #pragma unroll
                for (int s = 0; s < F::S; s++)
                    detail::hk_set(dst, i, j, k * F::S + s, base_types::convertor<T, U>::convert(run[s]));
            }
        }
    }
}

/**
 * @brief store() with the tile's rows chosen by index: tile row r goes to row
 * `idx[at + r]` of `dst`, from column `col`. An index outside [0, rows), or a
 * position at or past `count`, writes nothing. Two tile rows naming one row
 * race; which lands is unspecified.
 */
template<int TOPK = 0, int ADD = 0, int STAGED = 0, ducks::rt::all RT, ducks::gl::all GL,
         ducks::gl::all IL>
__device__ inline static void store_scatter(const GL &dst, const RT &src, const IL &idx,
                                            int at, int count, int col) {
    // STAGED > 0: the ids were staged in LDS by gather_prep<STAGED, 1>(idx, at0) (at0 a multiple
    // of STAGED, covering every position stored here): no global index load, so no vmcnt wait
    // that would drain the loads and stores in flight (K1 k1-prefill c36).
    // TOPK > 0: idx holds AITER's MoE sort ids (token | choice << 24) and dst is [TOPK * M, ..]
    // choice-major: the row is choice * M + token, nothing for a token >= M (padding).
    // ADD: the rows are added atomically (fp32, or bf16 pairs); with TOPK, dst is [M, ..] and the
    // row is the token's -- the top-k sum in whatever order the adds land.
    using F = detail::hk_frag<RT>;
    using T = typename F::T;
    using U = typename GL::dtype;
    static_assert(F::row, "scatter stores row-layout tiles");
    if constexpr (ADD) {
        static_assert(std::is_same_v<U, float> || (std::is_same_v<U, bf16> && F::S % 2 == 0),
                      "scatter-add: an fp32 global, or bf16 with even runs");
        // the adds sit in a per-row `if`: the tile's values are computed before it (hk_pinned)
        const RT tv = detail::hk_pinned(src);
        typedef __bf16 hk_bf16x2 __attribute__((ext_vector_type(2)));
        const int nrows = int(dst.template shape<2>());
        const uint64_t row_elems = uint64_t(dst.template stride<2>());
        U *base = reinterpret_cast<U *>(dst.raw_ptr);
        const int lane = laneid();
        #pragma unroll
        for (int i = 0; i < RT::height; i++) {
            int g = detail::hk_scatter_index<F::R, STAGED>(idx, at, count, i);
            if constexpr (TOPK > 0) g = (g < 0 || (g & 0xFFFFFF) >= nrows) ? -1 : (g & 0xFFFFFF);
            if (g < 0 || g >= nrows) continue;
            U *row = base + uint64_t(g) * row_elems + col;
            #pragma unroll
            for (int j = 0; j < RT::width; j++) {
                #pragma unroll
                for (int k = 0; k < F::K; k++) {
                    const int cy = detail::hk_frag_coord<RT>(k * F::S, lane).y;
                    U *p = row + j * F::C + cy;
                    if constexpr (std::is_same_v<U, float>) {
                        #pragma unroll
                        for (int s = 0; s < F::S; s++)
                            __hip_atomic_fetch_add(p + s, float(detail::hk_get(tv, i, j, k * F::S + s)),
                                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    } else {
                        #pragma unroll
                        for (int s = 0; s < F::S; s += 2) {
                            const hk_bf16x2 v = {__bf16(float(detail::hk_get(tv, i, j, k * F::S + s))),
                                                 __bf16(float(detail::hk_get(tv, i, j, k * F::S + s + 1)))};
                            __builtin_amdgcn_global_atomic_fadd_v2bf16(
                                (__attribute__((address_space(1))) hk_bf16x2 *)(p + s), v);
                        }
                    }
                }
            }
        }
        return;
    }
    uint32_t oob; int nrows;
    const detail::hk_i32x4 r = detail::hk_whole_srsrc(dst, oob, nrows);
    const uint32_t row_bytes = uint32_t(dst.template stride<2>()) * sizeof(U);
    const int lane = laneid();
#if defined(KITTENS_CDNA4)
    if constexpr (F::R == 16 && F::C == 16 && F::S == 4 && F::K == 1 && RT::width % 2 == 0 &&
                  std::is_same_v<T, float> && std::is_same_v<U, bf16>) {
        // 16x16 accumulator fragments (lane l: row l%16, 4 columns at 4(l/16)), stored as
        // bf16: v_permlane16_swap trades runs between lanes l and l+16 across a pair of
        // 16-column subtiles, so that each lane holds 8 consecutive columns -- lanes of
        // quarter q = l/16 start at 16(q&1) + 8(q>>1) of the pair's 32 -- and a row's 64 B
        // go out in 16-byte stores, half as many instructions as 8-byte ones (K1 k1-prefill
        // c35: the 8-byte scatter stores were a third of stage 2's time).
        const int q4 = lane / 16;
        const uint32_t start = uint32_t(16 * (q4 & 1) + 8 * (q4 >> 1));
        #pragma unroll
        for (int i = 0; i < RT::height; i++) {
            int g = detail::hk_scatter_index<F::R, STAGED>(idx, at, count, i);
            if constexpr (TOPK > 0) {
                const int m = nrows / TOPK, tok = g & 0xFFFFFF;
                g = (g < 0 || tok >= m) ? -1 : (g >> 24) * m + tok;
            }
            const bool in = g >= 0 && g < nrows;
            const uint32_t row_off = in ? uint32_t(g) * row_bytes + uint32_t(col) * sizeof(U) : oob;
            #pragma unroll
            for (int j = 0; j < RT::width; j += 2) {
                uint32_t a[2], b[2];
                #pragma unroll
                for (int d = 0; d < 2; d++) {
                    const bf16_2 pa = detail::hk_cvt_pk_bf16(src.tiles[i][j].data[d].x, src.tiles[i][j].data[d].y);
                    const bf16_2 pb = detail::hk_cvt_pk_bf16(src.tiles[i][j + 1].data[d].x,
                                                             src.tiles[i][j + 1].data[d].y);
                    __builtin_memcpy(&a[d], &pa, 4);
                    __builtin_memcpy(&b[d], &pb, 4);
                    const auto s = __builtin_amdgcn_permlane16_swap(a[d], b[d], false, true);
                    a[d] = s[0];
                    b[d] = s[1];
                }
                uint32_t chunk[4] = {a[0], a[1], b[0], b[1]};
                detail::hk_buf_store_bytes<16>(
                    r, in ? row_off + (uint32_t(j * F::C) + start) * sizeof(U) : oob, chunk);
            }
        }
        return;
    }
#endif
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        int g = detail::hk_scatter_index<F::R, STAGED>(idx, at, count, i);
        if constexpr (TOPK > 0) {
            const int m = nrows / TOPK, tok = g & 0xFFFFFF;
            g = (g < 0 || tok >= m) ? -1 : (g >> 24) * m + tok;
        }
        const bool in = g >= 0 && g < nrows;
        const uint32_t row_off = in ? uint32_t(g) * row_bytes + uint32_t(col) * sizeof(U) : oob;
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < F::K; k++) {
                const int cy = detail::hk_frag_coord<RT>(k * F::S, lane).y;
                U run[F::S];
                #pragma unroll
                for (int s = 0; s < F::S; s++)
                    run[s] = detail::hk_cvt<U, T>(detail::hk_get(src, i, j, k * F::S + s));
                detail::hk_buf_store_bytes<F::S * sizeof(U)>(
                    r, in ? row_off + uint32_t(j * F::C + cy) * sizeof(U) : oob, run);
            }
        }
    }
}

/* ------------------------------------------------------------------------
 * Random numbers: Triton's Philox, so a kernel and its Triton reference draw
 * the same numbers from the same seed and offsets (include/rdna3 random.cuh).
 * ---------------------------------------------------------------------- */

namespace detail {
__device__ inline uint32_t hk_philox_first(uint32_t c0, uint32_t k0, uint32_t k1, int rounds = 10) {
    uint32_t c1 = 0, c2 = 0, c3 = 0;
    #pragma unroll
    for (int r = 0; r < rounds; r++) {
        const uint32_t a0 = c0, a2 = c2;
        c0 = __umulhi(0xCD9E8D57u, a2) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, a0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * a2;
        c3 = 0xD2511F53u * a0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}
__device__ inline float hk_uniform_from(uint32_t x) {
    int32_t s = int32_t(x);
    s = s < 0 ? -s - 1 : s;
    float u = float(s) * 4.6566127342e-10f;
    // Rounded here, as Triton rounds it; contracted into a later subtraction
    // the product would skip this rounding.
    asm("" : "+v"(u));
    return u;
}
}  // namespace detail

/**
 * @brief Fill an fp32 tile with uniform [0, 1) draws: element (r, c) of the
 * tile draws counter `offset + r * row_stride + c`, as tl.rand would.
 */
template<ducks::rt::all RT>
__device__ inline void philox_uniform(RT &dst, uint32_t seed_lo, uint32_t seed_hi,
                                      int offset, int row_stride) {
    using F = detail::hk_frag<RT>;
    static_assert(std::is_same_v<typename F::T, float>, "draws go into an fp32 tile");
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < RT::height; i++)
        #pragma unroll
        for (int j = 0; j < RT::width; j++)
            #pragma unroll
            for (int e = 0; e < F::E; e++) {
                const int2 c = detail::hk_frag_coord<RT>(e, lane);
                const uint32_t ctr = uint32_t(offset + (i * F::R + c.x) * row_stride + j * F::C + c.y);
                detail::hk_set(dst, i, j, e,
                               detail::hk_uniform_from(detail::hk_philox_first(ctr, seed_lo, seed_hi)));
            }
}

/* ------------------------------------------------------------------------
 * Register strips (kittens::rs) on wave64: include/rdna3's rs, its warp ops
 * and its memory ops, with the two facts that differ changed.
 *
 *  - A chunk is 64 lanes x RUN: lane l owns, of every row, the runs
 *    [j*64*RUN + l*RUN, +RUN). A bf16 run of 8 is one buffer_load_dwordx4.
 *  - The row reduction's butterfly crosses 64 lanes: DPP quad_perm (xor 1,
 *    xor 2), row_half_mirror and row_mirror inside each 16-lane row, then
 *    v_permlane16_swap and v_permlane32_swap across rows (gfx950), or
 *    ds_bpermute (__shfl_xor) on gfx942, which has no swap. Each stage
 *    combines a value with a partner that combines the same pair, so every
 *    lane ends with the same bits, as on gfx11.
 *
 * Stores narrow through hk_cvt (bf16 rounds to nearest even; the CDNA
 * trees' convertor truncates), and the bounded forms go through hk_srsrc.
 * ---------------------------------------------------------------------- */

namespace ducks {
namespace rs {
struct identifier {};
struct cv_identifier {};
}  // namespace rs
}  // namespace ducks

/** @brief A strip's element type: the unpacked type, or bool for a mask (hk.map's b1 strips). */
template<typename T> struct rs_elem { using type = typename base_types::packing<T>::unpacked_type; };
template<> struct rs_elem<bool> { using type = bool; };

/** @brief One value per row of a strip, held by every lane of the warp. */
template<typename _T, int _length>
struct rs_cv {
    using identifier = ducks::rs::cv_identifier;
    static_assert(_length > 0, "a strip has at least one row");
    using T = typename rs_elem<_T>::type;
    using dtype = T;
    static constexpr int length    = _length;
    static constexpr int outer_dim = _length;
    static constexpr int inner_dim = 1;
    T data[outer_dim][inner_dim];
    __device__ inline       T* operator[](size_t idx)       { return &data[idx][0]; }
    __device__ inline const T* operator[](size_t idx) const { return &data[idx][0]; }
};

/** @brief ROWS rows of COLS columns, RUN contiguous elements of a row per
 * lane per 64*RUN-column chunk. */
template<typename _T, int _rows, int _cols, int _run = 8>
struct rs {
    using identifier = ducks::rs::identifier;
    using T = typename rs_elem<_T>::type;
    using dtype = T;
    static constexpr int rows   = _rows;
    static constexpr int cols   = _cols;
    static constexpr int run    = _run;
    static_assert(rows > 0, "a strip has at least one row");
    static_assert(run == 1 || run == 2 || run == 4 || run == 8 || run == 16,
                  "a run is a power of two of at most 16 elements");
    static constexpr int span   = run * WARP_THREADS;
    static_assert(cols % span == 0, "a strip is a whole number of chunks: COLS % (64*RUN) == 0");
    static constexpr int chunks = cols / span;
    static constexpr int elements_per_thread = rows * chunks * run;
    T data[rows][chunks][run];
    using col_vec = rs_cv<T, rows>;
    using row_vec = rs<T, 1, cols, run>;
};

namespace ducks {
namespace rs {
template<typename T>
concept all = requires { typename T::identifier; } &&
              std::is_same_v<typename T::identifier, identifier>;
template<typename T>
concept cv = requires { typename T::identifier; } &&
             std::is_same_v<typename T::identifier, cv_identifier>;
}  // namespace rs
}  // namespace ducks

namespace detail {

template<typename T> __device__ inline uint32_t hk_bits(const T &v) {
    if constexpr (sizeof(T) == 4) return __builtin_bit_cast(uint32_t, v);
    else { static_assert(sizeof(T) == 2, "a strip holds 16- and 32-bit elements");
           return uint32_t(__builtin_bit_cast(uint16_t, v)); }
}
template<typename T> __device__ inline T hk_from_bits(uint32_t u) {
    if constexpr (sizeof(T) == 4) return __builtin_bit_cast(T, u);
    else return __builtin_bit_cast(T, uint16_t(u));
}

/// v from the lane DPP control CTRL names (gfx9's encoding).
template<int CTRL, typename T> __device__ inline T hk_dpp(const T &v) {
    return hk_from_bits<T>(uint32_t(__builtin_amdgcn_mov_dpp(int(hk_bits(v)), CTRL, 0xF, 0xF, true)));
}

/// op of the value in this lane's 16-lane row and the one in the row XOR 1
/// (gfx950: v_permlane16_swap), or the half XOR 1 (v_permlane32_swap), in
/// that order in every lane.
template<typename op, int ROWS_APART, typename T> __device__ inline T hk_swap_op(const T &v) {
#if defined(KITTENS_CDNA4)
    const uint32_t u = hk_bits(v);
    if constexpr (ROWS_APART == 1) {
        const auto r = __builtin_amdgcn_permlane16_swap(u, u, false, true);
        return op::template op<T>(hk_from_bits<T>(r[0]), hk_from_bits<T>(r[1]));
    } else {
        const auto r = __builtin_amdgcn_permlane32_swap(u, u, false, true);
        return op::template op<T>(hk_from_bits<T>(r[0]), hk_from_bits<T>(r[1]));
    }
#else
    return op::template op<T>(v, hk_from_bits<T>(uint32_t(
        __shfl_xor(int(hk_bits(v)), 16 * ROWS_APART, WARP_THREADS))));
#endif
}

/// Fold N values with `op`, as a balanced tree.
template<typename op, typename T, int N>
__device__ inline T strip_tree(T (&a)[N]) {
    #pragma unroll
    for (int n = N; n > 1; n = (n + 1) / 2) {
        const int h = (n + 1) / 2;
        #pragma unroll
        for (int i = 0; i < n - h; i++) a[i] = op::template op<T>(a[i], a[i + h]);
    }
    return a[0];
}

/// Combine one value across all 64 lanes; every lane ends with the answer.
template<typename op, typename T>
__device__ inline T strip_butterfly(T v) {
    v = op::template op<T>(v, hk_dpp<0xB1>(v));    // quad_perm [1,0,3,2]: lane ^ 1
    v = op::template op<T>(v, hk_dpp<0x4E>(v));    // quad_perm [2,3,0,1]: lane ^ 2
    v = op::template op<T>(v, hk_dpp<0x141>(v));   // row_half_mirror: the other quad of 8
    v = op::template op<T>(v, hk_dpp<0x140>(v));   // row_mirror: the other 8 of 16
    v = hk_swap_op<op, 1>(v);
    return hk_swap_op<op, 2>(v);
}

/// op of v and the value lane ^ OFF holds, with the exact partner at every offset: DPP inside a
/// 16-lane row (row_shr / row_shl with bank masks for 8 and 4, quad_perm for 2 and 1 -- the
/// partners dsv41_mono's xshfl takes), the swaps across rows. strip_butterfly's mirrors are the
/// xor partners only when the lanes below agree, i.e. in ascending order; a descending order
/// needs these.
template<typename op, int OFF, typename T> __device__ inline T hk_xor_op(const T &v) {
    static_assert(OFF == 1 || OFF == 2 || OFF == 4 || OFF == 8 || OFF == 16 || OFF == 32,
                  "an xor offset is a power of two below 64");
    if constexpr (OFF >= 16) {
        return hk_swap_op<op, OFF / 16>(v);
    } else {
        const int x = int(hk_bits(v));
        int y;
        if constexpr (OFF == 8) {
            y = __builtin_amdgcn_update_dpp(x, x, 0x118, 0xF, 0xC, false);   // row_shr:8, lanes 8..15
            y = __builtin_amdgcn_update_dpp(y, x, 0x108, 0xF, 0x3, false);   // row_shl:8, lanes 0..7
        } else if constexpr (OFF == 4) {
            y = __builtin_amdgcn_update_dpp(x, x, 0x114, 0xF, 0xA, false);   // row_shr:4
            y = __builtin_amdgcn_update_dpp(y, x, 0x104, 0xF, 0x5, false);   // row_shl:4
        } else if constexpr (OFF == 2) {
            y = __builtin_amdgcn_update_dpp(x, x, 0x4E, 0xF, 0xF, false);    // quad_perm [2,3,0,1]
        } else {
            y = __builtin_amdgcn_update_dpp(x, x, 0xB1, 0xF, 0xF, false);    // quad_perm [1,0,3,2]
        }
        return op::template op<T>(v, hk_from_bits<T>(uint32_t(y)));
    }
}

template<typename RS, typename F>
__device__ inline void strip_each(F &&f) {
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            #pragma unroll
            for (int e = 0; e < RS::run; e++) f(r, j, e);
        }
    }
}

template<typename RS, typename U>
__device__ inline typename RS::T strip_scalar(const U &u) {
    using T = typename RS::T;
    if constexpr (std::is_same_v<U, T>) return u;
    else return hk_cvt<T, float>(float(u));
}

}  // namespace detail

/* ----------  maps  ---------- */

template<typename op, ducks::rs::all RS>
__device__ static inline void unary_map(RS &dst, const RS &src) {
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e]);
    });
}
template<typename op, ducks::rs::cv CV>
__device__ static inline void unary_map(CV &dst, const CV &src) {
    #pragma unroll
    for (int r = 0; r < CV::length; r++)
        dst.data[r][0] = op::template op<typename CV::T>(src.data[r][0]);
}

template<typename op, ducks::rs::all RS, typename U>
__device__ static inline void bin_map(RS &dst, const RS &lhs, const U &rhs) {
    using T = typename RS::T;
    if constexpr (std::is_same_v<U, RS>) {
        detail::strip_each<RS>([&](int r, int j, int e) {
            dst.data[r][j][e] = op::template op<T>(lhs.data[r][j][e], rhs.data[r][j][e]);
        });
    } else {
        const T s = detail::strip_scalar<RS>(rhs);
        detail::strip_each<RS>([&](int r, int j, int e) {
            dst.data[r][j][e] = op::template op<T>(lhs.data[r][j][e], s);
        });
    }
}
template<typename op, ducks::rs::cv CV, typename U>
__device__ static inline void bin_map(CV &dst, const CV &lhs, const U &rhs) {
    using T = typename CV::T;
    #pragma unroll
    for (int r = 0; r < CV::length; r++) {
        if constexpr (std::is_same_v<U, CV>)
            dst.data[r][0] = op::template op<T>(lhs.data[r][0], rhs.data[r][0]);
        else
            dst.data[r][0] = op::template op<T>(lhs.data[r][0], detail::strip_scalar<CV>(rhs));
    }
}

#define HK_STRIP_UNARY(name, opname)                                               \
    template<typename S> requires (ducks::rs::all<S> || ducks::rs::cv<S>)          \
    __device__ static inline void name(S &dst, const S &src) {                     \
        unary_map<base_ops::opname, S>(dst, src);                                   \
    }
HK_STRIP_UNARY(exp,   exp)
HK_STRIP_UNARY(exp2,  exp2)
HK_STRIP_UNARY(log,   log)
HK_STRIP_UNARY(log2,  log2)
HK_STRIP_UNARY(abs,   abs)
HK_STRIP_UNARY(relu,  relu)
HK_STRIP_UNARY(gelu,  gelu)
HK_STRIP_UNARY(dgelu, dgelu)
HK_STRIP_UNARY(sqrt,  sqrt)
HK_STRIP_UNARY(rsqrt, rsqrt)
HK_STRIP_UNARY(silu,  silu)
HK_STRIP_UNARY(step,  step)
#undef HK_STRIP_UNARY

#define HK_STRIP_FILL(name, opname)                                                \
    template<typename S> requires (ducks::rs::all<S> || ducks::rs::cv<S>)          \
    __device__ static inline void name(S &dst) {                                   \
        unary_map<base_ops::opname, S>(dst, dst);                                   \
    }
HK_STRIP_FILL(zero,      zero)
HK_STRIP_FILL(pos_infty, pos_infty)
HK_STRIP_FILL(neg_infty, neg_infty)
#undef HK_STRIP_FILL
/** @brief Fill with 1 (include/cdna4's base op is `one` too). */
template<typename S> requires (ducks::rs::all<S> || ducks::rs::cv<S>)
__device__ static inline void one(S &dst) {
    if constexpr (ducks::rs::all<S>)
        detail::strip_each<S>([&](int r, int j, int e) { dst.data[r][j][e] = typename S::T(1.f); });
    else
        for (int r = 0; r < S::length; r++) dst.data[r][0] = typename S::T(1.f);
}

/**
 * The fp32 max and min of hk's reductions: on gfx950 llvm.maximum / llvm.minimum, which it
 * selects as v_maximum3_f32 / v_minimum3_f32 as it does fmaxf as v_max3_f32. fmaxf is maxnum, and
 * in IEEE mode a maxnum operand that might be a signalling NaN is quieted first: a reduction of
 * 2k + 1 values pairs into max3s but for one two-operand max, whose operands -- MFMA results,
 * loop-carried values -- LLVM cannot prove quiet, so a v_max_f32 x, x each (attention's row max,
 * 3 instructions a reduction for 1). llvm.maximum propagates any NaN and needs no quieting. The
 * same on every value but NaN, which now propagates (torch.amax's), and -0 against +0, which is
 * now ordered (max -0, +0 is +0). Other types, and other CDNA targets, which have no
 * v_maximum3_f32 (gfx942 would expand each maximum into a compare and selects), keep the
 * library's base_ops::max / min.
 */
namespace base_ops {
struct hk_rmax {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return max::op<T>(a, b); }
};
struct hk_rmin {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return min::op<T>(a, b); }
};
#if defined(__gfx950__)
template<> __device__ inline float hk_rmax::op<float>(const float &a, const float &b) {
    return __builtin_elementwise_maximum(a, b);
}
template<> __device__ inline float2 hk_rmax::op<float2>(const float2 &a, const float2 &b) {
    return float2{__builtin_elementwise_maximum(a.x, b.x), __builtin_elementwise_maximum(a.y, b.y)};
}
template<> __device__ inline float hk_rmin::op<float>(const float &a, const float &b) {
    return __builtin_elementwise_minimum(a, b);
}
template<> __device__ inline float2 hk_rmin::op<float2>(const float2 &a, const float2 &b) {
    return float2{__builtin_elementwise_minimum(a.x, b.x), __builtin_elementwise_minimum(a.y, b.y)};
}
#endif
}  // namespace base_ops

/// The register tiles' row and column max and min through hk_rmax / hk_rmin: more constrained
/// than the library's (an fp32 tile), so such a tile picks them.
#define HK_RT_REDUCE(name, kind, opname)                                                        \
    template<ducks::rv::all V, ducks::rt::all T> requires (std::is_same_v<typename T::T, float>) \
    __device__ static inline void name(V &dst, const T &src) {                                  \
        kind##_reduce<base_ops::opname, V, T, true>(dst, src, dst);                             \
    }                                                                                           \
    template<ducks::rv::all V, ducks::rt::all T> requires (std::is_same_v<typename T::T, float>) \
    __device__ static inline void name(V &dst, const T &src, const V &acc) {                    \
        kind##_reduce<base_ops::opname, V, T, false>(dst, src, acc);                            \
    }
HK_RT_REDUCE(row_max, row, hk_rmax)
HK_RT_REDUCE(row_min, row, hk_rmin)
HK_RT_REDUCE(col_max, col, hk_rmax)
HK_RT_REDUCE(col_min, col, hk_rmin)
#undef HK_RT_REDUCE

#define HK_STRIP_BINARY(name, opname)                                              \
    template<typename S, typename U> requires (ducks::rs::all<S> || ducks::rs::cv<S>) \
    __device__ static inline void name(S &dst, const S &lhs, const U &rhs) {       \
        bin_map<base_ops::opname, S>(dst, lhs, rhs);                                \
    }
HK_STRIP_BINARY(add, sum)
HK_STRIP_BINARY(sub, sub)
HK_STRIP_BINARY(mul, mul)
HK_STRIP_BINARY(div, div)
HK_STRIP_BINARY(max, max)
HK_STRIP_BINARY(min, min)
#undef HK_STRIP_BINARY

/// The combine of two partial max / min reductions (cross_warp's, a vector folded in by max_):
/// base_ops::hk_rmax / hk_rmin, so a reduction split across warps or blocks keeps the NaN and
/// signed-zero rules of one within a wave. Register vectors and strip vectors.
#define HK_REDUCE_COMBINE(name, opname)                                                \
    template<ducks::rv::all T, typename U>                                              \
    __device__ static inline void name(T &dst, const T &lhs, const U &rhs) {            \
        bin_op<base_ops::opname, T>(dst, lhs, rhs);                                      \
    }                                                                                   \
    template<typename S, typename U> requires (ducks::rs::all<S> || ducks::rs::cv<S>)    \
    __device__ static inline void name(S &dst, const S &lhs, const U &rhs) {            \
        bin_map<base_ops::opname, S>(dst, lhs, rhs);                                     \
    }
HK_REDUCE_COMBINE(rmax, hk_rmax)
HK_REDUCE_COMBINE(rmin, hk_rmin)
#undef HK_REDUCE_COMBINE

namespace detail {
/** The whole tensor as one buffer: rows over batch and depth at the row stride (the descriptor's
 *  range is 32 bits: a tensor past 4 GB reads zeros and drops stores beyond it). */
template<ducks::gl::all GL> __device__ inline hk_i32x4 hk_elems_srsrc(const GL &g) {
    using U = typename GL::dtype;
    const uint64_t n = uint64_t(g.batch()) * g.depth() * g.rows() * uint64_t(g.row_stride());
    return hk_srsrc(g.raw_ptr, n * sizeof(U));
}
/// Any lane of the wave with an index >= 0 (take_elems on a wave with none polls nothing). Not for
/// load / store_elems: the branch splits a kernel's loads into blocks the scheduler cannot batch
/// across (DeepSeek-V4.1 K1's seam slice: +1 us).
template<typename I> __device__ inline bool hk_any_live(const I &idx) {
    bool live = false;
    strip_each<I>([&](int a, int j, int e) { live |= idx.data[a][j][e] >= 0; });
    return __any(live);
}
__device__ inline uint32_t hk_elem_off(int k, uint32_t size) {
    return k < 0 ? 0xFFFFFFFFu : uint32_t(k) * size;     // past the range: reads 0, stores nothing
}
}  // namespace detail
/** @brief hk lane_gather: element (r, c) of lane l = the same run element of lane lanes(r, c). */
template<ducks::rs::all D, ducks::rs::all I>
__device__ static inline void lane_gather(D &dst, const D &src, const I &lanes) {
    detail::strip_each<D>([&](int r, int j, int e) {
        const int v = __builtin_amdgcn_ds_bpermute(lanes.data[r][j][e] * 4,
                                                   int(detail::hk_bits(src.data[r][j][e])));
        dst.data[r][j][e] = detail::hk_from_bits<typename D::T>(uint32_t(v));
    });
}
namespace detail {
/** f(k, value): a register tile's or a strip's per-lane elements in lane_strip's order. */
template<typename S, typename F> __device__ inline void hk_lane_each(const S &src, F &&f) {
    if constexpr (ducks::rs::all<S>) {
        #pragma unroll
        for (int r = 0; r < S::rows; r++)
            #pragma unroll
            for (int j = 0; j < S::chunks; j++)
                #pragma unroll
                for (int e = 0; e < S::run; e++) f((r * S::chunks + j) * S::run + e, src.data[r][j][e]);
    } else {
        using G = hk_frag<S>;
        #pragma unroll
        for (int i = 0; i < S::height; i++)
            #pragma unroll
            for (int j = 0; j < S::width; j++)
                #pragma unroll
                for (int e = 0; e < G::E; e++) f((i * S::width + j) * G::E + e, hk_get(src, i, j, e));
    }
}
template<typename D> __device__ inline auto &hk_strip_at(D &d, int k) {
    return d.data[0][k / D::run][k % D::run];
}
}  // namespace detail
/** @brief hk lane_strip: the operand's registers, lane by lane in its per-lane order, as a 1-row strip. */
template<ducks::rs::all D, typename S>
__device__ static inline void lane_strip(D &dst, const S &src) {
    static_assert(D::rows == 1, "lane_strip gives a 1-row strip");
    detail::hk_lane_each(src, [&](int k, auto v) {
        detail::hk_strip_at(dst, k) = detail::hk_from_bits<typename D::T>(detail::hk_bits(v));
    });
}
/** @brief hk lane_coord: each element's operand row (AXIS 0) or column (1), in lane_strip's order. */
template<int AXIS, ducks::rs::all D, typename S>
__device__ static inline void lane_coord(D &dst, const S &) {
    static_assert(D::rows == 1, "lane_coord gives a 1-row strip");
    const int lane = laneid();
    if constexpr (ducks::rs::all<S>) {
        #pragma unroll
        for (int r = 0; r < S::rows; r++)
            #pragma unroll
            for (int j = 0; j < S::chunks; j++)
                #pragma unroll
                for (int e = 0; e < S::run; e++)
                    detail::hk_strip_at(dst, (r * S::chunks + j) * S::run + e) =
                        AXIS == 0 ? r : j * S::run * WARP_THREADS + lane * S::run + e;
    } else {
        using G = detail::hk_frag<S>;
        #pragma unroll
        for (int i = 0; i < S::height; i++)
            #pragma unroll
            for (int j = 0; j < S::width; j++)
                #pragma unroll
                for (int e = 0; e < G::E; e++) {
                    const int2 c = detail::hk_frag_coord<S>(e, lane);
                    detail::hk_strip_at(dst, (i * S::width + j) * G::E + e) =
                        AXIS == 0 ? i * G::R + c.x : j * G::C + c.y;
                }
    }
}
/** @brief hk lane_coord of a strip itself: element (r, c) = r (AXIS 0) or c (1), the column a
 * lane's run element sits in (hk.lane_coords(type); hk.iota is its column). */
template<int AXIS, ducks::rs::all D>
__device__ static inline void lane_coord(D &dst) {
    const int lane = laneid();
    detail::strip_each<D>([&](int r, int j, int e) {
        dst.data[r][j][e] = typename D::T(AXIS == 0 ? r : j * 64 * D::run + lane * D::run + e);
    });
}
/** @brief hk run_xor: element e of each lane's run = element e ^ MASK of the same run. */
template<int MASK, ducks::rs::all D>
__device__ static inline void run_xor(D &dst, const D &src) {
    static_assert(MASK > 0 && MASK < D::run, "run_xor's mask is below the run");
    D t = src;
    detail::strip_each<D>([&](int r, int j, int e) { dst.data[r][j][e] = t.data[r][j][e ^ MASK]; });
}

/** @brief hk post_elems: element (r, c) of val and `tag` to pair idx(r, c) of `box`, one 8-byte AUX store. */
template<int AUX, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void post_elems(const GL &box, const V &val, const I &idx, int tag) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a mailbox is int32 pairs");
    const detail::hk_i32x4 r = detail::hk_elems_srsrc(box);
    detail::strip_each<V>([&](int a, int j, int e) {
        const int k = idx.data[a][j][e];
        const uint32_t bits = detail::hk_bits(val.data[a][j][e]);
        detail::hk_buf_st64(uint64_t(bits) | (uint64_t(uint32_t(tag)) << 32), r,
                            detail::hk_elem_off(k, 8), 0, AUX);
    });
}
/** @brief hk post_elems(runs=True): a lane's run as consecutive pairs from idx of its first element, in
 * stores of up to 16 bytes (two pairs) -- masked by that first index. */
template<int AUX, bool BF16, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void post_elems_runs(const GL &box, const V &val, const I &idx, int tag) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a mailbox is int32 pairs");
    constexpr int PER = BF16 ? 4 : 2;           // values a 16-byte store carries (two pairs)
    static_assert(V::run % PER == 0, "runs of whole 16-byte pairs of pairs");
    const detail::hk_i32x4 r = detail::hk_elems_srsrc(box);
    #pragma unroll
    for (int a = 0; a < V::rows; a++)
        #pragma unroll
        for (int j = 0; j < V::chunks; j++) {
            const int k = idx.data[a][j][0];
            #pragma unroll
            for (int e = 0; e < V::run; e += PER) {
                uint32_t w[4];
                if constexpr (BF16) {          // two bf16 a pair, the first in the low half (the values' top halves)
                    w[0] = (detail::hk_bits(val.data[a][j][e]) >> 16) | (detail::hk_bits(val.data[a][j][e + 1]) & 0xFFFF0000u);
                    w[2] = (detail::hk_bits(val.data[a][j][e + 2]) >> 16) | (detail::hk_bits(val.data[a][j][e + 3]) & 0xFFFF0000u);
                } else {
                    w[0] = detail::hk_bits(val.data[a][j][e]);
                    w[2] = detail::hk_bits(val.data[a][j][e + 1]);
                }
                w[1] = w[3] = uint32_t(tag);
                detail::hk_buf_store_bytes<16, AUX>(r, k < 0 ? 0xFFFFFFFFu : uint32_t(k + e / (PER / 2)) * 8u, w);
            }
        }
}
/** @brief hk take_elems(runs=True): a lane's run from consecutive pairs from idx of its first element, in
 * loads of 16 bytes (two pairs); all polled together. */
template<int AUX, bool BF16, ducks::rs::all D, ducks::gl::all GL, ducks::rs::all I>
__device__ static inline void take_elems_runs(D &dst, const GL &box, const I &idx, int tag, long long timeout) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a mailbox is int32 pairs");
    constexpr int PER = BF16 ? 4 : 2;
    static_assert(D::run % PER == 0, "runs of whole 16-byte pairs of pairs");
    const detail::hk_i32x4 r = detail::hk_elems_srsrc(box);
    if (!detail::hk_any_live(idx)) {
        detail::strip_each<D>([&](int a, int j, int e) { dst.data[a][j][e] = typename D::T(0); });
        return;
    }
    const long long t0 = __builtin_amdgcn_s_memrealtime();
    while (true) {
        bool ready = true;
        #pragma unroll
        for (int a = 0; a < D::rows; a++)
            #pragma unroll
            for (int j = 0; j < D::chunks; j++) {
                const int k = idx.data[a][j][0];
                #pragma unroll
                for (int e = 0; e < D::run; e += PER) {
                    uint32_t w[4];
                    detail::hk_buf_load_bytes<16, AUX>(r, k < 0 ? 0xFFFFFFFFu : uint32_t(k + e / (PER / 2)) * 8u, w);
                    ready &= (k < 0) | ((int(w[1]) == tag) & (int(w[3]) == tag));
                    if constexpr (BF16) {
                        dst.data[a][j][e]     = detail::hk_from_bits<typename D::T>(w[0] << 16);
                        dst.data[a][j][e + 1] = detail::hk_from_bits<typename D::T>(w[0] & 0xFFFF0000u);
                        dst.data[a][j][e + 2] = detail::hk_from_bits<typename D::T>(w[2] << 16);
                        dst.data[a][j][e + 3] = detail::hk_from_bits<typename D::T>(w[2] & 0xFFFF0000u);
                    } else {
                        dst.data[a][j][e] = detail::hk_from_bits<typename D::T>(w[0]);
                        dst.data[a][j][e + 1] = detail::hk_from_bits<typename D::T>(w[2]);
                    }
                }
            }
        if (__all(ready)) break;
        if (timeout && __builtin_amdgcn_s_memrealtime() - t0 > timeout) __builtin_trap();
        __builtin_amdgcn_s_sleep(1);
        asm volatile("" ::: "memory");
    }
}
/** @brief hk take_elems: dst(r, c) = value of pair idx(r, c) once its tag is `tag` (all polled together). */
template<int AUX, ducks::rs::all D, ducks::gl::all GL, ducks::rs::all I>
__device__ static inline void take_elems(D &dst, const GL &box, const I &idx, int tag, long long timeout) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "a mailbox is int32 pairs");
    const detail::hk_i32x4 r = detail::hk_elems_srsrc(box);
    // a wave with no pair to take polls nothing (DeepSeek-V4.1 K1's norm: waves 4 .. 7 hold no
    // LIN; polling out of range they still held the CTA's barrier back)
    if (!detail::hk_any_live(idx)) {
        detail::strip_each<D>([&](int a, int j, int e) { dst.data[a][j][e] = typename D::T(0); });
        return;
    }
    const long long t0 = __builtin_amdgcn_s_memrealtime();
    while (true) {
        bool ready = true;
        detail::strip_each<D>([&](int a, int j, int e) {
            const int k = idx.data[a][j][e];
            // unconditional: a k < 0 offset is past the range and reads 0
            const uint64_t m = detail::hk_buf_ld64(r, detail::hk_elem_off(k, 8), 0, AUX);
            ready &= (k < 0) | (int(uint32_t(m >> 32)) == tag);
            dst.data[a][j][e] = detail::hk_from_bits<typename D::T>(uint32_t(m));
        });
        if (__all(ready)) break;   // every lane's pairs current
        if (timeout && __builtin_amdgcn_s_memrealtime() - t0 > timeout) __builtin_trap();
        __builtin_amdgcn_s_sleep(1);
        asm volatile("" ::: "memory");   // re-issued every round
    }
}
/** @brief hk lds_load_elems: dst(r, c) = src's element idx(r, c) (row-major, through its swizzle; OOB -> 0). */
template<ducks::rs::all D, typename ST, ducks::rs::all I>
__device__ static inline void lds_load_elems(D &dst, ST &src, const I &idx) {
    constexpr int N = ST::rows * ST::cols;
    detail::strip_each<D>([&](int a, int j, int e) {
        // branch-free: an index out of range reads element 0 and keeps 0. A guarded read put each in its
        // own exec-masked block, and the strip's reads went one at a time, each waiting out its latency.
        const int k = idx.data[a][j][e];
        const bool in = uint32_t(k) < uint32_t(N);
        const int kk = in ? k : 0;
        const float v = *detail::hk_st_elem(src, kk / ST::cols, kk % ST::cols);
        dst.data[a][j][e] = detail::hk_from_bits<typename D::T>(detail::hk_bits(in ? v : 0.f));
    });
}
/** @brief hk lds_load_elems(runs=True): each lane's run 4 elements at a time, one 16-byte read from each
 * 16-byte-aligned index (OOB -> 0). A 16x16 subtile's 16 columns are contiguous, so an aligned 4 is too. */
template<ducks::rs::all D, typename ST, ducks::rs::all I>
__device__ static inline void lds_load_elems_runs(D &dst, ST &src, const I &idx) {
    static_assert(D::run % 4 == 0, "runs of 4");
    constexpr int N = ST::rows * ST::cols;
    typedef float hk_f4 __attribute__((ext_vector_type(4)));
    typedef __attribute__((address_space(3))) hk_f4 hk_lds_f4;
    #pragma unroll
    for (int a = 0; a < D::rows; a++) {
        #pragma unroll
        for (int j = 0; j < D::chunks; j++) {
            #pragma unroll
            for (int q = 0; q < D::run / 4; q++) {
                const int k = idx.data[a][j][4 * q];
                const bool in = uint32_t(k) < uint32_t(N);
                const int kk = in ? k : 0;
                const hk_f4 v = *(hk_lds_f4 *)detail::hk_st_elem(src, kk / ST::cols, kk % ST::cols);
                #pragma unroll
                for (int e = 0; e < 4; e++)
                    dst.data[a][j][4 * q + e] = detail::hk_from_bits<typename D::T>(detail::hk_bits(in ? v[e] : 0.f));
            }
        }
    }
}
/** @brief hk lds_store_elems: val(r, c)'s bits to dst's element idx(r, c) (row-major; OOB: none). A bf16
 * tile takes the value (fp32) rounded to bf16 -- an MMA operand built element by element. */
template<typename ST, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void lds_store_elems(ST &dst, const V &val, const I &idx) {
    constexpr int N = ST::rows * ST::cols;
    detail::strip_each<V>([&](int a, int j, int e) {
        const int k = idx.data[a][j][e];
        if (k >= 0 && k < N) {
            const float v = detail::hk_from_bits<float>(detail::hk_bits(val.data[a][j][e]));
            if constexpr (std::is_same_v<typename ST::dtype, bf16>) {
                const int r = k / ST::cols, c = k % ST::cols;
                const int sub = (r / ST::underlying_subtile_rows) * ST::underlying_subtiles_per_row +
                                c / ST::underlying_subtile_cols;
                const uint32_t off = ST::swizzle({r % ST::underlying_subtile_rows, c % ST::underlying_subtile_cols});
                typedef __attribute__((address_space(3))) char lds_char;
                typedef __attribute__((address_space(3))) uint16_t lds_u16;
                const bf16 h = detail::hk_cvt<bf16, float>(v);
                *(lds_u16 *)((lds_char *)&dst.data[0] + sub * ST::underlying_subtile_bytes + off) =
                    __builtin_bit_cast(uint16_t, h);
            } else {
                *detail::hk_st_elem(dst, k / ST::cols, k % ST::cols) = v;
            }
        }
    });
}
#if defined(KITTENS_CDNA4)   // (hk_mxs and the f8f6f4 operand containers are gfx950's)
/** @brief hk.lane_view: a 1-row i32 strip's dwords as a 16x128 fp8 fragment -- eight a lane (1 x 512,
 * run 4: two runs, its K step's 16-byte halves at K 16 j .. and 64 + 16 j .., j = lane / 16), the fragment
 * mma_ABt_mx reads; or four (1 x 256, run 4), the first four dwords of the FP4 operand container. */
template<typename RT, ducks::rs::all S>
__device__ static inline void lane_view(RT &dst, const S &src) {
    static_assert(RT::height == 1 && RT::width == 1, "one 16x128 fragment");
    static_assert(S::rows == 1 && S::run == 4 && (S::chunks == 1 || S::chunks == 2),
                  "four or eight dwords a lane, in runs of four");
    if constexpr (S::chunks == 1) {
        int w[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) w[e] = int(src.data[0][0][e]);
        __builtin_memcpy(&dst.tiles[0][0].data[0], w, 16);
    } else {
        int w[8];
        #pragma unroll
        for (int e = 0; e < 4; e++) { w[e] = int(src.data[0][0][e]); w[4 + e] = int(src.data[0][1][e]); }
        __builtin_memcpy(&dst.tiles[0][0].data[0], w, 32);
    }
}
/** @brief hk.lane_view as scales: a lane's element (i32 strip 1 x 64, the E8M0 byte in bits 0..7) as the loose
 * scale register of row lane % 16, block lane / 16 of a 16x128 operand. */
template<ducks::rs::all S>
__device__ static inline void lane_view(hk_mxs<16, 4, 1> &dst, const S &src) {
    dst.reg[0][0] = uint32_t(int(src.data[0][0][0]) & 0xFF);
}
#endif
/** @brief hk lds_atomic_add_elems: val(r, c) (i32) added atomically (ds_add_u32) to dst's element idx(r, c) as
 * an i32 (row-major; OOB: none). */
template<typename ST, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void lds_atomic_add_elems(ST &dst, const V &val, const I &idx) {
    constexpr int N = ST::rows * ST::cols;
    detail::strip_each<V>([&](int a, int j, int e) {
        const int k = idx.data[a][j][e];
        if (k >= 0 && k < N) {
            typedef __attribute__((address_space(3))) int lds_int;
            __hip_atomic_fetch_add((lds_int *)detail::hk_st_elem(dst, k / ST::cols, k % ST::cols),
                                   int(val.data[a][j][e]), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
    });
}
/** @brief hk.load_elems: dst's element (r, c) = src's element idx(r, c), flat; out of range -> 0. */
template<int AUX = 0, ducks::rs::all D, ducks::gl::all GL, ducks::rs::all I>
__device__ static inline void load_elems(D &dst, const GL &src, const I &idx) {
    using U = typename GL::dtype;
    const detail::hk_i32x4 r = detail::hk_elems_srsrc(src);
    detail::strip_each<D>([&](int a, int j, int e) {
        dst.data[a][j][e] = detail::hk_cvt<typename D::T, U>(
            detail::hk_buf_load_elem<U, AUX>(r, detail::hk_elem_off(idx.data[a][j][e], sizeof(U))));
    });
}
/** @brief hk.store_elems(runs=True): each lane's run to the consecutive elements from idx of its first
 * element, as one store of the run's bytes (masked by that index). */
namespace detail {
/** One row of the tensor as the buffer: the base formed in 64 bits (a tensor may span far past a
 *  32-bit offset), the range that row. */
template<ducks::gl::all GL> __device__ inline hk_i32x4 hk_elems_srsrc_row(const GL &g, int row) {
    using U = typename GL::dtype;
    const uint64_t rs = uint64_t(g.row_stride()) * sizeof(U);
    return hk_srsrc(reinterpret_cast<const char *>(g.raw_ptr) + uint64_t(uint32_t(row)) * rs, rs);
}
}  // namespace detail
template<int AUX = 0, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void store_elems_runs(const GL &dst, const V &val, const I &idx,
                                               const detail::hk_i32x4 r);
template<int AUX = 0, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void store_elems_runs(const GL &dst, const V &val, const I &idx) {
    store_elems_runs<AUX>(dst, val, idx, detail::hk_elems_srsrc(dst));
}
/** @brief store_elems_runs, the indices counting from row `row` (hk_elems_srsrc_row). */
template<int AUX = 0, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void store_elems_runs(const GL &dst, const V &val, const I &idx, int row) {
    store_elems_runs<AUX>(dst, val, idx, detail::hk_elems_srsrc_row(dst, row));
}
template<int AUX, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void store_elems_runs(const GL &dst, const V &val, const I &idx,
                                               const detail::hk_i32x4 r) {
    using U = typename GL::dtype;
    #pragma unroll
    for (int a = 0; a < V::rows; a++) {
        #pragma unroll
        for (int j = 0; j < V::chunks; j++) {
            const uint32_t off = detail::hk_elem_off(idx.data[a][j][0], sizeof(U));
            if constexpr (std::is_same_v<U, fp8e4m3> && std::is_same_v<typename V::T, float> &&
                          V::run % 4 == 0) {
                // fp32 -> e4m3 four at a time into words (v_cvt_pk_fp8_f32, as the scalar path)
                uint32_t words[V::run / 4];
                #pragma unroll
                for (int q = 0; q < V::run / 4; q++) {
                    int w = __builtin_amdgcn_cvt_pk_fp8_f32(val.data[a][j][4 * q], val.data[a][j][4 * q + 1], 0, false);
                    w = __builtin_amdgcn_cvt_pk_fp8_f32(val.data[a][j][4 * q + 2], val.data[a][j][4 * q + 3], w, true);
                    words[q] = uint32_t(w);
                }
                detail::hk_buf_store_bytes<V::run, AUX>(r, off, words);
            } else {
                U run[V::run];
                #pragma unroll
                for (int e = 0; e < V::run; e++) run[e] = detail::hk_cvt<U, typename V::T>(val.data[a][j][e]);
                detail::hk_buf_store_bytes<V::run * sizeof(U), AUX>(r, off, run);
            }
        }
    }
}
/** @brief hk.store_elems: val's element (r, c) to dst's element idx(r, c), flat; out of range: none. */
template<int AUX = 0, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void store_elems(const GL &dst, const V &val, const I &idx,
                                          const detail::hk_i32x4 r);
template<int AUX = 0, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void store_elems(const GL &dst, const V &val, const I &idx) {
    store_elems<AUX>(dst, val, idx, detail::hk_elems_srsrc(dst));
}
/** @brief load_elems, the indices counting from row `row` (hk_elems_srsrc_row). */
template<int AUX = 0, ducks::rs::all D, ducks::gl::all GL, ducks::rs::all I>
__device__ static inline void load_elems(D &dst, const GL &src, const I &idx, int row) {
    using U = typename GL::dtype;
    const detail::hk_i32x4 r = detail::hk_elems_srsrc_row(src, row);
    detail::strip_each<D>([&](int a, int j, int e) {
        dst.data[a][j][e] = detail::hk_cvt<typename D::T, U>(
            detail::hk_buf_load_elem<U, AUX>(r, detail::hk_elem_off(idx.data[a][j][e], sizeof(U))));
    });
}
/** @brief hk.load_elems(runs=True): each lane's run from the consecutive elements from idx of its first
 * element, as one load of the run's bytes; a first index out of range reads the run as 0. */
template<int AUX = 0, ducks::rs::all D, ducks::gl::all GL, ducks::rs::all I>
__device__ static inline void load_elems_runs(D &dst, const GL &src, const I &idx, const detail::hk_i32x4 r) {
    using U = typename GL::dtype;
    #pragma unroll
    for (int a = 0; a < D::rows; a++) {
        #pragma unroll
        for (int j = 0; j < D::chunks; j++) {
            U run[D::run];
            detail::hk_buf_load_bytes<D::run * sizeof(U), AUX>(r, detail::hk_elem_off(idx.data[a][j][0], sizeof(U)), run);
            #pragma unroll
            for (int e = 0; e < D::run; e++) dst.data[a][j][e] = detail::hk_cvt<typename D::T, U>(run[e]);
        }
    }
}
template<int AUX = 0, ducks::rs::all D, ducks::gl::all GL, ducks::rs::all I>
__device__ static inline void load_elems_runs(D &dst, const GL &src, const I &idx) {
    load_elems_runs<AUX>(dst, src, idx, detail::hk_elems_srsrc(src));
}
/** @brief load_elems_runs, the indices counting from row `row` (hk_elems_srsrc_row). */
template<int AUX = 0, ducks::rs::all D, ducks::gl::all GL, ducks::rs::all I>
__device__ static inline void load_elems_runs(D &dst, const GL &src, const I &idx, int row) {
    load_elems_runs<AUX>(dst, src, idx, detail::hk_elems_srsrc_row(src, row));
}
/** @brief store_elems, the indices counting from row `row` (hk_elems_srsrc_row). */
template<int AUX = 0, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void store_elems(const GL &dst, const V &val, const I &idx, int row) {
    store_elems<AUX>(dst, val, idx, detail::hk_elems_srsrc_row(dst, row));
}
template<int AUX, ducks::gl::all GL, ducks::rs::all V, ducks::rs::all I>
__device__ static inline void store_elems(const GL &dst, const V &val, const I &idx,
                                          const detail::hk_i32x4 r) {
    using U = typename GL::dtype;
    detail::strip_each<V>([&](int a, int j, int e) {
        detail::hk_buf_store_elem<U, AUX>(r, detail::hk_elem_off(idx.data[a][j][e], sizeof(U)),
                                          detail::hk_cvt<U, typename V::T>(val.data[a][j][e]));
    });
}

/** @brief Each chunk's lane groups reduced: a lane's run folded (SEQ: left to right, else the
 * balanced tree), then combined over the xor offsets OFFS in that order (1 .. 2^(k-1): groups of
 * 2^k neighbouring lanes), the group's value written to every element of the lane's run -- an MX
 * group's amax (run 8, xor 1 2), a 64-dim block's (run 2, xor 1 .. 16). */
template<typename op, int SEQ, int... OFFS, ducks::rs::all RS>
__device__ static inline void group_reduce_ordered(RS &dst, const RS &src) {
    using T = typename RS::T;
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            T a[RS::run];
            #pragma unroll
            for (int e = 0; e < RS::run; e++) a[e] = src.data[r][j][e];
            T v;
            if constexpr (SEQ) {
                v = a[0];
                #pragma unroll
                for (int e = 1; e < RS::run; e++) v = op::template op<T>(v, a[e]);
            } else {
                v = detail::strip_tree<op>(a);
            }
            ((v = detail::hk_xor_op<op, OFFS>(v)), ...);
            #pragma unroll
            for (int e = 0; e < RS::run; e++) dst.data[r][j][e] = v;
        }
    }
}

/** @brief Copy a strip or a per-row vector, converting the element type. */
template<ducks::rs::all D, ducks::rs::all S>
__device__ static inline void copy(D &dst, const S &src) {
    static_assert(D::rows == S::rows && D::cols == S::cols && D::run == S::run,
                  "a strip copy keeps the shape and the run");
    detail::strip_each<D>([&](int r, int j, int e) {
        dst.data[r][j][e] = detail::hk_cvt<typename D::T, typename S::T>(src.data[r][j][e]);
    });
}
template<ducks::rs::cv D, ducks::rs::cv S>
__device__ static inline void copy(D &dst, const S &src) {
    static_assert(D::length == S::length, "a vector copy keeps the length");
    #pragma unroll
    for (int r = 0; r < D::length; r++)
        dst.data[r][0] = detail::hk_cvt<typename D::T, typename S::T>(src.data[r][0]);
}

/**
 * @brief The halves of each SPAN-column segment of every row exchanged, the first half negated
 * when NEG (NeoX's rotate_half): dst(r, c) = (NEG ? -1 : 1) * src(r, c + SPAN/2) in a segment's
 * first half and src(r, c - SPAN/2) in its second; SPAN = the row (COLS) exchanges the row's
 * halves. Halves whole chunks apart swap registers; halves 8, 16 or 32 lanes apart exchange
 * lanes -- DPP row_ror:8 inside a 16-lane row, v_permlane16_swap, v_permlane32_swap (gfx950;
 * ds_bpermute on gfx942) -- and the first half's sign is one XOR a value.
 */
template<int SPAN, bool NEG, ducks::rs::all RS>
__device__ static inline void swap_halves(RS &dst, const RS &src) {
    using T = typename RS::T;
    static_assert(RS::cols % SPAN == 0, "segments tile the row");
    constexpr uint32_t SIGN = !NEG ? 0u : (sizeof(T) == 4 ? 0x80000000u : 0x8000u);
    if constexpr (SPAN % (2 * RS::span) == 0) {
        constexpr int SEG = SPAN / RS::span, H = SEG / 2;
        RS tmp;
        detail::strip_each<RS>([&](int r, int j, int e) {
            const int s = j % SEG;
            if constexpr (NEG)
                tmp.data[r][j][e] = s < H
                    ? detail::hk_from_bits<T>(detail::hk_bits(src.data[r][j + H][e]) ^ SIGN)
                    : src.data[r][j - H][e];
            else
                tmp.data[r][j][e] = src.data[r][s < H ? j + H : j - H][e];
        });
        dst = tmp;
    } else {
        constexpr int LANES = SPAN / (2 * RS::run);
        static_assert(SPAN % (2 * RS::run) == 0 && (LANES == 8 || LANES == 16 || LANES == 32),
                      "a segment's halves 8, 16 or 32 lanes apart, or whole chunks");
        const int l = laneid();
        const uint32_t sign = NEG && (l % (2 * LANES)) < LANES ? SIGN : 0u;
        detail::strip_each<RS>([&](int r, int j, int e) {
            const uint32_t u = detail::hk_bits(src.data[r][j][e]);
            uint32_t p;
#if defined(KITTENS_CDNA4)
            if constexpr (LANES == 8) {
                p = uint32_t(__builtin_amdgcn_mov_dpp(int(u), 0x128, 0xF, 0xF, true));
            } else if constexpr (LANES == 16) {
                // the even row of each pair finds its partner's value in the second result,
                // the odd row in the first
                const auto q = __builtin_amdgcn_permlane16_swap(u, u, false, true);
                p = (l % 32) < 16 ? q[1] : q[0];
            } else {
                // vdst's upper half takes vsrc's lower and vsrc's lower vdst's upper: from
                // (u, u), lane l < 32 finds lane l + 32's value in the second result and lane
                // l + 32 finds lane l's in the first
                const auto q = __builtin_amdgcn_permlane32_swap(u, u, false, true);
                p = l < 32 ? q[1] : q[0];
            }
#else
            p = uint32_t(__shfl_xor(int(u), LANES, WARP_THREADS));
#endif
            if constexpr (NEG)
                dst.data[r][j][e] = detail::hk_from_bits<T>(p ^ sign);
            else
                dst.data[r][j][e] = detail::hk_from_bits<T>(p);
        });
    }
}

/* ----------  reductions  ---------- */

namespace detail {
template<typename op, bool reset, ducks::rs::cv CV, ducks::rs::all RS>
__device__ inline void strip_row_reduce(CV &dst, const RS &src, const CV &acc) {
    static_assert(CV::length == RS::rows, "one value per row of the strip");
    using T = typename RS::T;
    constexpr int N = RS::chunks * RS::run;
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        T a[N];
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            #pragma unroll
            for (int e = 0; e < RS::run; e++) a[j*RS::run + e] = src.data[r][j][e];
        }
        const T v = strip_butterfly<op>(strip_tree<op>(a));
        if constexpr (reset) dst.data[r][0] = v;
        else                 dst.data[r][0] = op::template op<T>(acc.data[r][0], v);
    }
}

/// A value the compiler can no longer see the making of: a product that feeds an add stays a
/// rounded product. HIP compiles with contraction on, and LLVM fuses `p = x * y; s + p` into one
/// fma -- across hk's ops, since they are inlined -- which would make an ordered sum of a product
/// another order (its first two summands fused as fma(x0, y0, x1 * y1), then fma(xi, yi, s)).
template<typename T> __device__ inline void hk_opaque(T &v) {
    if constexpr (sizeof(T) == 4) asm volatile("" : "+v"(v));
}

/// strip_row_reduce in a stated order: a lane's own values left to right (SEQ) or as the balanced
/// tree, then the lanes combined over the xor offsets OFFS in that order. The order is the
/// numerical contract when a result must match another kernel bit for bit (dsv41_mono's and
/// aiter's norms: sequential, then 32, 16, 8, 4, 2, 1). The values are the strip's as they are:
/// a product's rounding is not fused into the sum (strip_row_dot_ordered is the fused form).
template<typename op, bool reset, bool SEQ, int... OFFS, ducks::rs::cv CV, ducks::rs::all RS>
__device__ inline void strip_row_reduce_ordered(CV &dst, const RS &src, const CV &acc) {
    static_assert(CV::length == RS::rows, "one value per row of the strip");
    using T = typename RS::T;
    constexpr int N = RS::chunks * RS::run;
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        T a[N];
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            #pragma unroll
            for (int e = 0; e < RS::run; e++) {
                a[j*RS::run + e] = src.data[r][j][e];
                hk_opaque(a[j*RS::run + e]);
            }
        }
        T v;
        if constexpr (SEQ) {
            v = a[0];
            #pragma unroll
            for (int i = 1; i < N; i++) v = op::template op<T>(v, a[i]);
        } else {
            v = strip_tree<op>(a);
        }
        ((v = hk_xor_op<op, OFFS>(v)), ...);
        if constexpr (reset) dst.data[r][0] = v;
        else                 dst.data[r][0] = op::template op<T>(acc.data[r][0], v);
    }
}

/// The sum of a row's products src * times, the lane's own products fused into one running fma
/// the way LLVM contracts a compiled `s = a0 * b0 + a1 * b1; s += ai * bi` loop -- the second
/// product rounded and the first fused onto it, then each next one fused: s = fma(a0, b0, r(a1
/// b1)), s = fma(ai, bi, s) (a single product: r(a0 b0)) -- then the lanes summed over the xor
/// offsets OFFS in that order. dsv41_mono's sums of squares (q / kv norms) are this.
template<bool reset, int... OFFS, ducks::rs::cv CV, ducks::rs::all RS>
__device__ inline void strip_row_dot_ordered(CV &dst, const RS &src, const RS &times, const CV &acc) {
    _Pragma("clang fp contract(off)")
    static_assert(CV::length == RS::rows, "one value per row of the strip");
    using T = typename RS::T;
    static_assert(std::is_same_v<T, float>, "a fused sum of products is fp32's");
    constexpr int N = RS::chunks * RS::run;
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        T a[N], b[N];
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            #pragma unroll
            for (int e = 0; e < RS::run; e++) {
                a[j*RS::run + e] = src.data[r][j][e];
                b[j*RS::run + e] = times.data[r][j][e];
            }
        }
        T v;
        if constexpr (N == 1) {
            v = a[0] * b[0];
        } else {
            T p1 = a[1] * b[1];
            hk_opaque(p1);
            v = __builtin_fmaf(a[0], b[0], p1);
            #pragma unroll
            for (int i = 2; i < N; i++) v = __builtin_fmaf(a[i], b[i], v);
        }
        hk_opaque(v);
        ((v = hk_xor_op<base_ops::sum, OFFS>(v)), ...);
        if constexpr (reset) dst.data[r][0] = v;
        else                 dst.data[r][0] = acc.data[r][0] + v;
    }
}

template<typename op, bool reset, ducks::rs::all RV, ducks::rs::all RS>
__device__ inline void strip_col_reduce(RV &dst, const RS &src, const RV &acc) {
    static_assert(RV::rows == 1 && RV::cols == RS::cols && RV::run == RS::run,
                  "a column reduction gives the strip's row_vec");
    using T = typename RS::T;
    #pragma unroll
    for (int j = 0; j < RS::chunks; j++) {
        #pragma unroll
        for (int e = 0; e < RS::run; e++) {
            T v = src.data[0][j][e];
            #pragma unroll
            for (int r = 1; r < RS::rows; r++) v = op::template op<T>(v, src.data[r][j][e]);
            if constexpr (reset) dst.data[0][j][e] = v;
            else                 dst.data[0][j][e] = op::template op<T>(acc.data[0][j][e], v);
        }
    }
}
}  // namespace detail

#define HK_STRIP_REDUCE(name, opname, kind, VEC)                                   \
    template<ducks::rs::VEC V, ducks::rs::all RS>                                   \
    __device__ static inline void name(V &dst, const RS &src) {                    \
        detail::strip_##kind##_reduce<base_ops::opname, true>(dst, src, dst);      \
    }                                                                              \
    template<ducks::rs::VEC V, ducks::rs::all RS>                                   \
    __device__ static inline void name(V &dst, const RS &src, const V &acc) {      \
        detail::strip_##kind##_reduce<base_ops::opname, false>(dst, src, acc);     \
    }
HK_STRIP_REDUCE(row_sum,  sum, row, cv)
HK_STRIP_REDUCE(row_max,  hk_rmax, row, cv)
HK_STRIP_REDUCE(row_min,  hk_rmin, row, cv)
HK_STRIP_REDUCE(row_prod, mul, row, cv)
HK_STRIP_REDUCE(col_sum,  sum, col, all)
HK_STRIP_REDUCE(col_max,  hk_rmax, col, all)
HK_STRIP_REDUCE(col_min,  hk_rmin, col, all)
HK_STRIP_REDUCE(col_prod, mul, col, all)
#undef HK_STRIP_REDUCE

#define HK_STRIP_REDUCE_ORDERED(name, opname)                                       \
    template<int SEQ, int... OFFS, ducks::rs::cv V, ducks::rs::all RS>                \
    __device__ static inline void name##_ordered(V &dst, const RS &src) {            \
        detail::strip_row_reduce_ordered<base_ops::opname, true, bool(SEQ), OFFS...>(dst, src, dst); \
    }                                                                              \
    template<int SEQ, int... OFFS, ducks::rs::cv V, ducks::rs::all RS>                \
    __device__ static inline void name##_ordered(V &dst, const RS &src, const V &acc) { \
        detail::strip_row_reduce_ordered<base_ops::opname, false, bool(SEQ), OFFS...>(dst, src, acc); \
    }
HK_STRIP_REDUCE_ORDERED(row_sum,  sum)
/// row_sum_ordered of src * times with each product fused into the running sum: detail::strip_row_dot_ordered
template<int SEQ, int... OFFS, ducks::rs::cv V, ducks::rs::all RS>
__device__ static inline void row_sum_times_ordered(V &dst, const RS &src, const RS &times) {
    static_assert(SEQ, "a fused sum of products is left to right");
    detail::strip_row_dot_ordered<true, OFFS...>(dst, src, times, dst);
}
template<int SEQ, int... OFFS, ducks::rs::cv V, ducks::rs::all RS>
__device__ static inline void row_sum_times_ordered(V &dst, const RS &src, const RS &times, const V &acc) {
    static_assert(SEQ, "a fused sum of products is left to right");
    detail::strip_row_dot_ordered<false, OFFS...>(dst, src, times, acc);
}
HK_STRIP_REDUCE_ORDERED(row_max,  hk_rmax)
HK_STRIP_REDUCE_ORDERED(row_min,  hk_rmin)
HK_STRIP_REDUCE_ORDERED(row_prod, mul)
#undef HK_STRIP_REDUCE_ORDERED

/* ----------  broadcasts  ---------- */

template<typename op, ducks::rs::all RS, ducks::rs::cv CV>
__device__ static inline void row_map(RS &dst, const RS &src, const CV &v) {
    static_assert(CV::length == RS::rows, "one value per row of the strip");
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e], v.data[r][0]);
    });
}
template<typename op, ducks::rs::all RS>
__device__ static inline void col_map(RS &dst, const RS &src, const typename RS::row_vec &v) {
    detail::strip_each<RS>([&](int r, int j, int e) {
        dst.data[r][j][e] = op::template op<typename RS::T>(src.data[r][j][e], v.data[0][j][e]);
    });
}

#define HK_STRIP_BROADCAST(name, opname)                                           \
    template<ducks::rs::all RS, ducks::rs::cv CV>                                   \
    __device__ static inline void name##_row(RS &dst, const RS &src, const CV &v) { \
        row_map<base_ops::opname>(dst, src, v);                                     \
    }                                                                              \
    template<ducks::rs::all RS>                                                     \
    __device__ static inline void name##_col(RS &dst, const RS &src,               \
                                             const typename RS::row_vec &v) {      \
        col_map<base_ops::opname, RS>(dst, src, v);                                 \
    }
HK_STRIP_BROADCAST(add, sum)
HK_STRIP_BROADCAST(sub, sub)
HK_STRIP_BROADCAST(mul, mul)
HK_STRIP_BROADCAST(div, div)
#undef HK_STRIP_BROADCAST

/* ----------  fills  ---------- */

namespace detail {
template<bool left, ducks::rs::all RS>
__device__ inline void strip_axis_fill(RS &dst, const RS &src, int col_idx,
                                       const typename RS::T &val) {
    const int lane0 = laneid() * RS::run;
    detail::strip_each<RS>([&](int r, int j, int e) {
        const int c = j*RS::span + lane0 + e;
        const bool hit = left ? c < col_idx : c >= col_idx;
        dst.data[r][j][e] = hit ? val : src.data[r][j][e];
    });
}
}  // namespace detail

/// Columns col_idx and past it become `val`; the rest are copied from `src`.
template<ducks::rs::all RS>
__device__ static inline void right_fill(RS &dst, const RS &src, const int col_idx,
                                         const typename RS::T &val = 0) {
    detail::strip_axis_fill<false>(dst, src, col_idx, val);
}
/// Columns before col_idx become `val`; the rest are copied from `src`.
template<ducks::rs::all RS>
__device__ static inline void left_fill(RS &dst, const RS &src, const int col_idx,
                                        const typename RS::T &val = 0) {
    detail::strip_axis_fill<true>(dst, src, col_idx, val);
}

/* ----------  memory  ---------- */

namespace detail {

using hk_u32x4 = unsigned __attribute__((ext_vector_type(4)));
using hk_u32x2 = unsigned __attribute__((ext_vector_type(2)));

/// RUN elements at `p`, in the widest accesses the run's bytes allow.
template<typename U, int RUN>
__device__ inline void strip_read(U (&run)[RUN], const U *p) {
    constexpr int BYTES = RUN * sizeof(U);
    if constexpr (BYTES % 16 == 0) {
        #pragma unroll
        for (int k = 0; k < BYTES / 16; k++) {
            const hk_u32x4 v = *(reinterpret_cast<const hk_u32x4*>(p) + k);
            __builtin_memcpy((void*)&run[k * 16 / sizeof(U)], &v, 16);
        }
    } else if constexpr (BYTES == 8) {
        const hk_u32x2 v = *reinterpret_cast<const hk_u32x2*>(p);
        __builtin_memcpy((void*)&run[0], &v, 8);
    } else if constexpr (BYTES == 4) {
        const uint32_t v = *reinterpret_cast<const uint32_t*>(p);
        __builtin_memcpy((void*)&run[0], &v, 4);
    } else if constexpr (BYTES == 2) {
        const uint16_t v = *reinterpret_cast<const uint16_t*>(p);
        __builtin_memcpy((void*)&run[0], &v, 2);
    } else {
        static_assert(BYTES == 1, "a run is 1, 2, 4, 8 or a multiple of 16 bytes");
        __builtin_memcpy((void*)&run[0], (const void*)p, 1);
    }
}

template<typename U, int RUN>
__device__ inline void strip_write(U *p, const U (&run)[RUN]) {
    constexpr int BYTES = RUN * sizeof(U);
    if constexpr (BYTES % 16 == 0) {
        #pragma unroll
        for (int k = 0; k < BYTES / 16; k++) {
            hk_u32x4 v;
            __builtin_memcpy(&v, (const void*)&run[k * 16 / sizeof(U)], 16);
            *(reinterpret_cast<hk_u32x4*>(p) + k) = v;
        }
    } else if constexpr (BYTES == 8) {
        hk_u32x2 v; __builtin_memcpy(&v, (const void*)&run[0], 8);
        *reinterpret_cast<hk_u32x2*>(p) = v;
    } else if constexpr (BYTES == 4) {
        uint32_t v; __builtin_memcpy(&v, (const void*)&run[0], 4);
        *reinterpret_cast<uint32_t*>(p) = v;
    } else if constexpr (BYTES == 2) {
        uint16_t v; __builtin_memcpy(&v, (const void*)&run[0], 2);
        *reinterpret_cast<uint16_t*>(p) = v;
    } else {
        static_assert(BYTES == 1, "a run is 1, 2, 4, 8 or a multiple of 16 bytes");
        __builtin_memcpy((void*)p, (const void*)&run[0], 1);
    }
}

/// Past any buffer range this library builds: reads zero, writes nothing.
static constexpr uint32_t HK_STRIP_OOB = 0x80000000u;

/// The descriptor over a strip's rows that are in bounds, from column 0.
template<int ROWS, typename GL>
__device__ inline hk_i32x4 strip_srsrc(const GL &g, const coord<> &idx) {
    using U = typename GL::dtype;
    coord<> row0 = idx;
    row0.c = 0;
    const int left = int(g.rows()) - idx.r;
    const int valid = left <= 0 ? 0 : (left < ROWS ? left : ROWS);
    return hk_srsrc(&g[row0], uint64_t(valid) * uint64_t(g.row_stride()) * sizeof(U));
}

}  // namespace detail

/** @brief Global -> strip. Row r of the strip is row idx.r + r of `src`, from column idx.c. */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void load(RS &dst, const GL &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const U *base = &src[idx] + laneid() * RS::run;
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            U run[RS::run];
            detail::strip_read(run, base + r * stride + j * RS::span);
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                dst.data[r][j][e] = base_types::convertor<T, U>::convert(run[e]);
        }
    }
}

/** @brief load() of PERIOD columns repeated: strip column c is column
 * idx.c + c % PERIOD of the row. Unbounded. */
template<int PERIOD, ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void load_period(RS &dst, const GL &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    static_assert(PERIOD % RS::run == 0, "a period of whole runs");
    const U *base = &src[idx];
    const int lane0 = laneid() * RS::run;
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            U run[RS::run];
            detail::strip_read(run, base + r * stride + (j * RS::span + lane0) % PERIOD);
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                dst.data[r][j][e] = base_types::convertor<T, U>::convert(run[e]);
        }
    }
}

/** @brief Strip -> global, the inverse of load(). */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void store(const GL &dst, const RS &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    U *base = &dst[idx] + laneid() * RS::run;
    const int stride = dst.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            U run[RS::run];
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                run[e] = detail::hk_cvt<U, T>(src.data[r][j][e]);
            detail::strip_write(base + r * stride + j * RS::span, run);
        }
    }
}

/** @brief load(), predicated on the global's extent: rows past its last row
 * and runs at or past its last column read as zero and touch no memory. */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void load_bounded(RS &dst, const GL &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const detail::hk_i32x4 srsrc = detail::strip_srsrc<RS::rows>(src, idx);
    const int lane0 = laneid() * RS::run;
    const int cols_left = int(src.cols()) - idx.c;
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            const int c = j * RS::span + lane0;
            const uint32_t voff = c < cols_left ? uint32_t(r * stride + idx.c + c) * sizeof(U)
                                                : detail::HK_STRIP_OOB;
            U run[RS::run];
            detail::hk_buf_load_bytes<RS::run * sizeof(U)>(srsrc, voff, run);
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                dst.data[r][j][e] = base_types::convertor<T, U>::convert(run[e]);
        }
    }
}

/** @brief store(), predicated as load_bounded(). */
template<ducks::rs::all RS, ducks::gl::all GL>
__device__ inline static void store_bounded(const GL &dst, const RS &src, const coord<> &idx) {
    using T = typename RS::T;
    using U = typename GL::dtype;
    const detail::hk_i32x4 srsrc = detail::strip_srsrc<RS::rows>(dst, idx);
    const int lane0 = laneid() * RS::run;
    const int cols_left = int(dst.cols()) - idx.c;
    const int stride = dst.row_stride();
    #pragma unroll
    for (int r = 0; r < RS::rows; r++) {
        #pragma unroll
        for (int j = 0; j < RS::chunks; j++) {
            const int c = j * RS::span + lane0;
            const uint32_t voff = c < cols_left ? uint32_t(r * stride + idx.c + c) * sizeof(U)
                                                : detail::HK_STRIP_OOB;
            U run[RS::run];
            #pragma unroll
            for (int e = 0; e < RS::run; e++)
                run[e] = detail::hk_cvt<U, T>(src.data[r][j][e]);
            detail::hk_buf_store_bytes<RS::run * sizeof(U)>(srsrc, voff, run);
        }
    }
}

/** @brief Global -> one value per row: entry r is element (idx.r + r, idx.c). */
template<ducks::rs::cv CV, ducks::gl::all GL>
__device__ inline static void load(CV &dst, const GL &src, const coord<> &idx) {
    using U = typename GL::dtype;
    const U *base = &src[idx];
    const int stride = src.row_stride();
    #pragma unroll
    for (int r = 0; r < CV::length; r++)
        dst.data[r][0] = base_types::convertor<typename CV::T, U>::convert(base[r * stride]);
}

/** @brief Entry 0 of a uniform per-row vector -> one element, from lane 0. */
template<ducks::gl::all GL, ducks::rs::cv CV>
__device__ inline static void store_scalar(const GL &dst, const CV &src, const coord<> &idx) {
    using U = typename GL::dtype;
    if (laneid() != 0) return;
    *(U*)&dst[idx] = detail::hk_cvt<U, typename CV::T>(src.data[0][0]);
}

namespace detail {
/// An LDS vector's entries through an LDS pointer: through the generic one
/// the vector is reached by, they are flat_load/flat_store, which wait on
/// both counters and take the flat path.
template<typename U> __device__ inline __attribute__((address_space(3))) U *hk_lds(const U *p) {
    return (__attribute__((address_space(3))) U *)(uint32_t)(uintptr_t)p;
}
}  // namespace detail

/** @brief A per-row vector -> the first CV::length entries of an LDS vector,
 * from lane 0 (every lane holds the same values). */
template<ducks::sv::all SV, ducks::rs::cv CV>
__device__ inline static void store(SV &dst, const CV &src) {
    static_assert(SV::length >= CV::length, "the LDS vector holds every row's value");
    using U = typename SV::dtype;
    if (laneid() == 0) {
        auto *p = detail::hk_lds(&dst.data[0]);
        #pragma unroll
        for (int r = 0; r < CV::length; r++)
            p[r] = detail::hk_cvt<U, typename CV::T>(src.data[r][0]);
    }
}

/** @brief The first CV::length entries of an LDS vector -> a per-row vector, in every lane. */
template<ducks::rs::cv CV, ducks::sv::all SV>
__device__ inline static void load(CV &dst, const SV &src) {
    static_assert(SV::length >= CV::length, "the LDS vector holds every row's value");
    using U = typename SV::dtype;
    const auto *p = detail::hk_lds(&src.data[0]);
    #pragma unroll
    for (int r = 0; r < CV::length; r++) {
        const U u = p[r];
        dst.data[r][0] = base_types::convertor<typename CV::T, U>::convert(u);
    }
}

}  // namespace kittens
