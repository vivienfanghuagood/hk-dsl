// litmus_comm.hip: memory-model litmus tests on gfx950 (MI355X), between two GPUs over xGMI and between two
// XCDs of one GPU -- the ground truth hk.comm's lowering follows (docs/design/comm.md).
//
//   hipcc -O3 --offload-arch=gfx950 -o litmus tools/cdna-probes/litmus_comm.hip
//   ./litmus [ROUNDS [mp|xcd|tear|sb]]      (two visible GPUs for mp / tear / sb)
//
// Which store / publish / poll / read combinations are correct (never fail), and negative controls that show each test can see a violation.
//
// Cache-policy bits (LLVM CPol on gfx940+): SC0 = 1, NT = 2, SC1 = 16. {SC1,SC0} is the scope of a
// load / store: 00 wave, 01 workgroup, 10 device (agent), 11 system.
//
// Tests
//   MP   message passing: writer stores DATA (64 lanes x NW words) then FLAG; reader (which first READ the old
//        DATA, so its caches hold the stale lines) polls FLAG, then reads DATA. A round fails if any word is
//        old. Placement: push (DATA/FLAG in the reader's memory, the writer stores remotely), pull (in the
//        writer's memory, the reader loads remotely), xcd (one GPU, writer and reader workgroups on different
//        XCDs).
//   TEAR single-copy atomicity of 8 B / 16 B stores across the link: writer stores (r, r[, r, r]) in one store,
//        reader loads the same width; a load with unequal words is torn.
//   SB   store buffering: A: x = 1; fence; read y. B: y = 1; fence; read x. (0, 0) = SB outcome.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

typedef __amdgpu_buffer_rsrc_t rsrc_t;
__device__ inline rsrc_t rsrc(void* p) { return __builtin_amdgcn_make_buffer_rsrc(p, 0, 0x7FFFFFF0, 0x00020000); }
__device__ inline long long now() { return __builtin_amdgcn_s_memrealtime(); }   // 100 MHz

template <int AUX> __device__ inline void st(unsigned* p, unsigned v) { __builtin_amdgcn_raw_buffer_store_b32(v, rsrc(p), 0, 0, AUX); }
template <int AUX> __device__ inline unsigned ld(unsigned* p) { return __builtin_amdgcn_raw_buffer_load_b32(rsrc(p), 0, 0, AUX); }
__device__ inline void st_aux(unsigned* p, unsigned v, int aux) {
    switch (aux) { case 0: st<0>(p, v); break; case 1: st<1>(p, v); break; case 2: st<2>(p, v); break; case 16: st<16>(p, v); break; default: st<17>(p, v); }
}
__device__ inline unsigned ld_aux(unsigned* p, int aux) {
    switch (aux) { case 0: return ld<0>(p); case 1: return ld<1>(p); case 2: return ld<2>(p); case 16: return ld<16>(p); default: return ld<17>(p); }
}
// a plain load the compiler cannot CSE with another (buffer-load intrinsics are readonly: two at one address
// with no store between fold into one, which made the first run's plain re-reads return the primed value)
__device__ inline unsigned ld_plain(unsigned* p) {
    unsigned v; asm volatile("global_load_dword %0, %1, off\n s_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory"); return v;
}
__device__ inline void vm_drain() { __builtin_amdgcn_s_waitcnt(0x0F70); }   // vmcnt(0) (gfx9 encoding, others max)

// ---- writer publish modes
enum WMode {
    W_PLAIN = 0,      // data aux 0, flag aux 0, nothing between                         (negative control)
    W_NOWAIT_SYS,     // data aux 17, flag aux 17, no vmcnt wait between                  (negative control?)
    W_DRAIN_SYS,      // data aux 0, vmcnt(0), flag aux 17
    W_SYS_DRAIN_SYS,  // data aux 17, vmcnt(0), flag aux 17                               (dsv41_mono's CM_SYS publish)
    W_DEV_DRAIN_DEV,  // data aux 16, vmcnt(0), flag aux 16                               (CM_DEV publish, same device only)
    W_REL_SYS,        // plain C++ data stores, __hip_atomic_store(flag, release, system) (LLVM memory model)
    W_REL_AGENT,      // plain C++ data stores, __hip_atomic_store(flag, release, agent)
    W_NMODES
};
const char* wname[] = {"plain", "sys-nowait-sys", "plain-drain-sys", "sys-drain-sys", "dev-drain-dev", "c++release-sys", "c++release-agent"};
// ---- reader modes
enum RMode {
    R_PLAIN = 0,     // poll aux 0 (fresh load each time: volatile asm), read aux 0       (negative control)
    R_SYS_PLAIN,     // poll aux 17, read aux 0
    R_SYS_SYS,       // poll aux 17, read aux 17                                          (dsv41_mono)
    R_DEV_DEV,       // poll aux 16, read aux 16
    R_SYS_INV,       // poll aux 17, buffer_inv sc0 sc1, read aux 0
    R_ACQ_SYS,       // __hip_atomic_load(flag, acquire, system), plain C++ reads          (LLVM memory model)
    R_ACQ_AGENT,     // __hip_atomic_load(flag, acquire, agent), plain C++ reads
    R_NMODES
};
const char* rname[] = {"plain", "sys-poll/plain-read", "sys/sys", "dev/dev", "sys-poll/inv/plain", "c++acquire-sys", "c++acquire-agent"};

constexpr int NW = 4;          // words per lane per round
constexpr int SLOTS = 8;       // rounds rotate over SLOTS data blocks (64 lanes x NW words each)
constexpr long long TIMEOUT = 20000;   // 200 us of polling = a timed-out round (a healthy round takes ~3 us)

struct Shared {               // lives in one GPU's memory (placement decides which)
    unsigned flag[64];         // flag[0] used, own cache line
    unsigned ready[64];        // reader -> writer: ready for round r
    unsigned data[SLOTS][64 * NW];
};

__device__ void publish(Shared* s, unsigned* data, unsigned r, int mode) {
    const int lane = threadIdx.x;
    switch (mode) {
    case W_PLAIN:        for (int i = 0; i < NW; ++i) st<0>(&data[i * 64 + lane], r); break;
    case W_NOWAIT_SYS:   for (int i = 0; i < NW; ++i) st<17>(&data[i * 64 + lane], r); break;
    case W_DRAIN_SYS:    for (int i = 0; i < NW; ++i) st<0>(&data[i * 64 + lane], r); vm_drain(); break;
    case W_SYS_DRAIN_SYS:for (int i = 0; i < NW; ++i) st<17>(&data[i * 64 + lane], r); vm_drain(); break;
    case W_DEV_DRAIN_DEV:for (int i = 0; i < NW; ++i) st<16>(&data[i * 64 + lane], r); vm_drain(); break;
    default:             for (int i = 0; i < NW; ++i) data[i * 64 + lane] = r; break;
    }
    __builtin_amdgcn_s_barrier();   // one wave: a no-op ordering point
    if (lane == 0) {
        switch (mode) {
        case W_PLAIN:          st<0>(&s->flag[0], r); break;
        case W_DEV_DRAIN_DEV:  st<16>(&s->flag[0], r); break;
        case W_REL_SYS:        __hip_atomic_store(&s->flag[0], r, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); break;
        case W_REL_AGENT:      __hip_atomic_store(&s->flag[0], r, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); break;
        default:               st<17>(&s->flag[0], r); break;
        }
    }
}

// returns false on timeout
__device__ bool poll(Shared* s, unsigned r, int mode) {
    long long t0 = now();
    unsigned v;
    do {
        switch (mode) {
        case R_PLAIN: asm volatile("global_load_dword %0, %1, off\n s_waitcnt vmcnt(0)" : "=v"(v) : "v"(&s->flag[0]) : "memory"); break;
        case R_DEV_DEV: v = ld<16>(&s->flag[0]); break;
        case R_ACQ_SYS: v = __hip_atomic_load(&s->flag[0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); break;
        case R_ACQ_AGENT: v = __hip_atomic_load(&s->flag[0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT); break;
        default: v = ld<17>(&s->flag[0]); break;
        }
        if (v == r) break;
        __builtin_amdgcn_s_sleep(1);
    } while (now() - t0 < TIMEOUT);
    if (mode == R_SYS_INV) asm volatile("buffer_inv sc0 sc1" ::: "memory");
    return v == r;
}

__device__ unsigned read_data(unsigned* p, int mode) {
    switch (mode) {
    case R_SYS_SYS: return ld<17>(p);
    case R_DEV_DEV: return ld<16>(p);
    case R_PLAIN: case R_SYS_PLAIN: case R_SYS_INV: return ld_plain(p);
    default: return *(volatile unsigned*)p;
    }
}

// one wave each. writer: waits ready == r (in its own memory `wr_ready`, the reader stores it), publishes.
__global__ void k_writer(Shared* s, unsigned* wr_ready, int rounds, int wmode, unsigned* timeouts) {
    for (int r = 1; r <= rounds; ++r) {
        long long t0 = now();
        while (ld<17>(wr_ready) != (unsigned)r) {
            __builtin_amdgcn_s_sleep(1);
            if (now() - t0 > TIMEOUT) { if (threadIdx.x == 0) atomicAdd(timeouts, 1); break; }
        }
        publish(s, s->data[r % SLOTS], r, wmode);
    }
}
__global__ void k_reader(Shared* s, unsigned* wr_ready, int rounds, int rmode, unsigned* res /* [0] fails [1] timeouts [2] stale words */) {
    const int lane = threadIdx.x;
    for (int r = 1; r <= rounds; ++r) {
        unsigned* data = s->data[r % SLOTS];
        // prime: plain loads of the old values into this CU's L1 / this XCD's L2
        unsigned prime = 0;
        for (int i = 0; i < NW; ++i) prime += ld_plain(&data[i * 64 + lane]);
        if (prime == 0xFFFFFFFFu) res[3] = prime;
        vm_drain();
        if (lane == 0) { st<17>(wr_ready, r); }
        bool ok = poll(s, r, rmode);
        unsigned bad = 0;
        if (ok) for (int i = 0; i < NW; ++i) bad += read_data(&data[i * 64 + lane], rmode) != (unsigned)r;
        // wave reduce
        for (int o = 32; o; o >>= 1) bad += __shfl_xor(bad, o);
        if (lane == 0) {
            if (!ok) res[1]++;
            else if (bad) { res[0]++; res[2] += bad; }
        }
    }
}

// xcd variant: both on one GPU, launched as one grid of 16 workgroups; workgroup 0 writes, workgroup 1 reads
// (workgroups are dealt to the 8 XCDs round robin, so 0 and 1 are on different XCDs)
__global__ void k_xcd(Shared* s, unsigned* ready, int rounds, int wmode, int rmode, unsigned* res) {
    if (blockIdx.x == 0) {
        for (int r = 1; r <= rounds; ++r) {
            long long t0 = now();
            while (ld<17>(ready) != (unsigned)r) { __builtin_amdgcn_s_sleep(1); if (now() - t0 > TIMEOUT) { if (threadIdx.x == 0) res[4]++; break; } }
            publish(s, s->data[r % SLOTS], r, wmode);
        }
    } else if (blockIdx.x == 1) {
        const int lane = threadIdx.x;
        for (int r = 1; r <= rounds; ++r) {
            unsigned* data = s->data[r % SLOTS];
            unsigned prime = 0;
            for (int i = 0; i < NW; ++i) prime += ld_plain(&data[i * 64 + lane]);
            if (prime == 0xFFFFFFFFu) res[3] = prime;
            vm_drain();
            if (lane == 0) st<17>(ready, r);
            bool ok = poll(s, r, rmode);
            unsigned bad = 0;
            if (ok) for (int i = 0; i < NW; ++i) bad += read_data(&data[i * 64 + lane], rmode) != (unsigned)r;
            for (int o = 32; o; o >>= 1) bad += __shfl_xor(bad, o);
            if (lane == 0) { if (!ok) res[1]++; else if (bad) { res[0]++; res[2] += bad; } }
        }
    }
}

// ---- TEAR
__global__ void k_tear_writer(unsigned* p, int iters, int width, int aux, unsigned* stop) {
    typedef unsigned v2 __attribute__((ext_vector_type(2)));
    typedef unsigned v4 __attribute__((ext_vector_type(4)));
    const int lane = threadIdx.x;
    unsigned* q = p + lane * 4;
    for (unsigned r = 1; r <= (unsigned)iters; ++r) {
        if (width == 8) { v2 v = {r, r}; if (aux == 17) __builtin_amdgcn_raw_buffer_store_b64(v, rsrc(q), 0, 0, 17); else __builtin_amdgcn_raw_buffer_store_b64(v, rsrc(q), 0, 0, 0); }
        else { v4 v = {r, r, r, r}; if (aux == 17) __builtin_amdgcn_raw_buffer_store_b128(v, rsrc(q), 0, 0, 17); else __builtin_amdgcn_raw_buffer_store_b128(v, rsrc(q), 0, 0, 0); }
    }
    vm_drain();
    if (lane == 0) st<17>(stop, 1);
}
__global__ void k_tear_reader(unsigned* p, int width, unsigned* stop, unsigned* res /* [0] torn [1] loads [2] distinct changes */) {
    typedef unsigned v2 __attribute__((ext_vector_type(2)));
    typedef unsigned v4 __attribute__((ext_vector_type(4)));
    const int lane = threadIdx.x;
    unsigned* q = p + lane * 4;
    unsigned torn = 0, loads = 0, changes = 0, last = 0;
    long long t0 = now();
    while (ld<17>(stop) == 0 && now() - t0 < 500000000LL) {
        for (int k = 0; k < 64; ++k) {
            unsigned a, b, c = 0, d = 0;
            if (width == 8) { v2 v = __builtin_amdgcn_raw_buffer_load_b64(rsrc(q), 0, 0, 17); a = v.x; b = v.y; c = a; d = a; }
            else { v4 v = __builtin_amdgcn_raw_buffer_load_b128(rsrc(q), 0, 0, 17); a = v.x; b = v.y; c = v.z; d = v.w; }
            torn += (a != b) | (a != c) | (a != d);
            changes += a != last; last = a; ++loads;
        }
    }
    atomicAdd(&res[0], torn); atomicAdd(&res[1], loads); atomicAdd(&res[2], changes);
}

// ---- SB: each side stores its x then reads the other's, per round, after a start barrier on a counter.
__global__ void k_sb(unsigned* mine, unsigned* other, unsigned* go_mine, unsigned* go_other, int rounds, int fence, unsigned* out) {
    if (threadIdx.x) return;
    for (int r = 1; r <= rounds; ++r) {
        // start barrier: both sides announce r and wait for the other
        st<17>(go_other, r); vm_drain();
        long long t0 = now();
        while (ld<17>(go_mine) < (unsigned)r && now() - t0 < TIMEOUT) {}
        st<17>(mine, r);
        if (fence == 1) __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
        if (fence == 2) vm_drain();
        unsigned v = ld<17>(other);
        out[r] = v;
    }
}

static void* alloc(int dev, size_t n, int kind) {
    void* p; CK(hipSetDevice(dev));
    if (kind == 0) CK(hipMalloc(&p, n));
    else CK(hipExtMallocWithFlags(&p, n, kind == 1 ? hipDeviceMallocFinegrained : hipDeviceMallocUncached));
    CK(hipMemset(p, 0, n)); CK(hipDeviceSynchronize());
    return p;
}
const char* kname[] = {"coarse", "fine", "uncached"};

int main(int argc, char** argv) {
    int rounds = argc > 1 ? atoi(argv[1]) : 20000;
    std::string only = argc > 2 ? argv[2] : "";
    CK(hipSetDevice(0)); CK(hipDeviceEnablePeerAccess(1, 0));
    CK(hipSetDevice(1)); CK(hipDeviceEnablePeerAccess(0, 0));
    unsigned *res0, *res1;
    CK(hipSetDevice(0)); CK(hipMallocManaged(&res0, 4096));
    CK(hipSetDevice(1)); CK(hipMallocManaged(&res1, 4096));
    hipStream_t s0, s1;
    CK(hipSetDevice(0)); CK(hipStreamCreate(&s0)); CK(hipSetDevice(1)); CK(hipStreamCreate(&s1));

    // ---- MP across GPUs: writer on GPU 0, reader on GPU 1
    if (only.empty() || only == "mp") {
        printf("MP across GPUs (writer GPU0, reader GPU1), %d rounds: fails / timeouts / stale words\n", rounds);
        for (int place = 0; place < 2; ++place)
        for (int kind = 0; kind < 3; kind += 2) {
            // push: Shared in the reader's memory (GPU1); pull: in the writer's memory (GPU0)
            int home = place == 0 ? 1 : 0;
            Shared* sh = (Shared*)alloc(home, sizeof(Shared), kind);
            unsigned* wr_ready = (unsigned*)alloc(0, 256, kind);
            for (int w = 0; w < W_NMODES; ++w) for (int r = 0; r < R_NMODES; ++r) {
                if (w == W_DEV_DRAIN_DEV || r == R_DEV_DEV || w == W_REL_AGENT || r == R_ACQ_AGENT) continue;   // device scope: not for peers
                CK(hipSetDevice(home)); CK(hipMemset(sh, 0, sizeof(Shared))); CK(hipDeviceSynchronize());
                CK(hipSetDevice(0)); CK(hipMemset(wr_ready, 0, 256)); CK(hipDeviceSynchronize());
                memset(res0, 0, 64); memset(res1, 0, 64);
                CK(hipSetDevice(1)); k_reader<<<1, 64, 0, s1>>>(sh, wr_ready, rounds, r, res1);
                CK(hipSetDevice(0)); k_writer<<<1, 64, 0, s0>>>(sh, wr_ready, rounds, w, res0);
                CK(hipStreamSynchronize(s0)); CK(hipSetDevice(1)); CK(hipStreamSynchronize(s1));
                printf("  %-4s %-8s W %-16s R %-20s : %6u %6u %8u%s\n", place ? "pull" : "push", kname[kind], wname[w], rname[r],
                       res1[0], res1[1], res1[2], (res1[0] || res1[1]) ? "   <-- VIOLATION" : "");
                fflush(stdout);
            }
            CK(hipSetDevice(home)); CK(hipFree(sh)); CK(hipSetDevice(0)); CK(hipFree(wr_ready));
        }
    }
    // ---- MP across XCDs of GPU 0
    if (only.empty() || only == "xcd") {
        printf("MP across XCDs (GPU0, workgroups 0 -> 1), %d rounds\n", rounds);
        for (int kind = 0; kind < 3; kind += 2) {
            Shared* sh = (Shared*)alloc(0, sizeof(Shared), kind);
            unsigned* ready = (unsigned*)alloc(0, 256, kind);
            for (int w = 0; w < W_NMODES; ++w) for (int r = 0; r < R_NMODES; ++r) {
                CK(hipSetDevice(0)); CK(hipMemset(sh, 0, sizeof(Shared))); CK(hipMemset(ready, 0, 256)); CK(hipDeviceSynchronize());
                memset(res0, 0, 64);
                k_xcd<<<16, 64, 0, s0>>>(sh, ready, rounds, w, r, res0);
                CK(hipStreamSynchronize(s0));
                printf("  xcd  %-8s W %-16s R %-20s : %6u %6u %8u%s\n", kname[kind], wname[w], rname[r], res0[0], res0[1], res0[2],
                       (res0[0] || res0[1]) ? "   <-- VIOLATION" : "");
                fflush(stdout);
            }
            CK(hipFree(sh)); CK(hipFree(ready));
        }
    }
    // ---- TEAR: writer GPU0 stores into GPU1 memory, reader GPU1 loads locally
    if (only.empty() || only == "tear") {
        printf("TEAR (writer GPU0 remote-stores, reader GPU1 polls), 64 lanes x 1M stores\n");
        for (int kind = 0; kind < 3; kind += 2) for (int width : {8, 16}) for (int aux : {0, 17}) {
            unsigned* p = (unsigned*)alloc(1, 64 * 16, kind);
            unsigned* stop = (unsigned*)alloc(1, 256, kind);
            memset(res1, 0, 64);
            CK(hipSetDevice(1)); k_tear_reader<<<1, 64, 0, s1>>>(p, width, stop, res1);
            CK(hipSetDevice(0)); k_tear_writer<<<1, 64, 0, s0>>>(p, 1000000, width, aux, stop);
            CK(hipStreamSynchronize(s0)); CK(hipSetDevice(1)); CK(hipStreamSynchronize(s1));
            printf("  %-8s %2d B store aux %2d : torn %u of %u loads (%u value changes seen)%s\n", kname[kind], width, aux, res1[0], res1[1], res1[2],
                   res1[0] ? "   <-- TORN" : "");
            fflush(stdout);
            CK(hipFree(p)); CK(hipFree(stop));
        }
    }
    // ---- SB: x in GPU0 memory, y in GPU1 memory; each side stores locally, reads remotely
    if (only.empty() || only == "sb") {
        printf("SB across GPUs, %d rounds: count of (0,0) outcomes (each side read the other's old value)\n", rounds);
        for (int place = 0; place < 2; ++place)
        for (int fence = 0; fence < 3; ++fence) {
            // place 0: each side stores its own variable locally, loads the other's remotely
            // place 1: each side's variable lives on the OTHER GPU: the store is remote (slow), the load local
            unsigned* x = (unsigned*)alloc(place ? 1 : 0, 256, 2); unsigned* y = (unsigned*)alloc(place ? 0 : 1, 256, 2);
            unsigned* g0 = (unsigned*)alloc(0, 256, 2); unsigned* g1 = (unsigned*)alloc(1, 256, 2);
            unsigned *o0, *o1; CK(hipMallocManaged(&o0, 4 * (rounds + 1))); CK(hipMallocManaged(&o1, 4 * (rounds + 1)));
            CK(hipSetDevice(1)); k_sb<<<1, 64, 0, s1>>>(y, x, g1, g0, rounds, fence, o1);
            CK(hipSetDevice(0)); k_sb<<<1, 64, 0, s0>>>(x, y, g0, g1, rounds, fence, o0);
            CK(hipStreamSynchronize(s0)); CK(hipSetDevice(1)); CK(hipStreamSynchronize(s1));
            int both = 0;
            for (int r = 1; r <= rounds; ++r) both += o0[r] < (unsigned)r && o1[r] < (unsigned)r;
            const char* fn[] = {"none", "seq_cst system fence", "vmcnt(0)"};
            printf("  %-28s fence %-22s : %d / %d\n", place ? "remote store, local load" : "local store, remote load", fn[fence], both, rounds);
            fflush(stdout);
        }
    }
    return 0;
}
