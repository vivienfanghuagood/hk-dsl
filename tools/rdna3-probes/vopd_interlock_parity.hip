// vopd_interlock_parity.hip -- gfx11's VOPD interlock hazard, in the window that broke hk's strip RMSNorm.
//
// On gfx11.0 / gfx11.5, when X's and Y's SRC0, or their VSRC1, are distinct VGPRs with the same
// register-number parity, VOPD destination-buffer forwarding drops the interlock and an operand
// written just before can be read stale (llvm/llvm-project#220348, scoped by #225880). LLVM before
// that fix -- every ROCm release so far -- requires only different banks (the number mod 4).
//
// The window is AMD clang 23's (TheRock 7.14) for strip_addrmsw w16: v0 assembled by two v_mov_b16
// partial writes, three more 16-bit moves, the compiler's s_delay_alu, then
//   v_dual_add_f32 v0, v14, v22 :: v_dual_lshlrev_b32 v53, 16, v0      (VSRC1 v22 / v0: same parity)
// Each variant keeps the window and changes only which registers the pair reads or writes. It runs under
// a partial EXEC (ACTIVE_LANES of 32 run the asm) and checks Y's result, old v0 << 16, in every active lane.
//
//   hipcc --offload-arch=gfx1100 -O2 -DACTIVE_LANES=16 vopd_interlock_parity.hip -o vopd_interlock_parity
//   ./vopd_interlock_parity [launches]
//
// AMD Radeon Pro W7900D (gfx1100), TheRock 7.14 hipcc, 100 launches x 2^20 lanes, launches with a
// wrong lane:
//   ACTIVE_LANES   A: Y reads X's dst,   B: Y reads X's dst,   C: no X/Y dependency,   D: no dependency,
//                     VSRC1 same parity     parities differ      VSRC1 same parity       parities differ
//   16                  88-93                 0                    91                      0
//    1                  88-92                 0                    90                      0
//   32 (full EXEC)        0                   0                     0                      0
// On an AMD Radeon AI PRO R9700 (gfx1201), which has no such hazard, nothing fails.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef ACTIVE_LANES
#define ACTIVE_LANES 16
#endif

#define K(NAME, WINDOW)                                                                            \
  __global__ void NAME(const unsigned* a, const unsigned* b, unsigned* o, unsigned* o2) {          \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                                 \
    unsigned x = a[i], y = b[i], r = 0, r2 = 0;                                                    \
    if ((threadIdx.x & 31) < ACTIVE_LANES)                                                         \
      asm volatile("v_mov_b32 v22, %2\n v_mov_b32 v14, %3\n v_mov_b32 v23, %3\n"                   \
                   "v_mov_b32 v25, %2\n v_mov_b32 v15, %3\n" WINDOW                                \
                   "v_mov_b32 %0, v53\n v_mov_b32 %1, v0\n"                                        \
                   : "=v"(r), "=v"(r2)                                                             \
                   : "v"(x), "v"(y)                                                                \
                   : "v0", "v14", "v15", "v22", "v23", "v25", "v53");                              \
    o[i] = r;                                                                                      \
    o2[i] = r2;                                                                                    \
  }

// v<A> assembled from v22's halves, v22 shifted, three more 16-bit moves, the compiler's delay, then PAIR
#define WINDOW(A, PAIR)                                                                            \
  "v_mov_b16_e32 " A ".l, v22.h\n v_mov_b16_e32 " A ".h, v22.l\n v_lshlrev_b32_e32 v22, 16, v22\n"   \
  "v_mov_b16_e32 v15.h, v23.l\n v_mov_b16_e32 v23.l, v25.h\n v_mov_b16_e32 v23.h, v25.l\n"          \
  "s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)\n" PAIR "\n"

// A: the compiler's pair. Y reads X's destination; VSRC1 v22 / v0, both even.
K(k_a, WINDOW("v0", "v_dual_add_f32 v0, v14, v22 :: v_dual_lshlrev_b32 v53, 16, v0"))
// B: the same with v0 and v53 swapped. Y still reads X's destination; VSRC1 v22 / v53 differ.
K(k_b, WINDOW("v53", "v_dual_add_f32 v53, v14, v22 :: v_dual_lshlrev_b32 v0, 16, v53"))
// C: X writes v14 instead. X and Y are independent; VSRC1 v22 / v0, both even.
K(k_c, WINDOW("v0", "v_dual_add_f32 v14, v14, v22 :: v_dual_lshlrev_b32 v53, 16, v0"))
// D: as C with X reading v25. Independent; VSRC1 v25 / v0 differ.
K(k_d, WINDOW("v0", "v_dual_add_f32 v14, v14, v25 :: v_dual_lshlrev_b32 v53, 16, v0"))
K(k_a2, WINDOW("v0", "v_dual_add_f32 v0, v14, v22 :: v_dual_lshlrev_b32 v53, 16, v0"))
K(k_b2, WINDOW("v53", "v_dual_add_f32 v53, v14, v22 :: v_dual_lshlrev_b32 v0, 16, v53"))

int main(int argc, char** argv) {
  const int n = 1 << 20, reps = argc > 1 ? atoi(argv[1]) : 100;
  std::vector<unsigned> ha(n), hb(n), want(n), got(n), got2(n);
  srand(1);
  for (int i = 0; i < n; i++) {
    ha[i] = (unsigned)rand() * 2654435761u;
    hb[i] = 0x3f800000u + (rand() & 0xffff);
    want[i] = (ha[i] >> 16) << 16;  // the assembled register, (v22.l << 16) | v22.h, shifted left by 16
  }
  unsigned *a, *b, *o, *o2;
  if (hipMalloc(&a, n * 4) || hipMalloc(&b, n * 4) || hipMalloc(&o, n * 4) || hipMalloc(&o2, n * 4) ||
      hipMemcpy(a, ha.data(), n * 4, hipMemcpyHostToDevice) ||
      hipMemcpy(b, hb.data(), n * 4, hipMemcpyHostToDevice))
    return 2;
  void (*ks[])(const unsigned*, const unsigned*, unsigned*, unsigned*) = {
      k_a, k_b, k_c, k_d, k_a2, k_b2};
  const char* names[] = {"A Y reads X dst, VSRC1 v22/v0", "B Y reads X dst, VSRC1 v22/v53",
                         "C independent, VSRC1 v22/v0", "D independent, VSRC1 v25/v0", "A again", "B again"};
  printf("%d of 32 lanes active, %d launches of %d lanes\n", ACTIVE_LANES, reps, n);
  for (int v = 0; v < 6; v++) {
    long launches = 0, lanes = 0;
    for (int r = 0; r < reps; r++) {
      if (hipMemset(o, 0xff, n * 4)) return 2;
      ks[v]<<<n / 256, 256>>>(a, b, o, o2);
      if (hipMemcpy(got.data(), o, n * 4, hipMemcpyDeviceToHost) ||
          hipMemcpy(got2.data(), o2, n * 4, hipMemcpyDeviceToHost))
        return 2;
      long wrong = 0;
      for (int i = 0; i < n; i++)
        // Y's result: v53, or v0 where B swaps them
        wrong += (i & 31) < ACTIVE_LANES && (v == 1 || v == 5 ? got2[i] : got[i]) != want[i];
      lanes += wrong;
      launches += wrong > 0;
    }
    printf("%-32s launches wrong %ld/%d, lanes wrong %ld\n", names[v], launches, reps, lanes);
  }
  return 0;
}
