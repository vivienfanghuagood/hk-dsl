"""Does LLVM skip its LDS-DMA wait before a ds_read whose address is dynamic, when alias.scope metadata says the
read and the DMA do not alias? (A dynamic double buffer: the DMA fills slot 1-p while this iteration reads p.)"""
import sys

from hk2 import runtime as rt

TPL = """
target triple = "amdgcn-amd-amdhsa"
@lds = internal addrspace(3) global [16384 x i8] poison, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1), i16, i64, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
define amdgpu_kernel void @probe(ptr addrspace(1) %src, ptr addrspace(1) %out, i32 %par) #0 {
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  %rs = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %src, i16 0, i64 65536, i32 131072)
  %oth = sub i32 1, %par
  %ob = mul i32 %oth, 8192
  %dst = getelementptr i8, ptr addrspace(3) @lds, i32 %ob
  %vo = mul i32 %t, 16
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rs, ptr addrspace(3) %dst, i32 16, i32 %vo, i32 0, i32 0, i32 0) META_DMA
  %cb = mul i32 %par, 8192
  %t4 = mul i32 %t, 16
  %ro = add i32 %cb, %t4
  %rp = getelementptr i8, ptr addrspace(3) @lds, i32 %ro
  %x = load <4 x i32>, ptr addrspace(3) %rp, align 16 META_RD
  %op = getelementptr <4 x i32>, ptr addrspace(1) %out, i32 %t
  store <4 x i32> %x, ptr addrspace(1) %op
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }
!0 = distinct !{!0}
!1 = distinct !{!1, !0}
!2 = !{!1}
!3 = distinct !{!3, !0}
!5 = !{!3}
"""

for name, dma, rd in (("no metadata", "", ""),
                      ("alias scopes", ", !alias.scope !5, !noalias !2", ", !alias.scope !2, !noalias !5")):
    ir = TPL.replace("META_DMA", dma).replace("META_RD", rd)
    asm = rt.compile_ir(ir, "gfx950").asm
    body = asm.split("probe:")[1].split("s_endpgm")[0]
    seq = [l.strip() for l in body.splitlines() if any(k in l for k in ("buffer_load", "ds_read", "s_waitcnt"))]
    print(f"{name}:")
    for l in seq:
        print("   ", l)
sys.stdout.flush()
