"""Probe gfx950 buffer_load ... lds (LDS DMA): where each lane's 16 bytes land, and which waits LLVM inserts
when a DMA into one LDS buffer is followed by reads of another."""
import re

import torch

from hk2 import runtime as rt

IR = """
target triple = "amdgcn-amd-amdhsa"
@buf0 = internal addrspace(3) global [8192 x i8] poison, align 16
@buf1 = internal addrspace(3) global [8192 x i8] poison, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1), i16, i64, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
define amdgpu_kernel void @probe(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %perm) #0 {
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  %rs = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %src, i16 0, i64 65536, i32 131072)
  %pp = getelementptr i32, ptr addrspace(1) %perm, i32 %t
  %pv = load i32, ptr addrspace(1) %pp
  %vo = mul i32 %pv, 16
  ; DMA into buf1 (second instruction at +1024 with an immediate offset of 0 and a moved base)
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rs, ptr addrspace(3) @buf1, i32 16, i32 %vo, i32 0, i32 0, i32 0)
  %b1b = getelementptr i8, ptr addrspace(3) @buf1, i32 1024
  %vo2 = add i32 %vo, 1024
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rs, ptr addrspace(3) %b1b, i32 16, i32 %vo2, i32 0, i32 0, i32 0), !alias.scope !5, !noalias !2
  ; an unrelated read of buf0 after the DMA (a wavefront fence keeps it there)
  fence syncscope("wavefront") acquire
  %q0 = getelementptr i32, ptr addrspace(3) @buf0, i32 %t
  %x0 = load volatile i32, ptr addrspace(3) %q0, !alias.scope !2, !noalias !5
  %o0 = getelementptr i32, ptr addrspace(1) %out, i32 1024
  %o0t = getelementptr i32, ptr addrspace(1) %o0, i32 %t
  store i32 %x0, ptr addrspace(1) %o0t
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  ; read buf1 back linearly: 2048 bytes = 512 dwords, 8 per lane
  br label %copy
copy:
  %j = phi i32 [0, %0], [%jn, %copy]
  %idx0 = mul i32 %j, 64
  %idx = add i32 %idx0, %t
  %q = getelementptr i32, ptr addrspace(3) @buf1, i32 %idx
  %x = load i32, ptr addrspace(3) %q
  %o = getelementptr i32, ptr addrspace(1) %out, i32 %idx
  store i32 %x, ptr addrspace(1) %o
  %jn = add i32 %j, 1
  %more = icmp slt i32 %jn, 8
  br i1 %more, label %copy, label %done
done:
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }
!0 = distinct !{!0}
!1 = distinct !{!1, !0}
!2 = !{!1}
!3 = distinct !{!3, !0}
!5 = !{!3}
"""

co = rt.compile_ir(IR, "gfx950")
body = co.asm.split("probe:")[1] if "probe:" in co.asm else co.asm
for ln in body.splitlines():
    if re.search(r"buffer_load|ds_read|ds_load|s_waitcnt|s_barrier|m0", ln):
        print("   ", ln.strip())
k = rt.CodeObject(co.hsaco, "probe")
src = torch.arange(16384, dtype=torch.int32, device="cuda")  # dword i holds i
out = torch.full((2048,), -1, dtype=torch.int32, device="cuda")
perm = torch.tensor([(l * 5) % 64 for l in range(64)], dtype=torch.int32, device="cuda")  # lane l loads chunk 5l%64
k.launch((1, 1, 1), 64, rt.pack_args([("p", src.data_ptr()), ("p", out.data_ptr()), ("p", perm.data_ptr())]), 0,
         torch.cuda.current_stream().cuda_stream, 0)
torch.cuda.synchronize()
o = out[:512].view(128, 4).cpu()
print("LDS 16-byte slot -> source chunk (dword/4):")
print([int(o[s, 0]) // 4 for s in range(0, 70)])
ok = all(int(o[s, 0]) // 4 == (s * 5) % 64 for s in range(64)) and \
    all(int(o[64 + s, 0]) // 4 == 64 + (s * 5) % 64 for s in range(64))
print("slot l of instruction i holds what lane l loaded:", ok)
