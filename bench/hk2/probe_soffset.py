"""Is soffset part of the raw-buffer range check on gfx950 (buffer_load ... lds and plain loads)?"""
import torch

from hk2 import runtime as rt

IR = """
target triple = "amdgcn-amd-amdhsa"
@buf = internal addrspace(3) global [1024 x i8] poison, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1), i16, i64, i32)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8), i32, i32, i32)
define amdgpu_kernel void @probe(ptr addrspace(1) %src, ptr addrspace(1) %out, i32 %soff) #0 {
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  ; num_records = 512 bytes: lanes 0..31 are in range with soffset 0
  %rs = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %src, i16 0, i64 512, i32 131072)
  %vo = mul i32 %t, 16
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rs, ptr addrspace(3) @buf, i32 16, i32 %vo, i32 %soff, i32 0, i32 0)
  %x = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %rs, i32 %vo, i32 %soff, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 3952)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %q = getelementptr <4 x i32>, ptr addrspace(3) @buf, i32 %t
  %y = load <4 x i32>, ptr addrspace(3) %q
  %o1 = getelementptr <4 x i32>, ptr addrspace(1) %out, i32 %t
  store <4 x i32> %y, ptr addrspace(1) %o1
  %o2 = getelementptr <4 x i32>, ptr addrspace(1) %out, i32 64
  %o3 = getelementptr <4 x i32>, ptr addrspace(1) %o2, i32 %t
  store <4 x i32> %x, ptr addrspace(1) %o3
  ret void
}
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)
declare void @llvm.amdgcn.s.barrier()
attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }
"""
co = rt.compile_ir(IR, "gfx950")
k = rt.CodeObject(co.hsaco, "probe")
src = torch.arange(4096, dtype=torch.int32, device="cuda") + 1
for soff in (0, 256):
    out = torch.zeros(128, 4, dtype=torch.int32, device="cuda")
    k.launch((1, 1, 1), 64, rt.pack_args([("p", src.data_ptr()), ("p", out.data_ptr()), ("i", soff)]), 0,
             torch.cuda.current_stream().cuda_stream, 0)
    torch.cuda.synchronize()
    lds_nz = int((out[:64, 0] != 0).sum())
    reg_nz = int((out[64:, 0] != 0).sum())
    print(f"soffset {soff}: lanes with data -- DMA {lds_nz}, plain load {reg_nz} (32 = range-checked with soffset, "
          f"{32 if soff == 0 else 16} expected if soffset counts)")
