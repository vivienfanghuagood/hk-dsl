"""Probe gfx950's ds_read_b64_tr_b16: which LDS elements each lane receives, for a given per-lane address.

LDS holds a 64x64 matrix of i16 with element (r, c) = r*64 + c (row-major). Each lane passes the address of
(row = ADDR_ROW(lane), col = ADDR_COL(lane)); the four i16 it receives are decoded back to (r, c).
"""
import torch

from hk2 import runtime as rt

IR = """
target triple = "amdgcn-amd-amdhsa"
@lds = internal addrspace(3) global [4096 x i16] poison, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare <4 x i16> @llvm.amdgcn.ds.read.tr16.b64.v4i16(ptr addrspace(3))
define amdgpu_kernel void @probe(ptr addrspace(1) %out, ptr addrspace(1) %rows, ptr addrspace(1) %cols) #0 {
entry:
  %t = call i32 @llvm.amdgcn.workitem.id.x()
  br label %fill
fill:
  %j = phi i32 [0, %entry], [%jn, %fill]
  %base = mul i32 %t, 64
  %idx = add i32 %base, %j
  %v = trunc i32 %idx to i16
  %p = getelementptr i16, ptr addrspace(3) @lds, i32 %idx
  store i16 %v, ptr addrspace(3) %p
  %jn = add i32 %j, 1
  %more = icmp slt i32 %jn, 64
  br i1 %more, label %fill, label %go
go:
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %pr = getelementptr i32, ptr addrspace(1) %rows, i32 %t
  %pc = getelementptr i32, ptr addrspace(1) %cols, i32 %t
  %r = load i32, ptr addrspace(1) %pr
  %c = load i32, ptr addrspace(1) %pc
  %r64 = mul i32 %r, 64
  %e = add i32 %r64, %c
  %ap = getelementptr i16, ptr addrspace(3) @lds, i32 %e
  %x = call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64.v4i16(ptr addrspace(3) %ap)
  %o4 = mul i32 %t, 4
  %op = getelementptr i16, ptr addrspace(1) %out, i32 %o4
  store <4 x i16> %x, ptr addrspace(1) %op
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }
"""


def run(rows, cols):
    co = rt.compile_ir(IR, "gfx950")
    k = rt.CodeObject(co.hsaco, "probe")
    out = torch.zeros(64 * 4, dtype=torch.int16, device="cuda")
    r = torch.tensor(rows, dtype=torch.int32, device="cuda")
    c = torch.tensor(cols, dtype=torch.int32, device="cuda")
    k.launch((1, 1, 1), 64, rt.pack_args([("p", out.data_ptr()), ("p", r.data_ptr()), ("p", c.data_ptr())]), 0,
             torch.cuda.current_stream().cuda_stream, 0)
    torch.cuda.synchronize()
    o = out.view(64, 4).cpu().int()
    return [[(int(x) // 64, int(x) % 64) for x in o[l]] for l in range(64)]


# lane l points at row l % 16, column 4 * (l // 16): a 16 x 16 block, four columns per lane
res = run([l % 16 for l in range(64)], [4 * (l // 16) for l in range(64)])
for l in list(range(0, 18)) + [31, 32, 48, 63]:
    print(f"lane {l:2d} -> {res[l]}")
