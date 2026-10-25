// GPU "electric fence" for torch (CUDAPluggableAllocator): every allocation ends flush against the
// end of its own VMM mapping (to FENCE_ALIGN bytes), and the granule after it is reserved but never
// mapped. A kernel that reads or writes past the end of any tensor faults at once, deterministically,
// instead of silently touching the caching allocator's neighbour.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace {
struct Rec { void* base; size_t reserved; size_t mapped; hipMemGenericAllocationHandle_t h; };
std::mutex mu;
std::unordered_map<void*, Rec> live;
size_t align_of() { const char* e = getenv("FENCE_ALIGN"); return e ? strtoull(e, nullptr, 0) : 256; }
#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "fence: %s -> %s\n", #x, hipGetErrorString(_e)); abort(); } } while (0)
}

extern "C" void* fence_malloc(ssize_t size, int device, hipStream_t) {
    hipMemAllocationProp p = {};
    p.type = hipMemAllocationTypePinned;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = device;
    size_t g = 0;
    CK(hipMemGetAllocationGranularity(&g, &p, hipMemAllocationGranularityMinimum));
    size_t a = align_of(), need = ((size_t)(size > 0 ? size : 1) + a - 1) / a * a;
    size_t mapped = (need + g - 1) / g * g;
    Rec r{nullptr, mapped + g, mapped, {}};
    CK(hipMemAddressReserve(&r.base, r.reserved, g, nullptr, 0));
    CK(hipMemCreate(&r.h, mapped, &p, 0));
    CK(hipMemMap(r.base, mapped, 0, r.h, 0));
    hipMemAccessDesc d = {};
    d.location = p.location;
    d.flags = hipMemAccessFlagsProtReadWrite;
    CK(hipMemSetAccess(r.base, mapped, &d, 1));
    void* user = (char*)r.base + mapped - need;
    std::lock_guard<std::mutex> l(mu);
    live[user] = r;
    return user;
}

extern "C" void fence_free(void* ptr, ssize_t, int, hipStream_t stream) {
    Rec r;
    {
        std::lock_guard<std::mutex> l(mu);
        auto it = live.find(ptr);
        if (it == live.end()) { fprintf(stderr, "fence: free of unknown %p\n", ptr); abort(); }
        r = it->second;
        live.erase(it);
    }
    CK(hipStreamSynchronize(stream));   // the tensor's last use may still be in flight
    CK(hipMemUnmap(r.base, r.mapped));
    CK(hipMemRelease(r.h));
    CK(hipMemAddressFree(r.base, r.reserved));
}
