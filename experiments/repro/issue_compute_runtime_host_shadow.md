# Summary

On Arc Pro B70 (Battlemage G31), allocating device USM reduces system-wide host
`MemAvailable` at roughly 1:1 with the amount allocated, while the allocating
process's own RSS stays flat. The host cost appears at **allocation** time — no
device-side access to the memory is required — and it is returned on `free`.

For a large-model inference workload with two cards loaded, this becomes a hard
host-RAM ceiling: we lose ~24 GiB of host RAM to allocations that live in VRAM,
on a 59.44 GiB box, and the machine swaps while serving.

I could not find the duplication in compute-runtime itself (details and source
trace below — device USM looks correctly host-inaccessible and un-CPU-mapped), so
this is most likely below the UMD in the `xe` KMD / TTM BO backing path. Filing
here because this is where Level Zero/NEO behaviour on Intel GPUs gets triaged,
and because the source trace may save whoever picks this up a step. Happy to
re-file or cross-file wherever you prefer.

# Version

- NEO / compute-runtime: `intel-opencl-icd`, `libze-intel-gpu1`, `intel-ocloc` all `26.27.39122.12-1~26.04~ppa1`
- Level Zero loader: `libze1` / `libze-dev` `1.28.6-1~26.04~ppa1`
- SYCL device driver version string: `1.15.39122+12`
- oneAPI: 2026.1

# Environment

- GPU: 2x `Intel Corporation Battlemage G31 [8086:e223]` (Arc Pro B70), `global_mem_size` 31.89 GiB each
- Kernel: `7.0.0-30-generic`, Ubuntu 26.04 LTS, `xe` KMD
- Host RAM: `MemTotal` 59.44 GiB
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` (with V2 a plain `queue::memset` on device USM segfaults on this setup; unrelated to this report but needed to run the reproducer)

# Steps to reproduce

Standalone ~130-line SYCL program, no dependencies beyond SYCL. It allocates
device USM in chunks and samples `/proc/meminfo` and `/proc/self/status` after
each chunk, so process-local and system-wide effects are separated.

<details>
<summary><code>device_usm_host_shadow.cpp</code></summary>

```cpp
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr double kKiBPerGiB = 1024.0 * 1024.0;

struct HostMem {
    long mem_total = -1;
    long mem_free = -1;
    long mem_available = -1;
    long vm_rss = -1;
    long vm_hwm = -1;
};

long scan_kv(const char *path, const char *key) {
    std::ifstream f(path);
    if (!f) return -1;
    const std::size_t klen = std::strlen(key);
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, klen, key) != 0) continue;
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        return std::strtol(line.c_str() + colon + 1, nullptr, 10);
    }
    return -1;
}

HostMem sample() {
    HostMem m;
    m.mem_total = scan_kv("/proc/meminfo", "MemTotal:");
    m.mem_free = scan_kv("/proc/meminfo", "MemFree:");
    m.mem_available = scan_kv("/proc/meminfo", "MemAvailable:");
    m.vm_rss = scan_kv("/proc/self/status", "VmRSS:");
    m.vm_hwm = scan_kv("/proc/self/status", "VmHWM:");
    return m;
}

double gib(long kib) { return static_cast<double>(kib) / kKiBPerGiB; }

void report(const char *tag, const HostMem &base, const HostMem &now, double dev_gib) {
    std::printf("%-22s dev_usm=%6.2f GiB | MemAvailable=%7.2f GiB (d %+7.2f) | "
                "MemFree=%7.2f GiB (d %+7.2f) | VmRSS=%6.2f GiB (d %+6.2f)\n",
                tag, dev_gib, gib(now.mem_available),
                gib(now.mem_available) - gib(base.mem_available), gib(now.mem_free),
                gib(now.mem_free) - gib(base.mem_free), gib(now.vm_rss),
                gib(now.vm_rss) - gib(base.vm_rss));
    std::fflush(stdout);
}

long arg_long(int argc, char **argv, const char *flag, long fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::strtol(argv[i + 1], nullptr, 10);
    return fallback;
}

bool arg_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

}  // namespace

int main(int argc, char **argv) {
    const long total_gib = arg_long(argc, argv, "--gib", 16);
    const long chunk_gib = arg_long(argc, argv, "--chunk", 2);
    const long dev_index = arg_long(argc, argv, "--device", 0);
    const bool touch = !arg_flag(argc, argv, "--no-touch");

    std::vector<sycl::device> gpus;
    for (const auto &d : sycl::device::get_devices())
        if (d.is_gpu()) gpus.push_back(d);
    if (gpus.empty()) { std::fprintf(stderr, "no SYCL GPU devices found\n"); return 2; }

    sycl::queue q{gpus[static_cast<std::size_t>(dev_index)]};
    const auto &dev = q.get_device();

    std::printf("device[%ld]      : %s\n", dev_index,
                dev.get_info<sycl::info::device::name>().c_str());
    std::printf("driver         : %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("global_mem_size: %.2f GiB\n",
                static_cast<double>(dev.get_info<sycl::info::device::global_mem_size>())
                    / (1024.0 * 1024 * 1024));
    std::printf("plan           : %ld GiB device USM in %ld GiB chunks, touch=%s\n\n",
                total_gib, chunk_gib, touch ? "yes" : "no");

    const HostMem base = sample();
    std::printf("MemTotal       : %.2f GiB\n", gib(base.mem_total));
    report("baseline", base, base, 0.0);

    std::vector<void *> chunks;
    const std::size_t chunk_bytes = static_cast<std::size_t>(chunk_gib) * (1ull << 30);
    double allocated_gib = 0.0;

    for (long done = 0; done + chunk_gib <= total_gib; done += chunk_gib) {
        void *p = sycl::malloc_device(chunk_bytes, q);
        if (p == nullptr) {
            std::printf("\nmalloc_device failed at %.2f GiB -- stopping\n", allocated_gib);
            break;
        }
        chunks.push_back(p);
        allocated_gib += static_cast<double>(chunk_gib);
        if (touch) q.memset(p, 0xA5, chunk_bytes).wait();
        char tag[32];
        std::snprintf(tag, sizeof(tag), "after +%ld GiB", chunk_gib);
        report(tag, base, sample(), allocated_gib);
    }

    std::printf("\n");
    for (void *p : chunks) sycl::free(p, q);
    report("after free", base, sample(), 0.0);
    return 0;
}
```

</details>

Build and run:

```
icpx -fsycl -std=gnu++17 -O2 -Wall -Wextra -o device_usm_host_shadow device_usm_host_shadow.cpp
export SYCL_UR_USE_LEVEL_ZERO_V2=0
./device_usm_host_shadow --gib 8 --chunk 2 --device 1
./device_usm_host_shadow --gib 8 --chunk 2 --device 0 --no-touch
```

# Observed behavior

**Touched allocation, device 1 — `MemAvailable` tracks the allocation almost exactly, `VmRSS` does not move:**

```
device[1]      : Intel(R) Arc(TM) Pro B70 Graphics
driver         : 1.15.39122+12
global_mem_size: 31.89 GiB
plan           : 8 GiB device USM in 2 GiB chunks, touch=yes

MemTotal       : 59.44 GiB
baseline               dev_usm=  0.00 GiB | MemAvailable=  35.86 GiB (d   +0.00) | MemFree=  14.77 GiB (d   +0.00) | VmRSS=  0.15 GiB (d  +0.00)
after +2 GiB           dev_usm=  2.00 GiB | MemAvailable=  33.86 GiB (d   -2.00) | MemFree=  13.37 GiB (d   -1.39) | VmRSS=  0.15 GiB (d  +0.00)
after +2 GiB           dev_usm=  4.00 GiB | MemAvailable=  31.86 GiB (d   -4.00) | MemFree=  12.42 GiB (d   -2.35) | VmRSS=  0.15 GiB (d  +0.00)
after +2 GiB           dev_usm=  6.00 GiB | MemAvailable=  29.48 GiB (d   -6.38) | MemFree=  10.78 GiB (d   -3.99) | VmRSS=  0.15 GiB (d  +0.00)
after +2 GiB           dev_usm=  8.00 GiB | MemAvailable=  27.84 GiB (d   -8.02) | MemFree=  10.04 GiB (d   -4.73) | VmRSS=  0.15 GiB (d  +0.00)

after free             dev_usm=  0.00 GiB | MemAvailable=  35.83 GiB (d   -0.03) | MemFree=  18.03 GiB (d   +3.26) | VmRSS=  0.15 GiB (d  +0.00)
```

**Untouched allocation, device 0 — the host cost appears without any device access to the memory:**

```
plan           : 8 GiB device USM in 2 GiB chunks, touch=no

baseline               dev_usm=  0.00 GiB | MemAvailable=  34.05 GiB (d   +0.00) | MemFree=   9.27 GiB (d   +0.00) | VmRSS=  0.15 GiB (d  +0.00)
after +2 GiB           dev_usm=  2.00 GiB | MemAvailable=  33.86 GiB (d   -0.19) | MemFree=   9.71 GiB (d   +0.45) | VmRSS=  0.15 GiB (d  -0.00)
after +2 GiB           dev_usm=  4.00 GiB | MemAvailable=  31.83 GiB (d   -2.21) | MemFree=   8.61 GiB (d   -0.66) | VmRSS=  0.15 GiB (d  -0.00)
after +2 GiB           dev_usm=  6.00 GiB | MemAvailable=  29.86 GiB (d   -4.19) | MemFree=   7.83 GiB (d   -1.43) | VmRSS=  0.15 GiB (d  -0.00)
after +2 GiB           dev_usm=  8.00 GiB | MemAvailable=  27.85 GiB (d   -6.20) | MemFree=   6.66 GiB (d   -2.61) | VmRSS=  0.15 GiB (d  -0.00)
```

Summary of the three properties that seem diagnostic:

| property | observation |
|---|---|
| scaling | `MemAvailable` falls ~1:1 with device USM allocated (8 GiB → −8.02 GiB) |
| process RSS | flat at 0.15 GiB throughout — not a userspace staging copy |
| requires device access? | no — an untouched allocation still costs ~6.2 GiB of 8 GiB |
| both cards | reproduces on device 0 and device 1 independently |
| released on free | yes, returns to baseline (−0.03 GiB) |

There is measurement noise in these runs because an unrelated training job was
resident (visible in the `MemFree` column and in the first untouched chunk); the
`MemAvailable` slope and the flat `VmRSS` are the stable signals.

# Expected behavior

Device USM backed by VRAM should not consume host `MemAvailable` proportional to
its size. I would expect approximately zero host-RAM cost for an untouched
device allocation.

# What I was able to rule out, and where I think it lives

This does **not** look like NEO/UMD staging. At `32a2e4c2b0`:

- Device USM is created host-inaccessible with no CPU allocation:
  `shared/source/memory_manager/unified_memory_manager.cpp:570-571`
  (`unifiedMemoryProperties.flags.isHostInaccessibleAllocation = true;`) and
  `:607-608` (`allocData.cpuAllocation = nullptr;`).
  (Note for accuracy: the allocation type here resolves to
  `AllocationType::buffer` at `:1332-1351`, not `svmGpu`, unless a caller
  explicitly supplies a requested type at `:1347-1348`.)
- The Linux device pool creates a region GEM BO
  (`shared/source/os_interface/linux/drm_memory_manager.cpp:3018-3021`,
  `createBufferObjectInMemoryRegion`) and does not CPU-map it except when
  `allocationData.flags.requiresCpuAccess` (`:2832-2834`). There is a separate
  unconditional lock for `AllocationType::writeCombined` at `:2815-2830`, which
  is not the path here.
- Region selection / GEM-create dispatch:
  `shared/source/os_interface/linux/memory_info.cpp:121,152-162` and
  `shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp:748-749,762`
  (placement bitmask + `cpu_caching` + `DrmIoctl::gemCreate`).

Given process RSS stays flat, the duplicate is not in the allocating process's
address space at all, which points at kernel-side BO backing rather than
anything compute-runtime does.

`NEO_LOCAL_MEMORY_ALLOCATION_MODE` is not an available mitigation on this
platform, for two independent reasons:

- it is documented Windows-only —
  `level_zero/doc/experimental_extensions/LOCAL_MEMORY_ALLOCATION_MODE.md:11`
  ("At the moment this is supported on Windows only."), and the only production
  read of the flag is WDDM
  (`shared/source/os_interface/windows/wddm_memory_manager.cpp:65`);
- BMG reports local-only as not allowed:
  `shared/source/release_helpers/release_helper/release_helper_bmg_g31.cpp:18-20`
  returns `false` from `isLocalOnlyAllowed()` (G21 at
  `release_helper_bmg_g21.cpp:18-20` is the same), propagated at
  `shared/source/os_interface/linux/drm_memory_manager.cpp:3669-3672`.

# Impact

This is a capacity ceiling rather than a performance nit. Serving a large model
across two B70s, 47.4 of 59.4 GiB of host RAM is consumed once both cards are
loaded, leaving 2.78 GiB free while serving, and the box swaps under
long-context load. It also rules out otherwise-straightforward configurations —
adding a third card, or keeping host-resident copies of expert weights — purely
on host-RAM budget, not on VRAM.

# Questions

1. Is a ~1:1 host-RAM backing cost for VRAM-resident device USM expected on BMG
   with the `xe` KMD, or is this a bug?
2. If it is expected, is there a supported way to opt out on Linux (the
   equivalent of what `NEO_LOCAL_MEMORY_ALLOCATION_MODE` does on WDDM)?
3. Would you prefer this cross-filed against `xe`/DRM directly? Happy to do so
   and link back here.
