## Pre-submission Checklist

- [x] I am using the latest GPU driver version available for this distribution (`26.27.39122.12-1~26.04~ppa1`, intel-graphics PPA)
- [x] I have searched for similar issues and found none

## GPU Hardware

Intel Arc Pro B70 (Battlemage G31) — 2x installed, both reproduce independently.

## DRI Devices Information

```
$ ls -l /dev/dri/by-path/ | grep -E '11:00|15:00'
lrwxrwxrwx+1 root root  8 Aug 27 09:51 pci-0000:11:00.0-card -> ../card2
lrwxrwxrwx+1 root root 13 Aug 27 09:51 pci-0000:11:00.0-render -> ../renderD129
lrwxrwxrwx+1 root root  8 Aug 27 09:51 pci-0000:15:00.0-card -> ../card3
lrwxrwxrwx+1 root root 13 Aug 27 09:51 pci-0000:15:00.0-render -> ../renderD130
```

## GPU Detailed Information (lspci output)

```
$ lspci -nn | grep Battlemage
11:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]
15:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]
```

## Driver Version

`26.27.39122.12-1~26.04~ppa1` — SYCL device driver version string reports `1.15.39122+12`.

## Installed GPU Driver Packages

```
intel-ocloc                          26.27.39122.12-1~26.04~ppa1
intel-opencl-icd                     26.27.39122.12-1~26.04~ppa1
libigc2                              2.38.3-3~26.04
libigdgmm12:amd64                    22.10.0-1~26.04~ppa1
libze-dev:amd64                      1.28.6-1~26.04~ppa1
libze-intel-gpu-raytracing           1.2.4-1~26.04~ppa3
libze-intel-gpu1                     26.27.39122.12-1~26.04~ppa1
libze1:amd64                         1.28.6-1~26.04~ppa1
```

## Driver Installation Details

Installed from the Intel graphics PPA for Ubuntu 26.04 (`~26.04~ppa1` suffixed
packages above) via `apt`. Kernel-side is the in-tree `xe` KMD shipped with
kernel `7.0.0-30-generic`; no out-of-tree DKMS module.

## Linux Distribution

Ubuntu 26.04 LTS

## Kernel Version & Boot Parameters

```
$ uname -a
Linux 7.0.0-30-generic #30-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 31 18:22:54 UTC 2026 x86_64 GNU/Linux

$ cat /proc/cmdline
BOOT_IMAGE=/boot/vmlinuz-7.0.0-30-generic root=UUID=... ro quiet splash \
  crashkernel=2G-4G:320M,4G-32G:512M,32G-64G:1024M,64G-128G:2048M,128G-:4096M
```

KMD in use: `xe`.

## Actual Behavior

Allocating device USM with `sycl::malloc_device` reduces system-wide host
`MemAvailable` at approximately 1:1 with the amount allocated, even though the
allocation is expected to be VRAM-resident.

Three properties make this look like kernel-side BO backing rather than a
user-space staging copy:

| property | observation |
|---|---|
| scaling | 16 GiB of device USM costs 15.94–16.21 GiB of `MemAvailable` |
| requires device access? | **No** — an untouched allocation (no kernel, no memset) costs the full amount |
| process RSS | flat at 0.15 GiB across every sample |
| both cards | reproduces on `11:00.0` and `15:00.0` independently |
| released on free | yes, returns to baseline (+0.18 GiB) |

Measured on a fully idle machine (no inference server, no other GPU or host
load, both cards at 42 MiB used, `MemAvailable` 52.86 GiB of 59.44 GiB total):

**Untouched allocation, 4 GiB chunks, device 0** — no kernel ever launched
against the pointer:

```
plan           : 16 GiB device USM in 4 GiB chunks, touch=no

MemTotal       : 59.44 GiB
baseline               dev_usm=  0.00 GiB | MemAvailable=  52.86 GiB (d   +0.00) | MemFree=  51.39 GiB (d   +0.00) | VmRSS=  0.15 GiB (d  +0.00)
after +4 GiB           dev_usm=  4.00 GiB | MemAvailable=  48.87 GiB (d   -3.99) | MemFree=  47.40 GiB (d   -3.99) | VmRSS=  0.15 GiB (d  +0.00)
after +4 GiB           dev_usm=  8.00 GiB | MemAvailable=  44.93 GiB (d   -7.93) | MemFree=  43.47 GiB (d   -7.92) | VmRSS=  0.15 GiB (d  +0.00)
after +4 GiB           dev_usm= 12.00 GiB | MemAvailable=  40.95 GiB (d  -11.91) | MemFree=  39.49 GiB (d  -11.90) | VmRSS=  0.15 GiB (d  +0.00)
after +4 GiB           dev_usm= 16.00 GiB | MemAvailable=  36.64 GiB (d  -16.21) | MemFree=  35.53 GiB (d  -15.86) | VmRSS=  0.15 GiB (d  -0.00)

after free             dev_usm=  0.00 GiB | MemAvailable=  53.04 GiB (d   +0.18) | MemFree=  51.60 GiB (d   +0.20) | VmRSS=  0.15 GiB (d  -0.00)
```

**Touched allocation, 2 GiB chunks, device 0** (device-side `memset` per chunk):

```
after +2 GiB           dev_usm=  2.00 GiB | MemAvailable=  50.74 GiB (d   -2.00) | VmRSS=  0.15 GiB (d  +0.00)
after +2 GiB           dev_usm=  4.00 GiB | MemAvailable=  48.75 GiB (d   -4.00) | VmRSS=  0.15 GiB (d  +0.00)
after +2 GiB           dev_usm=  8.00 GiB | MemAvailable=  44.73 GiB (d   -8.01) | VmRSS=  0.15 GiB (d  +0.00)
after +2 GiB           dev_usm= 16.00 GiB | MemAvailable=  36.81 GiB (d  -15.94) | VmRSS=  0.15 GiB (d  +0.00)

after free             dev_usm=  0.00 GiB | MemAvailable=  52.87 GiB (d   +0.13) | VmRSS=  0.15 GiB (d  +0.00)
```

**Practical impact:** serving a large model across both cards consumes 47.4 of
59.4 GiB of host RAM once both are loaded, leaving 2.78 GiB free while running,
and the machine swaps under load. It also rules out configurations that are
otherwise well within VRAM budget (a third card, host-resident weight copies)
purely on host-RAM accounting.

## Expected Behavior

Device USM backed by VRAM should not consume host `MemAvailable` proportional to
its size. For an allocation that is never accessed from the device, I would
expect approximately zero host-RAM cost.

## Reproduction Rate

Always reproduces - 100%

## Steps to Reproduce

1. Build the reproducer below (SYCL only, no other dependencies):
   ```
   icpx -fsycl -std=gnu++17 -O2 -Wall -Wextra -o device_usm_host_shadow device_usm_host_shadow.cpp
   ```
2. `export SYCL_UR_USE_LEVEL_ZERO_V2=0`
   (unrelated to this report, but required on this setup — see Additional Notes)
3. Note `MemAvailable` from `/proc/meminfo`.
4. Run the untouched case, which is the clearest:
   ```
   ./device_usm_host_shadow --gib 16 --chunk 4 --device 0 --no-touch
   ```
5. Observe that `MemAvailable` falls ~1:1 with `dev_usm` while `VmRSS` stays flat.
6. Repeat with `--device 1` — same result.

## Is this a regression?

Not known. This is the only driver version I have tested, so I cannot say
whether earlier versions behaved differently.

## First Known Failing Driver Version

Unknown — see above. First (and only) version tested: `26.27.39122.12`.

## API Call Logs

Not captured. The behaviour is visible purely from allocation-time host memory
accounting and does not depend on any API returning an error — every call
succeeds. Happy to capture `ZE_DEBUG` / UR tracing if that would help.

## strace Logs

Not captured. Can provide on request — the relevant syscalls would be the
`DRM_IOCTL_XE_GEM_CREATE` sequence issued per chunk.

## System Logs / dmesg Output

No `xe` / DRM messages are emitted during the reproducer run — `dmesg` shows
nothing correlated with the allocations.

## Backtrace (if crash or hang occurred)

Not applicable — no crash or hang; this is a memory-accounting issue.

## Source Code / Reproducer

Standalone, ~130 lines, SYCL only. Separates process-local (`VmRSS`) from
system-wide (`MemAvailable`, `MemFree`) so a userspace staging copy can be
distinguished from kernel-side backing.

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
    long mem_total = -1, mem_free = -1, mem_available = -1, vm_rss = -1;
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

## Command Line / Application Details

```
export SYCL_UR_USE_LEVEL_ZERO_V2=0
./device_usm_host_shadow --gib 16 --chunk 4 --device 0 --no-touch   # clearest case
./device_usm_host_shadow --gib 16 --chunk 2 --device 0              # touched
./device_usm_host_shadow --gib 8  --chunk 2 --device 1              # other card
```

## oneAPI Version (if applicable)

oneAPI 2026.1 (`/opt/intel/oneapi/compiler/2026.1`), `icpx` from that toolchain.

## Screenshots / Video

Not applicable — all evidence is the textual output above.

## Additional Notes

**What I was able to rule out.** This does not appear to be NEO/UMD staging. At
`32a2e4c2b0`:

- Device USM is created host-inaccessible with no CPU allocation:
  `shared/source/memory_manager/unified_memory_manager.cpp:570-571`
  (`unifiedMemoryProperties.flags.isHostInaccessibleAllocation = true;`) and
  `:607-608` (`allocData.cpuAllocation = nullptr;`). For accuracy: the
  allocation type here resolves to `AllocationType::buffer` at `:1332-1351`,
  not `svmGpu`, unless a caller explicitly supplies a requested type at
  `:1347-1348`.
- The Linux device pool creates a region GEM BO
  (`shared/source/os_interface/linux/drm_memory_manager.cpp:3018-3021`,
  `createBufferObjectInMemoryRegion`) and does not CPU-map it except when
  `allocationData.flags.requiresCpuAccess` (`:2832-2834`). There is a separate
  unconditional lock for `AllocationType::writeCombined` at `:2815-2830`, which
  is not this path.
- Region selection / GEM-create dispatch:
  `shared/source/os_interface/linux/memory_info.cpp:121,152-162` and
  `shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp:748-749,762`.

Since process RSS stays flat, the duplicate is not in the allocating process's
address space at all, which points at kernel-side BO backing rather than
anything compute-runtime does. I am filing here because this is where Level
Zero/NEO behaviour on Intel GPUs is triaged and the source trace may save a
step — happy to re-file or cross-file against `xe`/DRM if you would prefer.

`NEO_LOCAL_MEMORY_ALLOCATION_MODE` does not appear usable as a mitigation here,
for two independent reasons: it is documented Windows-only
(`level_zero/doc/experimental_extensions/LOCAL_MEMORY_ALLOCATION_MODE.md:11`)
and the only production read of the flag is WDDM
(`shared/source/os_interface/windows/wddm_memory_manager.cpp:65`); and BMG
reports local-only as not allowed —
`shared/source/release_helpers/release_helper/release_helper_bmg_g31.cpp:18-20`
returns `false` from `isLocalOnlyAllowed()` (G21 at
`release_helper_bmg_g21.cpp:18-20` is the same), propagated at
`shared/source/os_interface/linux/drm_memory_manager.cpp:3669-3672`.

**Unrelated environment note:** `SYCL_UR_USE_LEVEL_ZERO_V2=0` is set because the
default v2 adapter segfaults on a plain device-USM `memset` with an
out-of-order queue on this setup. That is a separate problem and I filed it
against intel/llvm as
[intel/llvm#23034](https://github.com/intel/llvm/issues/23034); it is not
required to observe the behaviour reported here, only to run the reproducer as
written.

**Questions**

1. Is a ~1:1 host-RAM backing cost for VRAM-resident device USM expected on BMG
   with the `xe` KMD, or is this a bug?
2. If expected, is there a supported way to opt out on Linux (the equivalent of
   what `NEO_LOCAL_MEMORY_ALLOCATION_MODE` does on WDDM)?
