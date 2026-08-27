// device_usm_host_shadow.cpp
//
// Minimal standalone reproducer: measure host RAM consumed while allocating
// device USM on an Intel discrete GPU (observed on Arc Pro B70 / BMG).
//
// Expectation: device USM lives in VRAM and costs ~0 host RAM.
// Observed:    host MemAvailable falls roughly 1:1 with device USM allocated.
//
// The tool separates two things that are easy to conflate:
//   * process RSS  (VmRSS/VmHWM)  -- would implicate a userspace staging copy
//   * system-wide  (MemAvailable) -- implicates kernel-side BO backing
// Reporting both is the point: if RSS stays flat while MemAvailable drops, the
// duplicate is below the user-mode driver.
//
// It also frees everything and re-measures, to show whether the host-side
// footprint is returned at free() time or retained.
//
// Build: ./build_device_usm_host_shadow.sh
// Run:   ./device_usm_host_shadow --gib 24 --chunk 2 --device 0
//
// SPDX-License-Identifier: MIT

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

// One sample of host memory state, in KiB as reported by the kernel.
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
        // "<Key>: <value> kB"
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
    if (arg_flag(argc, argv, "--help")) {
        std::printf("usage: %s [--gib N] [--chunk N] [--device N] [--no-touch] [--keep]\n"
                    "  --gib N     total device USM to allocate (default 16)\n"
                    "  --chunk N   allocation granularity in GiB (default 2)\n"
                    "  --device N  index into the SYCL GPU device list (default 0)\n"
                    "  --no-touch  skip the device-side memset of each chunk\n"
                    "  --keep      hold the allocation and wait for stdin before freeing\n",
                    argv[0]);
        return 0;
    }

    const long total_gib = arg_long(argc, argv, "--gib", 16);
    const long chunk_gib = arg_long(argc, argv, "--chunk", 2);
    const long dev_index = arg_long(argc, argv, "--device", 0);
    const bool touch = !arg_flag(argc, argv, "--no-touch");

    if (total_gib <= 0 || chunk_gib <= 0) {
        std::fprintf(stderr, "--gib and --chunk must be positive\n");
        return 2;
    }

    std::vector<sycl::device> gpus;
    for (const auto &d : sycl::device::get_devices())
        if (d.is_gpu()) gpus.push_back(d);
    if (gpus.empty()) {
        std::fprintf(stderr, "no SYCL GPU devices found\n");
        return 2;
    }
    if (dev_index < 0 || static_cast<std::size_t>(dev_index) >= gpus.size()) {
        std::fprintf(stderr, "--device %ld out of range (%zu GPUs)\n", dev_index, gpus.size());
        return 2;
    }

    sycl::queue q{gpus[static_cast<std::size_t>(dev_index)]};
    const auto &dev = q.get_device();
    const auto vram = dev.get_info<sycl::info::device::global_mem_size>();

    std::printf("device[%ld]      : %s\n", dev_index,
                dev.get_info<sycl::info::device::name>().c_str());
    std::printf("driver         : %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("global_mem_size: %.2f GiB\n", static_cast<double>(vram) / (1024.0 * 1024 * 1024));
    std::printf("plan           : %ld GiB device USM in %ld GiB chunks, touch=%s\n\n", total_gib,
                chunk_gib, touch ? "yes" : "no");

    const HostMem base = sample();
    std::printf("MemTotal       : %.2f GiB\n", gib(base.mem_total));
    report("baseline", base, base, 0.0);

    std::vector<void *> chunks;
    const std::size_t chunk_bytes = static_cast<std::size_t>(chunk_gib) * (1ull << 30);
    double allocated_gib = 0.0;
    int rc = 0;

    for (long done = 0; done + chunk_gib <= total_gib; done += chunk_gib) {
        void *p = sycl::malloc_device(chunk_bytes, q);
        if (p == nullptr) {
            std::printf("\nmalloc_device failed at %.2f GiB -- stopping\n", allocated_gib);
            rc = 1;
            break;
        }
        chunks.push_back(p);
        allocated_gib += static_cast<double>(chunk_gib);

        if (touch) {
            // Force real backing store, not a lazy reservation.
            q.memset(p, 0xA5, chunk_bytes).wait();
        }

        char tag[32];
        std::snprintf(tag, sizeof(tag), "after +%ld GiB", chunk_gib);
        report(tag, base, sample(), allocated_gib);
    }

    if (arg_flag(argc, argv, "--keep")) {
        std::printf("\nholding %.2f GiB; press enter to free\n", allocated_gib);
        (void)std::getchar();
    }

    std::printf("\n");
    for (void *p : chunks) sycl::free(p, q);
    chunks.clear();
    report("after free", base, sample(), 0.0);

    // A second sample after the queue is torn down: some drivers only release
    // BO backing at context destruction.
    return rc;
}
