// l0v2_memset_crash.cpp
//
// Minimal reproducer: repeated sycl::malloc_device + queue::memset on an Intel
// discrete GPU aborts under the Level Zero v2 adapter, and succeeds under v1.
//
//   SYCL_UR_USE_LEVEL_ZERO_V2=1  -> fails
//   SYCL_UR_USE_LEVEL_ZERO_V2=0  -> passes
//
// Build: icpx -fsycl -std=gnu++17 -O2 -o l0v2_memset_crash l0v2_memset_crash.cpp
//
// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    // Default: 8 iterations of a 1 GiB device allocation + device-side fill.
    long iters = 8;
    std::size_t bytes = 1ull << 30;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0) iters = std::strtol(argv[i + 1], nullptr, 10);
        if (std::strcmp(argv[i], "--mib") == 0)
            bytes = static_cast<std::size_t>(std::strtol(argv[i + 1], nullptr, 10)) << 20;
    }

    std::vector<sycl::device> gpus;
    for (const auto &d : sycl::device::get_devices())
        if (d.is_gpu()) gpus.push_back(d);
    if (gpus.empty()) {
        std::fprintf(stderr, "no SYCL GPU devices found\n");
        return 2;
    }

    const bool in_order = [&] {
        for (int i = 1; i < argc; ++i)
            if (std::strcmp(argv[i], "--in-order") == 0) return true;
        return false;
    }();

    sycl::queue q = in_order ? sycl::queue{gpus[0], sycl::property::queue::in_order{}}
                             : sycl::queue{gpus[0]};

    std::printf("device : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("driver : %s\n",
                q.get_device().get_info<sycl::info::device::driver_version>().c_str());
    const char *v2 = std::getenv("SYCL_UR_USE_LEVEL_ZERO_V2");
    std::printf("SYCL_UR_USE_LEVEL_ZERO_V2 = %s\n", v2 ? v2 : "(unset)");
    std::printf("queue  : %s\n", in_order ? "in_order" : "out_of_order (default)");
    std::printf("plan   : %ld iterations, %zu MiB each, malloc_device + memset + wait + free\n\n",
                iters, bytes >> 20);
    std::fflush(stdout);

    // Hold each allocation so every iteration allocates fresh device memory,
    // rather than reusing a pointer the allocator just freed.
    std::vector<void *> held;

    for (long i = 0; i < iters; ++i) {
        void *p = sycl::malloc_device(bytes, q);
        if (p == nullptr) {
            std::printf("iter %2ld: malloc_device returned nullptr, stopping\n", i);
            break;
        }
        held.push_back(p);
        std::printf("iter %2ld: allocated %p, memset...", i, p);
        std::fflush(stdout);

        q.memset(p, 0xA5, bytes).wait();

        std::printf(" ok\n");
        std::fflush(stdout);
    }

    std::printf("\nfreeing %zu allocations\n", held.size());
    std::fflush(stdout);
    for (void *p : held) sycl::free(p, q);

    std::printf("PASS: completed %ld iterations without abort\n", iters);
    return 0;
}
