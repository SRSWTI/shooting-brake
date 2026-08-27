# Describe the bug

With the Level Zero **v2** adapter, a plain `sycl::queue::memset` on device USM
segfaults on the **second** call when the queue is the default out-of-order
queue. The crash is a jump to address `0x0` from inside
`ur_command_list_manager::isGraphCaptureActive`.

It is cleanly isolated to one configuration:

| queue | `SYCL_UR_USE_LEVEL_ZERO_V2=1` | `SYCL_UR_USE_LEVEL_ZERO_V2=0` |
|---|---|---|
| out-of-order (default) | **SIGSEGV** | PASS |
| `property::queue::in_order` | PASS | PASS |

First `memset` always succeeds; the second one crashes. Reducing the allocation
to 512 MiB or 1 GiB, and the iteration count to 2, does not change it.

# To reproduce

```cpp
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    long iters = 4;
    std::size_t bytes = 512ull << 20;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0) iters = std::strtol(argv[i + 1], nullptr, 10);
        if (std::strcmp(argv[i], "--mib") == 0)
            bytes = static_cast<std::size_t>(std::strtol(argv[i + 1], nullptr, 10)) << 20;
    }

    std::vector<sycl::device> gpus;
    for (const auto &d : sycl::device::get_devices())
        if (d.is_gpu()) gpus.push_back(d);
    if (gpus.empty()) { std::fprintf(stderr, "no SYCL GPU devices found\n"); return 2; }

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
    std::fflush(stdout);

    std::vector<void *> held;
    for (long i = 0; i < iters; ++i) {
        void *p = sycl::malloc_device(bytes, q);
        if (p == nullptr) { std::printf("iter %2ld: nullptr, stopping\n", i); break; }
        held.push_back(p);
        std::printf("iter %2ld: allocated %p, memset...", i, p);
        std::fflush(stdout);
        q.memset(p, 0xA5, bytes).wait();
        std::printf(" ok\n");
        std::fflush(stdout);
    }

    for (void *p : held) sycl::free(p, q);
    std::printf("PASS: completed %ld iterations without abort\n", iters);
    return 0;
}
```

```
icpx -fsycl -std=gnu++17 -O2 -o l0v2_memset_crash l0v2_memset_crash.cpp

SYCL_UR_USE_LEVEL_ZERO_V2=1 ./l0v2_memset_crash --iters 4 --mib 512              # SIGSEGV
SYCL_UR_USE_LEVEL_ZERO_V2=0 ./l0v2_memset_crash --iters 4 --mib 512              # PASS
SYCL_UR_USE_LEVEL_ZERO_V2=1 ./l0v2_memset_crash --iters 4 --mib 512 --in-order   # PASS
```

Observed:

```
device : Intel(R) Arc(TM) Pro B70 Graphics
driver : 1.15.39122+12
SYCL_UR_USE_LEVEL_ZERO_V2 = 1
queue  : out_of_order (default)

iter  0: allocated 0xffffd556b6400000, memset... ok
iter  1: allocated 0xffffd556d6400000, memset...
Segmentation fault
```

Exit status 139.

Backtrace:

```
Thread 1 "l0v2_memset_cra" received signal SIGSEGV, Segmentation fault.
0x0000000000000000 in ?? ()
#0  0x0000000000000000 in ?? ()
#1  0x00007ffff3f32414 in ur_command_list_manager::isGraphCaptureActive(bool*) ()
   from .../lib/libur_adapter_level_zero_v2.so.0
#2  0x00007ffff3f994f1 in v2::ur_queue_immediate_out_of_order_t::enqueueUSMFill(void*, unsigned long, void const*, unsigned long, unsigned int, ur_event_handle_t_* const*, ur_event_handle_t_**) ()
   from .../lib/libur_adapter_level_zero_v2.so.0
#3  0x00007ffff3f6a31b in ur::level_zero::urEnqueueUSMFill(ur_queue_handle_t_*, void*, unsigned long, void const*, unsigned long, unsigned int, ur_event_handle_t_* const*, ur_event_handle_t_**) ()
   from .../lib/libur_adapter_level_zero_v2.so.0
#4  0x00007ffff6964f36 in urEnqueueUSMFill () from .../lib/libur_loader.so.0
#5  0x00007ffff785af04 in sycl::_V1::detail::MemoryManager::fill_usm(...) from .../libsycl.so.9
#6  0x00007ffff78ac9f1 in sycl::_V1::detail::queue_impl::memset(...) from .../libsycl.so.9
#7  0x00007ffff79770d8 in sycl::_V1::queue::memset(...) from .../libsycl.so.9
#8  0x0000000000403016 in main ()
```

# Environment

- oneAPI 2026.1 (`/opt/intel/oneapi/compiler/2026.1`), `libsycl.so.9`
- GPU: Intel Arc Pro B70, `Battlemage G31` (`8086:e223`), device driver version `1.15.39122+12`
- NEO / compute-runtime `26.27.39122.12`, Level Zero loader `libze1 1.28.6`
- Kernel `7.0.0-30-generic`, Ubuntu 26.04 LTS, `xe` KMD
- Reproduces identically on both installed B70s

# Additional information

Frame #0 is address `0x0`, so `isGraphCaptureActive` appears to call through a
null function pointer rather than dereferencing a null object.

Two details that might narrow it down:

- It is specific to `ur_queue_immediate_out_of_order_t`. An in-order queue on
  the same adapter is fine, and that queue type is not in the failing frame.
- It always survives the first `memset` and dies on the second. If the
  out-of-order queue round-robins over a pool of command-list managers, that
  pattern would be consistent with only the first manager in the pool having its
  graph-capture state or hook initialized, with the second one reached on the
  second enqueue.

I could not confirm this against current `main`, and I want to flag that
honestly rather than guess: at `9e61eb36c42b` I cannot find
`isGraphCaptureActive` as a member of `ur_command_list_manager` at all. In that
tree it exists as `batch_manager::isGraphCaptureActive() const` in
`unified-runtime/source/adapters/level_zero/v2/queue_batched.hpp:157`, returning
a plain `bool` member, with a different signature from the
`isGraphCaptureActive(bool*)` in the shipped 2026.1 binary.

So this code path has evidently been restructured since the 2026.1 release, and
it is entirely possible this is already fixed on main. I'm filing it because it
is present in a shipped release, the repro is two allocations long, and if it
*is* already fixed then a confirmation plus the fixing commit would be useful to
anyone else hitting it on 2026.1. If you'd prefer this against a different
component or tracker, happy to move it.

**Workaround** for anyone who lands here: either use an in-order queue, or set
`SYCL_UR_USE_LEVEL_ZERO_V2=0`.
