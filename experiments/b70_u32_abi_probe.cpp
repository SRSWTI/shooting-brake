// Does the functor-field arg ABI hold when fields mix pointers and uint32?
// The baked-chain probe validated POINTER-ONLY functors; the baked MoE
// kernels add five u32 scalars and their parity fails. This answers whether
// scalars keep 1-field-per-arg indexing, in 20 seconds.
//
// Build: icpx -fsycl -O2 -std=c++20 experiments/b70_u32_abi_probe.cpp \
//   -o experiments/b70_u32_abi_probe -lze_loader
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>

struct MixedArgs {
  const float *x;        // arg 0
  float *y;              // arg 1
  std::uint32_t a;       // arg 2
  std::uint32_t b;       // arg 3
  std::uint32_t c;       // arg 4
  void operator()(sycl::nd_item<1> it) const {
    const std::size_t i = it.get_global_id(0);
    y[i] = x[i] * static_cast<float>(a) + static_cast<float>(b) * 1000.0f +
           static_cast<float>(c);
  }
};

int main() {
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
  auto *x = sycl::malloc_device<float>(256, q);
  auto *y = sycl::malloc_device<float>(256, q);
  std::vector<float> hx(256);
  for (int i = 0; i < 256; ++i) hx[i] = static_cast<float>(i);
  q.memcpy(x, hx.data(), sizeof(hx[0]) * 256).wait();
  const MixedArgs f{x, y, 7u, 3u, 11u};
  q.parallel_for(sycl::nd_range<1>(256, 256), f).wait();
  std::vector<float> ref(256);
  q.memcpy(ref.data(), y, sizeof(float) * 256).wait();

  auto kid = sycl::get_kernel_id<MixedArgs>();
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q.get_context(), {kid});
  auto zk = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      bundle.get_kernel(kid));
  ze_kernel_properties_t props{ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
  zeKernelGetProperties(zk, &props);
  std::printf("numKernelArgs=%u (fields: 5)\n", props.numKernelArgs);
  auto sp = [&](std::uint32_t i, const void *p) {
    return zeKernelSetArgumentValue(zk, i, sizeof(void *), &p);
  };
  auto su = [&](std::uint32_t i, std::uint32_t v) {
    return zeKernelSetArgumentValue(zk, i, sizeof(std::uint32_t), &v);
  };
  ze_result_t r0 = sp(0, x), r1 = sp(1, y);
  ze_result_t r2 = su(2, 7u), r3 = su(3, 3u), r4 = su(4, 11u);
  std::printf("set results: %x %x %x %x %x\n", r0, r1, r2, r3, r4);
  q.memset(y, 0, sizeof(float) * 256).wait();
  zeKernelSetGroupSize(zk, 256, 1, 1);
  auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      q.get_context());
  auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      q.get_device());
  ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.ordinal = 0;
  qd.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ze_command_list_handle_t list{};
  ze_command_queue_handle_t zq{};
  zeCommandQueueCreate(zctx, zdev, &qd, &zq);
  ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
  zeCommandListCreate(zctx, zdev, &ld, &list);
  ze_group_count_t gc{1, 1, 1};
  zeCommandListAppendLaunchKernel(list, zk, &gc, nullptr, 0, nullptr);
  zeCommandListClose(list);
  zeCommandQueueExecuteCommandLists(zq, 1, &list, nullptr);
  zeCommandQueueSynchronize(zq, UINT64_MAX);
  std::vector<float> got(256);
  q.memcpy(got.data(), y, sizeof(float) * 256).wait();
  int bad = 0;
  for (int i = 0; i < 256; ++i)
    if (got[i] != ref[i]) ++bad;
  std::printf("raw vs SYCL: %d/256 mismatched -> %s\n", bad,
              bad == 0 ? "U32 ABI OK" : "U32 ABI BROKEN");
  if (bad) std::printf("sample: got[1]=%f ref[1]=%f\n", got[1], ref[1]);
  return bad == 0 ? 0 : 1;
}
