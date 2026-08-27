#include "grouped_moe_onednn.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

namespace sb::xe2 {
namespace {

// Set once from SB_GROUPED_BACKEND; cleared forever on the first internal
// failure so a broken runtime degrades to the native pipeline, not to a
// per-dispatch throw/catch storm.
std::atomic<bool> g_armed{[] {
  const char* v = std::getenv("SB_GROUPED_BACKEND");
  const bool on = v && std::strcmp(v, "onednn") == 0;
  if (on) {
    std::fprintf(stderr, "[sb.grouped] backend=onednn armed\n");
  }
  return on;
}()};

void disarm(const char* what, const char* detail) {
  if (g_armed.exchange(false)) {
    std::fprintf(stderr,
                 "[sb.grouped] onednn backend DISARMED (%s: %s); native "
                 "pipeline takes over\n",
                 what, detail);
  }
}

// All oneDNN state for one provider queue. The provider owns one in-order
// queue per card for the life of the process, so nothing here is ever torn
// down deliberately; the bank the scale copies shadow lives exactly as long.
struct QueueCtx {
  dnnl::engine eng;
  dnnl::stream strm;
  sycl::queue q;

  struct Bound {
    dnnl::matmul prim;
    dnnl::memory src, wei, dst, scl;
  };
  // Keyed by (K, N): rows_cap is process-constant (chunk route capacity),
  // asserted on lookup rather than keyed, so a config change is loud.
  std::map<std::pair<int, int>, Bound> prims;
  int bound_rows_cap = -1;
  int bound_experts = -1;

  // Bank scale plane -> repacked canonical [E, K/16, N] device copy.
  //
  // The repack is deliberate and NOT removable by changing the bank layout:
  // oneDNN's grouped matmul requires dense canonical [E, K/16, N], while the
  // decode kernel (quixicore nvfp4_moe_split) walks K per output row and needs
  // the bank's [E, N, K/16]. Making the bank canonical costs decode >=1.78x
  // scale traffic on a leg already at 83.6% of the bandwidth ceiling, which is
  // far worse than this copy. Measured cost of the copy: +2.19 GiB per card.
  std::unordered_map<const void*, std::uint8_t*> scale_canon;

  explicit QueueCtx(sycl::queue& queue)
      : eng(dnnl::sycl_interop::make_engine(queue.get_device(),
                                            queue.get_context())),
        strm(dnnl::sycl_interop::make_stream(eng, queue)),
        q(queue) {}

  Bound& bound(int rows_cap, int experts, int K, int N) {
    if (bound_rows_cap < 0) {
      bound_rows_cap = rows_cap;
      bound_experts = experts;
    }
    if (bound_rows_cap != rows_cap || bound_experts != experts) {
      throw std::runtime_error("grouped geometry changed under a live cache");
    }
    auto it = prims.find({K, N});
    if (it != prims.end()) {
      return it->second;
    }
    using dtt = dnnl::memory::data_type;
    auto src_md =
        dnnl::memory::desc::grouped({rows_cap, K}, dtt::f16, 0, experts);
    auto dst_md =
        dnnl::memory::desc::grouped({rows_cap, N}, dtt::f32, 0, experts);
    // Bank plane [E, N, K] with K contiguous == acb on logical {E, K, N}.
    auto wei_md = dnnl::memory::desc({experts, K, N}, dtt::f4_e2m1,
                                     dnnl::memory::format_tag::acb);
    // Scales must be dense canonical [E, K/16, N]; the bank is [E, N, K/16],
    // hence canon_scales() below.
    auto scl_md = dnnl::memory::desc({experts, K / 16, N}, dtt::f8_e4m3,
                                     dnnl::memory::format_tag::abc);
    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2), {16, 1},
                    dtt::f8_e4m3, false);
    auto pd = dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
    Bound b{dnnl::matmul(pd),
            dnnl::memory(src_md, eng, {nullptr, nullptr}),
            dnnl::memory(wei_md, eng, nullptr),
            dnnl::memory(dst_md, eng, {nullptr, nullptr}),
            dnnl::memory(scl_md, eng, nullptr)};
    return prims.emplace(std::make_pair(K, N), std::move(b)).first->second;
  }

  // Returns the canonical [E, K/16, N] copy of a bank scale plane,
  // transposing it on `q` on first touch (in-order queue: subsequent GEMM
  // submissions see the finished copy without an explicit wait).
  std::uint8_t* canon_scales(const std::uint8_t* bank, int experts, int K,
                             int N) {
    auto it = scale_canon.find(bank);
    if (it != scale_canon.end()) {
      return it->second;
    }
    const int kg = K / 16;
    const std::size_t bytes =
        static_cast<std::size_t>(experts) * kg * N;
    std::uint8_t* canon = sycl::malloc_device<std::uint8_t>(bytes, q);
    if (canon == nullptr) {
      throw std::runtime_error("scale plane allocation failed");
    }
    q.parallel_for(sycl::range<3>(static_cast<std::size_t>(experts), kg, N),
                   [=](sycl::id<3> idx) {
                     const std::size_t e = idx[0], g = idx[1], n = idx[2];
                     canon[(e * kg + g) * N + n] = bank[(e * N + n) * kg + g];
                   });
    scale_canon.emplace(bank, canon);
    return canon;
  }
};

// One context per provider queue (== per card). Guarded: dispatch is
// per-card single-threaded, but two cards' service threads share this map.
std::mutex g_ctx_mu;
std::unordered_map<sycl::queue, QueueCtx> g_ctx;  // SYCL 2020 std::hash

QueueCtx& ctx_for(sycl::queue& q) {
  std::lock_guard<std::mutex> lock(g_ctx_mu);
  auto it = g_ctx.find(q);
  if (it == g_ctx.end()) {
    it = g_ctx.emplace(std::piecewise_construct, std::forward_as_tuple(q),
                       std::forward_as_tuple(q)).first;
  }
  return it->second;
}

}  // namespace

bool onednn_grouped_armed() noexcept {
  return g_armed.load(std::memory_order_relaxed);
}

bool onednn_grouped_gemm(sycl::queue& q, const sycl::half* act,
                         const std::uint8_t* wgt,
                         const std::uint8_t* scl_bank, float* out,
                         const std::int32_t* offs_ends, int rows_cap,
                         int experts, int K, int N) noexcept try {
  if (!onednn_grouped_armed() || rows_cap <= 0 || (K % 16) != 0) {
    return false;
  }
  QueueCtx& ctx = ctx_for(q);
  std::uint8_t* scl = ctx.canon_scales(scl_bank, experts, K, N);
  QueueCtx::Bound& b = ctx.bound(rows_cap, experts, K, N);

  auto* offs = const_cast<std::int32_t*>(offs_ends);
  b.src.set_data_handle(const_cast<sycl::half*>(act), 0);
  b.src.set_data_handle(offs, 1);
  b.wei.set_data_handle(const_cast<std::uint8_t*>(wgt), 0);
  b.dst.set_data_handle(out, 0);
  b.dst.set_data_handle(offs, 1);
  b.scl.set_data_handle(scl, 0);

  b.prim.execute(ctx.strm, {{DNNL_ARG_SRC, b.src},
                            {DNNL_ARG_WEIGHTS, b.wei},
                            {DNNL_ARG_DST, b.dst},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, b.scl}});
  return true;
} catch (const dnnl::error& e) {
  disarm("dnnl", e.what());
  return false;
} catch (const std::exception& e) {
  disarm("std", e.what());
  return false;
} catch (...) {
  disarm("unknown", "non-standard exception");
  return false;
}

}  // namespace sb::xe2
