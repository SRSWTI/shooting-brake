// Speed-only probe of Intel's vendored Xe2 grouped MoE GEMM at r15 geometry.
//
// WHY THIS EXISTS
// ---------------
// Our production B70 path is a per-route GEMV: for every (token, expert) pair
// it reads that expert's whole 5.0625 MiB of NVFP4 weights. At MNBT=512 one
// layer issues 512*10 = 5120 routes, ~2560 per card, over the 85 resident
// experts each card holds -- so every expert is dragged out of VRAM ~30x per
// layer per chunk. Measured 2026-08-13 the split path achieves 437.6 GB/s of
// the card's ~510-608, i.e. it is already bandwidth-saturated: there is no
// efficiency left, only bytes. Reading each expert once instead of ~30 times
// is the entire prize, worth ~22.6x on the leg that owns 86-92% of TTFT.
//
// Our own grouped attempt got the bytes right (28.6x fewer) and the memory
// pipeline catastrophically wrong: 9.9 GB/s, ~2% of card, so it LOST 1.55x.
// Intel's kernel has what ours lacks -- persistent atomic work queue, 2D block
// copies, six-K-tile prefetch, DPAS mainloop, grf_size=256.
//
// This probe answers ONE question before any porting work is done: does that
// kernel reach useful bandwidth AT OUR SHAPE? It deliberately measures speed
// only, with synthetic MXFP4 data the kernel already supports, because that
// needs no bank rebuild, no E4M3 scale port, and no correctness gate. If the
// answer is no, the whole plan dies here for the price of an afternoon.
//
// WHAT THIS PROBE IS NOT
// ----------------------
// It is NOT a correctness test and its numbers are NOT a promise about NVFP4.
// The vendored kernel implements MXFP4: E8M0 exponent-only scales at group 32
// (`B_DTYPE` has no NVFP4 entry, and the decode is `bits << 23`). Our bank is
// NVFP4: E4M3 scales at group 16 plus an FP32 global scale. Making it serve our
// bank needs an E4M3 decode path and either group-16 support or a re-quantised
// bank -- that work is only worth starting if the number below is good.
//
// A known risk this probe is shaped to expose: rows-per-expert. 2560 routes
// over 85 experts is ~30 rows each, and small M is exactly where grouped GEMMs
// lose. That is why it sweeps the m_8/m_16/m_32 policies alongside the big one
// rather than reporting a single figure.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cutlass/kernel_hardware_info.h"
#include "xe_2/gemm_xe2_policy.hpp"
#include "xe_2/grouped_gemm_xe2.hpp"

using namespace cute;

namespace {

// type tag so each instantiation gets a unique SYCL kernel name
template <typename, typename, typename, char, char, class, MoE::A_DTYPE,
          MoE::B_DTYPE>
class ProbeName;

struct Shape2 {
  int n;
  int k;
  const char* label;
};

struct Result {
  const char* policy;
  double ms;
  double gbps;
  bool ran;
  std::string note;
};

// Replicates MoEGEMMLauncher from grouped_gemm_xe2_interface.hpp without the
// torch dependency: same ranges, same sub_group_size<16>/grf_size<256>, same
// MoEGEMM entry. Kept deliberately faithful -- if this diverges the number is
// not comparable to Intel's own.
template <char layoutA, char layoutB, class policy, MoE::A_DTYPE ADT,
          MoE::B_DTYPE BDT, typename ElementA, typename ElementB,
          typename ElementS, typename ElementBI, typename ElementD>
void launch(sycl::queue& q, const ElementA* act, const ElementB* wgt,
            const ElementS* scl, const ElementBI* bias, ElementD* out,
            int gemm_n, int gemm_k, const int* rows_per_expert, int num_experts,
            int group_size, int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto threads_per_wg = size(mma);
  static constexpr int kMaxThreadsPerSM = 512;

  sycl::range<3> local(1, 1, threads_per_wg);
  sycl::range<3> global(1, sm_count * kMaxThreadsPerSM / threads_per_wg, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  using CopyA = typename policy::GmemTiledCopyA;
  using CopyB = typename policy::GmemTiledCopyB;
  using CopyD = typename policy::GmemTiledCopyD;

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<ProbeName<ElementA, ElementB, ElementD, layoutA, layoutB,
                               policy, ADT, BDT>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoE::MoEGEMM<ADT, BDT, CopyA, CopyB, CopyD, layoutA, layoutB, 'R'>(
              act, wgt, scl, bias, out, mma, rows_per_expert, num_experts,
              group_size, gemm_n, gemm_k, atomic_buffer, local_mem);
        });
  });
  q.wait_and_throw();
}

// BDT selects MXFP4 (E8M0 scales) or NVFP4 (E4M3 scales). ElementS stays
// uint8_t for both: the scale byte is raw either way, and the kernel does the
// format-specific decode internally.
template <class policy, MoE::B_DTYPE BDT = MoE::B_DTYPE::MXFP4>
Result bench(sycl::queue& q, const char* policy_name, int E, int M, int N,
             int K, int group_size, int iters) {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementS = uint8_t;   // MXFP4: E8M0 bits
  using ElementD = cutlass::half_t;

  const size_t a_elems = size_t(M) * K;
  const size_t b_bytes = size_t(E) * N * K / 2;             // 4-bit packed
  const size_t s_elems = size_t(E) * N * (K / group_size);  // [N, K/gs]/expert
  const size_t d_elems = size_t(M) * N;

  Result r{policy_name, 0.0, 0.0, false, ""};

  auto* A = sycl::malloc_device<ElementA>(a_elems, q);
  auto* B = sycl::malloc_device<uint8_t>(b_bytes, q);
  auto* S = sycl::malloc_device<ElementS>(s_elems, q);
  auto* D = sycl::malloc_device<ElementD>(d_elems, q);
  auto* rows = sycl::malloc_device<int>(E, q);
  auto* atomic_buf = sycl::malloc_device<int32_t>(1, q);
  // Real zeroed bias, NOT nullptr. The mainloop applies bias unconditionally,
  // so a null pointer faults inside the kernel and surfaces as a bare SIGSEGV
  // on the host with no diagnostic.
  auto* Bias = sycl::malloc_device<ElementA>(size_t(E) * N, q);
  if (!A || !B || !S || !D || !rows || !atomic_buf || !Bias) {
    r.note = "device allocation failed";
    return r;
  }

  // Host fill. Values are irrelevant to timing but must not be denormal-heavy
  // garbage that could trip slow paths, so scales sit near 1.0 (E8M0 bias 127).
  std::mt19937 rng(1234);
  std::vector<uint16_t> a_host(a_elems);
  for (auto& v : a_host) v = uint16_t(0x3800 | (rng() & 0x03FF));  // ~0.5..1
  std::vector<uint8_t> b_host(b_bytes);
  for (auto& v : b_host) v = uint8_t(rng() & 0xFF);
  std::vector<ElementS> s_host(s_elems, ElementS(127));

  // Even split, which is the optimistic case for the work queue. Uneven
  // routing is a separate question and a separate measurement.
  std::vector<int> rows_host(E, M / E);
  for (int i = 0; i < M % E; ++i) rows_host[i] += 1;

  int32_t zero = 0;
  q.memcpy(A, a_host.data(), a_elems * sizeof(uint16_t)).wait();
  q.memcpy(B, b_host.data(), b_bytes).wait();
  q.memcpy(S, s_host.data(), s_elems * sizeof(ElementS)).wait();
  q.memcpy(rows, rows_host.data(), E * sizeof(int)).wait();
  q.memcpy(atomic_buf, &zero, sizeof(int32_t)).wait();
  q.memset(Bias, 0, size_t(E) * N * sizeof(ElementA)).wait();
  q.memset(D, 0, d_elems * sizeof(ElementD)).wait();

  auto once = [&]() {
    q.memcpy(atomic_buf, &zero, sizeof(int32_t)).wait();
    launch<'R', 'R', policy, MoE::A_DTYPE::BITS16, BDT,
           ElementA, ElementB, ElementS, ElementA, ElementD>(
        q, A, reinterpret_cast<const ElementB*>(B), S, Bias, D, N, K, rows, E,
        group_size, atomic_buf);
  };

  try {
    once();  // warmup: JIT + first-touch
    once();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) once();
    auto t1 = std::chrono::steady_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.ms = total_ms / iters;
    // Bytes that MATTER: the weight+scale traffic a grouped pass should read
    // exactly once. Activations and output are small by comparison but counted
    // so the figure is honest rather than flattering.
    double bytes = double(b_bytes) + double(s_elems) +
                   double(a_elems) * 2 + double(d_elems) * 2;
    r.gbps = bytes / (r.ms * 1e-3) / 1e9;
    r.ran = true;
  } catch (sycl::exception const& e) {
    r.note = std::string("sycl: ") + e.what();
  } catch (std::exception const& e) {
    r.note = std::string("std: ") + e.what();
  }

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(S, q);
  sycl::free(D, q);
  sycl::free(rows, q);
  sycl::free(atomic_buf, q);
  sycl::free(Bias, q);
  return r;
}

int arg_int(int argc, char** argv, const char* flag, int dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return std::atoi(argv[i + 1]);
  return dflt;
}

}  // namespace

int main(int argc, char** argv) {
  // r15 at L=48: 218 experts, 48 local, 170 remote split 85/85 across cards.
  // gate/up are [N=1024, K=3072]; down is [N=3072, K=1024].
  const int E = arg_int(argc, argv, "--experts", 85);
  const int N = arg_int(argc, argv, "--n", 1024);
  const int K = arg_int(argc, argv, "--k", 3072);
  const int group_size = arg_int(argc, argv, "--group-size", 32);
  const int iters = arg_int(argc, argv, "--iters", 20);
  const int tokens = arg_int(argc, argv, "--tokens", 512);
  const int top_k = arg_int(argc, argv, "--top-k", 10);
  const int cards = arg_int(argc, argv, "--cards", 2);
  const int M = arg_int(argc, argv, "--m", tokens * top_k / cards);

  // Unbuffered: printf to a pipe is block-buffered, so a crash loses every
  // line already "printed". A silent exit 139 with no output is otherwise
  // indistinguishable from crashing before the first statement.
  setvbuf(stdout, nullptr, _IONBF, 0);

  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  printf("device      : %s\n",
              dev.get_info<sycl::info::device::name>().c_str());
  printf("geometry    : E=%d M=%d (%d tok x top%d / %d cards) N=%d K=%d gs=%d\n",
              E, M, tokens, top_k, cards, N, K, group_size);
  printf("rows/expert : %.1f  <- small M is where grouped GEMMs lose\n",
              double(M) / E);

  const double weight_mib = double(E) * N * K / 2 / 1048576.0;
  printf("weights     : %.1f MiB per grouped pass (read ONCE)\n", weight_mib);
  printf("split path  : %.1f MiB (each of %d routes reads a whole expert)\n",
              double(M) * N * K / 2 / 1048576.0, M);
  printf("amplification: %.1fx\n\n", double(M) / E);

  std::vector<Result> out;
  out.push_back(bench<MoE::w4a16_policy>(q, "w4a16_policy (128x256x32)", E, M, N, K, group_size, iters));
  out.push_back(bench<MoE::w4a16_policy_m_32>(q, "w4a16_policy_m_32 (32x64x32)", E, M, N, K, group_size, iters));
  out.push_back(bench<MoE::w4a16_policy_m_16>(q, "w4a16_policy_m_16 (16x64x32)", E, M, N, K, group_size, iters));
  out.push_back(bench<MoE::w4a16_policy_m_8>(q, "w4a16_policy_m_8 (8x64x32)", E, M, N, K, group_size, iters));
#ifdef XE2_NVFP4_FORK
  // tile_k=16 tiles, only present in our fork. These exist so that
  // tile_k == group_size for NVFP4's 16-element block scales, which keeps the
  // existing scale-reload gate correct without touching the mainloop. Run
  // these with --group-size 16; they are the ones that decide whether NVFP4
  // has a fast path at all.
  out.push_back(bench<MoE::w4a16_policy_k16>(q, "w4a16_policy_k16 (128x256x16)", E, M, N, K, group_size, iters));
  out.push_back(bench<MoE::w4a16_policy_m_32_k16>(q, "w4a16_policy_m_32_k16 (32x64x16)", E, M, N, K, group_size, iters));
  out.push_back(bench<MoE::w4a16_policy_m_16_k16>(q, "w4a16_policy_m_16_k16 (16x64x16)", E, M, N, K, group_size, iters));

  // The real target: NVFP4 (E4M3 scales) on the tile_k=16 tiles, which is the
  // only combination that is BOTH our bank's format and numerically correct at
  // group 16. Run with --group-size 16.
  out.push_back(bench<MoE::w4a16_policy_m_32_k16, MoE::B_DTYPE::NVFP4>(q, "NVFP4 m_32_k16 (32x64x16)", E, M, N, K, group_size, iters));
  out.push_back(bench<MoE::w4a16_policy_m_16_k16, MoE::B_DTYPE::NVFP4>(q, "NVFP4 m_16_k16 (16x64x16)", E, M, N, K, group_size, iters));
  out.push_back(bench<MoE::w4a16_policy_k16, MoE::B_DTYPE::NVFP4>(q, "NVFP4 k16 (128x256x16)", E, M, N, K, group_size, iters));
#endif

  printf("%-30s %10s %12s %14s\n", "policy", "ms", "GB/s", "us/token");
  double best = 0.0;
  for (auto& r : out) {
    if (!r.ran) {
      printf("%-30s %10s %12s   %s\n", r.policy, "-", "-", r.note.c_str());
      continue;
    }
    // 47 routed layers, 3 projections per layer, per card.
    double us_per_token = r.ms * 1000.0 * 47 * 3 / double(tokens);
    printf("%-30s %10.3f %12.1f %14.1f\n", r.policy, r.ms, r.gbps,
                us_per_token);
    if (r.gbps > best) best = r.gbps;
  }

  printf("\nreference: split path measured 437.6 GB/s (already saturated);\n");
  printf("           our own grouped attempt managed 9.9 GB/s and lost 1.55x.\n");
  printf("verdict   : best %.1f GB/s -> %s\n", best,
              best >= 200.0 ? "WORTH THE E4M3/group-16 PORT"
                            : "NOT worth porting; plan dies here");
  return 0;
}
