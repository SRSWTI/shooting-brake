// Correctness gate for the NVFP4 grouped path: does it compute the right product?
//
// Bench 24 proved the path runs at 340 GB/s on random bits. That proves bytes
// move, not that the arithmetic is right. Three of the four format questions
// are now settled against the real bank and the real checkpoint:
//
//   scale dtype   torch.float8_e4m3fn                -> E4M3 decode is correct
//   alpha         1/weight_global_scale, a MULTIPLIER -> w = e2m1 * s * alpha
//   scale layout  [N, K/16] row-major, byte-identical -> kernel indexes it as-is
//
// The fourth is the E2M1 nibble order: two 4-bit weights share a byte, and if
// cutlass's subbyte tensor reads them in the opposite order to the one
// compressed-tensors packed, every adjacent pair swaps. The output is then
// garbage at exactly the same speed, which is the worst possible failure mode.
//
// This settles it with controlled inputs rather than by fitting. E2M1 encodes
// magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} for bit patterns 0..7, so a byte of
// 0x21 holds one nibble worth 0.5 (0b001) and one worth 1.0 (0b010). Feed
// one-hot activations, set every block scale to E4M3 1.0 (0x38), and read the
// answer off the output:
//
//   out[k=0] == 0.5  ->  low nibble is the EVEN k   (little-endian nibbles)
//   out[k=0] == 1.0  ->  low nibble is the ODD  k   (our packing is swapped)
//
// Either result is fine and actionable. Not knowing is not.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include "cutlass/kernel_hardware_info.h"
#include "xe_2/gemm_xe2_policy.hpp"
#include "xe_2/grouped_gemm_xe2.hpp"

using namespace cute;

namespace {

template <typename, typename, typename, char, char, class, MoE::A_DTYPE,
          MoE::B_DTYPE>
class VerifyName;

// Faithful copy of MoEGEMMLauncher minus torch. Any divergence here invalidates
// the comparison, so it is kept identical to the vendored launcher.
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
  sycl::range<3> local(1, 1, threads_per_wg);
  sycl::range<3> global(1, sm_count * 512 / threads_per_wg, 1);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};
  using CopyA = typename policy::GmemTiledCopyA;
  using CopyB = typename policy::GmemTiledCopyB;
  using CopyD = typename policy::GmemTiledCopyD;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<VerifyName<ElementA, ElementB, ElementD, layoutA, layoutB,
                                policy, ADT, BDT>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoE::MoEGEMM<ADT, BDT, CopyA, CopyB, CopyD, layoutA, layoutB, 'R'>(
              act, wgt, scl, bias, out, mma, rows_per_expert, num_experts,
              group_size, gemm_n, gemm_k, atomic_buffer, local_mem);
        });
  });
  q.wait_and_throw();
}

// E2M1: 3 magnitude bits -> {0, .5, 1, 1.5, 2, 3, 4, 6}, top bit is sign.
float e2m1_to_float(uint8_t nib) {
  static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = mag[nib & 0x7];
  return (nib & 0x8) ? -v : v;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);

  const int E = 1;
  const int N = 64;
  const int K = 64;          // 4 scale groups of 16
  const int gs = 16;
  const int M = 8;           // one-hot rows probing k = 0..7

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementS = uint8_t;
  using ElementD = cutlass::half_t;

  sycl::queue q{sycl::gpu_selector_v};
  printf("device : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  printf("shape  : E=%d M=%d N=%d K=%d group_size=%d\n\n", E, M, N, K, gs);

  const size_t a_elems = size_t(M) * K;
  const size_t b_bytes = size_t(E) * N * K / 2;
  const size_t s_elems = size_t(E) * N * (K / gs);
  const size_t d_elems = size_t(M) * N;

  auto* A = sycl::malloc_device<ElementA>(a_elems, q);
  auto* B = sycl::malloc_device<uint8_t>(b_bytes, q);
  auto* S = sycl::malloc_device<ElementS>(s_elems, q);
  auto* D = sycl::malloc_device<ElementD>(d_elems, q);
  auto* Bias = sycl::malloc_device<ElementA>(size_t(E) * N, q);
  auto* rows = sycl::malloc_device<int>(E, q);
  auto* atom = sycl::malloc_device<int32_t>(1, q);

  // A: row r is one-hot at k = r, so out[r][n] == dequant(w[n][r]).
  std::vector<uint16_t> a_host(a_elems, 0);
  const uint16_t half_one = 0x3C00;
  for (int r = 0; r < M; ++r) a_host[size_t(r) * K + r] = half_one;

  // B: every byte 0x21 -> nibbles {1, 2} -> magnitudes {0.5, 1.0}.
  std::vector<uint8_t> b_host(b_bytes, 0x21);

  // Scales: E4M3 1.0 is exponent 7 (bias 7), mantissa 0 -> 0b0_0111_000 = 0x38.
  std::vector<ElementS> s_host(s_elems, 0x38);

  std::vector<int> rows_host{M};
  int32_t zero = 0;
  q.memcpy(A, a_host.data(), a_elems * 2).wait();
  q.memcpy(B, b_host.data(), b_bytes).wait();
  q.memcpy(S, s_host.data(), s_elems).wait();
  q.memcpy(rows, rows_host.data(), E * sizeof(int)).wait();
  q.memcpy(atom, &zero, sizeof(int32_t)).wait();
  q.memset(Bias, 0, size_t(E) * N * sizeof(ElementA)).wait();
  q.memset(D, 0, d_elems * sizeof(ElementD)).wait();

  printf("E4M3 scale byte 0x38 should decode to 1.0\n");
  printf("weight byte 0x21 holds nibbles 0x1 (%.1f) and 0x2 (%.1f)\n\n",
         e2m1_to_float(1), e2m1_to_float(2));

  try {
    launch<'R', 'C', MoE::w4a16_policy_m_32_k16, MoE::A_DTYPE::BITS16,
           MoE::B_DTYPE::NVFP4, ElementA, ElementB, ElementS, ElementA,
           ElementD>(q, A, reinterpret_cast<const ElementB*>(B), S, Bias, D, N,
                     K, rows, E, gs, atom);
  } catch (sycl::exception const& e) {
    printf("KERNEL THREW: %s\n", e.what());
    return 1;
  }

  std::vector<uint16_t> d_host(d_elems);
  q.memcpy(d_host.data(), D, d_elems * 2).wait();

  auto h2f = [](uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp(float(m), -24);
    uint32_t bits = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  };

  printf("%-6s %-12s %s\n", "k", "out[k][0]", "reading");
  int low_even = 0, low_odd = 0, other = 0;
  for (int r = 0; r < M; ++r) {
    float v = h2f(d_host[size_t(r) * N + 0]);
    const char* verdict;
    if (std::fabs(v - (r % 2 == 0 ? 0.5f : 1.0f)) < 1e-3) {
      verdict = "low nibble = EVEN k"; ++low_even;
    } else if (std::fabs(v - (r % 2 == 0 ? 1.0f : 0.5f)) < 1e-3) {
      verdict = "low nibble = ODD k"; ++low_odd;
    } else {
      verdict = "UNEXPECTED"; ++other;
    }
    printf("%-6d %-12.4f %s\n", r, v, verdict);
  }

  printf("\n");
  if (other) {
    printf("VERDICT: BROKEN -- %d of %d rows produced neither 0.5 nor 1.0.\n",
           other, M);
    printf("         The fault is upstream of nibble order: scale decode,\n"
           "         addressing or layout. Do NOT proceed to integration.\n");
  } else if (low_even == M) {
    printf("VERDICT: CORRECT as written. Low nibble is even k, which is what\n"
           "         compressed-tensors packs. Kernel arithmetic is sound;\n"
           "         proceed to route sorting.\n");
  } else if (low_odd == M) {
    printf("VERDICT: NIBBLES SWAPPED. Every adjacent weight pair is\n"
           "         transposed. One-line fix in the dequant, then re-run.\n");
  } else {
    printf("VERDICT: INCONSISTENT (%d even, %d odd) -- not a nibble-order\n"
           "         question. Suspect tiling or scale-group addressing.\n",
           low_even, low_odd);
  }

  sycl::free(A, q); sycl::free(B, q); sycl::free(S, q); sycl::free(D, q);
  sycl::free(Bias, q); sycl::free(rows, q); sycl::free(atom, q);
  return other ? 1 : 0;
}
