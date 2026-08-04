// Shooting Brake Phase 1 — B70 Provider Test
//
// Loads NVFP4 expert weights from expert_bank.bin, uploads to B70,
// runs QuixiCore nvfp4_moe_split with test input, reports timing.
//
// Build:
//   source /opt/intel/oneapi/compiler/2026.1/env/vars.sh
//   make -C phase1
//
// Run:
//   ./phase1/b70_provider_test

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sycl/sycl.hpp>
#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"

// ─── Expert bank header ────────────────────────────────────────────────────
#pragma pack(push, 1)
struct ExpertBankHeader {
  char magic[8];
  uint32_t num_layers;
  uint32_t experts_per_layer;
  uint32_t K;          // hidden_size = 2048
  uint32_t I;          // intermediate = 512
  uint32_t reserved;
  uint64_t w13_bytes;  // per expert
  uint64_t s13_bytes;
  uint64_t w2_bytes;
  uint64_t s2_bytes;
};
#pragma pack(pop)
static_assert(sizeof(ExpertBankHeader) == 60, "packed header must be 60 bytes");
static constexpr size_t HEADER_SIZE = sizeof(ExpertBankHeader);

// ─── Helper: find Intel B70 ────────────────────────────────────────────────
static sycl::device find_b70() {
  auto platforms = sycl::platform::get_platforms();
  for (auto& p : platforms) {
    auto devices = p.get_devices();
    for (auto& d : devices) {
      std::string name = d.get_info<sycl::info::device::name>();
      if (name.find("B70") != std::string::npos ||
          name.find("Arc") != std::string::npos) {
        printf("Found device: %s\n", name.c_str());
        printf("  Compute units: %u\n",
               d.get_info<sycl::info::device::max_compute_units>());
        printf("  Memory: %zu MB\n",
               d.get_info<sycl::info::device::global_mem_size>() / (1024*1024));
        return d;
      }
    }
  }
  fprintf(stderr, "FATAL: No Intel B70/Arc device found.\n");
  exit(1);
}

int main() {
  printf("=== Shooting Brake Phase 1 — B70 Provider Test ===\n\n");

  // ── 1. Find and select B70 ───────────────────────────────────────────────
  sycl::device b70 = find_b70();
  sycl::context ctx(b70);
  sycl::queue q(ctx, b70, sycl::property::queue::enable_profiling{});
  printf("Queue created on B70.\n\n");

  // ── 2. Open expert bank ──────────────────────────────────────────────────
  const char* bank_path = "phase1/expert_bank.bin";
  int fd = open(bank_path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "FATAL: Cannot open %s\n", bank_path);
    exit(1);
  }

  struct stat st;
  fstat(fd, &st);
  size_t file_size = st.st_size;
  printf("Expert bank: %s (%.2f GiB)\n", bank_path, file_size / 1024.0/1024.0/1024.0);

  void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    fprintf(stderr, "FATAL: mmap failed\n");
    exit(1);
  }

  const auto* hdr = reinterpret_cast<const ExpertBankHeader*>(mapped);
  if (memcmp(hdr->magic, "SBEXP001", 8) != 0) {
    fprintf(stderr, "FATAL: Bad magic in expert bank\n");
    exit(1);
  }

  size_t num_layers = hdr->num_layers;
  size_t E = hdr->experts_per_layer;
  size_t K = hdr->K;
  size_t I = hdr->I;
  size_t total_experts = num_layers * E;

  size_t expert_stride = hdr->w13_bytes + hdr->s13_bytes +
                         hdr->w2_bytes + hdr->s2_bytes + 8;
  const size_t expected_file_size =
      HEADER_SIZE + total_experts * expert_stride;
  if (file_size != expected_file_size) {
    fprintf(stderr,
            "FATAL: Expert bank size mismatch: got %zu bytes, expected %zu\n",
            file_size, expected_file_size);
    exit(1);
  }
  const uint8_t* expert_base = reinterpret_cast<const uint8_t*>(mapped) + HEADER_SIZE;

  printf("Header: layers=%zu, experts/layer=%zu, K=%zu, I=%zu\n",
         num_layers, E, K, I);
  printf("Total experts: %zu (NVFP4 layers 0-%zu)\n", total_experts, num_layers-1);
  printf("Per-expert stride: %zu bytes (%.2f MiB)\n", expert_stride,
         expert_stride / 1024.0/1024.0);
  printf("Expert weight total: %.2f GiB\n\n",
         total_experts * expert_stride / 1024.0/1024.0/1024.0);

  // ── 3. Upload all experts to B70 ─────────────────────────────────────────
  // For now, upload one layer (256 experts) to test.
  // Full upload can be done once we validate correctness.
  size_t upload_layers = 1;  // start with 1 layer
  size_t upload_experts = upload_layers * E;

  printf("Uploading %zu experts (1 layer) to B70...\n", upload_experts);

  size_t w13_total = upload_experts * hdr->w13_bytes;
  size_t s13_total = upload_experts * hdr->s13_bytes;
  size_t w2_total = upload_experts * hdr->w2_bytes;
  size_t s2_total = upload_experts * hdr->s2_bytes;

  auto t0 = std::chrono::high_resolution_clock::now();

  uint8_t* d_w13 = sycl::malloc_device<uint8_t>(w13_total, q);
  uint8_t* d_s13 = sycl::malloc_device<uint8_t>(s13_total, q);
  uint8_t* d_w2 = sycl::malloc_device<uint8_t>(w2_total, q);
  uint8_t* d_s2 = sycl::malloc_device<uint8_t>(s2_total, q);
  float* d_w13_global = sycl::malloc_device<float>(upload_experts, q);
  float* d_w2_global = sycl::malloc_device<float>(upload_experts, q);

  // Copy expert data to device, reassembling into contiguous arrays
  // Temporarily stage on host
  auto* h_w13 = new uint8_t[w13_total];
  auto* h_s13 = new uint8_t[s13_total];
  auto* h_w2 = new uint8_t[w2_total];
  auto* h_s2 = new uint8_t[s2_total];
  auto* h_w13g = new float[upload_experts];
  auto* h_w2g = new float[upload_experts];

  for (size_t e = 0; e < upload_experts; e++) {
    const uint8_t* src = expert_base + e * expert_stride;
    size_t off = 0;
    memcpy(h_w13 + e * hdr->w13_bytes, src + off, hdr->w13_bytes); off += hdr->w13_bytes;
    memcpy(h_s13 + e * hdr->s13_bytes, src + off, hdr->s13_bytes); off += hdr->s13_bytes;
    memcpy(h_w2 + e * hdr->w2_bytes, src + off, hdr->w2_bytes); off += hdr->w2_bytes;
    memcpy(h_s2 + e * hdr->s2_bytes, src + off, hdr->s2_bytes); off += hdr->s2_bytes;
    // Stored globals are the kernel's dequant multipliers: 1 / checkpoint global scale.
    memcpy(&h_w13g[e], src + off, sizeof(float));
    memcpy(&h_w2g[e], src + off + sizeof(float), sizeof(float));
  }

  q.memcpy(d_w13, h_w13, w13_total).wait();
  q.memcpy(d_s13, h_s13, s13_total).wait();
  q.memcpy(d_w2, h_w2, w2_total).wait();
  q.memcpy(d_s2, h_s2, s2_total).wait();
  q.memcpy(d_w13_global, h_w13g, upload_experts * sizeof(float)).wait();
  q.memcpy(d_w2_global, h_w2g, upload_experts * sizeof(float)).wait();

  auto t1 = std::chrono::high_resolution_clock::now();
  double upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("Upload done in %.1f ms (%.1f MB/s)\n\n", upload_ms,
         (w13_total + s13_total + w2_total + s2_total) / (upload_ms / 1000.0) / 1e6);

  // ── 4. Preallocate buffers ───────────────────────────────────────────────
  size_t M = 1;        // tokens
  size_t top_k = 8;    // experts per token

  // Activation: [M, K] fp16
  sycl::half* d_hidden = sycl::malloc_device<sycl::half>(M * K, q);
  // Route IDs: [M, top_k] int32
  int32_t* d_topk_ids = sycl::malloc_device<int32_t>(M * top_k, q);
  // Route weights: [M, top_k] float32
  float* d_topk_weights = sycl::malloc_device<float>(M * top_k, q);
  // Scratch: [M*top_k, 2*I] float32
  float* d_scratch = sycl::malloc_device<float>(M * top_k * 2 * I, q);
  // Output: [M, K] float32
  float* d_output = sycl::malloc_device<float>(M * K, q);

  // ── 5. Load golden reference ─────────────────────────────────────────────
  FILE* gf = fopen("phase1/golden_reference.bin", "rb");
  if (!gf) { fprintf(stderr, "FATAL: Cannot open golden_reference.bin\n"); return 1; }
  uint32_t g_K, g_I, g_topk;
  fread(&g_K, 4, 1, gf); fread(&g_I, 4, 1, gf); fread(&g_topk, 4, 1, gf);
  assert(g_K == K && g_I == I && g_topk == 1);

  auto* h_hidden = new sycl::half[K];
  fread(h_hidden, sizeof(sycl::half), K, gf);
  int32_t golden_expert; float golden_weight;
  fread(&golden_expert, 4, 1, gf); fread(&golden_weight, 4, 1, gf);
  auto* golden_output = new float[K];
  fread(golden_output, sizeof(float), K, gf);
  fclose(gf);
  printf("Golden: expert=%d weight=%.1f, output range [%.6f, %.6f]\n",
         golden_expert, golden_weight,
         *std::min_element(golden_output, golden_output+K),
         *std::max_element(golden_output, golden_output+K));

  q.memcpy(d_hidden, h_hidden, M * K * sizeof(sycl::half)).wait();
  // Route: expert 0 only, weight 1.0 (others -1 = invalid/skipped)
  int32_t h_ids[8] = {0, -1, -1, -1, -1, -1, -1, -1};
  float h_wts[8] = {1.0f, 0, 0, 0, 0, 0, 0, 0};
  q.memcpy(d_topk_ids, h_ids, M * top_k * sizeof(int32_t)).wait();
  q.memcpy(d_topk_weights, h_wts, M * top_k * sizeof(float)).wait();

  // ── 6. Run nvfp4_moe_split ───────────────────────────────────────────────
  printf("Running nvfp4_moe_split (M=1, expert 0, weight 1.0)...\n");
  auto run = [&]() {
    quixicore::xpu::ops::nvfp4_moe_split(
        q, d_hidden, d_topk_ids, d_topk_weights,
        d_w13, d_s13, d_w13_global, d_w2, d_s2, d_w2_global,
        d_scratch, d_output, M, upload_experts, top_k, K, I,
        quixicore::xpu::DType::f16, true,
        quixicore::xpu::Variant::sycl, true);
    q.wait();
  };
  run(); // warmup
  double total_us = 0;
  for (int r = 0; r < 50; r++) {
    auto s0 = std::chrono::high_resolution_clock::now();
    run();
    auto s1 = std::chrono::high_resolution_clock::now();
    total_us += std::chrono::duration<double, std::micro>(s1 - s0).count();
  }
  printf("M=1: %.1f us mean over 50 runs\n\n", total_us / 50);

  // ── 7. Compare against golden ────────────────────────────────────────────
  auto* h_output = new float[M * K];
  q.memcpy(h_output, d_output, M * K * sizeof(float)).wait();
  // Tolerance: element is bad if abs_err > atol + rtol*|golden|
  // Pre-defined before seeing results: atol=1e-6, rtol=1e-2
  constexpr float ATOL = 1e-6f, RTOL = 1e-2f;
  float max_abs = 0, max_rel = 0;
  double sq_err = 0, sq_gold = 0;
  int n_bad = 0, n_nonfinite = 0;
  for (size_t i = 0; i < K; i++) {
    if (!std::isfinite(h_output[i])) n_nonfinite++;
    float ae = std::abs(h_output[i] - golden_output[i]);
    float tol = ATOL + RTOL * std::abs(golden_output[i]);
    if (ae > tol) n_bad++;
    max_abs = std::max(max_abs, ae);
    if (std::abs(golden_output[i]) > 1e-10f)
      max_rel = std::max(max_rel, ae / std::abs(golden_output[i]));
    sq_err += (double)(ae * ae);
    sq_gold += (double)(golden_output[i] * golden_output[i]);
  }
  double rmse = std::sqrt(sq_err / K), sig = std::sqrt(sq_gold / K);
  printf("=== Correctness vs golden (atol=%.0e, rtol=%.0e) ===\n", ATOL, RTOL);
  printf("  Max abs: %.2e  Max rel: %.2f%%  RMSE: %.2e  Signal: %.2e  Bad: %d  Nonfinite: %d\n",
         max_abs, max_rel*100, rmse, sig, n_bad, n_nonfinite);
  printf("  B70[0:4]:    %.6f %.6f %.6f %.6f\n", h_output[0], h_output[1], h_output[2], h_output[3]);
  printf("  Golden[0:4]: %.6f %.6f %.6f %.6f\n", golden_output[0], golden_output[1], golden_output[2], golden_output[3]);
  bool pass = (n_bad == 0 && n_nonfinite == 0);
  if (!pass) { printf("FAIL: %d bad, %d nonfinite (tol=%.0e+%.0e*|g|)\n", n_bad, n_nonfinite, ATOL, RTOL); return 1; }
  printf("  PASS (all within tol, RMSE/signal=%.4f)\n\n", rmse/(sig+1e-30));


  // ── 8. Test repeated-reference batches and representative prefill ─────────
  for (size_t test_M : {size_t{2}, size_t{4}, size_t{8}, size_t{16},
                        size_t{32}, size_t{128}}) {
    sycl::free(d_hidden, q);
    sycl::free(d_topk_ids, q);
    sycl::free(d_topk_weights, q);
    sycl::free(d_scratch, q);
    sycl::free(d_output, q);

    d_hidden = sycl::malloc_device<sycl::half>(test_M * K, q);
    d_topk_ids = sycl::malloc_device<int32_t>(test_M * top_k, q);
    d_topk_weights = sycl::malloc_device<float>(test_M * top_k, q);
    d_scratch = sycl::malloc_device<float>(test_M * top_k * 2 * I, q);
    d_output = sycl::malloc_device<float>(test_M * K, q);

    auto* h_hidden_big = new sycl::half[test_M * K];
    auto* h_ids_big = new int32_t[test_M * top_k];
    auto* h_wts_big = new float[test_M * top_k];
    auto* h_output_big = new float[test_M * K];
    constexpr float duplicate_route_weights[8] = {
        0.03f, 0.07f, 0.11f, 0.13f, 0.17f, 0.19f, 0.23f, 0.07f};
    const bool exercise_all_routes = test_M == 8;
    for (size_t row = 0; row < test_M; ++row) {
      memcpy(h_hidden_big + row * K, h_hidden, K * sizeof(sycl::half));
      for (size_t route = 0; route < top_k; ++route) {
        h_ids_big[row * top_k + route] =
            exercise_all_routes || route == 0 ? 0 : -1;
        h_wts_big[row * top_k + route] = exercise_all_routes
            ? duplicate_route_weights[route]
            : (route == 0 ? 1.0f : 0.0f);
      }
    }
    q.memcpy(d_hidden, h_hidden_big, test_M * K * sizeof(sycl::half)).wait();
    q.memcpy(d_topk_ids, h_ids_big, test_M * top_k * sizeof(int32_t)).wait();
    q.memcpy(d_topk_weights, h_wts_big, test_M * top_k * sizeof(float)).wait();

    const bool use_split = test_M <= 32;
    auto run_batch = [&]() {
      if (use_split) {
        quixicore::xpu::ops::nvfp4_moe_split(
            q, d_hidden, d_topk_ids, d_topk_weights,
            d_w13, d_s13, d_w13_global, d_w2, d_s2, d_w2_global,
            d_scratch, d_output, test_M, upload_experts, top_k, K, I,
            quixicore::xpu::DType::f16, true,
            quixicore::xpu::Variant::sycl, true);
      } else {
        quixicore::xpu::ops::nvfp4_moe_fused(
            q, d_hidden, d_topk_ids, d_topk_weights,
            d_w13, d_s13, d_w13_global, d_w2, d_s2, d_w2_global,
            d_output, test_M, upload_experts, top_k, K, I,
            quixicore::xpu::DType::f16, true,
            quixicore::xpu::Variant::sycl, true);
      }
      q.wait();
    };

    run_batch();
    total_us = 0;
    for (int run_index = 0; run_index < 20; ++run_index) {
      auto s0 = std::chrono::high_resolution_clock::now();
      run_batch();
      auto s1 = std::chrono::high_resolution_clock::now();
      total_us +=
          std::chrono::duration<double, std::micro>(s1 - s0).count();
    }

    q.memcpy(h_output_big, d_output, test_M * K * sizeof(float)).wait();
    size_t batch_bad = 0;
    size_t batch_nonfinite = 0;
    float batch_max_abs = 0.0f;
    for (size_t row = 0; row < test_M; ++row) {
      for (size_t col = 0; col < K; ++col) {
        const float actual = h_output_big[row * K + col];
        const float expected = golden_output[col];
        if (!std::isfinite(actual)) {
          ++batch_nonfinite;
          continue;
        }
        const float abs_error = std::abs(actual - expected);
        batch_max_abs = std::max(batch_max_abs, abs_error);
        if (abs_error > ATOL + RTOL * std::abs(expected))
          ++batch_bad;
      }
    }
    printf("M=%3zu %-5s: %.1f us mean, max_abs=%.2e, bad=%zu, nonfinite=%zu\n",
           test_M, use_split ? "split" : "fused", total_us / 20,
           batch_max_abs, batch_bad, batch_nonfinite);
    if (batch_bad != 0 || batch_nonfinite != 0) {
      fprintf(stderr, "FAIL: batched correctness failed at M=%zu\n", test_M);
      return 1;
    }

    delete[] h_hidden_big;
    delete[] h_ids_big;
    delete[] h_wts_big;
    delete[] h_output_big;
  }

  printf("\n=== Phase 1 provider test PASSED (all checked outputs within tolerance) ===\n");

  // ── Cleanup ──────────────────────────────────────────────────────────────
  sycl::free(d_w13, q);
  sycl::free(d_s13, q);
  sycl::free(d_w2, q);
  sycl::free(d_s2, q);
  sycl::free(d_w13_global, q);
  sycl::free(d_w2_global, q);
  sycl::free(d_hidden, q);
  sycl::free(d_topk_ids, q);
  sycl::free(d_topk_weights, q);
  sycl::free(d_scratch, q);
  sycl::free(d_output, q);

  delete[] h_w13; delete[] h_s13; delete[] h_w2; delete[] h_s2;
  delete[] h_w13g; delete[] h_w2g;
  delete[] h_hidden; delete[] h_output;

  munmap(mapped, file_size);
  close(fd);

  return 0;
}
