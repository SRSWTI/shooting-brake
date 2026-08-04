/*
 * b70_expert_sycl.cpp — B70 expert compute proof-of-concept.
 *
 * Allocates N expert weight matrices on B70 VRAM (FP32 synthetic),
 * receives a 2048-dim activation from host, computes gate/up/SiLU/down
 * for K assigned experts, returns one weighted partial.
 *
 * Measures per-expert and per-layer compute time on the B70.
 *
 * Build:
 *   source /opt/intel/oneapi/2026.1/setvars.sh
 *   icpx -O3 -fsycl b70_expert_sycl.cpp \
 *       -I$MKLROOT/include -L$MKLROOT/lib/intel64 \
 *       -lmkl_sycl_blas -lsycl \
 *       -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
 *       -o b70_expert_sycl
 */
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

namespace mkl = oneapi::mkl;

int main() {
    const int H = 2048;   // hidden size
    const int I = 512;    // intermediate per expert
    const int N_EXP = 32; // experts on B70
    const int TOPK_B70 = 3; // experts per layer on B70
    const int N_LAYERS = 40;
    const int N_TOKENS = 32;
    const int ITERS = 100;

    // Create SYCL queue on B70
    auto platforms = sycl::platform::get_platforms();
    sycl::device b70_dev;
    bool found = false;
    for (auto &p : platforms) {
        auto devs = p.get_devices();
        for (auto &d : devs) {
            if (d.is_gpu() && d.get_info<sycl::info::device::vendor_id>() == 0x8086) {
                auto name = d.get_info<sycl::info::device::name>();
                auto mem = d.get_info<sycl::info::device::global_mem_size>();
                printf("Found: %s (%.1f GB)\n", name.c_str(), mem / 1e9);
                if (mem > 8e9 && mem < 40e9) {  // B70 = ~32GB, not iGPU
                    b70_dev = d;
                    found = true;
                }
            }
        }
    }
    if (!found) { fprintf(stderr, "B70 not found\n"); return 1; }

    sycl::queue q(b70_dev);
    printf("Queue on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    printf("H=%d I=%d experts=%d topk_b70=%d\n\n", H, I, N_EXP, TOPK_B70);

    // Allocate expert weights on B70 (FP32 synthetic)
    // gate[e]: [I, H], up[e]: [I, H], down[e]: [H, I]
    // Stored row-major, gate/up transposed during GEMM
    float *d_gate[N_EXP], *d_up[N_EXP], *d_down[N_EXP];
    for (int e = 0; e < N_EXP; e++) {
        d_gate[e] = sycl::malloc_device<float>(I * H, q);
        d_up[e]   = sycl::malloc_device<float>(I * H, q);
        d_down[e] = sycl::malloc_device<float>(H * I, q);
    }
    // Fill with random data on host, copy up
    float *tmp = new float[I * H];
    for (int e = 0; e < N_EXP; e++) {
        for (int i = 0; i < I * H; i++) tmp[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        q.memcpy(d_gate[e], tmp, I * H * sizeof(float)).wait();
        q.memcpy(d_up[e], tmp, I * H * sizeof(float)).wait();
        for (int i = 0; i < H * I; i++) tmp[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        q.memcpy(d_down[e], tmp, H * I * sizeof(float)).wait();
    }
    delete[] tmp;

    // Activation and workspace on device
    float *d_x  = sycl::malloc_device<float>(H, q);     // activation [H]
    float *d_g  = sycl::malloc_device<float>(I, q);      // gate output [I]
    float *d_u  = sycl::malloc_device<float>(I, q);      // up output [I]
    float *d_a  = sycl::malloc_device<float>(I, q);      // activation(g)*u [I]
    float *d_y  = sycl::malloc_device<float>(H, q);      // expert output [H]
    float *d_out = sycl::malloc_device<float>(H, q);     // accumulated partial [H]
    float *h_x  = sycl::malloc_host<float>(H, q);        // host activation
    float *h_out = sycl::malloc_host<float>(H, q);       // host partial

    for (int i = 0; i < H; i++) h_x[i] = 0.1f;
    q.memcpy(d_x, h_x, H * sizeof(float)).wait();

    printf("=== B70 Expert Compute (FP32, oneMKL BLAS) ===\n");

    // Single expert timing
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERS; iter++) {
        int e = iter % N_EXP;
        // gate: d_g[1,I] = d_x[1,H] @ d_gate[e][I,H]^T
        mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans,
            1, I, H, 1.0f, d_x, H, d_gate[e], H, 0.0f, d_g, I);
        // up: d_u[1,I] = d_x[1,H] @ d_up[e][I,H]^T
        mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans,
            1, I, H, 1.0f, d_x, H, d_up[e], H, 0.0f, d_u, I);
        q.wait();
        // SiLU(g) * u  — custom kernel
        q.parallel_for(I, [=](auto i) { d_a[i] = d_g[i] / (1.0f + sycl::exp(-d_g[i])) * d_u[i]; }).wait();
        // down: d_y[1,H] = d_a[1,I] @ d_down[e][H,I]^T
        mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans,
            1, H, I, 1.0f, d_a, I, d_down[e], I, 0.0f, d_y, H);
        q.wait();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_per_expert = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
    printf("Single expert: %.1f us (gate+up+SiLU+down)\n", us_per_expert);

    // Full B70 partial: TOPK_B70 experts with routing weights
    float weights[8] = {0.15f, 0.12f, 0.10f}; // routing weights for 3 experts
    int eids[8] = {0, 1, 2};

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERS; iter++) {
        // Zero accumulator
        q.parallel_for(H, [=](auto i) { d_out[i] = 0.0f; }).wait();

        for (int k = 0; k < TOPK_B70; k++) {
            int e = eids[k] % N_EXP;
            float w = weights[k];
            mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans,
                1, I, H, 1.0f, d_x, H, d_gate[e], H, 0.0f, d_g, I);
            mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans,
                1, I, H, 1.0f, d_x, H, d_up[e], H, 0.0f, d_u, I);
            q.parallel_for(I, [=](auto i) { d_a[i] = d_g[i] / (1.0f + sycl::exp(-d_g[i])) * d_u[i]; });
            mkl::blas::row_major::gemm(q, mkl::transpose::nontrans, mkl::transpose::trans,
                1, H, I, w, d_a, I, d_down[e], I, 1.0f, d_out, H);
        }
        q.wait();
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double us_per_partial = std::chrono::duration<double, std::micro>(t3 - t2).count() / ITERS;
    printf("%d-expert partial: %.1f us\n", TOPK_B70, us_per_partial);

    // Per-layer and per-token
    double us_per_layer = us_per_partial;
    double ms_per_token = us_per_layer * N_LAYERS / 1e3;
    printf("\nPer layer: %.1f us\n", us_per_layer);
    printf("Per token (40 layers): %.2f ms\n", ms_per_token);
    printf("Projected decode (compute only): %.0f tok/s\n", 1e3 / ms_per_token);

    // Transport overhead (from earlier measurement)
    double transport_per_layer = 43.5; // us
    double total_per_layer = (us_per_layer > transport_per_layer) ? us_per_layer : transport_per_layer;
    printf("\nWith transport (%.1f us/layer, overlapped):\n", transport_per_layer);
    printf("Per token: %.2f ms → %.0f tok/s\n", total_per_layer * N_LAYERS / 1e3,
           1e3 / (total_per_layer * N_LAYERS / 1e3));

    // Correctness check: compare with CPU reference
    printf("\n=== Correctness check ===\n");
    // Copy B70 output to host
    q.memcpy(h_out, d_out, H * sizeof(float)).wait();
    // CPU reference for one expert
    float *cpu_gate = new float[I * H];
    float *cpu_up = new float[I * H];
    float *cpu_down = new float[H * I];
    // Fill same as expert 0
    srand(0);
    for (int i = 0; i < I * H; i++) cpu_gate[i] = cpu_up[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    for (int i = 0; i < H * I; i++) cpu_down[i] = (float)(rand() % 100) / 100.0f - 0.5f;

    float cpu_out[H]; memset(cpu_out, 0, sizeof(cpu_out));
    for (int k = 0; k < TOPK_B70; k++) {
        float w = weights[k];
        float g[I], u[I], a[I], y[H];
        for (int o = 0; o < I; o++) {
            float gs = 0, us = 0;
            for (int ii = 0; ii < H; ii++) { gs += h_x[ii] * cpu_gate[o*H+ii]; us += h_x[ii] * cpu_up[o*H+ii]; }
            g[o] = gs; u[o] = us; a[o] = gs / (1.0f + expf(-gs)) * us;
        }
        for (int o = 0; o < H; o++) {
            float s = 0;
            for (int ii = 0; ii < I; ii++) s += a[ii] * cpu_down[o*I+ii];
            cpu_out[o] += w * s;
        }
    }
    float maxerr = 0;
    for (int i = 0; i < H; i++) {
        float err = fabsf(h_out[i] - cpu_out[i]);
        if (err > maxerr) maxerr = err;
    }
    printf("Max abs error vs CPU: %e (%s)\n", maxerr, maxerr < 1e-3 ? "PASS" : "CHECK");

    // Cleanup
    for (int e = 0; e < N_EXP; e++) {
        sycl::free(d_gate[e], q); sycl::free(d_up[e], q); sycl::free(d_down[e], q);
    }
    sycl::free(d_x, q); sycl::free(d_g, q); sycl::free(d_u, q);
    sycl::free(d_a, q); sycl::free(d_y, q); sycl::free(d_out, q);
    sycl::free(h_x, q); sycl::free(h_out, q);
    delete[] cpu_gate; delete[] cpu_up; delete[] cpu_down;
    printf("\n[done]\n");
    return 0;
}
