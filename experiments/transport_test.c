/*
 * transport_test.c — Cross-vendor DMA latency: 5090 ↔ B70 via pinned host.
 *
 * Links both CUDA runtime and Level Zero loader in one process.
 * Measures real round-trip latency for the Shooting Brake activation/partial
 * transport path: CUDA VRAM → pinned host → Level Zero B70 VRAM → pinned host → CUDA VRAM.
 *
 * Build:
 *   gcc -O2 transport_test.c -o transport_test \
 *       -I/usr/include/level_zero -lze_loader -lcudart -lm
 *
 * Run:
 *   ./transport_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <level_zero/ze_api.h>
#include <math.h>
#include <cuda_runtime.h>

#define HIDDEN 2048
#define ACT_BYTES  (HIDDEN * 2)   /* BF16 activation = 4 KB */
#define OUT_BYTES  (HIDDEN * 4)   /* FP32 partial = 8 KB */
#define ITERS 1000
#define WARMUP 50

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

#define ZE_CHECK(call) do { \
    ze_result_t r = (call); \
    if (r != ZE_RESULT_SUCCESS) { \
        fprintf(stderr, "Level Zero error 0x%x at %s:%d\n", r, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    cudaError_t r = (call); \
    if (r != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(r), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

int main(void) {
    /* ---- Initialize Level Zero ---- */
    ZE_CHECK(zeInit(0));

    uint32_t n_drivers = 0;
    ZE_CHECK(zeDriverGet(&n_drivers, NULL));
    ze_driver_handle_t driver;
    ZE_CHECK(zeDriverGet(&n_drivers, &driver));

    uint32_t n_devices = 0;
    ZE_CHECK(zeDeviceGet(driver, &n_devices, NULL));
    printf("Level Zero: %u device(s)\n", n_devices);

    ze_device_handle_t devices[8];
    ZE_CHECK(zeDeviceGet(driver, &n_devices, devices));

    /* Find the B70 (Battlemage) — look for discrete GPU */
    ze_device_handle_t b70 = NULL;
    for (uint32_t i = 0; i < n_devices; i++) {
        ze_device_properties_t props;
        memset(&props, 0, sizeof(props));
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        ZE_CHECK(zeDeviceGetProperties(devices[i], &props));
        printf("  L0 dev %u: %s (type %u, vendor 0x%x)\n", i, props.name, props.type, props.vendorId);
        if (props.type == ZE_DEVICE_TYPE_GPU && props.vendorId == 0x8086) {
            b70 = devices[i];
        }
    }
    if (!b70) { fprintf(stderr, "No Intel GPU found via Level Zero\n"); return 1; }

    ze_context_desc_t ctx_desc = { .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC, .pNext = NULL, .flags = 0 };
    ze_context_handle_t ctx;
    ZE_CHECK(zeContextCreate(driver, &ctx_desc, &ctx));

    /* Create command queue on B70 */
    ze_command_queue_desc_t q_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext = NULL,
        .ordinal = 0,  /* compute ordinal */
        .index = 0,
        .flags = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_DEFAULT,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    ze_command_queue_handle_t b70_queue;
    ZE_CHECK(zeCommandQueueCreate(ctx, b70, &q_desc, &b70_queue));

    /* Allocate device memory on B70 for activation and partial */
    void *b70_act = NULL, *b70_out = NULL;
    ze_device_mem_alloc_desc_t d_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
        .ordinal = 0
    };
    ZE_CHECK(zeMemAllocDevice(ctx, &d_desc, ACT_BYTES, 64, b70, &b70_act));
    ZE_CHECK(zeMemAllocDevice(ctx, &d_desc, OUT_BYTES, 64, b70, &b70_out));
    printf("B70 device memory: act=%p out=%p\n", b70_act, b70_out);

    /* Allocate pinned host memory (CUDA portable) */
    void *host_act = NULL, *host_out = NULL;
    CU_CHECK(cudaHostAlloc(&host_act, ACT_BYTES, cudaHostAllocPortable | cudaHostAllocMapped));
    CU_CHECK(cudaHostAlloc(&host_out, OUT_BYTES, cudaHostAllocPortable | cudaHostAllocMapped));
    memset(host_act, 0x42, ACT_BYTES);
    memset(host_out, 0x00, OUT_BYTES);

    /* Allocate CUDA device memory (simulating 5090 activation + result) */
    void *cuda_act = NULL, *cuda_out = NULL;
    CU_CHECK(cudaMalloc(&cuda_act, ACT_BYTES));
    CU_CHECK(cudaMalloc(&cuda_out, OUT_BYTES));
    CU_CHECK(cudaMemset(cuda_act, 0x55, ACT_BYTES));

    /* Create Level Zero command list for B70 copies */
    ze_command_list_desc_t cl_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = NULL,
        .commandQueueGroupOrdinal = 0,
        .flags = 0
    };
    ze_command_list_handle_t b70_cl;
    ZE_CHECK(zeCommandListCreate(ctx, b70, &cl_desc, &b70_cl));

    printf("\n=== Transport Test: 5090 → pinned host → B70 → pinned host → 5090 ===\n");
    printf("Activation: %d bytes (BF16), Partial: %d bytes (FP32)\n", ACT_BYTES, OUT_BYTES);
    printf("Iterations: %d (after %d warmup)\n\n", ITERS, WARMUP);

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        /* 5090 → pinned host */
        CU_CHECK(cudaMemcpy(host_act, cuda_act, ACT_BYTES, cudaMemcpyDeviceToHost));
        CU_CHECK(cudaStreamSynchronize(0));

        /* pinned host → B70 */
        ZE_CHECK(zeCommandListReset(b70_cl));
        ZE_CHECK(zeCommandListAppendMemoryCopy(b70_cl, b70_act, host_act, ACT_BYTES, NULL, 0, NULL));
        ZE_CHECK(zeCommandListAppendBarrier(b70_cl, NULL, 0, NULL));
        ZE_CHECK(zeCommandListAppendMemoryCopy(b70_cl, host_out, b70_out, OUT_BYTES, NULL, 0, NULL));
        ZE_CHECK(zeCommandListClose(b70_cl));
        ZE_CHECK(zeCommandQueueExecuteCommandLists(b70_queue, 1, &b70_cl, NULL));
        ZE_CHECK(zeCommandQueueSynchronize(b70_queue, UINT64_MAX));

        /* pinned host → 5090 */
        CU_CHECK(cudaMemcpy(cuda_out, host_out, OUT_BYTES, cudaMemcpyHostToDevice));
        CU_CHECK(cudaStreamSynchronize(0));
    }

    /* Measure full round-trip */
    double t_start = now_us();
    for (int i = 0; i < ITERS; i++) {
        CU_CHECK(cudaMemcpy(host_act, cuda_act, ACT_BYTES, cudaMemcpyDeviceToHost));
        CU_CHECK(cudaStreamSynchronize(0));

        ZE_CHECK(zeCommandListReset(b70_cl));
        ZE_CHECK(zeCommandListAppendMemoryCopy(b70_cl, b70_act, host_act, ACT_BYTES, NULL, 0, NULL));
        ZE_CHECK(zeCommandListAppendBarrier(b70_cl, NULL, 0, NULL));
        ZE_CHECK(zeCommandListAppendMemoryCopy(b70_cl, host_out, b70_out, OUT_BYTES, NULL, 0, NULL));
        ZE_CHECK(zeCommandListClose(b70_cl));
        ZE_CHECK(zeCommandQueueExecuteCommandLists(b70_queue, 1, &b70_cl, NULL));
        ZE_CHECK(zeCommandQueueSynchronize(b70_queue, UINT64_MAX));

        CU_CHECK(cudaMemcpy(cuda_out, host_out, OUT_BYTES, cudaMemcpyHostToDevice));
        CU_CHECK(cudaStreamSynchronize(0));
    }
    double t_total = now_us() - t_start;
    double t_per_trip = t_total / ITERS;

    printf("Full round-trip: %.1f us/call\n", t_per_trip);
    printf("  Per layer (40 layers): %.2f ms\n", t_per_trip * 40 / 1e3);
    printf("  Per token transport:   %.2f ms\n", t_per_trip * 40 / 1e3);

    /* Measure individual components */
    /* 5090 → pinned only */
    t_start = now_us();
    for (int i = 0; i < ITERS; i++) {
        CU_CHECK(cudaMemcpy(host_act, cuda_act, ACT_BYTES, cudaMemcpyDeviceToHost));
        CU_CHECK(cudaStreamSynchronize(0));
    }
    double t_d2h = (now_us() - t_start) / ITERS;
    printf("\n5090 → pinned host (D2H):  %.1f us\n", t_d2h);

    /* pinned → 5090 only */
    t_start = now_us();
    for (int i = 0; i < ITERS; i++) {
        CU_CHECK(cudaMemcpy(cuda_out, host_out, OUT_BYTES, cudaMemcpyHostToDevice));
        CU_CHECK(cudaStreamSynchronize(0));
    }
    double t_h2d = (now_us() - t_start) / ITERS;
    printf("pinned host → 5090 (H2D):  %.1f us\n", t_h2d);

    /* pinned → B70 only */
    t_start = now_us();
    for (int i = 0; i < ITERS; i++) {
        ZE_CHECK(zeCommandListReset(b70_cl));
        ZE_CHECK(zeCommandListAppendMemoryCopy(b70_cl, b70_act, host_act, ACT_BYTES, NULL, 0, NULL));
        ZE_CHECK(zeCommandListClose(b70_cl));
        ZE_CHECK(zeCommandQueueExecuteCommandLists(b70_queue, 1, &b70_cl, NULL));
        ZE_CHECK(zeCommandQueueSynchronize(b70_queue, UINT64_MAX));
    }
    double t_h2b70 = (now_us() - t_start) / ITERS;
    printf("pinned host → B70 (H2D):   %.1f us\n", t_h2b70);

    /* B70 → pinned only */
    t_start = now_us();
    for (int i = 0; i < ITERS; i++) {
        ZE_CHECK(zeCommandListReset(b70_cl));
        ZE_CHECK(zeCommandListAppendMemoryCopy(b70_cl, host_out, b70_out, OUT_BYTES, NULL, 0, NULL));
        ZE_CHECK(zeCommandListClose(b70_cl));
        ZE_CHECK(zeCommandQueueExecuteCommandLists(b70_queue, 1, &b70_cl, NULL));
        ZE_CHECK(zeCommandQueueSynchronize(b70_queue, UINT64_MAX));
    }
    double t_b702h = (now_us() - t_start) / ITERS;
    printf("B70 → pinned host (D2H):   %.1f us\n", t_b702h);

    printf("\n=== Summary ===\n");
    printf("Component breakdown (should sum to ~round-trip):\n");
    printf("  5090 D2H:     %.1f us\n", t_d2h);
    printf("  host → B70:   %.1f us\n", t_h2b70);
    printf("  B70 → host:   %.1f us\n", t_b702h);
    printf("  host → 5090:  %.1f us\n", t_h2d);
    printf("  Sum:          %.1f us\n", t_d2h + t_h2b70 + t_b702h + t_h2d);
    printf("  Measured RT:  %.1f us\n", t_per_trip);
    printf("\nFor Qwen 3.6 (40 layers, ~2-3 B70 experts/layer):\n");
    printf("  Transport per token: %.2f ms\n", t_per_trip * 40 / 1e3);
    printf("  B70 compute per token (3 exp × 45us × 40 layers): %.2f ms\n", 3 * 45.0 * 40 / 1e3);
    printf("  Combined (overlapped): ~%.2f ms\n",
           ( (t_per_trip * 40 / 1e3) > (3 * 45.0 * 40 / 1e3) ) ? (t_per_trip * 40 / 1e3) : (3 * 45.0 * 40 / 1e3));

    /* Cleanup */
    ZE_CHECK(zeCommandListDestroy(b70_cl));
    ZE_CHECK(zeMemFree(ctx, b70_act));
    ZE_CHECK(zeMemFree(ctx, b70_out));
    ZE_CHECK(zeCommandQueueDestroy(b70_queue));
    ZE_CHECK(zeContextDestroy(ctx));
    CU_CHECK(cudaFree(cuda_act));
    CU_CHECK(cudaFree(cuda_out));
    CU_CHECK(cudaFreeHost(host_act));
    CU_CHECK(cudaFreeHost(host_out));

    printf("\n[done] transport test complete\n");
    return 0;
}
