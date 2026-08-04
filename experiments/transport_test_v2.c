#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <level_zero/ze_api.h>
#include <cuda_runtime.h>

#define H 2048
#define ACT_B  (H * 2)
#define OUT_B  (H * 4)
#define ITERS 1000
#define WARMUP 50

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}
#define ZC(c) do{ze_result_t _r=(c); if(_r!=ZE_RESULT_SUCCESS){fprintf(stderr,"ZE err 0x%x @%d\n",_r,__LINE__);exit(1);}}while(0)
#define CC(c) do{cudaError_t _r=(c); if(_r!=cudaSuccess){fprintf(stderr,"CUDA: %s @%d\n",cudaGetErrorString(_r),__LINE__);exit(1);}}while(0)

int main(void) {
    ZC(zeInit(0));
    uint32_t ndr=0; ze_driver_handle_t drv;
    ZC(zeDriverGet(&ndr,&drv));

    uint32_t ndev=0;
    ZC(zeDeviceGet(drv,&ndev,NULL));
    ze_device_handle_t devs[8], b70=NULL;
    ZC(zeDeviceGet(drv,&ndev,devs));
    for(uint32_t i=0;i<ndev;i++){
        ze_device_properties_t p={.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        ZC(zeDeviceGetProperties(devs[i],&p));
        printf("L0 dev %u: %s (type %u vendor 0x%x)\n",i,p.name,p.type,p.vendorId);
        if(p.type==ZE_DEVICE_TYPE_GPU && p.vendorId==0x8086) b70=devs[i];
    }
    if(!b70){fprintf(stderr,"no Intel GPU\n");return 1;}

    /* Find copy queue group */
    uint32_t nqg=0;
    ZC(zeDeviceGetCommandQueueGroupProperties(b70,&nqg,NULL));
    ze_command_queue_group_properties_t qgp[8];
    for(uint32_t i=0;i<nqg;i++) qgp[i]=(ze_command_queue_group_properties_t){.stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES};
    ZC(zeDeviceGetCommandQueueGroupProperties(b70,&nqg,qgp));
    uint32_t copy_ord=0; int has_copy=0, compute_ord=0;
    for(uint32_t i=0;i<nqg;i++){
        int is_compute = qgp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
        int is_copy = qgp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
        printf("  group %u: compute=%d copy=%d\n",i,is_compute,is_copy);
        if(is_copy && !has_copy){copy_ord=i;has_copy=1;}
        if(is_compute) compute_ord=i;
    }

    ze_context_desc_t cd={.stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx;
    ZC(zeContextCreate(drv,&cd,&ctx));

    /* Immediate command list on copy group (lowest latency) */
    ze_command_queue_desc_t qd={
        .stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext=NULL,
        .ordinal=has_copy?copy_ord:compute_ord,
        .index=0,.flags=0,
        .mode=ZE_COMMAND_QUEUE_MODE_DEFAULT,
        .priority=ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    ze_command_list_handle_t imm;
    ZC(zeCommandListCreateImmediate(ctx,b70,&qd,&imm));
    printf("Immediate CL on group %u (%s)\n",has_copy?copy_ord:compute_ord,has_copy?"COPY":"COMPUTE");

    /* Device memory */
    ze_device_mem_alloc_desc_t dd={.stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void *b70_a,*b70_o;
    ZC(zeMemAllocDevice(ctx,&dd,ACT_B,64,b70,&b70_a));
    ZC(zeMemAllocDevice(ctx,&dd,OUT_B,64,b70,&b70_o));

    /* Pinned host */
    void *ha,*ho;
    CC(cudaHostAlloc(&ha,ACT_B,cudaHostAllocPortable));
    CC(cudaHostAlloc(&ho,OUT_B,cudaHostAllocPortable));
    memset(ha,0x42,ACT_B);

    /* CUDA buffers */
    void *ca,*co;
    CC(cudaMalloc(&ca,ACT_B));
    CC(cudaMalloc(&co,OUT_B));

    printf("\n=== Transport Test v2 (immediate CL) ===\n");
    printf("Activation: %d B, Partial: %d B, Iters: %d\n",ACT_B,OUT_B,ITERS);

    /* Warmup */
    for(int i=0;i<WARMUP;i++){
        CC(cudaMemcpy(ha,ca,ACT_B,cudaMemcpyDeviceToHost));
        CC(cudaStreamSynchronize(0));
        ZC(zeCommandListAppendMemoryCopy(imm,b70_a,ha,ACT_B,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
        ZC(zeCommandListAppendMemoryCopy(imm,ho,b70_o,OUT_B,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
        CC(cudaMemcpy(co,ho,OUT_B,cudaMemcpyHostToDevice));
        CC(cudaStreamSynchronize(0));
    }

    /* Full round-trip */
    double t0=now_us();
    for(int i=0;i<ITERS;i++){
        CC(cudaMemcpy(ha,ca,ACT_B,cudaMemcpyDeviceToHost));
        CC(cudaStreamSynchronize(0));
        ZC(zeCommandListAppendMemoryCopy(imm,b70_a,ha,ACT_B,NULL,0,NULL));
        ZC(zeCommandListAppendBarrier(imm,NULL,0,NULL));
        ZC(zeCommandListAppendMemoryCopy(imm,ho,b70_o,OUT_B,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
        CC(cudaMemcpy(co,ho,OUT_B,cudaMemcpyHostToDevice));
        CC(cudaStreamSynchronize(0));
    }
    double t_rt=(now_us()-t0)/ITERS;
    printf("\nFull round-trip: %.1f us\n",t_rt);

    /* B70 H2D only */
    t0=now_us();
    for(int i=0;i<ITERS;i++){
        ZC(zeCommandListAppendMemoryCopy(imm,b70_a,ha,ACT_B,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
    }
    double t_h2d=(now_us()-t0)/ITERS;
    printf("pinned → B70 (4KB): %.1f us (%.1f MB/s)\n",t_h2d,ACT_B/t_h2d);

    /* B70 D2H only */
    t0=now_us();
    for(int i=0;i<ITERS;i++){
        ZC(zeCommandListAppendMemoryCopy(imm,ho,b70_o,OUT_B,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
    }
    double t_d2h=(now_us()-t0)/ITERS;
    printf("B70 → pinned (8KB): %.1f us (%.1f MB/s)\n",t_d2h,OUT_B/t_d2h);

    /* Bandwidth: 1MB transfer */
    int mb=1<<20;
    void *b70_mb,*h_mb;
    ZC(zeMemAllocDevice(ctx,&dd,mb,64,b70,&b70_mb));
    CC(cudaHostAlloc(&h_mb,mb,cudaHostAllocPortable));
    /* warmup */
    for(int i=0;i<5;i++){
        ZC(zeCommandListAppendMemoryCopy(imm,b70_mb,h_mb,mb,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
    }
    t0=now_us();
    for(int i=0;i<200;i++){
        ZC(zeCommandListAppendMemoryCopy(imm,b70_mb,h_mb,mb,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
    }
    double t_mb=(now_us()-t0)/200;
    printf("\nB70 H2D 1MB: %.1f us (%.0f MB/s)\n",t_mb,mb/t_mb);

    /* 5090 D2H + H2D for comparison */
    t0=now_us();
    for(int i=0;i<ITERS;i++){
        CC(cudaMemcpy(ha,ca,ACT_B,cudaMemcpyDeviceToHost));
        CC(cudaStreamSynchronize(0));
    }
    printf("5090 D2H (4KB): %.1f us\n",(now_us()-t0)/ITERS);

    printf("\n=== Qwen 3.6 projection ===\n");
    printf("Per layer transport: %.1f us\n",t_rt);
    printf("Per token (40 layers): %.2f ms\n",t_rt*40/1e3);
    printf("B70 compute (3 exp × 40 layers): %.2f ms\n",3*45.0*40/1e3);
    double transport_tok = t_rt*40/1e3;
    double compute_tok = 3*45.0*40/1e3;
    printf("Overlapped: %.2f ms → ~%.0f tok/s (transport-bound)\n",
           (transport_tok>compute_tok)?transport_tok:compute_tok,
           1e3/((transport_tok>compute_tok)?transport_tok:compute_tok));

    ZC(zeMemFree(ctx,b70_a));ZC(zeMemFree(ctx,b70_o));
    ZC(zeMemFree(ctx,b70_mb));
    ZC(zeCommandListDestroy(imm));
    ZC(zeContextDestroy(ctx));
    CC(cudaFree(ca));CC(cudaFree(co));
    CC(cudaFreeHost(ha));CC(cudaFreeHost(ho));CC(cudaFreeHost(h_mb));
    printf("\n[done]\n");
    return 0;
}
