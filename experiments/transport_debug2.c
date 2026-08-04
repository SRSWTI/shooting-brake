#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <level_zero/ze_api.h>
#include <cuda_runtime.h>

#define ZC(c) do{ze_result_t _r=(c); if(_r!=ZE_RESULT_SUCCESS){fprintf(stderr,"ZE err 0x%x @%d\n",_r,__LINE__);exit(1);}}while(0)
#define CC(c) do{cudaError_t _r=(c); if(_r!=cudaSuccess){fprintf(stderr,"CUDA: %s @%d\n",cudaGetErrorString(_r),__LINE__);exit(1);}}while(0)

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

int main(void) {
    ZC(zeInit(0));

    /* Enumerate ALL drivers properly */
    uint32_t ndr=0;
    ZC(zeDriverGet(&ndr,NULL));
    printf("Drivers: %u\n",ndr);
    ze_driver_handle_t drvs[8];
    ZC(zeDriverGet(&ndr,drvs));

    /* Find B70 across all drivers */
    ze_driver_handle_t b70_drv=NULL;
    ze_device_handle_t b70=NULL;
    for(uint32_t d=0; d<ndr; d++){
        uint32_t ndev=0;
        ZC(zeDeviceGet(drvs[d],&ndev,NULL));
        ze_device_handle_t devs[8];
        ZC(zeDeviceGet(drvs[d],&ndev,devs));
        for(uint32_t i=0;i<ndev;i++){
            ze_device_properties_t p={.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            ZC(zeDeviceGetProperties(devs[i],&p));
            printf("  drv %u dev %u: %-40s type=%u vendor=0x%x\n",d,i,p.name,p.type,p.vendorId);
            /* B70 = discrete Intel GPU (vendor 0x8086, not integrated) */
            if(p.vendorId==0x8086 && p.type==ZE_DEVICE_TYPE_GPU){
                /* Check if discrete (not integrated) */
                ze_device_memory_properties_t mp={.stype=ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES};
                uint32_t nmem=0;
                zeDeviceGetMemoryProperties(devs[i],&nmem,NULL);
                ze_device_memory_properties_t mps[4];
                for(uint32_t m=0;m<nmem&&m<4;m++) mps[m]=(ze_device_memory_properties_t){.stype=ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES};
                zeDeviceGetMemoryProperties(devs[i],&nmem,mps);
                unsigned long long maxmem=0;
                for(uint32_t m=0;m<nmem;m++) if(mps[m].totalSize>maxmem) maxmem=mps[m].totalSize;
                printf("    max memory: %.1f GB\n",maxmem/1e9);
                if(maxmem > 8e9 && maxmem < 40e9){ /* >8GB = discrete B70, not iGPU */
                    b70=devs[i]; b70_drv=drvs[d];
                    printf("    → SELECTED as B70\n");
                }
            }
        }
    }
    if(!b70){fprintf(stderr,"B70 not found\n");return 1;}

    /* Queue groups */
    uint32_t nqg=0;
    ZC(zeDeviceGetCommandQueueGroupProperties(b70,&nqg,NULL));
    ze_command_queue_group_properties_t qgp[8];
    for(uint32_t i=0;i<nqg;i++) qgp[i]=(ze_command_queue_group_properties_t){.stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES};
    ZC(zeDeviceGetCommandQueueGroupProperties(b70,&nqg,qgp));
    uint32_t use_ord=0;
    for(uint32_t i=0;i<nqg;i++){
        int cp=qgp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
        int co=qgp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
        printf("  qgroup %u: compute=%d copy=%d queues=%u\n",i,co,cp,qgp[i].numQueues);
        if(cp) use_ord=i;
        if(!cp && !use_ord) use_ord=i;
    }

    /* Context */
    ze_context_desc_t cd={.stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx;
    ZC(zeContextCreate(b70_drv,&cd,&ctx));

    /* Immediate command list */
    ze_command_queue_desc_t qd={
        .stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,.pNext=NULL,
        .ordinal=use_ord,.index=0,.flags=0,
        .mode=ZE_COMMAND_QUEUE_MODE_DEFAULT,
        .priority=ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    ze_command_list_handle_t imm;
    ZC(zeCommandListCreateImmediate(ctx,b70,&qd,&imm));
    printf("Immediate CL on group %u\n\n",use_ord);

    /* Allocs */
    ze_device_mem_alloc_desc_t dd={.stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void *b70_a,*b70_o;
    ZC(zeMemAllocDevice(ctx,&dd,4096,64,b70,&b70_a));
    ZC(zeMemAllocDevice(ctx,&dd,8192,64,b70,&b70_o));

    void *ha,*ho;
    CC(cudaHostAlloc(&ha,4096,cudaHostAllocPortable));
    CC(cudaHostAlloc(&ho,8192,cudaHostAllocPortable));
    memset(ha,0x42,4096);

    void *ca,*co;
    CC(cudaMalloc(&ca,4096));
    CC(cudaMalloc(&co,8192));

    printf("=== Transport Test (immediate CL) ===\n");
    printf("Activation 4KB, Partial 8KB, 1000 iters\n\n");

    /* Warmup */
    for(int i=0;i<50;i++){
        CC(cudaMemcpy(ha,ca,4096,cudaMemcpyDeviceToHost));
        CC(cudaStreamSynchronize(0));
        ZC(zeCommandListAppendMemoryCopy(imm,b70_a,ha,4096,NULL,0,NULL));
        ZC(zeCommandListAppendBarrier(imm,NULL,0,NULL));
        ZC(zeCommandListAppendMemoryCopy(imm,ho,b70_o,8192,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
        CC(cudaMemcpy(co,ho,8192,cudaMemcpyHostToDevice));
        CC(cudaStreamSynchronize(0));
    }

    /* Full round-trip */
    double t0=now_us();
    for(int i=0;i<1000;i++){
        CC(cudaMemcpy(ha,ca,4096,cudaMemcpyDeviceToHost));
        CC(cudaStreamSynchronize(0));
        ZC(zeCommandListAppendMemoryCopy(imm,b70_a,ha,4096,NULL,0,NULL));
        ZC(zeCommandListAppendBarrier(imm,NULL,0,NULL));
        ZC(zeCommandListAppendMemoryCopy(imm,ho,b70_o,8192,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
        CC(cudaMemcpy(co,ho,8192,cudaMemcpyHostToDevice));
        CC(cudaStreamSynchronize(0));
    }
    double t_rt=(now_us()-t0)/1000;
    printf("Full round-trip: %.1f us\n",t_rt);

    /* B70 H2D only */
    t0=now_us();
    for(int i=0;i<1000;i++){
        ZC(zeCommandListAppendMemoryCopy(imm,b70_a,ha,4096,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
    }
    printf("pinned → B70 (4KB): %.1f us (%.0f MB/s)\n",(now_us()-t0)/1000,4096/((now_us()-t0)/1000));

    /* B70 D2H only */
    t0=now_us();
    for(int i=0;i<1000;i++){
        ZC(zeCommandListAppendMemoryCopy(imm,ho,b70_o,8192,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
    }
    printf("B70 → pinned (8KB): %.1f us\n",(now_us()-t0)/1000);

    /* 1MB bandwidth */
    int mb=1<<20;
    void *b70_mb,*h_mb;
    ZC(zeMemAllocDevice(ctx,&dd,mb,64,b70,&b70_mb));
    CC(cudaHostAlloc(&h_mb,mb,cudaHostAllocPortable));
    for(int i=0;i<5;i++){ZC(zeCommandListAppendMemoryCopy(imm,b70_mb,h_mb,mb,NULL,0,NULL));ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));}
    t0=now_us();
    for(int i=0;i<200;i++){ZC(zeCommandListAppendMemoryCopy(imm,b70_mb,h_mb,mb,NULL,0,NULL));ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));}
    double t_mb=(now_us()-t0)/200;
    printf("\nB70 H2D 1MB: %.1f us (%.0f MB/s)\n",t_mb,mb/t_mb);

    printf("\n=== Qwen 3.6 projection ===\n");
    printf("Transport per token (40 layers): %.2f ms\n",t_rt*40/1e3);
    printf("B70 compute per token: %.2f ms\n",3*45.0*40/1e3);

    ZC(zeMemFree(ctx,b70_a));ZC(zeMemFree(ctx,b70_o));ZC(zeMemFree(ctx,b70_mb));
    ZC(zeCommandListDestroy(imm));ZC(zeContextDestroy(ctx));
    CC(cudaFree(ca));CC(cudaFree(co));
    CC(cudaFreeHost(ha));CC(cudaFreeHost(ho));CC(cudaFreeHost(h_mb));
    printf("\n[done]\n");
    return 0;
}
