#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <level_zero/ze_api.h>
#include <cuda_runtime.h>

#define ZC(c) do{ze_result_t _r=(c); if(_r!=ZE_RESULT_SUCCESS){fprintf(stderr,"ZE err 0x%x @%d\n",_r,__LINE__);exit(1);}}while(0)
#define CC(c) do{cudaError_t _r=(c); if(_r!=cudaSuccess){fprintf(stderr,"CUDA: %s @%d\n",cudaGetErrorString(_r),__LINE__);exit(1);}}while(0)

int main(void) {
    printf("step 1: zeInit\n"); fflush(stdout);
    ZC(zeInit(0));

    printf("step 2: zeDriverGet\n"); fflush(stdout);
    uint32_t ndr=0; ze_driver_handle_t drv;
    ZC(zeDriverGet(&ndr,&drv));
    printf("  drivers: %u\n",ndr);

    printf("step 3: zeDeviceGet\n"); fflush(stdout);
    uint32_t ndev=0;
    ZC(zeDeviceGet(drv,&ndev,NULL));
    printf("  devices: %u\n",ndev);
    ze_device_handle_t devs[8];
    ZC(zeDeviceGet(drv,&ndev,devs));

    ze_device_handle_t b70=NULL;
    for(uint32_t i=0;i<ndev;i++){
        ze_device_properties_t p={.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        ZC(zeDeviceGetProperties(devs[i],&p));
        printf("  dev %u: %s type=%u vendor=0x%x\n",i,p.name,p.type,p.vendorId);
        if(p.type==ZE_DEVICE_TYPE_GPU && p.vendorId==0x8086) b70=devs[i];
    }
    if(!b70){fprintf(stderr,"no Intel GPU\n");return 1;}

    printf("step 4: queue groups\n"); fflush(stdout);
    uint32_t nqg=0;
    ZC(zeDeviceGetCommandQueueGroupProperties(b70,&nqg,NULL));
    ze_command_queue_group_properties_t qgp[8];
    for(uint32_t i=0;i<nqg;i++) qgp[i]=(ze_command_queue_group_properties_t){.stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES};
    ZC(zeDeviceGetCommandQueueGroupProperties(b70,&nqg,qgp));
    uint32_t use_ord=0;
    for(uint32_t i=0;i<nqg;i++){
        int cp=qgp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
        int co=qgp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
        printf("  group %u: compute=%d copy=%d engines=%u\n",i,co,cp,qgp[i].numQueues);
        if(cp) use_ord=i;
    }

    printf("step 5: context\n"); fflush(stdout);
    ze_context_desc_t cd={.stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx;
    ZC(zeContextCreate(drv,&cd,&ctx));

    printf("step 6: immediate CL (group %u)\n",use_ord); fflush(stdout);
    ze_command_queue_desc_t qd={
        .stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,.pNext=NULL,
        .ordinal=use_ord,.index=0,.flags=0,
        .mode=ZE_COMMAND_QUEUE_MODE_DEFAULT,
        .priority=ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    ze_command_list_handle_t imm;
    ZC(zeCommandListCreateImmediate(ctx,b70,&qd,&imm));
    printf("  imm=%p\n",imm);

    printf("step 7: alloc device mem\n"); fflush(stdout);
    ze_device_mem_alloc_desc_t dd={.stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void *b70_a;
    ZC(zeMemAllocDevice(ctx,&dd,4096,64,b70,&b70_a));
    printf("  b70_a=%p\n",b70_a);

    printf("step 8: alloc pinned\n"); fflush(stdout);
    void *ha;
    CC(cudaHostAlloc(&ha,4096,cudaHostAllocPortable));
    memset(ha,0x42,4096);
    printf("  ha=%p\n",ha);

    printf("step 9: first copy\n"); fflush(stdout);
    ZC(zeCommandListAppendMemoryCopy(imm,b70_a,ha,4096,NULL,0,NULL));
    printf("  copy appended\n"); fflush(stdout);
    ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
    printf("  synced OK\n"); fflush(stdout);

    printf("step 10: timing loop\n"); fflush(stdout);
    double t0=(double)clock();
    for(int i=0;i<100;i++){
        ZC(zeCommandListAppendMemoryCopy(imm,b70_a,ha,4096,NULL,0,NULL));
        ZC(zeCommandListHostSynchronize(imm,UINT64_MAX));
    }
    printf("  100 copies: %.2f s\n",(double)(clock()-t0)/CLOCKS_PER_SEC);

    ZC(zeMemFree(ctx,b70_a));
    ZC(zeCommandListDestroy(imm));
    ZC(zeContextDestroy(ctx));
    CC(cudaFreeHost(ha));
    printf("[done]\n");
    return 0;
}
