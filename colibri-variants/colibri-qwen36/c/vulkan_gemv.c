/* vulkan_gemv.c -- Vulkan compute backend for qwen36 MoE expert GEMVs.
 * See vulkan_gemv.h for the design. Builds standalone with -D VG_SELFTEST for a
 * CPU-vs-GPU correctness check; linked into qwen36.c / qwen36_serve.c otherwise.
 *
 * Vulkan is loaded DYNAMICALLY (dlopen/LoadLibrary "vulkan-1") so the default
 * binary keeps zero compile-time dependency on Vulkan and silently falls back to
 * CPU when the loader or a compute device is absent.
 *
 * LOCAL EXTENSION (multi-GPU): the backend now drives up to VG_MAX_DEV Vulkan
 * devices in parallel. Experts are sharded by owner = eid % n_devices; each
 * device holds only its own experts' weights and computes only its share of a
 * (token, layer). Phase submissions go to all queues first and are then waited
 * on together, so the GPUs overlap. Device selection prefers DISCRETE_GPU and
 * skips CPU-type devices (llvmpipe); COLIBRI_GPUS=n limits the device count
 * (COLIBRI_GPUS=1 restores the old single-device behaviour).
 * Also fixed here: 64-bit slot offsets (the original 32-bit wbase arithmetic
 * overflows and aliases slots once cap*layers*slot_bytes exceeds 4 GB). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#define VK_LIB "vulkan-1.dll"
static void* dl_open(const char*n){ return LoadLibraryA(n); }
static void* dl_sym(void*l,const char*n){ return (void*)GetProcAddress((HMODULE)l,n); }
static void  dl_close(void*l){ FreeLibrary((HMODULE)l); }
#else
#include <dlfcn.h>
#define VK_LIB "libvulkan.so"
static void* dl_open(const char*n){ return dlopen(n,RTLD_NOW); }
static void* dl_sym(void*l,const char*n){ return dlsym(l,n); }
static void  dl_close(void*l){ dlclose(l); }
#endif

#include "vulkan_gemv.h"
#include "vulkan_core.h"
#include "vulkan_gemv_spv.h"
#include "vulkan_gemv_idp_spv.h"
#include "vulkan_gemv_int4_spv.h"

/* ---------- loaded Vulkan function pointers ---------- */
#define VK_FNS \
  X(vkCreateInstance) \
  X(vkEnumeratePhysicalDevices) \
  X(vkGetPhysicalDeviceQueueFamilyProperties) \
  X(vkGetPhysicalDeviceMemoryProperties) \
  X(vkGetPhysicalDeviceProperties) \
  X(vkGetPhysicalDeviceFeatures2) \
  X(vkDestroyInstance) \
  X(vkCreateDevice) \
  X(vkDestroyDevice) \
  X(vkGetDeviceQueue) \
  X(vkCreateBuffer) \
  X(vkDestroyBuffer) \
  X(vkGetBufferMemoryRequirements) \
  X(vkAllocateMemory) \
  X(vkFreeMemory) \
  X(vkBindBufferMemory) \
  X(vkMapMemory) \
  X(vkUnmapMemory) \
  X(vkFlushMappedMemoryRanges) \
  X(vkInvalidateMappedMemoryRanges) \
  X(vkCreateShaderModule) \
  X(vkDestroyShaderModule) \
  X(vkCreateDescriptorSetLayout) \
  X(vkDestroyDescriptorSetLayout) \
  X(vkCreateDescriptorPool) \
  X(vkDestroyDescriptorPool) \
  X(vkAllocateDescriptorSets) \
  X(vkUpdateDescriptorSets) \
  X(vkCreatePipelineLayout) \
  X(vkDestroyPipelineLayout) \
  X(vkCreateComputePipelines) \
  X(vkDestroyPipeline) \
  X(vkCreateCommandPool) \
  X(vkDestroyCommandPool) \
  X(vkAllocateCommandBuffers) \
  X(vkFreeCommandBuffers) \
  X(vkResetCommandBuffer) \
  X(vkBeginCommandBuffer) \
  X(vkEndCommandBuffer) \
  X(vkCmdBindPipeline) \
  X(vkCmdBindDescriptorSets) \
  X(vkCmdDispatch) \
  X(vkQueueSubmit) \
  X(vkQueueWaitIdle)

#define X(name) static PFN_##name g_##name = NULL;
VK_FNS
#undef X

static void* g_lib = NULL;
static int g_vg_ok = 0;

/* ---------- static sizes ---------- */
static int g_hidden=0, g_inter=0, g_cap=0, g_topk=0, g_nlayers=0, g_nslots=0;
static uint64_t g_tick = 0;
static int g_use_idp=0;              /* 1 if IDP active (on ALL devices) */
static int g_use_int4=0;             /* 1 if int4 path active (on ALL devices) */
static int g_weight_bits=8;          /* 4 or 8 */
static int g_dbg=0;                  /* set by MOE_DBG env */

/* per-slot byte/float layout (precomputed) */
static uint64_t g_slot_wbytes = 0;   /* bytes per slot in w  (3 * D * Ih [/2 for int4]) */
static uint32_t g_slot_sfloats = 0;  /* floats per slot in s (3 * D) */

/* persistent buffers (host-visible + coherent) */
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *ptr; VkDeviceSize size; } Buf;

/* GPU slot table (mirror of CPU LRU; one table per device, experts live only
 * on their owning device = eid % g_ndev) */
typedef struct {
    int layer, eid, valid;
    int pool;                          /* index into the device-local weight pool, -1 = not resident */
    uint64_t used;
    uint32_t woff_g, woff_u, woff_d;   /* uint offsets into w (pool-relative) */
    uint32_t soff_g, soff_u, soff_d;   /* float offsets into s (pool-relative) */
} GSlot;

/* ---------- per-device context (multi-GPU extension) ---------- */
#define VG_MAX_DEV 4
#define VG_KMAX 64                    /* max experts per (token, layer) call */
typedef struct {
    VkPhysicalDevice pd;
    VkDevice         dev;
    VkQueue          q;
    uint32_t         qf;
    VkShaderModule   sm, sm_idp, sm_int4;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool dpool;
    VkDescriptorSet  ds;
    VkPipelineLayout pl;
    VkPipeline       pipe, pipe_idp, pipe_int4;
    VkCommandPool    cpool;
    VkCommandBuffer  cmd;
    Buf w, s, x, y, meta;
    GSlot *slot;
    /* device-local weight pool: the precompiled GEMV shaders address at most
     * 4 GB into the weight buffer (32-bit byte addressing), so the full slot
     * space cannot be mirrored 1:1. Weights live in a compact pool of
     * pool_n slots with LRU reuse; a pool miss just re-memcpys the expert
     * from host RAM into the pinned mirror. */
    int  pool_n;
    int *pool_gidx;        /* pool slot -> global slot idx (-1 free) */
    GSlot **pool_owner;    /* pool slot -> owning GSlot (for eviction) */
    uint32_t vendor_id, driver_ver;
    char name[256];
} VgDev;
static VgDev g_d[VG_MAX_DEV];
static int   g_ndev = 0;
static VkInstance g_inst = VK_NULL_HANDLE;

static int vg_owner(int eid){ return g_ndev>1 ? (eid % g_ndev) : 0; }

#define CHECK(r,msg) do{ if((r)!=VK_SUCCESS){ fprintf(stderr,"[vg] %s failed (code %d)\n", msg, (int)(r)); goto fail; } }while(0)
static uint64_t align_up64(uint64_t x, uint64_t a){ return (x + a - 1) & ~(a - 1); }

static VkResult vg_create_buf(VgDev *dv, VkDeviceSize size, VkBufferUsageFlags usage, Buf *b){
    VkBufferCreateInfo bi; memset(&bi,0,sizeof bi);
    bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size=size; bi.usage=usage; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VkResult r=g_vkCreateBuffer(dv->dev,&bi,NULL,&b->buf);
    if(r!=VK_SUCCESS) return r;
    VkMemoryRequirements req; g_vkGetBufferMemoryRequirements(dv->dev,b->buf,&req);
    VkPhysicalDeviceMemoryProperties mp; g_vkGetPhysicalDeviceMemoryProperties(dv->pd,&mp);
    uint32_t mi=~0u;
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((req.memoryTypeBits&(1u<<i)) &&
           (mp.memoryTypes[i].propertyFlags &
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
           ==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
            mi=i; break;
        }
    }
    if(mi==~0u){ fprintf(stderr,"[vg] no host-visible coherent memory type\n"); return VK_ERROR_UNKNOWN; }
    VkMemoryAllocateInfo ai; memset(&ai,0,sizeof ai);
    ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize=req.size; ai.memoryTypeIndex=mi;
    r=g_vkAllocateMemory(dv->dev,&ai,NULL,&b->mem);
    if(r!=VK_SUCCESS) return r;
    r=g_vkBindBufferMemory(dv->dev,b->buf,b->mem,0);
    if(r!=VK_SUCCESS) return r;
    b->size=req.size;
    return g_vkMapMemory(dv->dev,b->mem,0,req.size,0,&b->ptr);
}

static void vg_flush(VgDev *dv, Buf *b, VkDeviceSize off, VkDeviceSize len){
    if(len==0) return;
    VkMappedMemoryRange r; memset(&r,0,sizeof r);
    r.sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE; r.memory=b->mem; r.offset=off; r.size=len;
    g_vkFlushMappedMemoryRanges(dv->dev,1,&r);
}
static void vg_invalidate(VgDev *dv, Buf *b, VkDeviceSize off, VkDeviceSize len){
    if(len==0) return;
    VkMappedMemoryRange r; memset(&r,0,sizeof r);
    r.sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE; r.memory=b->mem; r.offset=off; r.size=len;
    g_vkInvalidateMappedMemoryRanges(dv->dev,1,&r);
}

static int vg_load_lib(void){
    g_lib=dl_open(VK_LIB);
#if !defined(_WIN32)
    if(!g_lib) g_lib=dl_open("libvulkan.so.1"); /* LOCAL FIX: runtime-only systems ship the .so.1, not the dev symlink */
#endif
    if(!g_lib){ fprintf(stderr,"[vg] cannot load %s\n", VK_LIB); return -1; }
#define X(name) \
    g_##name=(PFN_##name)dl_sym(g_lib,#name); \
    if(!g_##name){ fprintf(stderr,"[vg] missing %s\n", #name); return -1; }
    VK_FNS
#undef X
    return 0;
}

/* ---------- per-device teardown ---------- */
static void vg_dev_shutdown(VgDev *dv){
    if(!dv->dev){ memset(dv,0,sizeof *dv); return; }
    if(dv->slot){ free(dv->slot); dv->slot=NULL; }
    if(dv->pool_gidx){ free(dv->pool_gidx); dv->pool_gidx=NULL; }
    if(dv->pool_owner){ free(dv->pool_owner); dv->pool_owner=NULL; }
    if(dv->cmd)  g_vkFreeCommandBuffers(dv->dev,dv->cpool,1,&dv->cmd);
    if(dv->cpool)g_vkDestroyCommandPool(dv->dev,dv->cpool,NULL);
    if(dv->pipe) g_vkDestroyPipeline(dv->dev,dv->pipe,NULL);
    if(dv->pipe_idp) g_vkDestroyPipeline(dv->dev,dv->pipe_idp,NULL);
    if(dv->pipe_int4) g_vkDestroyPipeline(dv->dev,dv->pipe_int4,NULL);
    if(dv->sm_idp) g_vkDestroyShaderModule(dv->dev,dv->sm_idp,NULL);
    if(dv->sm_int4) g_vkDestroyShaderModule(dv->dev,dv->sm_int4,NULL);
    if(dv->pl)   g_vkDestroyPipelineLayout(dv->dev,dv->pl,NULL);
    if(dv->dpool)g_vkDestroyDescriptorPool(dv->dev,dv->dpool,NULL);
    if(dv->dsl)  g_vkDestroyDescriptorSetLayout(dv->dev,dv->dsl,NULL);
    if(dv->sm)   g_vkDestroyShaderModule(dv->dev,dv->sm,NULL);
    if(dv->y.buf){ g_vkDestroyBuffer(dv->dev,dv->y.buf,NULL); g_vkFreeMemory(dv->dev,dv->y.mem,NULL); }
    if(dv->x.buf){ g_vkDestroyBuffer(dv->dev,dv->x.buf,NULL); g_vkFreeMemory(dv->dev,dv->x.mem,NULL); }
    if(dv->s.buf){ g_vkDestroyBuffer(dv->dev,dv->s.buf,NULL); g_vkFreeMemory(dv->dev,dv->s.mem,NULL); }
    if(dv->w.buf){ g_vkDestroyBuffer(dv->dev,dv->w.buf,NULL); g_vkFreeMemory(dv->dev,dv->w.mem,NULL); }
    if(dv->meta.buf){ g_vkDestroyBuffer(dv->dev,dv->meta.buf,NULL); g_vkFreeMemory(dv->dev,dv->meta.mem,NULL); }
    g_vkDestroyDevice(dv->dev,NULL);
    memset(dv,0,sizeof *dv);
}

/* ---------- per-device init: logical device, pipelines, buffers ----------
 * Returns 0 on success. On failure the device is torn down and skipped. */
static int vg_dev_init(VgDev *dv, VkPhysicalDevice pd, const char *name,
                       uint32_t vendor_id, uint32_t driver_ver,
                       int idp_supported_dev, int int8_supported_dev){
    memset(dv,0,sizeof *dv);
    dv->pd=pd; dv->qf=~0u;
    dv->vendor_id=vendor_id; dv->driver_ver=driver_ver;
    snprintf(dv->name,sizeof dv->name,"%s",name);

    uint32_t nf=0; g_vkGetPhysicalDeviceQueueFamilyProperties(pd,&nf,NULL);
    VkQueueFamilyProperties *qfp=malloc(nf*sizeof(VkQueueFamilyProperties));
    g_vkGetPhysicalDeviceQueueFamilyProperties(pd,&nf,qfp);
    for(uint32_t i=0;i<nf;i++){ if(qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT){ dv->qf=i; break; } }
    free(qfp);
    if(dv->qf==~0u){ fprintf(stderr,"[vg] %s: no compute queue family\n", name); return -1; }

    float qpri=1.0f;
    VkDeviceQueueCreateInfo dq; memset(&dq,0,sizeof dq);
    dq.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; dq.queueFamilyIndex=dv->qf;
    dq.queueCount=1; dq.pQueuePriorities=&qpri;

    VkPhysicalDeviceFeatures2 f2; memset(&f2,0,sizeof f2);
    f2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dpf; memset(&dpf,0,sizeof dpf);
    dpf.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    VkPhysicalDeviceShaderFloat16Int8Features f16i8; memset(&f16i8,0,sizeof f16i8);
    f16i8.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    f2.pNext=&f16i8; f16i8.pNext=&dpf;
    dpf.shaderIntegerDotProduct = idp_supported_dev ? VK_TRUE : VK_FALSE;
    f16i8.shaderInt8 = int8_supported_dev ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo dci; memset(&dci,0,sizeof dci);
    dci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.queueCreateInfoCount=1;
    dci.pQueueCreateInfos=&dq; dci.pNext=&f2;
    CHECK(g_vkCreateDevice(pd,&dci,NULL,&dv->dev),"vkCreateDevice");
    g_vkGetDeviceQueue(dv->dev,dv->qf,0,&dv->q);

    /* float-path shader module */
    VkShaderModuleCreateInfo smi; memset(&smi,0,sizeof smi);
    smi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize=GEMV_SPV_LEN; smi.pCode=(const uint32_t*)GEMV_SPV;
    CHECK(g_vkCreateShaderModule(dv->dev,&smi,NULL,&dv->sm),"vkCreateShaderModule");

    /* descriptor set: 5 storage buffers (meta,x,w,scale,y) */
    VkDescriptorSetLayoutBinding bnd[5];
    for(int b=0;b<5;b++){ memset(&bnd[b],0,sizeof bnd[b]); bnd[b].binding=b;
        bnd[b].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bnd[b].descriptorCount=1;
        bnd[b].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dlci; memset(&dlci,0,sizeof dlci);
    dlci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dlci.bindingCount=5; dlci.pBindings=bnd;
    CHECK(g_vkCreateDescriptorSetLayout(dv->dev,&dlci,NULL,&dv->dsl),"desc layout");

    VkDescriptorPoolSize dps; dps.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; dps.descriptorCount=5;
    VkDescriptorPoolCreateInfo dpci; memset(&dpci,0,sizeof dpci);
    dpci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dpci.maxSets=1;
    dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    CHECK(g_vkCreateDescriptorPool(dv->dev,&dpci,NULL,&dv->dpool),"desc pool");

    VkDescriptorSetAllocateInfo dsai; memset(&dsai,0,sizeof dsai);
    dsai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dsai.descriptorPool=dv->dpool;
    dsai.descriptorSetCount=1; dsai.pSetLayouts=&dv->dsl;
    CHECK(g_vkAllocateDescriptorSets(dv->dev,&dsai,&dv->ds),"desc alloc");

    /* pipelines */
    VkPipelineShaderStageCreateInfo stage; memset(&stage,0,sizeof stage);
    stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; stage.module=dv->sm; stage.pName="main";
    VkPipelineLayoutCreateInfo plci; memset(&plci,0,sizeof plci);
    plci.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; plci.setLayoutCount=1; plci.pSetLayouts=&dv->dsl;
    CHECK(g_vkCreatePipelineLayout(dv->dev,&plci,NULL,&dv->pl),"pipeline layout");
    VkComputePipelineCreateInfo cpci; memset(&cpci,0,sizeof cpci);
    cpci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; cpci.stage=stage; cpci.layout=dv->pl;
    CHECK(g_vkCreateComputePipelines(dv->dev,VK_NULL_HANDLE,1,&cpci,NULL,&dv->pipe),"compute pipeline");

    /* int4 path (Shader-only, unpack nibbles + float GEMV) */
    if(g_weight_bits==4){
        VkShaderModuleCreateInfo i4smi; memset(&i4smi,0,sizeof i4smi);
        i4smi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        i4smi.codeSize=GEMV_INT4_SPV_LEN; i4smi.pCode=(const uint32_t*)GEMV_INT4_SPV;
        if(g_vkCreateShaderModule(dv->dev,&i4smi,NULL,&dv->sm_int4)==VK_SUCCESS){
            VkPipelineShaderStageCreateInfo i4stg; memset(&i4stg,0,sizeof i4stg);
            i4stg.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            i4stg.stage=VK_SHADER_STAGE_COMPUTE_BIT; i4stg.module=dv->sm_int4; i4stg.pName="main";
            VkComputePipelineCreateInfo i4cpi; memset(&i4cpi,0,sizeof i4cpi);
            i4cpi.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; i4cpi.stage=i4stg; i4cpi.layout=dv->pl;
            if(g_vkCreateComputePipelines(dv->dev,VK_NULL_HANDLE,1,&i4cpi,NULL,&dv->pipe_int4)!=VK_SUCCESS){
                g_vkDestroyShaderModule(dv->dev,dv->sm_int4,NULL); dv->sm_int4=VK_NULL_HANDLE;
            }
        }
        if(!dv->pipe_int4){ fprintf(stderr,"[vg] %s: int4 pipeline unavailable\n", name); goto fail; }
    }

    /* integer dot-product path (int8 weights only); per-device support was
     * checked by the caller, AMD 0x800184 blacklist applies as before */
    if(g_weight_bits!=4 && idp_supported_dev && int8_supported_dev){
        VkShaderModuleCreateInfo ismi; memset(&ismi,0,sizeof ismi);
        ismi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ismi.codeSize=GEMV_IDP_SPV_LEN; ismi.pCode=(const uint32_t*)GEMV_IDP_SPV;
        if(g_vkCreateShaderModule(dv->dev,&ismi,NULL,&dv->sm_idp)==VK_SUCCESS){
            VkPipelineShaderStageCreateInfo istg; memset(&istg,0,sizeof istg);
            istg.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            istg.stage=VK_SHADER_STAGE_COMPUTE_BIT; istg.module=dv->sm_idp; istg.pName="main";
            VkComputePipelineCreateInfo icpi; memset(&icpi,0,sizeof icpi);
            icpi.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; icpi.stage=istg; icpi.layout=dv->pl;
            if(g_vkCreateComputePipelines(dv->dev,VK_NULL_HANDLE,1,&icpi,NULL,&dv->pipe_idp)!=VK_SUCCESS){
                g_vkDestroyShaderModule(dv->dev,dv->sm_idp,NULL); dv->sm_idp=VK_NULL_HANDLE;
            }
        }
    }

    /* command pool + buffer */
    VkCommandPoolCreateInfo cp2; memset(&cp2,0,sizeof cp2);
    cp2.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cp2.queueFamilyIndex=dv->qf;
    CHECK(g_vkCreateCommandPool(dv->dev,&cp2,NULL,&dv->cpool),"cmd pool");
    VkCommandBufferAllocateInfo cbai; memset(&cbai,0,sizeof cbai);
    cbai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cbai.commandPool=dv->cpool;
    cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    CHECK(g_vkAllocateCommandBuffers(dv->dev,&cbai,&dv->cmd),"cmd alloc");

    /* persistent buffers (host-visible coherent). Same global slot-index space
     * on every device; only slots owned by this device are ever written, so the
     * resident share is ~1/n of the buffer. */
    {
        /* pool: as many slots as fit below the 4 GB shader addressing limit */
        uint64_t limit = 3750ull*1024*1024;   /* margin below 4 GiB */
        uint64_t maxs  = g_slot_wbytes ? limit/g_slot_wbytes : 0;
        dv->pool_n = (int)((uint64_t)g_nslots < maxs ? (uint64_t)g_nslots : maxs);
        if(dv->pool_n <= 2*g_topk){ fprintf(stderr,"[vg] %s: slot too large for pool\n", dv->name); goto fail; }
        VkDeviceSize wsz=(VkDeviceSize)((uint64_t)dv->pool_n*g_slot_wbytes);
        if(wsz==0) wsz=16;
        CHECK(vg_create_buf(dv,wsz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dv->w),"buf w");
        VkDeviceSize ssz=(VkDeviceSize)dv->pool_n*g_slot_sfloats*sizeof(float);
        if(ssz==0) ssz=16;
        CHECK(vg_create_buf(dv,ssz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dv->s),"buf s");
        VkDeviceSize xsz=(VkDeviceSize)(g_hidden + g_topk*g_inter + 16)*sizeof(float);
        CHECK(vg_create_buf(dv,xsz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &dv->x),"buf x");
        VkDeviceSize ysz=(VkDeviceSize)(g_topk*g_hidden + 16)*sizeof(float);
        CHECK(vg_create_buf(dv,ysz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &dv->y),"buf y");
        VkDeviceSize msz=(VkDeviceSize)(1 + 6*2*g_topk + 16)*sizeof(uint32_t);
        CHECK(vg_create_buf(dv,msz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dv->meta),"buf meta");
    }

    /* bind descriptor buffers */
    {
        VkBuffer bufs[5]={dv->meta.buf,dv->x.buf,dv->w.buf,dv->s.buf,dv->y.buf};
        VkDescriptorBufferInfo dbi[5];
        VkWriteDescriptorSet wds[5];
        for(int b=0;b<5;b++){ dbi[b].buffer=bufs[b]; dbi[b].offset=0; dbi[b].range=VK_WHOLE_SIZE;
            memset(&wds[b],0,sizeof wds[b]); wds[b].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds[b].dstSet=dv->ds; wds[b].dstBinding=b; wds[b].descriptorCount=1;
            wds[b].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[b].pBufferInfo=&dbi[b]; }
        g_vkUpdateDescriptorSets(dv->dev,5,wds,0,NULL);
    }

    dv->slot=calloc((size_t)g_nslots, sizeof(GSlot));
    dv->pool_gidx=malloc((size_t)dv->pool_n*sizeof(int));
    dv->pool_owner=calloc((size_t)dv->pool_n, sizeof(GSlot*));
    if(!dv->slot || !dv->pool_gidx || !dv->pool_owner){ fprintf(stderr,"[vg] OOM slot table\n"); goto fail; }
    for(int i=0;i<g_nslots;i++) dv->slot[i].pool=-1;
    for(int i=0;i<dv->pool_n;i++) dv->pool_gidx[i]=-1;
    return 0;

fail:
    vg_dev_shutdown(dv);
    return -1;
}

int vg_init(const vg_cfg *cfg){
    if(g_vg_ok) return 0;
    if(vg_load_lib()!=0) return -1;

    g_hidden=cfg->hidden; g_inter=cfg->inter; g_cap=cfg->cap;
    if(getenv("MOE_DBG")) g_dbg=1;
    g_topk=cfg->topk; g_nlayers=cfg->n_layers;
    g_weight_bits = (cfg->weight_bits==4) ? 4 : 8;   /* 0/other -> 8 */
    g_nslots=g_nlayers*g_cap;
    if(g_nslots<=0 || g_hidden<=0 || g_inter<=0) return -1;
    if(g_topk>VG_KMAX){ fprintf(stderr,"[vg] topk %d > VG_KMAX\n", g_topk); return -1; }

    /* 64-bit slot layout (LOCAL FIX: the original 32-bit math overflows for
     * cap*layers*slot_bytes > 4 GB and silently aliases slots) */
    g_slot_wbytes = 3ull * (uint64_t)g_hidden * (uint64_t)g_inter
                    * (g_weight_bits==4 ? 1u : 2u) / 2u;
    g_slot_wbytes = align_up64(g_slot_wbytes, 4u);
    g_slot_sfloats = 3u * (uint32_t)g_hidden;

    /* instance — request 1.1 so vkGetPhysicalDeviceFeatures2 works */
    VkApplicationInfo ai; memset(&ai,0,sizeof ai);
    ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci; memset(&ci,0,sizeof ci);
    ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ci.pApplicationInfo=&ai;
    VkResult ir = g_vkCreateInstance(&ci,NULL,&g_inst);
    if(ir!=VK_SUCCESS){
        ai.apiVersion=VK_API_VERSION_1_0;
        if(g_vkCreateInstance(&ci,NULL,&g_inst)!=VK_SUCCESS){
            fprintf(stderr,"[vg] vkCreateInstance failed\n"); return -1; }
    }

    /* ---------- device selection (multi-GPU extension) ----------
     * Rank: DISCRETE_GPU (2) first, then INTEGRATED (1), then VIRTUAL (3);
     * CPU-type devices (llvmpipe) are never auto-selected. COLIBRI_GPUS=n
     * caps the count (default: all discrete GPUs, or the single best device). */
    {
        uint32_t ndev=0; g_vkEnumeratePhysicalDevices(g_inst,&ndev,NULL);
        if(ndev==0){ fprintf(stderr,"[vg] no physical devices\n"); goto fail_inst; }
        VkPhysicalDevice *pds=malloc(ndev*sizeof(VkPhysicalDevice));
        g_vkEnumeratePhysicalDevices(g_inst,&ndev,pds);

        typedef struct { VkPhysicalDevice pd; VkPhysicalDeviceProperties props;
                         int idp, int8s; int rank; } Cand;
        Cand *cand=calloc(ndev,sizeof(Cand)); int ncand=0;
        for(uint32_t i=0;i<ndev;i++){
            VkPhysicalDeviceProperties pp; memset(&pp,0,sizeof pp);
            g_vkGetPhysicalDeviceProperties(pds[i],&pp);
            int rank;
            switch((int)pp.deviceType){
                case 2 /*DISCRETE*/:   rank=0; break;
                case 1 /*INTEGRATED*/: rank=1; break;
                case 3 /*VIRTUAL*/:    rank=2; break;
                default:               rank=99; break;  /* CPU/other: skip */
            }
            if(rank==99) continue;
            /* feature probe (idp/int8) per device */
            VkPhysicalDeviceFeatures2 f2; memset(&f2,0,sizeof f2);
            f2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            VkPhysicalDeviceShaderIntegerDotProductFeatures dpf; memset(&dpf,0,sizeof dpf);
            dpf.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
            VkPhysicalDeviceShaderFloat16Int8Features f16i8; memset(&f16i8,0,sizeof f16i8);
            f16i8.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            f2.pNext=&f16i8; f16i8.pNext=&dpf;
            g_vkGetPhysicalDeviceFeatures2(pds[i],&f2);
            cand[ncand].pd=pds[i]; cand[ncand].props=pp;
            cand[ncand].idp=(int)dpf.shaderIntegerDotProduct;
            cand[ncand].int8s=(int)f16i8.shaderInt8;
            cand[ncand].rank=rank; ncand++;
        }
        free(pds);
        if(ncand==0){ fprintf(stderr,"[vg] no usable (non-CPU) Vulkan device\n"); free(cand); goto fail_inst; }
        /* stable sort by rank (insertion, ncand is tiny) */
        for(int i=1;i<ncand;i++){ Cand t=cand[i]; int j=i-1;
            while(j>=0 && cand[j].rank>t.rank){ cand[j+1]=cand[j]; j--; } cand[j+1]=t; }

        int n_best_rank=0; for(int i=0;i<ncand;i++) if(cand[i].rank==cand[0].rank) n_best_rank++;
        int want = n_best_rank;                          /* default: all best-rank GPUs */
        const char *ge=getenv("COLIBRI_GPUS");
        if(ge && atoi(ge)>0) want=atoi(ge);
        if(want>ncand) want=ncand;
        if(want>VG_MAX_DEV) want=VG_MAX_DEV;

        /* IDP policy: only if every selected device supports it (keeps one
         * uniform compute path). AMD 0x800184 blacklist as before. */
        int all_idp=1;
        for(int i=0;i<want;i++){
            int black = (cand[i].props.vendorID==0x1002u && cand[i].props.driverVersion==0x800184u);
            const char *idp_env=getenv("COLIBRI_IDP");
            int force_on=(idp_env&&idp_env[0]=='1'), force_off=(idp_env&&idp_env[0]=='0');
            int ok = cand[i].idp && cand[i].int8s && !force_off && (!black || force_on);
            if(!ok) all_idp=0;
        }

        g_ndev=0;
        for(int i=0;i<want;i++){
            VgDev *dv=&g_d[g_ndev];
            if(vg_dev_init(dv,cand[i].pd,cand[i].props.deviceName,
                           cand[i].props.vendorID,cand[i].props.driverVersion,
                           (g_weight_bits!=4)&&all_idp?cand[i].idp:0,
                           (g_weight_bits!=4)&&all_idp?cand[i].int8s:0)==0){
                fprintf(stderr,"[vg] device %d: %s (vendor 0x%X, type %d)\n",
                        g_ndev, dv->name, dv->vendor_id, (int)cand[i].props.deviceType);
                g_ndev++;
            } else {
                fprintf(stderr,"[vg] device init failed for %s -> skipped\n",
                        cand[i].props.deviceName);
            }
        }
        free(cand);
        if(g_ndev==0){ fprintf(stderr,"[vg] all device inits failed\n"); goto fail_inst; }
    }

    /* uniform path flags across devices */
    g_use_int4 = (g_weight_bits==4);            /* devices without int4 were dropped */
    g_use_idp  = 1;
    for(int d=0;d<g_ndev;d++) if(!g_d[d].pipe_idp) g_use_idp=0;
    if(g_weight_bits==4) g_use_idp=0;
    if(g_weight_bits!=4 && !g_use_idp)
        fprintf(stderr,"[vg] int8 dot-product path off (unsupported/blacklisted on >=1 device) -> float path\n");

    g_vg_ok=1;
    fprintf(stderr,"[vg] Vulkan GEMV backend ready: %d device(s), %d layers x %d slots, pool %d slots (%.1f MB)/device%s\n",
            g_ndev, g_nlayers, g_nslots, g_d[0].pool_n,
            (double)((uint64_t)g_d[0].pool_n*g_slot_wbytes)/(1024.0*1024.0),
            g_use_int4 ? " [int4 unpack+float GEMV ACTIVE]"
            : g_use_idp ? " [int8 dot-product OpSDotKHR ACTIVE]" : " [float path]");
    return 0;

fail_inst:
    vg_shutdown();
    return -1;
}

int vg_ready(void){ return g_vg_ok; }
int vg_use_int4(void){ return g_use_int4; }

void vg_shutdown(void){
    for(int d=0;d<VG_MAX_DEV;d++) vg_dev_shutdown(&g_d[d]);
    g_ndev=0;
    if(g_inst){ g_vkDestroyInstance(g_inst,NULL); g_inst=VK_NULL_HANDLE; }
    if(g_lib){ dl_close(g_lib); g_lib=NULL; }
    g_vg_ok=0;
}

/* Reserve a pool slot for global slot idx on device dv (LRU eviction).
 * Returns the pool index; the evicted GSlot (if any) is unmapped. */
static int vg_pool_place(VgDev *dv, GSlot *s, int idx){
    if(s->pool>=0 && dv->pool_gidx[s->pool]==idx) return s->pool;
    int best=-1; uint64_t best_used=~0ull;
    for(int i=0;i<dv->pool_n;i++){
        if(dv->pool_gidx[i]<0){ best=i; break; }
        GSlot *o=dv->pool_owner[i];
        uint64_t u=o?o->used:0;
        if(u<best_used){ best_used=u; best=i; }
    }
    GSlot *victim=dv->pool_owner[best];
    if(victim && victim!=s) victim->pool=-1;
    dv->pool_gidx[best]=idx; dv->pool_owner[best]=s; s->pool=best;
    return best;
}

/* Upload one expert's weights/scales into the OWNING device's pool slot.
 * 64-bit host offsets; shader offsets stay below 4 GB by pool construction. */
void vg_expert_loaded(int layer, int eid, int li,
                      const int8_t *g, const int8_t *u, const int8_t *d,
                      const float *gs, const float *us, const float *ds){
    if(!g_vg_ok) return;
    int idx = layer*g_cap + li;
    if(idx<0 || idx>=g_nslots) return;
    int owner=vg_owner(eid);
    VgDev *dv=&g_d[owner];
    /* the CPU LRU reassigned this slot: stale entries for the previous eid may
     * survive on other devices — invalidate them so the owner scan is unique */
    for(int od=0;od<g_ndev;od++) if(od!=owner) g_d[od].slot[idx].valid=0;
    GSlot *s=&dv->slot[idx];
    s->used=++g_tick;                       /* fresh tick BEFORE placing: never evicted by itself */
    int pool = vg_pool_place(dv,s,idx);
    uint64_t wbase = (uint64_t)pool * g_slot_wbytes;          /* bytes */
    uint64_t sbase = (uint64_t)pool * g_slot_sfloats;          /* floats */
    uint32_t D=(uint32_t)g_hidden, Ih=(uint32_t)g_inter;
    uint64_t gbytes = (uint64_t)D*Ih*sizeof(int8_t);
    memcpy((uint8_t*)dv->w.ptr + wbase,              g,  gbytes);
    memcpy((uint8_t*)dv->w.ptr + wbase + gbytes,     u,  gbytes);
    memcpy((uint8_t*)dv->w.ptr + wbase + 2*gbytes,   d,  gbytes);
    memcpy((float*)dv->s.ptr + sbase,          gs, Ih*sizeof(float));
    memcpy((float*)dv->s.ptr + sbase + Ih,     us, Ih*sizeof(float));
    memcpy((float*)dv->s.ptr + sbase + 2*Ih,   ds, D *sizeof(float));
    vg_flush(dv,&dv->w, wbase, 3u*gbytes);
    vg_flush(dv,&dv->s, (VkDeviceSize)(sbase*sizeof(float)), (VkDeviceSize)(2u*Ih + D)*sizeof(float));

    s->woff_g = (uint32_t)(wbase/4u);
    s->woff_u = (uint32_t)((wbase + gbytes)/4u);
    s->woff_d = (uint32_t)((wbase + 2*gbytes)/4u);
    s->soff_g = (uint32_t)sbase; s->soff_u = (uint32_t)(sbase + Ih); s->soff_d = (uint32_t)(sbase + 2*Ih);
    s->layer=layer; s->eid=eid; s->valid=1; s->used=++g_tick;
}

void vg_expert_ensure(int layer, int li, int eid,
                      const int8_t *g, const int8_t *u, const int8_t *d,
                      const float *gs, const float *us, const float *ds){
    if(!g_vg_ok) return;
    int idx = layer*g_cap + li;
    if(idx<0 || idx>=g_nslots) return;
    VgDev *dv=&g_d[vg_owner(eid)];
    GSlot *s=&dv->slot[idx];
    if(s->valid && s->eid==eid && s->pool>=0 && dv->pool_gidx[s->pool]==idx){ s->used=++g_tick; return; }
    vg_expert_loaded(layer, eid, li, g, u, d, gs, us, ds);
}

/* int4 variant: weights packed 2 nibbles/byte. */
void vg_expert_loaded_int4(int layer, int eid, int li,
                           const uint8_t *g, const uint8_t *u, const uint8_t *d,
                           const float *gs, const float *us, const float *ds){
    if(!g_vg_ok) return;
    int idx = layer*g_cap + li;
    if(idx<0 || idx>=g_nslots) return;
    int owner=vg_owner(eid);
    VgDev *dv=&g_d[owner];
    for(int od=0;od<g_ndev;od++) if(od!=owner) g_d[od].slot[idx].valid=0;
    GSlot *s=&dv->slot[idx];
    s->used=++g_tick;
    int pool = vg_pool_place(dv,s,idx);
    uint64_t wbase = (uint64_t)pool * g_slot_wbytes;          /* bytes */
    uint64_t sbase = (uint64_t)pool * g_slot_sfloats;          /* floats */
    uint32_t D=(uint32_t)g_hidden, Ih=(uint32_t)g_inter;
    uint64_t gbytes = (uint64_t)D*Ih/2u;                      /* int4: half of int8 */
    memcpy((uint8_t*)dv->w.ptr + wbase,              g,  gbytes);
    memcpy((uint8_t*)dv->w.ptr + wbase + gbytes,     u,  gbytes);
    memcpy((uint8_t*)dv->w.ptr + wbase + 2*gbytes,   d,  gbytes);
    memcpy((float*)dv->s.ptr + sbase,          gs, Ih*sizeof(float));
    memcpy((float*)dv->s.ptr + sbase + Ih,     us, Ih*sizeof(float));
    memcpy((float*)dv->s.ptr + sbase + 2*Ih,   ds, D *sizeof(float));
    vg_flush(dv,&dv->w, wbase, 3u*gbytes);
    vg_flush(dv,&dv->s, (VkDeviceSize)(sbase*sizeof(float)), (VkDeviceSize)(2u*Ih + D)*sizeof(float));

    s->woff_g = (uint32_t)(wbase/4u);
    s->woff_u = (uint32_t)((wbase + gbytes)/4u);
    s->woff_d = (uint32_t)((wbase + 2*gbytes)/4u);
    s->soff_g = (uint32_t)sbase; s->soff_u = (uint32_t)(sbase + Ih); s->soff_d = (uint32_t)(sbase + 2*Ih);
    s->layer=layer; s->eid=eid; s->valid=1; s->used=++g_tick;
}

void vg_expert_ensure_int4(int layer, int li, int eid,
                           const uint8_t *g, const uint8_t *u, const uint8_t *d,
                           const float *gs, const float *us, const float *ds){
    if(!g_vg_ok) return;
    int idx = layer*g_cap + li;
    if(idx<0 || idx>=g_nslots) return;
    VgDev *dv=&g_d[vg_owner(eid)];
    GSlot *s=&dv->slot[idx];
    if(s->valid && s->eid==eid && s->pool>=0 && dv->pool_gidx[s->pool]==idx){ s->used=++g_tick; return; }
    vg_expert_loaded_int4(layer, eid, li, g, u, d, gs, us, ds);
}

/* Quantize a float vector to packed int8 (4 int8/uint32, LE) with a symmetric
 * per-vector scale (absmax/127). Returns the scale. */
static float vg_quantize_pack(const float *x, int n, uint32_t *dst_packed){
    float amax=0.0f;
    for(int i=0;i<n;i++){ float a=x[i]<0?-x[i]:x[i]; if(a>amax) amax=a; }
    float s = amax>1e-9f ? amax/127.0f : 1.0f;
    int np=(n+3)/4;
    for(int g=0;g<np;g++){
        uint32_t w=0;
        for(int j=0;j<4;j++){
            int idx=g*4+j; int8_t v=0;
            if(idx<n){ int q=(int)(x[idx]/s + (x[idx]>=0?0.5f:-0.5f));
                       if(q>127)q=127; if(q<-127)q=-127; v=(int8_t)q; }
            w |= ((uint32_t)(unsigned char)v) << (j*8);
        }
        dst_packed[g]=w;
    }
    return s;
}
static uint32_t vg_f2u(float f){ uint32_t u; memcpy(&u,&f,4); return u; }

/* Record + submit one dispatch on a device WITHOUT waiting; call vg_wait()
 * after all devices were submitted so they overlap (multi-GPU extension). */
static int vg_dispatch_submit(VgDev *dv, uint32_t nmat, uint32_t stride,
                              const uint32_t *meta, uint32_t total){
    uint32_t groups=(total+63u)/64u;
    uint32_t nwords=1u+stride*nmat;
    memcpy(dv->meta.ptr, meta, (size_t)nwords*sizeof(uint32_t));
    vg_flush(dv,&dv->meta, 0, (VkDeviceSize)nwords*sizeof(uint32_t));

    VkPipeline pipe = g_use_int4 ? dv->pipe_int4 : (g_use_idp ? dv->pipe_idp : dv->pipe);
    VkCommandBufferBeginInfo bbi; memset(&bbi,0,sizeof bbi);
    bbi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if(g_vkResetCommandBuffer(dv->cmd,0)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] ResetCmdBuffer FAIL\n"); return -1; }
    if(g_vkBeginCommandBuffer(dv->cmd,&bbi)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] BeginCmdBuffer FAIL\n"); return -1; }
    g_vkCmdBindPipeline(dv->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    g_vkCmdBindDescriptorSets(dv->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,dv->pl,0,1,&dv->ds,0,NULL);
    g_vkCmdDispatch(dv->cmd,groups,1,1);
    if(g_vkEndCommandBuffer(dv->cmd)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] EndCmdBuffer FAIL\n"); return -1; }
    VkSubmitInfo si; memset(&si,0,sizeof si);
    si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount=1; si.pCommandBuffers=&dv->cmd;
    if(g_vkQueueSubmit(dv->q,1,&si,VK_NULL_HANDLE)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] QueueSubmit FAIL\n"); return -1; }
    return 0;
}
static int vg_wait(VgDev *dv){
    if(g_vkQueueWaitIdle(dv->q)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] QueueWaitIdle FAIL\n"); return -1; }
    return 0;
}

/* Run one (token, layer) of the routed-expert forward across all devices.
 * Experts are partitioned by owner (eid % g_ndev); each phase is submitted to
 * every participating queue first and waited on together. */
void vg_moe_run(int layer, int K, const int *handles, const float *val,
                const float *xs, float *out){
    (void)layer;
    if(!g_vg_ok) return;
    if(K<=0 || K>VG_KMAX) return;
    const uint32_t D=(uint32_t)g_hidden, Ih=(uint32_t)g_inter;
    uint32_t *meta=malloc((size_t)(1u+6u*2u*(uint32_t)VG_KMAX)*sizeof(uint32_t));
    if(!meta) return;

    /* partition: kl[d] = global k-indices owned by device d (in call order) */
    int kl[VG_MAX_DEV][VG_KMAX]; int Kl[VG_MAX_DEV];
    for(int d=0;d<g_ndev;d++) Kl[d]=0;
    for(int k=0;k<K;k++){
        /* owner from the slot table: every device indexes the same global slot
         * space, so look the eid up on each device (valid only on the owner). */
        int owner=0;
        if(g_ndev>1){
            owner=-1;
            for(int d=0;d<g_ndev;d++){
                GSlot *s=&g_d[d].slot[handles[k]];
                if(s->valid && s->pool>=0 && vg_owner(s->eid)==d){ owner=d; break; }
            }
            if(owner<0){ free(meta); return; }  /* not uploaded -> let CPU path handle */
        }
        kl[owner][Kl[owner]++]=k;
    }

    if(g_use_idp){
        /* ---- IDP path: activations quantized to packed int8, OpSDotKHR ---- */
        /* Phase 1: gate + up, shared activation xs (same quantization on all
         * devices so results match the single-device path bit-for-bit) */
        uint32_t *xq_tmp=malloc(((size_t)D/4u+4)*sizeof(uint32_t));
        if(!xq_tmp){ free(meta); return; }
        float s1 = vg_quantize_pack(xs,(int)D,xq_tmp);
        for(int d=0;d<g_ndev;d++){
            if(!Kl[d]) continue;
            VgDev *dv=&g_d[d];
            memcpy(dv->x.ptr, xq_tmp, (size_t)(D/4u)*4u);
            vg_flush(dv,&dv->x, 0, (VkDeviceSize)(D/4u)*4u);
            uint32_t nm1=(uint32_t)(2*Kl[d]);
            meta[0]=nm1;
            for(int lk=0;lk<Kl[d];lk++){
                GSlot *s=&dv->slot[handles[kl[d][lk]]];
                uint32_t m=(uint32_t)(2*lk);
                meta[1+6*m+0]=0;        meta[1+6*m+1]=s->woff_g; meta[1+6*m+2]=s->soff_g;
                meta[1+6*m+3]=D/4u;     meta[1+6*m+4]=Ih;        meta[1+6*m+5]=vg_f2u(s1);
                uint32_t m2=(uint32_t)(2*lk+1);
                meta[1+6*m2+0]=0;       meta[1+6*m2+1]=s->woff_u; meta[1+6*m2+2]=s->soff_u;
                meta[1+6*m2+3]=D/4u;    meta[1+6*m2+4]=Ih;       meta[1+6*m2+5]=vg_f2u(s1);
            }
            if(vg_dispatch_submit(dv,nm1,6,meta,(uint32_t)(2*Kl[d])*Ih)!=0){ free(xq_tmp); free(meta); return; }
        }
        float *gact=malloc((size_t)K*Ih*sizeof(float));
        float *sks=malloc((size_t)K*sizeof(float));
        if(!gact||!sks){ free(gact); free(sks); free(xq_tmp); free(meta); return; }
        for(int d=0;d<g_ndev;d++){
            if(!Kl[d]) continue;
            VgDev *dv=&g_d[d];
            vg_wait(dv);
            vg_invalidate(dv,&dv->y,0,(VkDeviceSize)((uint32_t)(2*Kl[d])*Ih)*sizeof(float));
            float *y=(float*)dv->y.ptr;
            for(int lk=0;lk<Kl[d];lk++){
                const float *gk=y+(size_t)(2*lk)*Ih;
                const float *uk=y+(size_t)(2*lk+1)*Ih;
                float *ak=gact+(size_t)kl[d][lk]*Ih;
                for(uint32_t i=0;i<Ih;i++){ float gv=gk[i]; ak[i]=gv/(1.0f+expf(-gv))*uk[i]; }
            }
        }
        /* Phase 2: down, per-expert activation (own scale) */
        for(int d=0;d<g_ndev;d++){
            if(!Kl[d]) continue;
            VgDev *dv=&g_d[d];
            uint32_t *xq=(uint32_t*)dv->x.ptr;
            uint32_t nm2=(uint32_t)Kl[d];
            meta[0]=nm2;
            uint32_t xoff_k=0;
            for(int lk=0;lk<Kl[d];lk++){
                int k=kl[d][lk];
                float sk=vg_quantize_pack(gact+(size_t)k*Ih,(int)Ih, xq+(size_t)lk*(Ih/4u));
                sks[k]=sk;
                GSlot *s=&dv->slot[handles[k]];
                uint32_t m=(uint32_t)lk;
                meta[1+6*m+0]=xoff_k;   meta[1+6*m+1]=s->woff_d; meta[1+6*m+2]=s->soff_d;
                meta[1+6*m+3]=Ih/4u;    meta[1+6*m+4]=D;         meta[1+6*m+5]=vg_f2u(sk);
                xoff_k += Ih/4u;
            }
            vg_flush(dv,&dv->x,0,(VkDeviceSize)xoff_k*4u);
            if(vg_dispatch_submit(dv,nm2,6,meta,(uint32_t)Kl[d]*D)!=0){ free(gact); free(sks); free(xq_tmp); free(meta); return; }
        }
        for(int d=0;d<g_ndev;d++){
            if(!Kl[d]) continue;
            VgDev *dv=&g_d[d];
            vg_wait(dv);
            vg_invalidate(dv,&dv->y,0,(VkDeviceSize)((uint32_t)Kl[d]*D)*sizeof(float));
            float *y=(float*)dv->y.ptr;
            for(int lk=0;lk<Kl[d];lk++){
                int k=kl[d][lk];
                const float *hk=y+(size_t)lk*D; float w=val[k];
                for(uint32_t dd=0;dd<D;dd++) out[dd]+= w*hk[dd];
            }
        }
        free(gact); free(sks); free(xq_tmp); free(meta);
        return;
    }

    /* ---- int4 path (unpack nibbles + float GEMV) and float path share the
     * same two-phase structure; they differ only in meta contents ---- */
    int int4 = g_use_int4;

    /* Phase 1: gate + up */
    for(int d=0;d<g_ndev;d++){
        if(!Kl[d]) continue;
        VgDev *dv=&g_d[d];
        float *x=(float*)dv->x.ptr;
        memcpy(x, xs, D*sizeof(float));
        vg_flush(dv,&dv->x, 0, (VkDeviceSize)D*sizeof(float));
        uint32_t nm1=(uint32_t)(2*Kl[d]);
        meta[0]=nm1;
        for(int lk=0;lk<Kl[d];lk++){
            GSlot *s=&dv->slot[handles[kl[d][lk]]];
            uint32_t m=(uint32_t)(2*lk), m2=(uint32_t)(2*lk+1);
            if(int4){
                meta[1+5*m+0]=0;  meta[1+5*m+1]=s->woff_g;  meta[1+5*m+2]=s->soff_g;  meta[1+5*m+3]=D/8u; meta[1+5*m+4]=Ih;
                meta[1+5*m2+0]=0; meta[1+5*m2+1]=s->woff_u; meta[1+5*m2+2]=s->soff_u; meta[1+5*m2+3]=D/8u; meta[1+5*m2+4]=Ih;
            } else {
                meta[1+5*m+0]=0;  meta[1+5*m+1]=s->woff_g;  meta[1+5*m+2]=s->soff_g;  meta[1+5*m+3]=D;    meta[1+5*m+4]=Ih;
                meta[1+5*m2+0]=0; meta[1+5*m2+1]=s->woff_u; meta[1+5*m2+2]=s->soff_u; meta[1+5*m2+3]=D;   meta[1+5*m2+4]=Ih;
            }
        }
        if(vg_dispatch_submit(dv,nm1,5,meta,(uint32_t)(2*Kl[d])*Ih)!=0){ free(meta); return; }
    }
    /* wait + silu(g)*u into each device's x (after the xs prefix) */
    for(int d=0;d<g_ndev;d++){
        if(!Kl[d]) continue;
        VgDev *dv=&g_d[d];
        vg_wait(dv);
        vg_invalidate(dv,&dv->y,0,(VkDeviceSize)((uint32_t)(2*Kl[d])*Ih)*sizeof(float));
        float *y=(float*)dv->y.ptr;
        float *x=(float*)dv->x.ptr;
        if(g_dbg && d==0){
            static int pc=0;
            if(pc++<1)
                fprintf(stderr,"[vg-dbg] Phase1 gate[0..3]=%.4f %.4f %.4f %.4f  up[0..3]=%.4f %.4f %.4f %.4f\n",
                        y[0],y[1],y[2],y[3], y[Ih],y[Ih+1],y[Ih+2],y[Ih+3]);
        }
        for(int lk=0;lk<Kl[d];lk++){
            const float *gk = y + (size_t)(2*lk)*Ih;
            const float *uk = y + (size_t)(2*lk+1)*Ih;
            float *ak = x + D + (size_t)lk*Ih;
            for(uint32_t i=0;i<Ih;i++){
                float gv=gk[i]; float a=gv/(1.0f+expf(-gv)); ak[i]=a*uk[i];
            }
        }
    }
    /* Phase 2: down */
    for(int d=0;d<g_ndev;d++){
        if(!Kl[d]) continue;
        VgDev *dv=&g_d[d];
        vg_flush(dv,&dv->x, (VkDeviceSize)D*sizeof(float), (VkDeviceSize)((uint32_t)Kl[d]*Ih)*sizeof(float));
        uint32_t nm2=(uint32_t)Kl[d];
        meta[0]=nm2;
        for(int lk=0;lk<Kl[d];lk++){
            GSlot *s=&dv->slot[handles[kl[d][lk]]];
            uint32_t m=(uint32_t)lk;
            meta[1+5*m+0]=D + (uint32_t)lk*Ih; meta[1+5*m+1]=s->woff_d; meta[1+5*m+2]=s->soff_d;
            meta[1+5*m+3]= int4 ? Ih/8u : Ih;  meta[1+5*m+4]=D;
        }
        if(vg_dispatch_submit(dv,nm2,5,meta,(uint32_t)Kl[d]*D)!=0){ free(meta); return; }
    }
    for(int d=0;d<g_ndev;d++){
        if(!Kl[d]) continue;
        VgDev *dv=&g_d[d];
        vg_wait(dv);
        vg_invalidate(dv,&dv->y,0,(VkDeviceSize)((uint32_t)Kl[d]*D)*sizeof(float));
        float *y=(float*)dv->y.ptr;
        for(int lk=0;lk<Kl[d];lk++){
            int k=kl[d][lk];
            const float *hk = y + (size_t)lk*D; float w=val[k];
            for(uint32_t dd=0;dd<D;dd++) out[dd]+= w*hk[dd];
        }
    }
    free(meta);
}

/* ===================================================================== */
#ifdef VG_SELFTEST
/* CPU reference (mirrors qwen36.c matmul_q, no NEON) */
static void ref_matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O){
    for(int o=0;o<O;o++){
        const int8_t *w=q+(int64_t)o*I; double acc=0;
        for(int i=0;i<I;i++) acc+=(double)x[i]*(double)w[i];
        y[o]=(float)(acc*scale[o]);
    }
}
static float silu(float z){ return z/(1.0f+expf(-z)); }
/* symmetric int8 quantization scale (matches vg_quantize_pack) */
static float ref_qscale(const float *x, int n){
    float amax=0; for(int i=0;i<n;i++){ float a=x[i]<0?-x[i]:x[i]; if(a>amax)amax=a; }
    return amax>1e-9f ? amax/127.0f : 1.0f;
}
/* CPU int8-activation matmul (mirrors the IDP GPU path exactly) */
static void ref_mq_i8(float *y, const float *x, float xs, const int8_t *q, const float *wscale, int I, int O){
    for(int o=0;o<O;o++){
        const int8_t *w=q+(int64_t)o*I; long acc=0;
        for(int i=0;i<I;i++){ int vi=(int)(x[i]/xs + (x[i]>=0?0.5f:-0.5f)); if(vi>127)vi=127; if(vi<-127)vi=-127; acc+=(long)vi*(long)w[i]; }
        y[o]=wscale[o]*xs*(float)acc;
    }
}

int main(void){
    srand(20260724);
    int wbits = 4;   /* default: verify the int4 unpack+float path (AMD-safe) */
    const char *wbenv=getenv("COLIBRI_WBITS");
    if(wbenv) wbits = atoi(wbenv);
    /* Real-model dimensions (D=4096,Ih=512,K=8,cap=8) can be forced via env
     * (TD/TIH/TK/TCAP/TNL) to reproduce the "self-test passes, real model
     * outputs zero" bug without loading the 35B container. */
    int D_test = getenv("TD")   ? atoi(getenv("TD"))   : 96;
    int Ih_test= getenv("TIH")  ? atoi(getenv("TIH"))  : 40;
    int K_test = getenv("TK")   ? atoi(getenv("TK"))   : 4;
    int C_test = getenv("TCAP") ? atoi(getenv("TCAP")) : 6;
    int NL_test= getenv("TNL")  ? atoi(getenv("TNL"))  : 2;
    int L0_test= getenv("TL0")  ? atoi(getenv("TL0"))  : 0;   /* base layer: exercises HIGH buffer offsets */
    vg_cfg cfg; cfg.n_layers=NL_test; cfg.hidden=D_test; cfg.inter=Ih_test; cfg.cap=C_test; cfg.topk=K_test; cfg.weight_bits=wbits;
    if(vg_init(&cfg)!=0){ fprintf(stderr,"SELFTEST: vg_init failed (no GPU?) -> cannot verify\n"); return 2; }

    int D=cfg.hidden, Ih=cfg.inter, K=cfg.topk;
    /* build K experts (use layers 0,1) with random integer weights in [-8,7]
     * -- valid as both int4 and int8, so the same values feed both paths. */
    int nE = 2*cfg.cap;
    int nwb = Ih*D;                 /* weights per matrix (gate/up/down) */
    int8_t  *Eg=malloc((size_t)nE*nwb), *Eu=malloc((size_t)nE*nwb), *Ed=malloc((size_t)nE*nwb);
    uint8_t *Eg4=malloc((size_t)nE*nwb/2), *Eu4=malloc((size_t)nE*nwb/2), *Ed4=malloc((size_t)nE*nwb/2);
    float   *Egs=malloc((size_t)nE*Ih*4), *Eus=malloc((size_t)nE*Ih*4), *Eds=malloc((size_t)nE*D*4);
    for(int e=0;e<nE;e++){
        for(int j=0;j<nwb;j++){
            int8_t v=(int8_t)((rand()%16)-8);   /* signed int4 [-8,7] */
            Eg[e*nwb+j]=v; Eu[e*nwb+j]=v; Ed[e*nwb+j]=v;
        }
        /* pack 2 int4 per byte: even index -> low nibble, odd -> high nibble (LE) */
        for(int j=0;j<nwb;j+=2){
            uint8_t lo,hi;
            lo=(uint8_t)(Eg[e*nwb+j]&0xF);   hi=(uint8_t)(Eg[e*nwb+j+1]&0xF); Eg4[e*(nwb/2)+j/2]=(hi<<4)|lo;
            lo=(uint8_t)(Eu[e*nwb+j]&0xF);   hi=(uint8_t)(Eu[e*nwb+j+1]&0xF); Eu4[e*(nwb/2)+j/2]=(hi<<4)|lo;
            lo=(uint8_t)(Ed[e*nwb+j]&0xF);   hi=(uint8_t)(Ed[e*nwb+j+1]&0xF); Ed4[e*(nwb/2)+j/2]=(hi<<4)|lo;
        }
        for(int j=0;j<Ih;j++){ Egs[e*Ih+j]=0.01f+((float)rand()/RAND_MAX)*0.09f; Eus[e*Ih+j]=Egs[e*Ih+j]; }
        for(int j=0;j<D;j++)  Eds[e*D+j]=0.01f+((float)rand()/RAND_MAX)*0.09f;
    }

    float *xs=malloc(D*sizeof(float));
    for(int i=0;i<D;i++) xs[i]=((float)rand()/RAND_MAX)*2.f-1.f;
    float val[K]; float sv=0; for(int k=0;k<K;k++){ val[k]=((float)rand()/RAND_MAX)+0.1f; sv+=val[k]; } for(int k=0;k<K;k++) val[k]/=sv;

    /* upload experts (li = e % cap, layer = e / cap) */
    int handles[K];
    for(int k=0;k<K;k++){
        int e=k; int layer=L0_test + e/cfg.cap, li=e%cfg.cap;
        if(wbits==4)
            vg_expert_loaded_int4(layer, e, li, Eg4+e*(nwb/2), Eu4+e*(nwb/2), Ed4+e*(nwb/2),
                                   Egs+e*Ih, Eus+e*Ih, Eds+e*D);
        else
            vg_expert_loaded(layer, e, li, Eg+e*nwb, Eu+e*nwb, Ed+e*nwb,
                             Egs+e*Ih, Eus+e*Ih, Eds+e*D);
        handles[k]=layer*cfg.cap+li;
    }

    /* GPU run */
    float *out_gpu=calloc(D,sizeof(float));
    vg_moe_run(0, K, handles, val, xs, out_gpu);

    /* CPU references */
    float *g=malloc(Ih*sizeof(float)), *u=malloc(Ih*sizeof(float)), *hh=malloc(D*sizeof(float));
    float *out_cpu=calloc(D,sizeof(float));       /* float activation reference */
    float *out_cpu_i8=calloc(D,sizeof(float));     /* int8 activation reference (matches IDP) */
    float s1 = ref_qscale(xs, D);                   /* shared activation scale for gate+up */
    for(int k=0;k<K;k++){
        int e=k;
        /* float path reference */
        ref_matmul_q(g, xs, Eg+e*Ih*D, Egs+e*Ih, D, Ih);
        ref_matmul_q(u, xs, Eu+e*Ih*D, Eus+e*Ih, D, Ih);
        for(int i=0;i<Ih;i++) g[i]=silu(g[i])*u[i];
        ref_matmul_q(hh, g, Ed+e*Ih*D, Eds+e*D, Ih, D);
        for(int d=0;d<D;d++) out_cpu[d]+=val[k]*hh[d];
        /* int8-activation reference (mirrors IDP: gate/up share s1, down per-expert) */
        ref_mq_i8(g, xs, s1, Eg+e*Ih*D, Egs+e*Ih, D, Ih);
        ref_mq_i8(u, xs, s1, Eu+e*Ih*D, Eus+e*Ih, D, Ih);
        for(int i=0;i<Ih;i++) g[i]=silu(g[i])*u[i];
        float sk = ref_qscale(g, Ih);
        ref_mq_i8(hh, g, sk, Ed+e*Ih*D, Eds+e*D, Ih, D);
        for(int d=0;d<D;d++) out_cpu_i8[d]+=val[k]*hh[d];
    }

    /* choose the reference that matches the active GPU path */
    float *out_ref = g_use_idp ? out_cpu_i8 : out_cpu;
    double dot=0,n1=0,n2=0;
    for(int d=0;d<D;d++){ dot+=(double)out_ref[d]*out_gpu[d]; n1+=(double)out_ref[d]*out_ref[d]; n2+=(double)out_gpu[d]*out_gpu[d]; }
    double cos = dot/sqrt(n1*n2);
    printf("devices=%d weight_bits=%d  int4_active=%d  idp_active=%d\n", g_ndev, wbits, g_use_int4, g_use_idp);
    printf("CPU[0..3] = %.5f %.5f %.5f %.5f\n", out_ref[0],out_ref[1],out_ref[2],out_ref[3]);
    printf("GPU[0..3] = %.5f %.5f %.5f %.5f\n", out_gpu[0],out_gpu[1],out_gpu[2],out_gpu[3]);
    printf("cosine(GPU, %s-ref) = %.6f\n", g_use_idp?"int8-act":(g_use_int4?"int4":"float"), cos);

    /* second run (cached experts) should still match */
    float *out_gpu2=calloc(D,sizeof(float));
    vg_moe_run(0, K, handles, val, xs, out_gpu2);
    double dot2=0,n22=0; for(int d=0;d<D;d++){ dot2+=(double)out_ref[d]*out_gpu2[d]; n22+=(double)out_gpu2[d]*out_gpu2[d]; }
    double cos2=dot2/sqrt(n1*n22);
    printf("cosine(GPU cached, %s-ref) = %.6f\n", g_use_idp?"int8-act":(g_use_int4?"int4":"float"), cos2);

    int pass = (cos>0.999 && cos2>0.999);
    printf(pass?"RESULT: PASS (GPU batched MoE matches CPU)\n":"RESULT: FAIL\n");

    free(Eg);free(Eu);free(Ed);free(Eg4);free(Eu4);free(Ed4);free(Egs);free(Eus);free(Eds);free(xs);
    free(g);free(u);free(hh);free(out_cpu);free(out_cpu_i8);free(out_gpu);free(out_gpu2);
    vg_shutdown();
    return pass?0:1;
}
#endif /* VG_SELFTEST */
