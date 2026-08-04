#include "qwen36_cuda.h"
#include "qwen36_tier.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ColiCudaTensor *q,*k,*v,*o,*router;
    ColiCudaTensor *sh_g,*sh_u,*sh_d;
    ColiCudaTensor *dn_qkv,*dn_z,*dn_b,*dn_a,*dn_out;
    float *in_norm,*post_norm,*q_norm,*k_norm,*router_bias,*shared_gate;
    float *dn_conv,*dn_dt,*dn_alog,*dn_norm;
} QwenCudaLayer;

struct QwenCudaState {
    QwenCudaConfig c;
    QwenCudaLayer *layer;
    ColiCudaTensor *embed,*lm_head;
    float *final_norm;
    int device,ready,request_enabled,state_valid,in_txn,kv_len,kv_cap;
    uint64_t epoch;
    size_t reserved;
    float **k_cache,**v_cache;
    float **dn_rec,**dn_conv_state,**dn_rec_backup,**dn_conv_backup;
    float *x,*nrm,*tmp,*moe,*logits;
    float *q,*k,*v,*dn_qkv,*dn_z,*dn_b,*dn_a,*dn_outv;
    float *sh_g,*sh_u,*cpu_stage;
    int *route_ids;
    float *route_weights;
    int *ids_host,*route_ids_host;
    float *route_weights_host,*x_host,*partial_host;
    QwenCudaCpuRoutes cpu_routes;
    QwenCudaRouteCommit route_commit;
    void *opaque;
};

static void *dalloc(QwenCudaState *s,size_t bytes){
    void *p=coli_cuda_pipe_alloc(s->device,bytes);
    if(p) s->reserved+=bytes;
    return p;
}
static float *raw_upload(QwenCudaState *s,const float *src,size_t n){
    if(!src||!n) return NULL;
    float *p=(float*)dalloc(s,n*sizeof(float));
    if(!p) return NULL;
    if(!coli_cuda_pipe_upload(s->device,p,src,n*sizeof(float))){
        coli_cuda_pipe_free(s->device,p);
        s->reserved-=n*sizeof(float);
        return NULL;
    }
    return p;
}
static int tensor_upload(QwenCudaState *s,ColiCudaTensor **out,const QwenCudaWeight *w){
    if(!w||!w->weights||w->input<1||w->output<1) return 0;
    int ok=coli_cuda_tensor_upload_g(out,w->weights,w->scales,w->fmt,w->input,
                                     w->output,s->device,w->group_size);
    if(ok) s->reserved+=coli_cuda_tensor_bytes(*out);
    return ok;
}
static void tensor_drop(ColiCudaTensor **p){ if(*p){coli_cuda_tensor_free(*p);*p=NULL;} }
static void raw_drop(QwenCudaState *s,float **p){ if(*p){coli_cuda_pipe_free(s->device,*p);*p=NULL;} }

static int upload_layer(QwenCudaState *s,QwenCudaLayer *d,const QwenCudaLayerWeights *h){
    const QwenCudaConfig *c=&s->c;
    if(!tensor_upload(s,&d->router,&h->router)||
       !tensor_upload(s,&d->sh_g,&h->shared_gate)||
       !tensor_upload(s,&d->sh_u,&h->shared_up)||
       !tensor_upload(s,&d->sh_d,&h->shared_down)) return 0;
    d->in_norm=raw_upload(s,h->input_norm,c->hidden);
    d->post_norm=raw_upload(s,h->post_norm,c->hidden);
    d->router_bias=h->router_bias?raw_upload(s,h->router_bias,c->experts):NULL;
    d->shared_gate=h->shared_gate_weight?raw_upload(s,h->shared_gate_weight,c->hidden):NULL;
    if(!d->in_norm||!d->post_norm||(h->router_bias&&!d->router_bias)||
       (h->shared_gate_weight&&!d->shared_gate)) return 0;
    if(h->is_attention){
        if(!tensor_upload(s,&d->q,&h->q)||!tensor_upload(s,&d->k,&h->k)||
           !tensor_upload(s,&d->v,&h->v)||!tensor_upload(s,&d->o,&h->o)) return 0;
        d->q_norm=raw_upload(s,h->q_norm,c->head_dim);
        d->k_norm=raw_upload(s,h->k_norm,c->k_head_dim);
        return d->q_norm&&d->k_norm;
    }
    if(!tensor_upload(s,&d->dn_qkv,&h->dn_qkv)||!tensor_upload(s,&d->dn_z,&h->dn_z)||
       !tensor_upload(s,&d->dn_b,&h->dn_b)||!tensor_upload(s,&d->dn_a,&h->dn_a)||
       !tensor_upload(s,&d->dn_out,&h->dn_out)) return 0;
    d->dn_conv=raw_upload(s,h->dn_conv,(size_t)c->dn_conv_dim*c->dn_conv_kernel);
    d->dn_dt=raw_upload(s,h->dn_dt_bias,c->dn_value_heads);
    d->dn_alog=raw_upload(s,h->dn_a_log,c->dn_value_heads);
    d->dn_norm=raw_upload(s,h->dn_norm,c->dn_value_dim);
    return d->dn_conv&&d->dn_dt&&d->dn_alog&&d->dn_norm;
}

QwenCudaState *qwen_cuda_create(const QwenCudaConfig *cfg,
        const QwenCudaLayerWeights *layers,const QwenCudaWeight *embedding,
        const QwenCudaWeight *lm_head,const float *final_norm,int initial_kv_capacity,
        QwenCudaCpuRoutes cpu_routes,QwenCudaRouteCommit route_commit,void *opaque){
    const char *on=getenv("COLI_CUDA");
    if(!on||*on!='1'||!cfg||!layers||!embedding||!lm_head||!final_norm) return NULL;
    int devices[16],nd=0;
    char list[128]; const char *env=getenv("COLI_GPUS");
    snprintf(list,sizeof(list),"%s",env&&*env?env:"0");
    for(char *p=strtok(list,",");p&&nd<16;p=strtok(NULL,",")) devices[nd++]=atoi(p);
    if(!nd||!coli_cuda_init(devices,nd)) return NULL;
    QwenCudaState *s=(QwenCudaState*)calloc(1,sizeof(*s));
    if(!s) return NULL;
    s->c=*cfg; s->device=devices[0]; s->request_enabled=1;
    s->cpu_routes=cpu_routes; s->route_commit=route_commit; s->opaque=opaque;
    s->layer=(QwenCudaLayer*)calloc((size_t)cfg->layers,sizeof(*s->layer));
    s->k_cache=(float**)calloc((size_t)cfg->layers,sizeof(float*));
    s->v_cache=(float**)calloc((size_t)cfg->layers,sizeof(float*));
    s->dn_rec=(float**)calloc((size_t)cfg->layers,sizeof(float*));
    s->dn_conv_state=(float**)calloc((size_t)cfg->layers,sizeof(float*));
    s->dn_rec_backup=(float**)calloc((size_t)cfg->layers,sizeof(float*));
    s->dn_conv_backup=(float**)calloc((size_t)cfg->layers,sizeof(float*));
    if(!s->layer||!s->k_cache||!s->v_cache||!s->dn_rec||!s->dn_conv_state||
       !s->dn_rec_backup||!s->dn_conv_backup) goto fail;
    if(!tensor_upload(s,&s->embed,embedding)||!tensor_upload(s,&s->lm_head,lm_head)) goto fail;
    s->final_norm=raw_upload(s,final_norm,cfg->hidden); if(!s->final_norm) goto fail;
    for(int i=0;i<cfg->layers;i++) if(!upload_layer(s,&s->layer[i],&layers[i])) goto fail;

    size_t D=(size_t)cfg->hidden,qsz=(size_t)cfg->q_heads*cfg->q_head_dim;
    size_t kvsz=(size_t)cfg->kv_heads*cfg->k_head_dim;
    size_t vals=(size_t)cfg->dn_value_heads*cfg->dn_value_dim;
    s->x=(float*)dalloc(s,D*4); s->nrm=(float*)dalloc(s,D*4);
    s->tmp=(float*)dalloc(s,D*4); s->moe=(float*)dalloc(s,D*4);
    s->logits=(float*)dalloc(s,(size_t)cfg->vocab*4);
    s->q=(float*)dalloc(s,qsz*4); s->k=(float*)dalloc(s,kvsz*4); s->v=(float*)dalloc(s,kvsz*4);
    s->dn_qkv=(float*)dalloc(s,(size_t)cfg->dn_conv_dim*4);
    s->dn_z=(float*)dalloc(s,vals*4); s->dn_b=(float*)dalloc(s,(size_t)cfg->dn_value_heads*4);
    s->dn_a=(float*)dalloc(s,(size_t)cfg->dn_value_heads*4); s->dn_outv=(float*)dalloc(s,vals*4);
    s->sh_g=(float*)dalloc(s,(size_t)cfg->shared_intermediate*4);
    s->sh_u=(float*)dalloc(s,(size_t)cfg->shared_intermediate*4);
    s->cpu_stage=(float*)dalloc(s,D*4);
    s->route_ids=(int*)dalloc(s,(size_t)cfg->topk*sizeof(int));
    s->route_weights=(float*)dalloc(s,(size_t)cfg->topk*4);
    s->ids_host=(int*)malloc(sizeof(int));
    s->route_ids_host=(int*)malloc((size_t)cfg->topk*sizeof(int));
    s->route_weights_host=(float*)malloc((size_t)cfg->topk*4);
    s->x_host=(float*)malloc(D*4); s->partial_host=(float*)malloc(D*4);
    if(!s->x||!s->nrm||!s->tmp||!s->moe||!s->logits||!s->q||!s->k||!s->v||
       !s->dn_qkv||!s->dn_z||!s->dn_b||!s->dn_a||!s->dn_outv||!s->sh_g||!s->sh_u||
       !s->cpu_stage||!s->route_ids||!s->route_weights||!s->ids_host||
       !s->route_ids_host||!s->route_weights_host||!s->x_host||!s->partial_host) goto fail;
    for(int i=0;i<cfg->layers;i++) if(!layers[i].is_attention){
        size_t rb=(size_t)cfg->dn_value_heads*cfg->dn_key_dim*cfg->dn_value_dim*4;
        size_t cb=(size_t)cfg->dn_conv_dim*(cfg->dn_conv_kernel-1)*4;
        s->dn_rec[i]=(float*)dalloc(s,rb); s->dn_rec_backup[i]=(float*)dalloc(s,rb);
        s->dn_conv_state[i]=(float*)dalloc(s,cb); s->dn_conv_backup[i]=(float*)dalloc(s,cb);
        if(!s->dn_rec[i]||!s->dn_rec_backup[i]||!s->dn_conv_state[i]||!s->dn_conv_backup[i]||
           !coli_cuda_pipe_zero(s->device,s->dn_rec[i],rb)||
           !coli_cuda_pipe_zero(s->device,s->dn_conv_state[i],cb)) goto fail;
    }
    if(initial_kv_capacity<1) initial_kv_capacity=1;
    if(!qwen_cuda_ensure_kv(s,initial_kv_capacity)) goto fail;
    s->ready=1; s->state_valid=1;
    fprintf(stderr,"[qwen-cuda] dense/state/scratch reserved %.2f GB, KV capacity %d\n",
            s->reserved/1073741824.0,s->kv_cap);
    return s;
fail:
    qwen_cuda_destroy(s);
    return NULL;
}

void qwen_cuda_destroy(QwenCudaState *s){
    if(!s) return;
    for(int i=0;i<s->c.layers;i++){
        QwenCudaLayer *l=s->layer?&s->layer[i]:NULL;
        if(l){ tensor_drop(&l->q);tensor_drop(&l->k);tensor_drop(&l->v);tensor_drop(&l->o);
            tensor_drop(&l->router);tensor_drop(&l->sh_g);tensor_drop(&l->sh_u);tensor_drop(&l->sh_d);
            tensor_drop(&l->dn_qkv);tensor_drop(&l->dn_z);tensor_drop(&l->dn_b);tensor_drop(&l->dn_a);tensor_drop(&l->dn_out);
            raw_drop(s,&l->in_norm);raw_drop(s,&l->post_norm);raw_drop(s,&l->q_norm);raw_drop(s,&l->k_norm);
            raw_drop(s,&l->router_bias);raw_drop(s,&l->shared_gate);raw_drop(s,&l->dn_conv);
            raw_drop(s,&l->dn_dt);raw_drop(s,&l->dn_alog);raw_drop(s,&l->dn_norm); }
        if(s->k_cache) raw_drop(s,&s->k_cache[i]); if(s->v_cache) raw_drop(s,&s->v_cache[i]);
        if(s->dn_rec) raw_drop(s,&s->dn_rec[i]); if(s->dn_conv_state) raw_drop(s,&s->dn_conv_state[i]);
        if(s->dn_rec_backup) raw_drop(s,&s->dn_rec_backup[i]); if(s->dn_conv_backup) raw_drop(s,&s->dn_conv_backup[i]);
    }
    tensor_drop(&s->embed); tensor_drop(&s->lm_head); raw_drop(s,&s->final_norm);
    raw_drop(s,&s->x);raw_drop(s,&s->nrm);raw_drop(s,&s->tmp);raw_drop(s,&s->moe);raw_drop(s,&s->logits);
    raw_drop(s,&s->q);raw_drop(s,&s->k);raw_drop(s,&s->v);raw_drop(s,&s->dn_qkv);raw_drop(s,&s->dn_z);
    raw_drop(s,&s->dn_b);raw_drop(s,&s->dn_a);raw_drop(s,&s->dn_outv);raw_drop(s,&s->sh_g);raw_drop(s,&s->sh_u);
    raw_drop(s,&s->cpu_stage);raw_drop(s,(float**)&s->route_ids);raw_drop(s,&s->route_weights);
    free(s->ids_host);free(s->route_ids_host);free(s->route_weights_host);free(s->x_host);free(s->partial_host);
    free(s->layer);free(s->k_cache);free(s->v_cache);free(s->dn_rec);free(s->dn_conv_state);
    free(s->dn_rec_backup);free(s->dn_conv_backup);free(s);
}

int qwen_cuda_ready(const QwenCudaState *s){ return s&&s->ready; }
size_t qwen_cuda_reserved_bytes(const QwenCudaState *s){ return s?s->reserved:0; }
int qwen_cuda_kv_len(const QwenCudaState *s){ return s?s->kv_len:0; }
void qwen_cuda_disable_request(QwenCudaState *s){ if(s) s->request_enabled=0; }
int qwen_cuda_request_enabled(const QwenCudaState *s){ return s&&s->ready&&s->request_enabled; }

int qwen_cuda_ensure_kv(QwenCudaState *s,int cap){
    if(!s||cap<1) return 0;
    if(cap<=s->kv_cap) return 1;
    float **nk=(float**)calloc((size_t)s->c.layers,sizeof(float*));
    float **nv=(float**)calloc((size_t)s->c.layers,sizeof(float*));
    if(!nk||!nv){free(nk);free(nv);return 0;}
    size_t row=(size_t)s->c.kv_heads*s->c.k_head_dim;
    for(int i=0;i<s->c.layers;i++) if(s->layer[i].q){
        size_t bytes=row*(size_t)cap*4;
        nk[i]=(float*)coli_cuda_pipe_alloc(s->device,bytes);
        nv[i]=(float*)coli_cuda_pipe_alloc(s->device,bytes);
        if(!nk[i]||!nv[i]) goto fail;
        if(s->kv_len){
            for(int h=0;h<s->c.kv_heads;h++){
                size_t used=(size_t)s->kv_len*s->c.k_head_dim*4;
                if(!coli_cuda_pipe_copy(s->device,nk[i]+(size_t)h*cap*s->c.k_head_dim,
                       s->k_cache[i]+(size_t)h*s->kv_cap*s->c.k_head_dim,used)||
                   !coli_cuda_pipe_copy(s->device,nv[i]+(size_t)h*cap*s->c.k_head_dim,
                       s->v_cache[i]+(size_t)h*s->kv_cap*s->c.k_head_dim,used)) goto fail;
            }
        }
    }
    for(int i=0;i<s->c.layers;i++) if(s->layer[i].q){
        raw_drop(s,&s->k_cache[i]);raw_drop(s,&s->v_cache[i]);
        s->k_cache[i]=nk[i];s->v_cache[i]=nv[i];
        s->reserved+=2*row*(size_t)cap*4;
    }
    free(nk);free(nv);s->kv_cap=cap;return 1;
fail:
    for(int i=0;i<s->c.layers;i++){if(nk[i])coli_cuda_pipe_free(s->device,nk[i]);if(nv[i])coli_cuda_pipe_free(s->device,nv[i]);}
    free(nk);free(nv);return 0;
}

int qwen_cuda_reset(QwenCudaState *s){
    if(!s||!s->ready) return 0;
    for(int i=0;i<s->c.layers;i++) if(s->dn_rec[i]){
        size_t rb=(size_t)s->c.dn_value_heads*s->c.dn_key_dim*s->c.dn_value_dim*4;
        size_t cb=(size_t)s->c.dn_conv_dim*(s->c.dn_conv_kernel-1)*4;
        if(!coli_cuda_pipe_zero(s->device,s->dn_rec[i],rb)||
           !coli_cuda_pipe_zero(s->device,s->dn_conv_state[i],cb)) return 0;
    }
    s->kv_len=0;s->request_enabled=1;s->state_valid=1;s->epoch++;
    return coli_cuda_pipe_sync(s->device);
}

static int snapshot(QwenCudaState *s){
    for(int i=0;i<s->c.layers;i++) if(s->dn_rec[i]){
        size_t rb=(size_t)s->c.dn_value_heads*s->c.dn_key_dim*s->c.dn_value_dim*4;
        size_t cb=(size_t)s->c.dn_conv_dim*(s->c.dn_conv_kernel-1)*4;
        if(!coli_cuda_pipe_copy(s->device,s->dn_rec_backup[i],s->dn_rec[i],rb)||
           !coli_cuda_pipe_copy(s->device,s->dn_conv_backup[i],s->dn_conv_state[i],cb)) return 0;
    }
    return 1;
}
static int rollback(QwenCudaState *s){
    int ok=1;
    for(int i=0;i<s->c.layers;i++) if(s->dn_rec[i]){
        size_t rb=(size_t)s->c.dn_value_heads*s->c.dn_key_dim*s->c.dn_value_dim*4;
        size_t cb=(size_t)s->c.dn_conv_dim*(s->c.dn_conv_kernel-1)*4;
        ok&=coli_cuda_pipe_copy(s->device,s->dn_rec[i],s->dn_rec_backup[i],rb);
        ok&=coli_cuda_pipe_copy(s->device,s->dn_conv_state[i],s->dn_conv_backup[i],cb);
    }
    ok&=coli_cuda_pipe_sync(s->device);s->in_txn=0;return ok;
}

static int cpu_partial(QwenCudaState *s,int layer,uint32_t mask){
    if(!mask) return 1;
    if(!s->cpu_routes) return 0;
    memset(s->partial_host,0,(size_t)s->c.hidden*4);
    if(!s->cpu_routes(s->opaque,layer,s->route_ids_host,s->route_weights_host,
                      s->c.topk,mask,s->x_host,s->partial_host)) return 0;
    return coli_cuda_pipe_upload(s->device,s->cpu_stage,s->partial_host,
                                 (size_t)s->c.hidden*4)&&
           coli_cuda_pipe_add(s->device,s->moe,s->cpu_stage,(size_t)s->c.hidden);
}

static int one_token(QwenCudaState *s,int id,int pos,int *pending_ids,
                     float *pending_weights,int token_index){
    const QwenCudaConfig *c=&s->c;
    if(!coli_cuda_pipe_embedding(s->embed,s->x,&id,1)) return 0;
    for(int i=0;i<c->layers;i++){
        QwenCudaLayer *l=&s->layer[i];
        if(!coli_cuda_qwen_rmsnorm(s->device,s->nrm,s->x,l->in_norm,1,c->hidden,c->rms_eps)) return 0;
        if(l->q){
            if(!coli_cuda_pipe_gemm(l->q,s->q,s->nrm,1)||!coli_cuda_pipe_gemm(l->k,s->k,s->nrm,1)||
               !coli_cuda_pipe_gemm(l->v,s->v,s->nrm,1)||
               !coli_cuda_qwen_gqa(s->device,s->tmp,s->q,s->k,s->v,l->q_norm,l->k_norm,
                    s->k_cache[i],s->v_cache[i],1,pos,s->kv_cap,c->q_heads,c->kv_heads,
                    c->head_dim,c->q_head_dim,c->k_head_dim,c->rotary_dim,c->rms_eps,c->rope_theta)||
               !coli_cuda_pipe_gemm(l->o,s->moe,s->tmp,1)) return 0;
        }else{
            if(!coli_cuda_pipe_gemm(l->dn_qkv,s->dn_qkv,s->nrm,1)||
               !coli_cuda_pipe_gemm(l->dn_z,s->dn_z,s->nrm,1)||
               !coli_cuda_pipe_gemm(l->dn_b,s->dn_b,s->nrm,1)||
               !coli_cuda_pipe_gemm(l->dn_a,s->dn_a,s->nrm,1)||
               !coli_cuda_qwen_deltanet(s->device,s->dn_outv,s->dn_qkv,s->dn_z,s->dn_b,s->dn_a,
                    l->dn_conv,l->dn_dt,l->dn_alog,l->dn_norm,s->dn_conv_state[i],s->dn_rec[i],
                    1,c->dn_value_heads,c->dn_key_heads,c->dn_key_dim,c->dn_value_dim,
                    c->dn_conv_kernel,c->dn_conv_dim,c->rms_eps)||
               !coli_cuda_pipe_gemm(l->dn_out,s->moe,s->dn_outv,1)) return 0;
        }
        if(!coli_cuda_pipe_add(s->device,s->x,s->moe,(size_t)c->hidden)||
           !coli_cuda_qwen_rmsnorm(s->device,s->nrm,s->x,l->post_norm,1,c->hidden,c->rms_eps)||
           !coli_cuda_qwen_router(l->router,l->router_bias,s->nrm,1,c->topk,c->groups,
                c->topk_groups,s->route_ids,s->route_weights,s->route_ids_host,
                s->route_weights_host,NULL)||
           !coli_cuda_pipe_zero(s->device,s->moe,(size_t)c->hidden*4)) return 0;
        size_t po=((size_t)token_index*c->layers+i)*c->topk;
        memcpy(pending_ids+po,s->route_ids_host,(size_t)c->topk*sizeof(int));
        memcpy(pending_weights+po,s->route_weights_host,(size_t)c->topk*4);
        QtDevIssue issue;
        int issued=qt_issue_dev(&issue,i,s->route_ids_host,s->route_weights_host,c->topk,
                                s->device,s->nrm,s->x_host);
        if(!issued&&!coli_cuda_pipe_download(s->device,s->nrm,s->x_host,(size_t)c->hidden*4)) return 0;
        if(!coli_cuda_pipe_gemm(l->sh_g,s->sh_g,s->nrm,1)||
           !coli_cuda_pipe_gemm(l->sh_u,s->sh_u,s->nrm,1)||
           !coli_cuda_pipe_silu_mul(s->device,s->sh_g,s->sh_u,(size_t)c->shared_intermediate)||
           !coli_cuda_pipe_gemm(l->sh_d,s->tmp,s->sh_g,1)){
            if(issued) qt_abort_dev(&issue); return 0;
        }
        if(l->shared_gate){
            if(!coli_cuda_qwen_shared_gate_add(s->device,s->moe,s->tmp,s->nrm,
                                                l->shared_gate,1,c->hidden)){
                if(issued)qt_abort_dev(&issue);return 0;
            }
        }else if(!coli_cuda_pipe_add(s->device,s->moe,s->tmp,(size_t)c->hidden)){
            if(issued)qt_abort_dev(&issue);return 0;
        }
        uint32_t cpu=issued?qt_take_dev(&issue,s->moe):
                         (c->topk==32?0xffffffffu:((1u<<c->topk)-1u));
        if(!cpu_partial(s,i,cpu)||!coli_cuda_pipe_add(s->device,s->x,s->moe,(size_t)c->hidden)) return 0;
    }
    return 1;
}

int qwen_cuda_step(QwenCudaState *s,const int *ids,int S,int pos,float *logits_host,
                   int *greedy_id){
    if(!s||!s->ready||!s->state_valid||!s->request_enabled||!ids||S<1||
       !logits_host||!greedy_id||pos!=s->kv_len||pos+S>s->kv_cap||s->in_txn) return 0;
    size_t routes=(size_t)S*s->c.layers*s->c.topk;
    int *pending_ids=(int*)malloc(routes*sizeof(int));
    float *pending_weights=(float*)malloc(routes*4);
    if(!pending_ids||!pending_weights){free(pending_ids);free(pending_weights);return 0;}
    if(!snapshot(s)){free(pending_ids);free(pending_weights);return 0;}
    s->in_txn=1;
    for(int t=0;t<S;t++) if(!one_token(s,ids[t],pos+t,pending_ids,pending_weights,t)) goto fail;
    if(!coli_cuda_qwen_rmsnorm(s->device,s->nrm,s->x,s->final_norm,1,s->c.hidden,s->c.rms_eps)||
       !coli_cuda_pipe_gemm(s->lm_head,s->logits,s->nrm,1)||
       !coli_cuda_pipe_argmax(s->device,s->logits,s->c.vocab,greedy_id,NULL)||
       !coli_cuda_pipe_download(s->device,s->logits,logits_host,(size_t)s->c.vocab*4)||
       !coli_cuda_pipe_sync(s->device)) goto fail;
    s->kv_len=pos+S;s->epoch++;s->in_txn=0;
    if(s->route_commit) for(int t=0;t<S;t++) for(int i=0;i<s->c.layers;i++){
        size_t o=((size_t)t*s->c.layers+i)*s->c.topk;
        s->route_commit(s->opaque,i,pending_ids+o,pending_weights+o,s->c.topk);
    }
    free(pending_ids);free(pending_weights);return 1;
fail:
    if(!rollback(s)){s->state_valid=0;s->request_enabled=0;}
    free(pending_ids);free(pending_weights);return 0;
}

int qwen_cuda_export_state(QwenCudaState *s,float **kh,float **vh,float **rh,float **ch,
                           int len,int host_cap){
    if(!s||!s->state_valid||len<0||len>s->kv_len||host_cap<len) return 0;
    for(int i=0;i<s->c.layers;i++) if(s->layer[i].q){
        for(int h=0;h<s->c.kv_heads;h++){
            size_t n=(size_t)len*s->c.k_head_dim;
            if(!coli_cuda_pipe_download(s->device,s->k_cache[i]+(size_t)h*s->kv_cap*s->c.k_head_dim,
                                        kh[i]+(size_t)h*host_cap*s->c.k_head_dim,n*4)||
               !coli_cuda_pipe_download(s->device,s->v_cache[i]+(size_t)h*s->kv_cap*s->c.k_head_dim,
                                        vh[i]+(size_t)h*host_cap*s->c.k_head_dim,n*4)) return 0;
        }
    }else{
        size_t rn=(size_t)s->c.dn_value_heads*s->c.dn_key_dim*s->c.dn_value_dim;
        size_t cn=(size_t)s->c.dn_conv_dim*(s->c.dn_conv_kernel-1);
        if(!coli_cuda_pipe_download(s->device,s->dn_rec[i],rh[i],rn*4)||
           !coli_cuda_pipe_download(s->device,s->dn_conv_state[i],ch[i],cn*4)) return 0;
    }
    return 1;
}
int qwen_cuda_import_state(QwenCudaState *s,float **kh,float **vh,float **rh,float **ch,
                           int len,int host_cap){
    if(!s||len<0||len>s->kv_cap||host_cap<len) return 0;
    for(int i=0;i<s->c.layers;i++) if(s->layer[i].q){
        for(int h=0;h<s->c.kv_heads;h++){
            size_t n=(size_t)len*s->c.k_head_dim;
            if(!coli_cuda_pipe_upload(s->device,s->k_cache[i]+(size_t)h*s->kv_cap*s->c.k_head_dim,
                                      kh[i]+(size_t)h*host_cap*s->c.k_head_dim,n*4)||
               !coli_cuda_pipe_upload(s->device,s->v_cache[i]+(size_t)h*s->kv_cap*s->c.k_head_dim,
                                      vh[i]+(size_t)h*host_cap*s->c.k_head_dim,n*4)) return 0;
        }
    }else{
        size_t rn=(size_t)s->c.dn_value_heads*s->c.dn_key_dim*s->c.dn_value_dim;
        size_t cn=(size_t)s->c.dn_conv_dim*(s->c.dn_conv_kernel-1);
        if(!coli_cuda_pipe_upload(s->device,s->dn_rec[i],rh[i],rn*4)||
           !coli_cuda_pipe_upload(s->device,s->dn_conv_state[i],ch[i],cn*4)) return 0;
    }
    s->kv_len=len;return coli_cuda_pipe_sync(s->device);
}
