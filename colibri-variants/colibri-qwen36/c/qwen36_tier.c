/* qwen36_tier.c — M-QTIER: VRAM-Experten-Tier (Implementierung). Siehe Header. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "qwen36_tier.h"
#include "backend_cuda.h"
#include "b70_tier.h"

#define QT_MAX_DEV 8
#define QT_QCAP 48            /* Upload-Queue-Tiefe (Staging ~1,6 MB/Eintrag) */

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    uint32_t heat;
    uint8_t resident, queued, planned;
    /* RAM-Zeiger (Slots sind bei cap==n_experts nie evicted) — für Warmstart,
     * Lookahead und LFRU-Swaps ohne Engine-Callback */
    const uint8_t *g4,*u4,*d4; const float *gs,*us,*ds;
} QSlot;

static struct {
    int on, nl, ne, D, Ih, topk, ndev;
    int egs; size_t sc_gu, sc_d;   /* expert group size + per-matrix scale counts (gs64) */
    int dev[QT_MAX_DEV];
    size_t budget[QT_MAX_DEV], used[QT_MAX_DEV];
    size_t exp_bytes;                     /* VRAM-Bedarf je Experte (geschätzt) */
    size_t cuda_target, cuda_planned;      /* explicit heat-ordered ownership split */
    QSlot *slot;                          /* [nl*ne] */
    pthread_mutex_t mx;
    pthread_t th;
    int th_stop;
    /* Upload-Queue (Ring) mit Staging-Kopien */
    struct { int layer, eid; uint8_t *w; float *s; int v_layer, v_eid; } q[QT_QCAP];
    int qh, qt_, qn;
    pthread_cond_t cv;
    /* Statistik */
    uint64_t hits[QT_MAX_DEV], b70_hits, direct_cpu, recovery_cpu;
    uint64_t uploads, q_full_skips;
    /* Issue-Zustand des Decode-Threads (single-threaded) */
    int is_cnt[QT_MAX_DEV];
    int is_k[QT_MAX_DEV][32];
    float *is_x;                          /* count*D Replikate je Device */
    /* Device-input transaction state. Scratch is reserved during init, before
     * expert budgets are measured and before any CUDA/B70 warmstart. */
    float *dev_slots, *dev_acc, *dev_stage;
    int dev_issue_devices[QT_MAX_DEV], dev_issue_count;
    float dev_weights[32];
    uint64_t dev_serial;
    int dev_pending;
    const float *dev_x;
    float *dev_x_host;
    int dev_host_staged;
    /* M3 */
    int *fill_order; int fill_cur;        /* Warmstart-Reihenfolge (Heat desc) */
    int issue_open;                       /* Schutz: kein tensor_free während Issue */
    pthread_cond_t cv_take;               /* signalisiert qt_take-Ende + Queue-Platz */
    uint64_t tick, swaps, pf_hits, pf_notes;
    uint32_t *heat0;                      /* geladene Heat-Tabelle (HEAT_FILE) */
    int inflight;                         /* Uploads in-flight (dequeued, not yet resident) */
} G;

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return eid % G.ndev; }

static int qt_init_fail(const char *why,int sync_inited){
    if(why) fprintf(stderr,"[qtier] %s -> CPU fallback\n",why);
    if(sync_inited){
        pthread_cond_destroy(&G.cv_take);
        pthread_cond_destroy(&G.cv);
        pthread_mutex_destroy(&G.mx);
    }
    if(G.dev_slots) coli_cuda_pipe_free(G.dev[0],G.dev_slots);
    if(G.dev_acc) coli_cuda_pipe_free(G.dev[0],G.dev_acc);
    if(G.dev_stage) coli_cuda_pipe_free(G.dev[0],G.dev_stage);
    free(G.slot); free(G.is_x); free(G.fill_order); free(G.heat0);
    size_t tensor_count=0,tensor_bytes=0;
    coli_cuda_stats(-1,&tensor_count,&tensor_bytes);
    if(!tensor_count) coli_cuda_shutdown();
    memset(&G,0,sizeof G);
    return 0;
}

/* Staging: packed int4 (g|u|d) Zweierkomplement -> Offset-Binary (XOR 0x88)
 * + Scales (gs|us|ds) kopieren. */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                  const float *gs,const float *us,const float *ds){
    size_t mb = (size_t)G.D*G.Ih/2;
    const uint64_t X=0x8888888888888888ull;
    const uint64_t *sg=(const uint64_t*)g4,*su=(const uint64_t*)u4,*sd=(const uint64_t*)d4;
    uint64_t *w0=(uint64_t*)dw,*w1=(uint64_t*)(dw+mb),*w2=(uint64_t*)(dw+2*mb);
    for(size_t i=0;i<mb/8;i++){ w0[i]=sg[i]^X; w1[i]=su[i]^X; w2[i]=sd[i]^X; }
    memcpy(dsc,                 gs, G.sc_gu*sizeof(float));
    memcpy(dsc+G.sc_gu,         us, G.sc_gu*sizeof(float));
    memcpy(dsc+2*G.sc_gu,       ds, G.sc_d *sizeof(float));
}

static void *uploader(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&G.mx);
        while(G.qn==0 && !G.th_stop) pthread_cond_wait(&G.cv,&G.mx);
        if(G.th_stop && G.qn==0){ pthread_mutex_unlock(&G.mx); return NULL; }
        int layer=G.q[G.qh].layer, eid=G.q[G.qh].eid;
        int vl=G.q[G.qh].v_layer, ve=G.q[G.qh].v_eid;
        uint8_t *w=G.q[G.qh].w; float *sc=G.q[G.qh].s;
        G.qh=(G.qh+1)%QT_QCAP; G.qn--;
        G.inflight++;                             /* track in-flight upload (fix #4) */
        pthread_cond_broadcast(&G.cv_take);          /* Queue-Platz frei */
        if(ve>=0){
            /* LFRU-Swap: Victim erst freigeben, wenn keine Issue offen ist */
            while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
            QSlot *v=qs(vl,ve);
            ColiCudaTensor *a=v->tg,*b=v->tu,*ct=v->td;
            v->tg=v->tu=v->td=NULL;
            pthread_mutex_unlock(&G.mx);
            if(a)coli_cuda_tensor_free(a); if(b)coli_cuda_tensor_free(b); if(ct)coli_cuda_tensor_free(ct);
        } else pthread_mutex_unlock(&G.mx);

        int dv = G.dev[home(eid)];
        size_t mb=(size_t)G.D*G.Ih/2;
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok;
        if(G.egs){
            ok = coli_cuda_tensor_upload_g(&tg, w,      sc,             4, G.D,  G.Ih, dv, G.egs)
              && coli_cuda_tensor_upload_g(&tu, w+mb,   sc+G.sc_gu,     4, G.D,  G.Ih, dv, G.egs)
              && coli_cuda_tensor_upload_g(&td, w+2*mb, sc+2*G.sc_gu,   4, G.Ih, G.D,  dv, G.egs);
        } else {
            ok = coli_cuda_tensor_upload(&tg, w,      sc,          2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   sc+G.Ih,     2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, sc+2*G.Ih,   2, G.Ih, G.D,  dv);
        }
        free(w); free(sc);
        pthread_mutex_lock(&G.mx);
        QSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid);
                /* Normal/planned fills reserved one expert. A swap is budget
                 * neutral, so subtracting on its upload failure underflows. */
                if(ve<0&&G.used[hd]>=G.exp_bytes) G.used[hd]-=G.exp_bytes;
                G.budget[hd]=G.used[hd];   /* VRAM real full: stop new fills */
                if(tg) coli_cuda_tensor_free(tg);  /* fix #3: free partial uploads */
                if(tu) coli_cuda_tensor_free(tu);
                if(td) coli_cuda_tensor_free(td); }
        s->queued=0;
        G.inflight--;                               /* upload done (fix #4) */
        pthread_cond_broadcast(&G.cv_take);         /* wake qt_fill_wait */
        pthread_mutex_unlock(&G.mx);
    }
}

int qt_init(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs){
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(cap != ne){
        fprintf(stderr,"[qtier] cap=%d != n_experts=%d -> Tier aus (volle RAM-Residenz nötig, Z5)\n",cap,ne);
        return 0;
    }
    if(topk>32){ fprintf(stderr,"[qtier] topk>32 nicht unterstützt\n"); return 0; }
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk;

    /* Geräte: COLI_GPUS="0,1"; a missing override must remain valid on the
     * common single-GPU host, so default to device 0 only. */
    const char *gl=getenv("COLI_GPUS");
    char buf[128]; snprintf(buf,sizeof buf,"%s", gl?gl:"0");
    for(char *t=strtok(buf,","); t && G.ndev<QT_MAX_DEV; t=strtok(NULL,","))
        G.dev[G.ndev++]=atoi(t);
    if(!coli_cuda_init(G.dev,G.ndev)){ fprintf(stderr,"[qtier] coli_cuda_init FAIL -> CPU\n"); return 0; }
    int have=coli_cuda_device_count();
    if(have<G.ndev){ G.ndev=have; }
    if(G.ndev<1){ fprintf(stderr,"[qtier] keine CUDA-Devices -> CPU\n"); return 0; }
#ifdef COLI_QWEN_CUDA_STATE
    /* Resident execution scratch is part of the dense/state reservation, not
     * the expert budget. Reserve it before querying remaining free VRAM. */
    for(int i=0;i<G.ndev;i++)
        if(!coli_cuda_expert_group_resident_reserve(G.dev[i],D,Ih))
            return qt_init_fail("resident scratch reservation failed",0);
    G.dev_slots=coli_cuda_pipe_alloc(G.dev[0],(size_t)G.ndev*D*sizeof(float));
    G.dev_acc=coli_cuda_pipe_alloc(G.dev[0],(size_t)D*sizeof(float));
    G.dev_stage=coli_cuda_pipe_alloc(G.dev[0],(size_t)D*sizeof(float));
    if(!G.dev_slots||!G.dev_acc||!G.dev_stage)
        return qt_init_fail("device-input scratch reservation failed",0);
#endif

    /* Budget is capped against memory remaining after every dense/state/scratch
     * allocation. A user request can lower this cap, never exceed it. */
    G.egs = expert_gs;
    G.sc_gu = expert_gs ? (size_t)Ih * ((D + expert_gs - 1)/expert_gs) : (size_t)Ih;
    G.sc_d  = expert_gs ? (size_t)D  * ((Ih + expert_gs - 1)/expert_gs) : (size_t)D;
    G.exp_bytes = 3ull*D*Ih/2 + (2*G.sc_gu+G.sc_d)*sizeof(float) + 4096;
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        size_t usable=freeb>(1ull<<30)?freeb-(1ull<<30):0;
        size_t requested=(bg&&strcmp(bg,"auto")&&atof(bg)>0)
                         ?(size_t)(atof(bg)*1024.0*1024.0*1024.0):usable;
        size_t b=requested<usable?requested:usable;
        G.budget[i]=b;
        fprintf(stderr,"[qtier] dev %d: %.1f GB remaining, final expert budget %.1f GB (~%zu experts)\n",
                G.dev[i],freeb/1073741824.0,b/1073741824.0,b/G.exp_bytes);
    }
    G.slot=calloc((size_t)nl*ne,sizeof(QSlot));
    G.is_x=malloc((size_t)32*D*sizeof(float));
    if(!G.slot||!G.is_x) return qt_init_fail("host tier-state allocation failed",0);
    /* M3: gelernte Heat laden (HEAT_FILE), Fill-Reihenfolge + Startwerte */
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"rb");
        if(f){
            uint32_t hdr[3]={0,0,0};
            if(fread(hdr,4,3,f)==3 && hdr[0]==0x51544831u && hdr[1]==(uint32_t)nl && hdr[2]==(uint32_t)ne){
                G.heat0=malloc((size_t)nl*ne*4);
                if(G.heat0 && fread(G.heat0,4,(size_t)nl*ne,f)==(size_t)nl*ne){
                    for(size_t i=0;i<(size_t)nl*ne;i++) G.slot[i].heat=G.heat0[i]>>1; /* Decay */
                    fprintf(stderr,"[qtier] HEAT_FILE geladen: %s\n",hf);
                } else { free(G.heat0); G.heat0=NULL; }
            }
            fclose(f);
        }
    }
    pthread_mutex_init(&G.mx,NULL); pthread_cond_init(&G.cv,NULL); pthread_cond_init(&G.cv_take,NULL);
    if(pthread_create(&G.th,NULL,uploader,NULL)!=0)
        return qt_init_fail("uploader thread creation failed",1);
    G.on=1;
    b70_tier_init(nl, ne, D, Ih, topk, expert_gs);
    size_t total_experts=(size_t)nl*ne;
    size_t b70_capacity=(size_t)b70_tier_capacity();
    if(b70_capacity>total_experts) b70_capacity=total_experts;
    G.cuda_target=total_experts-b70_capacity;
    fprintf(stderr,
            "[qtier] ownership target: CUDA %zu (%.1f%%) | B70 %zu (%.1f%%)\n",
            G.cuda_target,100.0*G.cuda_target/total_experts,
            b70_capacity,100.0*b70_capacity/total_experts);
    fprintf(stderr,"[qtier] VRAM-Experten-Tier aktiv: %d Device(s), %.2f MB/Experte\n",
            G.ndev, G.exp_bytes/1048576.0);
    return 1;
}

int qt_ready(void){ return G.on; }

/* Ist (layer,eid) aktuell im VRAM resident? (für RAM-Freigabe, Maßnahme 2) */
int qt_is_resident(int layer,int eid){
    if(!G.on) return 0;
    pthread_mutex_lock(&G.mx);
    int r = qs(layer,eid)->resident;
    pthread_mutex_unlock(&G.mx);
    return r;
}

/* intern: unter gehaltenem G.mx einreihen. victim=-1: normaler Upload
 * (Budget wird reserviert); victim>=0: Swap (Budget neutral). */
static int enqueue_locked(int layer,int eid,int v_layer,int v_eid,int reserved){
    QSlot *s=qs(layer,eid);
    if(s->resident||s->queued||!s->g4) return 0;
    if(G.qn>=QT_QCAP){ G.q_full_skips++; return 0; }
    int hd=home(eid);
    if(!reserved && v_eid<0 && G.used[hd]+G.exp_bytes>G.budget[hd]) return 0;
    size_t mb=(size_t)G.D*G.Ih/2;
    uint8_t *w=malloc(3*mb); float *sc=malloc((2*G.sc_gu+G.sc_d)*sizeof(float));
    if(!w||!sc){ free(w); free(sc); return 0; }
    if(!reserved && v_eid<0) G.used[hd]+=G.exp_bytes;
    s->queued=1;
    stage(w,sc,s->g4,s->u4,s->d4,s->gs,s->us,s->ds);
    G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid; G.q[G.qt_].w=w; G.q[G.qt_].s=sc;
    G.q[G.qt_].v_layer=v_layer; G.q[G.qt_].v_eid=v_eid;
    G.qt_=(G.qt_+1)%QT_QCAP; G.qn++;
    pthread_cond_signal(&G.cv);
    return 1;
}

void qt_note(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    if(s->heat<0xFFFFFFFFu) s->heat++;
    if(!b70_tier_is_resident(layer,eid)) enqueue_locked(layer,eid,-1,-1,0);
    else b70_tier_note(layer,eid,g4,u4,d4,gs,us,ds);
    pthread_mutex_unlock(&G.mx);
}

/* M3: blockierende Variante für den Warmstart (wartet auf Queue-Platz). */
void qt_note_block(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    if(b70_tier_is_resident(layer,eid)){
        b70_tier_note(layer,eid,g4,u4,d4,gs,us,ds);
        pthread_mutex_unlock(&G.mx);
        return;
    }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    enqueue_locked(layer,eid,-1,-1,0);
    pthread_mutex_unlock(&G.mx);
}

/* M3: Warmstart-Reihenfolge — Heat absteigend (HEAT_FILE), sonst natürlich.
 * Liefert 0, wenn alle Budgets voll oder Liste erschöpft. */
static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    return ha<hb ? 1 : ha>hb ? -1 : 0;
}
int qt_fill_next(int *layer,int *eid){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order){
        G.fill_order=malloc(n*sizeof(int));
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
        G.fill_cur=0;
    }
    while((size_t)G.fill_cur<n){
        int gi=G.fill_order[G.fill_cur];
        int l=gi/G.ne, e=gi%G.ne, hd=home(e);
        QSlot *s=qs(l,e);
        G.fill_cur++;
        if(s->resident||s->queued) continue;
        int use_cuda=G.cuda_planned<G.cuda_target &&
                     G.used[hd]+G.exp_bytes<=G.budget[hd];
        if(use_cuda) G.cuda_planned++;
        else if(!b70_tier_claim(l,e)) continue;
        *layer=l; *eid=e;
        pthread_mutex_unlock(&G.mx);
        return 1;
    }
    pthread_mutex_unlock(&G.mx);
    return 0;
}

/* M3b (Parallelisierung): plant das komplette Warmstart-Set in einem Zug —
 * Heat-Reihenfolge und Budget-Reservierung exakt wie qt_fill_next, aber ohne
 * zu laden. Die Experten werden danach von mehreren Threads geladen und per
 * qt_note_planned eingereiht. */
int qt_plan_fill(int *layers,int *eids,int max){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    int cnt=0;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order){
        G.fill_order=malloc(n*sizeof(int));
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
        G.fill_cur=0;
    }
    while((size_t)G.fill_cur<n && cnt<max){
        int gi=G.fill_order[G.fill_cur++];
        int l=gi/G.ne, e=gi%G.ne, hd=home(e);
        QSlot *s=qs(l,e);
        if(s->resident||s->queued||s->planned) continue;
        if(G.cuda_planned<G.cuda_target &&
           G.used[hd]+G.exp_bytes<=G.budget[hd]){
            G.used[hd]+=G.exp_bytes;
            G.cuda_planned++;             /* hottest entries stay on CUDA */
        } else if(!b70_tier_claim(l,e)) {
            continue;
        }
        s->planned=1;
        layers[cnt]=l; eids[cnt]=e; cnt++;
    }
    pthread_mutex_unlock(&G.mx);
    return cnt;
}

/* Threadsicher (von mehreren Ladethreads aufrufbar): stage + einreihen eines
 * per qt_plan_fill reservierten Experten; blockiert nur bei voller Queue. */
void qt_note_planned(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    if(b70_tier_is_resident(layer,eid)){
        b70_tier_note(layer,eid,g4,u4,d4,gs,us,ds);
        s->planned=0;
        pthread_mutex_unlock(&G.mx);
        return;
    }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    if(!enqueue_locked(layer,eid,-1,-1,1)){
        if(s->planned) G.used[home(eid)]-=G.exp_bytes;
    }
    s->planned=0;
    pthread_mutex_unlock(&G.mx);
}

/* M3: wartet, bis die Upload-Queue leer ist (Ende Warmstart). */
void qt_fill_wait(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    while((G.qn>0 || G.inflight>0) && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
}

/* M3: LFRU-Swap-Prüfung (alle 16 Ticks = Tokens): je Device kältester
 * Resident vs. heißester Nicht-Resident mit Hysterese (tier.h-Semantik). */
static void qt_lfru_tick_locked(void){
    if(++G.tick % 16) return;
    size_t n=(size_t)G.nl*G.ne;
    for(int di=0;di<G.ndev;di++){
        int cold=-1, hot=-1; uint32_t ch=0, hh=0;
        for(size_t i=0;i<n;i++){
            QSlot *s=&G.slot[i];
            int e=(int)(i%G.ne);
            if(home(e)!=di) continue;
            if(s->resident && !s->queued){ if(cold<0||s->heat<ch){ cold=(int)i; ch=s->heat; } }
            else if(!s->resident && !s->queued && s->g4 &&
                    !b70_tier_is_resident((int)(i/G.ne),e)){
                if(hot<0||s->heat>hh){ hot=(int)i; hh=s->heat; }
            }
        }
        if(cold<0||hot<0) continue;
        if(hh<=ch+(ch>>2)+4) continue;                    /* Hysterese wie tier.h */
        QSlot *v=&G.slot[cold];
        v->resident=0;                                    /* ab sofort CPU-Fallback */
        if(enqueue_locked(hot/G.ne,hot%G.ne,cold/G.ne,cold%G.ne,0)) G.swaps++;
        else v->resident=1;                               /* Queue voll: zurücknehmen */
    }
}

uint32_t qt_issue(int layer,const int *eids,const float *weights,int K,const float *x){
    if(!G.on||K>32) return 0;
    uint32_t mask=0;
    ColiCudaTensor *tg[QT_MAX_DEV][32],*tu[QT_MAX_DEV][32],*td[QT_MAX_DEV][32];
    static int rows[32]={0};
    if(!rows[0]) for(int i=0;i<32;i++) rows[i]=1;
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    b70_tier_record(layer,eids,weights,K,x,&mask);
    G.b70_hits += (uint64_t)__builtin_popcount(mask);
    pthread_mutex_lock(&G.mx);
    if(layer==0) qt_lfru_tick_locked();
    G.issue_open=1;
    for(int k=0;k<K;k++){
        if(mask & (1u<<k)) continue;
        QSlot *s=qs(layer,eids[k]);
        if(s->resident){
            int di=home(eids[k]); int c=G.is_cnt[di];
            tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
            G.is_k[di][c]=k; G.is_cnt[di]=c+1;
            mask|=1u<<k;
        }
    }
    pthread_mutex_unlock(&G.mx);

    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x + (size_t)di*8*G.D;               /* je Device eigener x-Block */
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        if(!coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr)){
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        } else {
            G.hits[di] += (uint64_t)c;
        }
    }
    G.direct_cpu += (uint64_t)(K-__builtin_popcount(mask));
    return mask;
}

/* Returns mask of experts that were issued to the GPU but whose results could
 * not be collected (e.g. stream sync error). The caller must recompute those
 * on CPU to preserve correctness (fix #5: no silent expert drops). */
uint32_t qt_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    uint32_t fail=0;
    if(!G.on) return 0;
    if(mask) for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        const float *y=coli_cuda_expert_group_take(G.dev[di]);
        if(!y){ for(int j=0;j<c;j++) fail|=1u<<G.is_k[di][j]; continue; }
        for(int j=0;j<c;j++){
            float w=val[G.is_k[di][j]];
            const float *row=y+(size_t)j*G.D;
            for(int d=0;d<G.D;d++) out[d]+=w*row[d];
        }
        G.is_cnt[di]=0;
    }
    fail |= b70_tier_collect(val,K,out,G.D);
    G.recovery_cpu += (uint64_t)__builtin_popcount(fail);
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
    return fail;
}

static void qt_dev_close(void){
    pthread_mutex_lock(&G.mx);
    G.dev_pending=0;
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}

int qt_issue_dev(QtDevIssue *issue,int layer,const int *eids,const float *weights,
                 int K,int home_device,const float *x_dev,float *x_host){
    if(!issue||!G.on||!eids||!weights||!x_dev||!x_host||K<1||K>32||
       home_device!=G.dev[0]) return 0;
    memset(issue,0,sizeof(*issue));
    ColiCudaTensor *tg[QT_MAX_DEV][32],*tu[QT_MAX_DEV][32],*td[QT_MAX_DEV][32];
    float ww[QT_MAX_DEV][32];
    int route[QT_MAX_DEV][32],cnt[QT_MAX_DEV]={0};
    uint32_t b70_candidates=0,cuda_mask=0;
    pthread_mutex_lock(&G.mx);
    if(G.issue_open||G.dev_pending){ pthread_mutex_unlock(&G.mx); return 0; }
    G.issue_open=1;
    if(layer==0) qt_lfru_tick_locked();
    for(int k=0;k<K;k++){
        QSlot *s=qs(layer,eids[k]);
        if(s->resident){
            int di=home(eids[k]),c=cnt[di]++;
            tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
            ww[di][c]=weights[k]; route[di][c]=k;
            cuda_mask|=1u<<k;
        }else if(b70_tier_is_resident(layer,eids[k])) b70_candidates|=1u<<k;
    }
    pthread_mutex_unlock(&G.mx);

    uint32_t all=K==32?0xffffffffu:((1u<<K)-1u);
    int host_staged=(b70_candidates||(all&~(b70_candidates|cuda_mask)));
    if(host_staged&&!coli_cuda_pipe_download(home_device,x_dev,x_host,
                                              (size_t)G.D*sizeof(float))){
        qt_dev_close();
        return 0;
    }
    uint32_t b70_mask=0;
    if(b70_candidates) b70_tier_record(layer,eids,weights,K,x_host,&b70_mask);
    b70_mask&=b70_candidates;

    G.dev_issue_count=0;
    for(int di=0;di<G.ndev;di++){
        if(!cnt[di]) continue;
        float *slot=G.dev_slots+(size_t)G.dev_issue_count*G.D;
        if(coli_cuda_expert_group_resident_issue(tg[di],tu[di],td[di],ww[di],
                                                  cnt[di],home_device,x_dev,slot)){
            G.dev_issue_devices[G.dev_issue_count++]=G.dev[di];
            G.hits[di]+=(uint64_t)cnt[di];
        }else{
            for(int j=0;j<cnt[di];j++) cuda_mask&=~(1u<<route[di][j]);
        }
    }
    uint32_t cpu_mask=all&~(cuda_mask|b70_mask);
    memcpy(G.dev_weights,weights,(size_t)K*sizeof(float));
    issue->serial=++G.dev_serial;
    issue->cuda_mask=cuda_mask; issue->b70_mask=b70_mask; issue->cpu_mask=cpu_mask;
    issue->home_device=home_device; issue->K=K;
    G.dev_pending=1;
    G.dev_x=x_dev; G.dev_x_host=x_host; G.dev_host_staged=host_staged||cpu_mask;
    if(cpu_mask&&!host_staged&&
       !coli_cuda_pipe_download(home_device,x_dev,x_host,(size_t)G.D*sizeof(float))){
        qt_abort_dev(issue);
        return 0;
    }
    G.b70_hits+=(uint64_t)__builtin_popcount(b70_mask);
    G.direct_cpu+=(uint64_t)__builtin_popcount(cpu_mask);
    return 1;
}

uint32_t qt_take_dev(QtDevIssue *issue,float *out_dev){
    if(!issue||!out_dev||!G.dev_pending||issue->serial!=G.dev_serial) return 0xffffffffu;
    uint32_t fail=issue->cpu_mask;
    if(issue->cuda_mask){
        if(!coli_cuda_expert_group_resident_take(issue->home_device,G.dev_issue_devices,
                G.dev_issue_count,G.dev_slots,G.dev_acc,G.D)||
           !coli_cuda_pipe_add(issue->home_device,out_dev,G.dev_acc,(size_t)G.D))
            fail|=issue->cuda_mask;
    }
    if(issue->b70_mask){
        float *host=G.is_x;
        memset(host,0,(size_t)G.D*sizeof(float));
        uint32_t bf=b70_tier_collect(G.dev_weights,issue->K,host,G.D);
        fail|=bf;
        if((issue->b70_mask&~bf) &&
           (!coli_cuda_pipe_upload(issue->home_device,G.dev_stage,host,
                                   (size_t)G.D*sizeof(float))||
            !coli_cuda_pipe_add(issue->home_device,out_dev,G.dev_stage,(size_t)G.D)))
            fail|=issue->b70_mask;
    }
    if(fail&&!G.dev_host_staged&&
       !coli_cuda_pipe_download(issue->home_device,G.dev_x,G.dev_x_host,
                                (size_t)G.D*sizeof(float)))
        fail|=issue->cuda_mask|issue->b70_mask;
    G.recovery_cpu+=(uint64_t)__builtin_popcount(fail&~issue->cpu_mask);
    qt_dev_close();
    return fail;
}

void qt_abort_dev(QtDevIssue *issue){
    if(!G.on||!G.dev_pending) return;
    if(issue&&issue->serial&&issue->serial!=G.dev_serial) return;
    if(G.dev_issue_count)
        (void)coli_cuda_expert_group_resident_take(issue?issue->home_device:G.dev[0],
                G.dev_issue_devices,G.dev_issue_count,G.dev_slots,G.dev_acc,G.D);
    if(b70_tier_on()){
        memset(G.is_x,0,(size_t)G.D*sizeof(float));
        (void)b70_tier_collect(G.dev_weights,issue?issue->K:G.topk,G.is_x,G.D);
    }
    qt_dev_close();
}

void qt_stats(void){
    if(!G.on) return;
    uint64_t hits=0; size_t res=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++) res += G.slot[i].resident;
    fprintf(stderr,"[qtier] resident %zu/%d Experten | uploads %llu | actual CPU fallback %llu (%llu recovery) | B70 routes %llu | q_skips %llu\n",
            res, G.nl*G.ne, (unsigned long long)G.uploads,
            (unsigned long long)(G.direct_cpu+G.recovery_cpu),
            (unsigned long long)G.recovery_cpu,
            (unsigned long long)G.b70_hits,
            (unsigned long long)G.q_full_skips);
    for(int i=0;i<G.ndev;i++){
        size_t tc=0,tb=0; coli_cuda_stats(G.dev[i],&tc,&tb);
        hits+=G.hits[i];
        fprintf(stderr,"[qtier]   dev %d: hits %llu | %zu Tensoren, %.2f GB VRAM belegt (Budget %.2f GB)\n",
                G.dev[i], (unsigned long long)G.hits[i], tc, tb/1073741824.0, G.budget[i]/1073741824.0);
    }
    double tot=(double)(hits+G.b70_hits+G.direct_cpu);
    fprintf(stderr,
            "[qtier] route share: CUDA %.1f%% | B70 %.1f%% | CPU %.1f%% | LFRU-Swaps %llu\n",
            tot>0?100.0*hits/tot:0.0, tot>0?100.0*G.b70_hits/tot:0.0,
            tot>0?100.0*G.direct_cpu/tot:0.0, (unsigned long long)G.swaps);
    { uint64_t calls=0,ex=0,rows=0; double h2d=0,kms=0,d2h=0;
      coli_cuda_group_stats(&calls,&ex,&rows,&h2d,&kms,&d2h);
      if(calls) fprintf(stderr,"[qtier] group_stats: %llu Calls, %llu Experten | h2d %.0f ms, kernel %.0f ms, d2h %.0f ms\n",
              (unsigned long long)calls,(unsigned long long)ex,h2d,kms,d2h); }
    b70_tier_stats();
}

void qt_shutdown(void){
    if(!G.on) return;
    if(G.dev_pending){
        QtDevIssue pending={G.dev_serial,0,0,0,G.dev[0],G.topk};
        qt_abort_dev(&pending);
    }
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"wb");
        if(f){
            uint32_t hdr[3]={0x51544831u,(uint32_t)G.nl,(uint32_t)G.ne};
            fwrite(hdr,4,3,f);
            for(size_t i=0;i<(size_t)G.nl*G.ne;i++) fwrite(&G.slot[i].heat,4,1,f);
            fclose(f);
            fprintf(stderr,"[qtier] HEAT_FILE gespeichert: %s\n",hf);
        }
    }
    pthread_mutex_lock(&G.mx); G.th_stop=1; pthread_cond_signal(&G.cv); pthread_mutex_unlock(&G.mx);
    pthread_join(G.th,NULL);
    coli_cuda_pipe_free(G.dev[0],G.dev_slots);
    coli_cuda_pipe_free(G.dev[0],G.dev_acc);
    coli_cuda_pipe_free(G.dev[0],G.dev_stage);
    G.on=0;
    coli_cuda_shutdown();
    b70_tier_shutdown();
}
