/* qwen36_tier.h — M-QTIER (R2): VRAM-Experten-Tier für qwen36 auf Basis des
 * Upstream-CUDA-Backends (backend_cuda.cu / coli_cuda_*).
 *
 * Konzept (Colibrì „route → union → place → overlap → learn", RAM→VRAM-Stufe):
 *  - Jeder Experte hat ein Heimat-Device (eid % n_gpus), keine Duplikate (E1/E3).
 *  - Routing-Hitze (Heat) entscheidet, wer VRAM verdient; M2 = Warmup-Füllung
 *    bis Budget, M3 ergänzt LFRU-Swaps + Prefetch.
 *  - Uploads laufen in einem Hintergrund-Thread über Staging-Kopien; das
 *    Decodieren blockiert nie auf einen Upload (E2).
 *  - VRAM-Miss ⇒ Aufrufer rechnet den Experten auf der CPU, überlappend mit
 *    den GPU-Gruppen (qt_issue … CPU … qt_take).
 *
 * Aktivierung: COLI_CUDA=1 [COLI_GPUS=0,1] [CUDA_EXPERT_GB=<G>|auto]
 */
#ifndef QWEN36_TIER_H
#define QWEN36_TIER_H
#include <stdint.h>
#include <stddef.h>

/* One complete device-input expert transaction. Masks are disjoint and cover
 * every selected route. `serial` prevents a stale take/abort from consuming a
 * later issue. */
typedef struct {
    uint64_t serial;
    uint32_t cuda_mask, b70_mask, cpu_mask;
    int home_device, K;
} QtDevIssue;

/* The Qwen engine calls this interface unconditionally. CUDA builds link
 * qwen36_tier.c; CPU-only builds use these zero-cost stubs so the documented
 * `make qwen36` path remains dependency-free. */
#ifdef COLI_CUDA
/* Init nach Modell-Load. Gibt 1 zurück, wenn der Tier aktiv ist.
 * cap_experts_per_layer muss == n_experts sein (volle RAM-Residenz, Z5);
 * sonst bleibt der Tier aus (Upload-Zeiger könnten sonst evicted werden). */
int  qt_init(int n_layers, int n_experts, int hidden, int inter,
             int cap_experts_per_layer, int topk, int expert_gs);
int  qt_ready(void);
int  qt_is_resident(int layer, int eid);
void qt_shutdown(void);

/* Pro geroutetem Experten einmal je Token aufrufen (Zeiger auf die RAM-Slots,
 * int4-packed + per-row-Scales). Aktualisiert Heat und stößt ggf. einen
 * Hintergrund-Upload an (nicht blockierend). */
void qt_note(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);

/* Startet die GPU-Gruppen für die residenten der K Experten (asynchron, beide
 * Devices parallel). Rückgabe: Bitmaske der k, die die GPU übernimmt.
 * Danach: Misses auf der CPU rechnen, dann qt_take() mit derselben Maske. */
uint32_t qt_issue(int layer, const int *eids, const float *weights,
                  int K, const float *x);

/* Sammelt die GPU-Ergebnisse ein und akkumuliert val[k]*y_k in out[hidden].
 * Rückgabe: Bitmaske der k, die zwar an die GPU geschickt wurden, aber nicht
 * eingesammelt werden konnten (z.B. Stream-Sync-Fehler). Der Aufrufer muss
 * diese Experten auf der CPU nachrechnen. */
uint32_t qt_take(uint32_t mask, const float *val, int K, float *out);

/* Additive state-owner path. CUDA-resident experts consume x_dev directly.
 * B70/CPU routes cause one exact activation row to be staged into x_host.
 * take adds the CUDA and B70 weighted partials into out_dev and returns only
 * routes that the caller must recompute on CPU. */
int qt_issue_dev(QtDevIssue *issue,int layer,const int *eids,
                 const float *weights,int K,int home_device,
                 const float *x_dev,float *x_host);
uint32_t qt_take_dev(QtDevIssue *issue,float *out_dev);
void qt_abort_dev(QtDevIssue *issue);

/* M3: Warmstart — nächster vorzuladender Experte (Heat-Reihenfolge, bis
 * Budgets voll). Engine lädt den RAM-Slot und ruft qt_note_block. */
int  qt_fill_next(int *layer, int *eid);
/* M3b: komplettes Warmstart-Set planen (Heat-Reihenfolge, Budget reserviert);
 * danach dürfen MEHRERE Threads laden und qt_note_planned aufrufen. */
int  qt_plan_fill(int *layers, int *eids, int max);
void qt_note_planned(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);
void qt_note_block(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);
void qt_fill_wait(void);   /* wartet, bis die Upload-Queue leer ist */

/* Telemetriezeile (stderr): Residenz, Hits/Misses, Uploads je Device. */
void qt_stats(void);
#else
static inline int qt_init(int nl, int ne, int d, int ih, int cap, int topk, int gs)
{ (void)nl; (void)ne; (void)d; (void)ih; (void)cap; (void)topk; (void)gs; return 0; }
static inline int qt_ready(void) { return 0; }
static inline int qt_is_resident(int layer, int eid)
{ (void)layer; (void)eid; return 0; }
static inline void qt_shutdown(void) {}
static inline void qt_note(int layer, int eid, const uint8_t *g4, const uint8_t *u4,
                           const uint8_t *d4, const float *gs, const float *us, const float *ds)
{ (void)layer; (void)eid; (void)g4; (void)u4; (void)d4; (void)gs; (void)us; (void)ds; }
static inline uint32_t qt_issue(int layer, const int *eids, const float *weights,
                                int k, const float *x)
{ (void)layer; (void)eids; (void)weights; (void)k; (void)x; return 0; }
static inline uint32_t qt_take(uint32_t mask, const float *val, int k, float *out)
{ (void)mask; (void)val; (void)k; (void)out; return 0; }
static inline int qt_issue_dev(QtDevIssue *issue,int layer,const int *eids,
                               const float *weights,int k,int home_device,
                               const float *x_dev,float *x_host)
{ (void)issue; (void)layer; (void)eids; (void)weights; (void)k; (void)home_device;
  (void)x_dev; (void)x_host; return 0; }
static inline uint32_t qt_take_dev(QtDevIssue *issue,float *out_dev)
{ (void)out_dev; return issue ? issue->cpu_mask : 0; }
static inline void qt_abort_dev(QtDevIssue *issue) { (void)issue; }
static inline int qt_fill_next(int *layer, int *eid)
{ (void)layer; (void)eid; return 0; }
static inline int qt_plan_fill(int *layers, int *eids, int max)
{ (void)layers; (void)eids; (void)max; return 0; }
static inline void qt_note_planned(int layer, int eid, const uint8_t *g4,
                                   const uint8_t *u4, const uint8_t *d4,
                                   const float *gs, const float *us, const float *ds)
{ (void)layer; (void)eid; (void)g4; (void)u4; (void)d4; (void)gs; (void)us; (void)ds; }
static inline void qt_note_block(int layer, int eid, const uint8_t *g4,
                                 const uint8_t *u4, const uint8_t *d4,
                                 const float *gs, const float *us, const float *ds)
{ (void)layer; (void)eid; (void)g4; (void)u4; (void)d4; (void)gs; (void)us; (void)ds; }
static inline void qt_fill_wait(void) {}
static inline void qt_stats(void) {}
#endif

#endif
