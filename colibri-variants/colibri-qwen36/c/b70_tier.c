/* b70_tier.c — B70 expert tier: converts Colibri signed-S4 weights to the
 * N-major INT4 layout consumed by llm-scaler's decode kernels. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>
#include "b70_tier.h"

/* INT4 SYCL library function pointers */
typedef int  (*fn_init)(int,int,int,int,int);
typedef int  (*fn_upload)(int,const uint32_t*,const uint16_t*,
                         const uint32_t*,const uint16_t*);
typedef int  (*fn_issue)(const float*,const int*,const float*,int);
typedef int  (*fn_take)(float*);
typedef void (*fn_shutdown)(void);

static struct {
    int on, initialized, failed;
    int nl, ne, D, Ih, topk, egs;
    int capacity, claimed;
    int *slot_of;              /* [nl*ne]: compact B70 slot, -1 = not owned */
    uint8_t *ready;            /* [nl*ne]: weights uploaded successfully */
    void *lib;
    fn_init     b_init;
    fn_upload   b_upload;
    fn_issue    b_issue;
    fn_take     b_take;
    fn_shutdown b_shutdown;
    /* Per-call state (set in record, used in collect) */
    float *saved_x;            /* saved activation [D] */
    int saved_eids[32];        /* B70-local expert IDs */
    int saved_k[32];           /* position in K */
    int saved_n;               /* count */
} B;

static uint16_t fp16_bits(float value) {
    _Float16 half = (_Float16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static uint32_t marlin_shuffle(uint32_t packed) {
    static const uint8_t source_position[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    uint32_t shuffled = 0;
    for (int output_position = 0; output_position < 8; output_position++) {
        uint32_t nibble =
            (packed >> (source_position[output_position] * 4)) & 0x0f;
        shuffled |= nibble << (output_position * 4);
    }
    return shuffled;
}

int b70_convert_expert_s4(const uint8_t *gate, const uint8_t *up,
                          const uint8_t *down, const float *gate_scales,
                          const float *up_scales, const float *down_scales,
                          int hidden, int intermediate, int group_size,
                          uint32_t *gate_up_out, uint16_t *gate_up_scales_out,
                          uint32_t *down_out, uint16_t *down_scales_out) {
    if (!gate || !up || !down || !gate_scales || !up_scales || !down_scales ||
        !gate_up_out || !gate_up_scales_out || !down_out || !down_scales_out ||
        hidden <= 0 || intermediate <= 0 || group_size <= 0 ||
        (hidden & 1) || (intermediate & 1) ||
        hidden % group_size || intermediate % group_size)
        return 0;

    int gate_packed = hidden / 8;
    int down_packed = intermediate / 8;
    int gate_groups = hidden / group_size;
    int down_groups = intermediate / group_size;
    int gate_up_rows = 2 * intermediate;

    /* IPEX K-major layout used by llm-scaler's SLM ESIMD decode kernels.
     * Colibri input is N-major signed S4. XOR converts each nibble to q+8;
     * marlin_shuffle matches IPEX's physical nibble order. */
    for (int row = 0; row < intermediate; row++) {
        for (int packed_k = 0; packed_k < gate_packed; packed_k++) {
            uint32_t gate_word, up_word;
            memcpy(&gate_word,
                   gate + (size_t)row * (hidden / 2) + packed_k * 4, 4);
            memcpy(&up_word,
                   up + (size_t)row * (hidden / 2) + packed_k * 4, 4);
            gate_up_out[(size_t)packed_k * gate_up_rows + row] =
                marlin_shuffle(gate_word ^ 0x88888888u);
            gate_up_out[(size_t)packed_k * gate_up_rows + intermediate + row] =
                marlin_shuffle(up_word ^ 0x88888888u);
        }
        for (int group = 0; group < gate_groups; group++) {
            size_t source = (size_t)row * gate_groups + group;
            gate_up_scales_out[(size_t)group * gate_up_rows + row] =
                fp16_bits(gate_scales[source]);
            gate_up_scales_out[
                (size_t)group * gate_up_rows + intermediate + row] =
                fp16_bits(up_scales[source]);
        }
    }
    for (int row = 0; row < hidden; row++) {
        for (int packed_k = 0; packed_k < down_packed; packed_k++) {
            uint32_t word;
            memcpy(&word,
                   down + (size_t)row * (intermediate / 2) + packed_k * 4, 4);
            down_out[(size_t)packed_k * hidden + row] =
                marlin_shuffle(word ^ 0x88888888u);
        }
        for (int group = 0; group < down_groups; group++) {
            size_t source = (size_t)row * down_groups + group;
            down_scales_out[(size_t)group * hidden + row] =
                fp16_bits(down_scales[source]);
        }
    }
    return 1;
}

int b70_tier_on(void) { return B.on; }
int b70_tier_capacity(void) { return B.on ? B.capacity : 0; }

int b70_tier_init(int nl, int ne, int D, int Ih, int topk, int egs) {
    const char *e = getenv("COLI_B70");
    if (!e || *e != '1') return 0;
    if (egs <= 0 || D <= 0 || Ih <= 0 || D % egs || Ih % egs) {
        fprintf(stderr,
                "[b70_tier] unsupported dimensions: D=%d Ih=%d group_size=%d\n",
                D, Ih, egs);
        return 0;
    }

    /* Capacity only. The tier planner assigns the cold complement after
     * reserving an explicit hot CUDA ownership quota. */
    const char *bs = getenv("B70_EXPERTS_PER_LAYER");
    int per_layer = bs ? atoi(bs) : (ne * 3 / 10);
    if (per_layer < 1 || per_layer > ne) per_layer = ne / 3;
    B.capacity = nl * per_layer;

    B.nl = nl; B.ne = ne; B.D = D; B.Ih = Ih; B.topk = topk; B.egs = egs;
    B.on = 1;

    /* dlopen the SYCL library */
    B.lib = dlopen("libb70_moe.so", RTLD_NOW | RTLD_GLOBAL);
    if (!B.lib) {
        /* Try relative path */
        B.lib = dlopen("./libb70_moe.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!B.lib) {
        fprintf(stderr, "[b70_tier] dlopen failed: %s\n", dlerror());
        B.on = 0; return 0;
    }
    B.b_init     = (fn_init)     dlsym(B.lib, "b70_moe_init");
    B.b_upload   = (fn_upload)   dlsym(B.lib, "b70_moe_upload");
    B.b_issue    = (fn_issue)    dlsym(B.lib, "b70_moe_issue");
    B.b_take     = (fn_take)     dlsym(B.lib, "b70_moe_take");
    B.b_shutdown = (fn_shutdown) dlsym(B.lib, "b70_moe_shutdown");
    if (!B.b_init || !B.b_upload || !B.b_issue || !B.b_take ||
        !B.b_shutdown) {
        fprintf(stderr, "[b70_tier] missing symbols in libb70_moe.so\n");
        B.on = 0;
        b70_tier_shutdown();
        return 0;
    }

    size_t entries = (size_t)nl * ne;
    B.slot_of = malloc(entries * sizeof(int));
    B.ready = calloc(entries, 1);
    if (!B.slot_of || !B.ready) {
        fprintf(stderr, "[b70_tier] placement-map allocation failed\n");
        B.on = 0;
        b70_tier_shutdown();
        return 0;
    }
    for (size_t index = 0; index < entries; index++)
        B.slot_of[index] = -1;

    if (B.b_init(B.capacity, D, Ih, topk, egs) != 0) {
        fprintf(stderr, "[b70_tier] B70 init failed\n");
        B.on = 0;
        b70_tier_shutdown();
        return 0;
    }
    B.initialized = 1;

    B.saved_x = malloc((size_t)D * sizeof(float));
    if (!B.saved_x) {
        fprintf(stderr, "[b70_tier] activation-buffer allocation failed\n");
        B.on = 0;
        b70_tier_shutdown();
        return 0;
    }
    fprintf(stderr, "[b70_tier] active: capacity %d compact B70 slots\n",
            B.capacity);
    return 1;
}

int b70_tier_claim(int layer, int eid) {
    if (!B.on || layer < 0 || layer >= B.nl || eid < 0 || eid >= B.ne)
        return 0;
    size_t index = (size_t)layer * B.ne + eid;
    if (B.slot_of[index] >= 0) return 1;
    if (B.claimed >= B.capacity) return 0;
    B.slot_of[index] = B.claimed++;
    return 1;
}

void b70_tier_note(int layer, int eid,
                   const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
                   const float *gs, const float *us, const float *ds) {
    if (!B.on) return;
    if (B.slot_of[(size_t)layer * B.ne + eid] < 0) return;
    if (B.ready[layer * B.ne + eid]) return;

    size_t matrix_bytes = (size_t)B.D * B.Ih / 2;
    int gate_groups = B.D / B.egs;
    int down_groups = B.Ih / B.egs;
    uint32_t *gate_up = malloc(2 * matrix_bytes);
    uint32_t *down = malloc(matrix_bytes);
    uint16_t *gate_up_scales =
        malloc((size_t)2 * B.Ih * gate_groups * sizeof(uint16_t));
    uint16_t *down_scales =
        malloc((size_t)B.D * down_groups * sizeof(uint16_t));

    int converted = gate_up && down && gate_up_scales && down_scales &&
        b70_convert_expert_s4(g4, u4, d4, gs, us, ds, B.D, B.Ih, B.egs,
                              gate_up, gate_up_scales, down, down_scales);
    int b70_id = B.slot_of[(size_t)layer * B.ne + eid];
    int uploaded = converted &&
        B.b_upload(b70_id, gate_up, gate_up_scales, down, down_scales) == 0;
    free(gate_up);
    free(down);
    free(gate_up_scales);
    free(down_scales);
    if (uploaded)
        B.ready[layer * B.ne + eid] = 1;
    if (!uploaded) {
        fprintf(stderr, "[b70_tier] INT4 upload failed for layer=%d expert=%d\n",
                layer, eid);
        B.failed = 1;
        B.on = 0;
    }
}

int b70_tier_is_resident(int layer, int eid) {
    if (!B.on || layer < 0 || layer >= B.nl || eid < 0 || eid >= B.ne)
        return 0;
    return B.slot_of[(size_t)layer * B.ne + eid] >= 0;
}

void b70_tier_record(int layer, const int *eids, const float *weights,
                     int K, const float *x, uint32_t *mask) {
    if (!B.on) { B.saved_n = 0; return; }
    B.saved_n = 0;
    memcpy(B.saved_x, x, B.D * sizeof(float));
    float selected_weights[32];
    for (int k = 0; k < K; k++) {
        if (b70_tier_is_resident(layer, eids[k]) &&
            B.ready[layer * B.ne + eids[k]]) {
            B.saved_eids[B.saved_n] =
                B.slot_of[(size_t)layer * B.ne + eids[k]];
            B.saved_k[B.saved_n] = k;
            selected_weights[B.saved_n] = weights[k];
            B.saved_n++;
        }
    }
    if (B.saved_n &&
        B.b_issue(B.saved_x, B.saved_eids, selected_weights, B.saved_n) == 0) {
        for (int j = 0; j < B.saved_n; j++)
            *mask |= 1u << B.saved_k[j];
    } else if (B.saved_n) {
        fprintf(stderr, "[b70_tier] issue failed; disabling B70 tier\n");
        B.failed = 1;
        B.on = 0;
        B.saved_n = 0;
    }
}

uint32_t b70_tier_collect(const float *val, int K, float *out, int D) {
    (void)val;
    (void)K;
    if (!B.on || B.saved_n == 0) return 0;
    uint32_t fail = 0;
    float *partial = malloc((size_t)D * sizeof(float));
    if (!partial || B.b_take(partial) != 0) {
        for (int j = 0; j < B.saved_n; j++)
            fail |= 1u << B.saved_k[j];
        fprintf(stderr, "[b70_tier] dispatch failed; disabling B70 tier\n");
        B.failed = 1;
        B.on = 0;
    } else {
        for (int d = 0; d < D; d++)
            out[d] += partial[d];
    }
    free(partial);
    B.saved_n = 0;
    return fail;
}

void b70_tier_stats(void) {
    if (!B.on) return;
    int total = B.nl * B.ne;
    fprintf(stderr,
            "[b70_tier] %d/%d compact slots claimed (%.1f%% of %d experts), %d dispatched last call\n",
            B.claimed, B.capacity, total ? 100.0 * B.claimed / total : 0.0,
            total, B.saved_n);
}

void b70_tier_shutdown(void) {
    if (B.initialized && B.b_shutdown) B.b_shutdown();
    if (B.lib) dlclose(B.lib);
    free(B.slot_of);
    free(B.ready);
    free(B.saved_x);
    memset(&B, 0, sizeof(B));
}
