// Laguna Spark decode bench: drives LagunaBackend (the REAL hybrid/spark
// path used by dflash_server), honoring all DFLASH_* env knobs:
//   DFLASH_LAGUNA_HOTNESS=<csv>     calibrated placement
//   DFLASH_EXPERT_BUDGET_PCT=60     pinned-hot fraction
//   DFLASH_LAGUNA_CACHE_SLOTS=16    cache ring slots/layer
//   DFLASH_LAGUNA_PROFILE=1         cold-experts/token profiling
//   DFLASH_LAGUNA_NO_SINGLE_GRAPH=1 per-layer fallback (for trace capture)
//   DFLASH_LAGUNA_PREGATE_TRACE=<f> pregate trace capture (fallback path)
//
// Usage: bench_laguna_spark <laguna.gguf> [prompt_N=128] [n_gen=256]

#include "laguna_backend.h"
#include "dflash27b.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <laguna.gguf> [prompt_N=128] [n_gen=256] [repetitions=1]\n",
                     argv[0]);
        return 2;
    }
    const int prompt_N    = std::max(1, (argc >= 3) ? std::atoi(argv[2]) : 128);
    const int n_gen       = std::max(1, (argc >= 4) ? std::atoi(argv[3]) : 256);
    const int repetitions = std::max(1, (argc >= 5) ? std::atoi(argv[4]) : 1);

    LagunaBackendArgs args;
    args.target_path = argv[1];
    args.max_ctx     = prompt_N + n_gen + 64;

    LagunaBackend be(args);
    if (!be.init()) {
        std::fprintf(stderr, "backend init failed\n");
        return 1;
    }
    be.print_ready_banner();

    // BOS + fake tokens (same seeding as bench_laguna_generate so the
    // routing trajectory is comparable across configs). DFLASH_BENCH_MIX=1
    // uses a deterministic varied prompt instead (non-degenerate continuation,
    // for exactness comparisons between decode paths).
    GenerateRequest req;
    req.prompt.resize((size_t)prompt_N, 1972);
    req.prompt[0] = 2;  // laguna bos
    if (std::getenv("DFLASH_BENCH_MIX")) {
        int64_t seed = 1;
        if (const char * s = std::getenv("DFLASH_BENCH_SEED")) seed = std::atoll(s);
        for (int i = 1; i < prompt_N; ++i)
            req.prompt[(size_t)i] = 1000 + (int32_t)((((int64_t)i + seed * 7919) * 2654435761LL) % 50000);
    }
    req.n_gen = n_gen;
    req.stream = false;

    DaemonIO io{};
    std::vector<double> prefill_tps;
    std::vector<double> decode_tps;
    prefill_tps.reserve((size_t)repetitions);
    decode_tps.reserve((size_t)repetitions);
    uint64_t reference_hash = 0;
    bool exact_across_runs = true;

    for (int run = 0; run < repetitions; ++run) {
        GenerateResult r = be.generate(req, io);
        if (!r.ok()) {
            std::fprintf(stderr, "generate run %d failed: %s (%s)\n", run + 1,
                         r.error_code().data(), r.error_detail().data());
            return 1;
        }
        const int nd = (int)r.tokens.size();
        const double pf_tps = prompt_N / std::max(1e-9, r.prefill_s);
        const double dec_tps = nd / std::max(1e-9, r.decode_s);
        prefill_tps.push_back(pf_tps);
        decode_tps.push_back(dec_tps);
        std::printf("[spark-bench] run=%d prefill N=%d in %.3fs (%.1f tok/s)\n",
                    run + 1, prompt_N, r.prefill_s, pf_tps);
        std::printf("[spark-bench] run=%d decoded %d tokens in %.3fs (%.1f tok/s)\n",
                    run + 1, nd, r.decode_s, dec_tps);
        std::printf("[spark-bench] run=%d first ids:", run + 1);
        for (int i = 0; i < nd && i < 16; ++i) std::printf(" %d", r.tokens[(size_t)i]);
        std::printf("\n");

        // FNV-1a over the full generated sequence: exactness fingerprint.
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < nd; ++i) {
            uint32_t v = (uint32_t)r.tokens[(size_t)i];
            for (int b = 0; b < 4; ++b) { h ^= (v >> (8*b)) & 0xff; h *= 1099511628211ULL; }
        }
        if (run == 0) reference_hash = h;
        exact_across_runs = exact_across_runs && h == reference_hash;
        std::printf("[spark-bench] run=%d ids_hash=%016llx n=%d\n",
                    run + 1, (unsigned long long)h, nd);
    }

    if (repetitions >= 3) {
        double pf_sum = 0.0;
        double dec_sum = 0.0;
        double dec_min = decode_tps[2];
        for (int run = 2; run < repetitions; ++run) {
            pf_sum += prefill_tps[(size_t)run];
            dec_sum += decode_tps[(size_t)run];
            dec_min = std::min(dec_min, decode_tps[(size_t)run]);
        }
        const int steady_runs = repetitions - 2;
        std::printf("[spark-bench] summary first_prefill=%.1f first_decode=%.1f "
                    "steady_prefill=%.1f steady_decode=%.1f steady_decode_min=%.1f "
                    "steady_runs=%d warmup_excluded=2 exact=%s\n",
                    prefill_tps[0], decode_tps[0],
                    pf_sum / steady_runs, dec_sum / steady_runs, dec_min,
                    steady_runs, exact_across_runs ? "YES" : "NO");
    } else {
        std::printf("[spark-bench] summary first_prefill=%.1f first_decode=%.1f "
                    "steady_runs=0 warmup_excluded=0 exact=%s\n",
                    prefill_tps[0], decode_tps[0],
                    exact_across_runs ? "YES" : "NO");
    }
    return exact_across_runs ? 0 : 1;
}
