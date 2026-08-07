#include "cpu_expert_capi.h"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <functional>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SB_SPIN_HINT() _mm_pause()
#else
#define SB_SPIN_HINT() ((void)0)
#endif

namespace {

constexpr size_t kHugePage = 2u << 20;  // 2 MiB
constexpr size_t kAlign = 64;           // cache line

// bf16 occupies the high 16 bits of the fp32 it rounds to, so widening is a
// shift. Written through memcpy because type punning through a pointer cast
// is UB; every compiler folds this to a single move.
inline float bf16_to_f32(uint16_t v) noexcept {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// NVFP4 element decode: 1 sign bit + 3 magnitude bits per weight.
//
// Magnitudes 2..7 lie on a regular ladder -- {1, 1.5, 2, 3, 4, 6} is exactly
// what `0x3F000000 + (mag << 22)` produces when bitcast to fp32 -- so the
// common case is one shift, one add and a bitcast, with no lookup table to
// pull through cache. E2M1 has no subnormal exponent field, so magnitudes 0
// and 1 fall off that ladder and are patched to 0.0 and 0.5.
//
// Written branch-light on purpose: this runs once per weight, and the whole
// point of storing NVFP4 rather than bf16 is that the tier becomes
// compute-sensitive once it stops being starved for bandwidth.
inline float e2m1_to_f32(uint32_t nib) noexcept {
  const uint32_t mag = nib & 0x7u;
  uint32_t bits = 0x3F000000u + (mag << 22);
  if (mag < 2u) bits = (mag == 0u) ? 0u : 0x3F000000u;
  bits |= (nib & 0x8u) << 28;  // sign bit 3 -> bit 31
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Block scales are stored as raw float8_e4m3fn bytes, matching what the
// checkpoint carries. Decoded through a 256-entry table built once: the
// scale is read once per 16 weights, so the lookup is 1/16th the traffic of
// the element decode and costs nothing measurable.
//
// e4m3fn has no infinities; exponent 15 with mantissa 7 is NaN. That is left
// as NaN rather than clamped, so a corrupt scale surfaces loudly instead of
// quietly biasing an expert.
inline float e4m3_to_f32(uint8_t v) noexcept {
  const uint32_t sign = static_cast<uint32_t>(v & 0x80u) << 24;
  const uint32_t exp = (v >> 3) & 0x0Fu;
  const uint32_t man = v & 0x07u;
  uint32_t bits;
  if (exp == 0u) {
    if (man == 0u) {
      bits = sign;  // +/- 0
    } else {
      // Subnormal: 2^-6 * (man / 8).
      const float f = std::ldexp(static_cast<float>(man), -9);
      std::memcpy(&bits, &f, sizeof(bits));
      bits |= sign;
    }
  } else if (exp == 0x0Fu && man == 0x07u) {
    bits = 0x7FC00000u | sign;  // NaN
  } else {
    bits = sign | ((exp + 120u) << 23) | (man << 20);  // bias 7 -> 127
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

inline float silu(float x) noexcept { return x / (1.0f + std::exp(-x)); }

// ---------------------------------------------------------------------------
// Hugepage arena
// ---------------------------------------------------------------------------

// Expert weights are read start-to-finish on every activation, so the access
// pattern is pure streaming over a large region -- exactly the case where 4 KiB
// pages waste TLB entries. MADV_HUGEPAGE asks for 2 MiB mappings instead.
//
// The mapping is reserved up front but only backed on first touch, so sizing
// generously costs address space rather than resident memory.
class HugeArena {
 public:
  explicit HugeArena(size_t bytes) {
    size_ = (bytes + kHugePage - 1) & ~(kHugePage - 1);
    base_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
      base_ = nullptr;
      size_ = 0;
      return;
    }
    // Advisory: a kernel built without THP simply ignores it.
    madvise(base_, size_, MADV_HUGEPAGE);
  }

  ~HugeArena() {
    if (base_) munmap(base_, size_);
  }

  HugeArena(const HugeArena&) = delete;
  HugeArena& operator=(const HugeArena&) = delete;

  bool valid() const noexcept { return base_ != nullptr; }

  void* alloc(size_t bytes) noexcept {
    const size_t start = (used_ + kAlign - 1) & ~(kAlign - 1);
    if (start + bytes > size_) return nullptr;
    used_ = start + bytes;
    return static_cast<char*>(base_) + start;
  }

  size_t used() const noexcept { return used_; }

  // Borrowed; valid for the arena's lifetime. The prefill path hands this
  // to cudaHostRegister so the DMA engine can read experts in place.
  const void* base() const noexcept { return base_; }
  size_t capacity() const noexcept { return size_; }

 private:
  void* base_ = nullptr;
  size_t size_ = 0;
  size_t used_ = 0;
};

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

// Splits a range across persistent workers. Workers spin briefly before
// parking on a condition variable: back-to-back dispatches (the steady state)
// hit the spin and avoid the ~10-30us futex wake, while an idle host does not
// burn cores that vLLM's scheduler needs.
class ThreadPool {
 public:
  explicit ThreadPool(size_t n) : nthreads_(n < 1 ? 1 : n) {
    // Worker 0 is the calling thread, so only n-1 are spawned.
    threads_.reserve(nthreads_ - 1);
    for (size_t i = 1; i < nthreads_; ++i) {
      threads_.emplace_back([this, i] { worker(i); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      running_ = false;
    }
    cv_.notify_all();
    for (std::thread& t : threads_) {
      if (t.joinable()) t.join();
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  size_t size() const noexcept { return nthreads_; }

  // Runs fn(begin, end) over a partition of [0, total). The caller executes
  // chunk 0 itself, so a single-threaded pool never touches the mutex.
  void parallel_for(size_t total, const std::function<void(size_t, size_t)>& fn) {
    if (total == 0) return;
    if (nthreads_ == 1 || total == 1) {
      fn(0, total);
      return;
    }

    task_ = &fn;
    total_.store(total, std::memory_order_relaxed);
    pending_.store(nthreads_ - 1, std::memory_order_release);
    {
      std::lock_guard<std::mutex> guard(mutex_);
      ++generation_;
    }
    cv_.notify_all();

    run_chunk(0, total);

    // Wait for the spawned workers.
    while (pending_.load(std::memory_order_acquire) != 0) SB_SPIN_HINT();
    task_ = nullptr;
  }

 private:
  void run_chunk(size_t id, size_t total) {
    const size_t per = (total + nthreads_ - 1) / nthreads_;
    const size_t begin = std::min(id * per, total);
    const size_t end = std::min(begin + per, total);
    if (begin < end) (*task_)(begin, end);
  }

  void worker(size_t id) {
    uint64_t seen = 0;
    for (;;) {
      // Spin first: a burst of parallel_for calls (one per projection, three
      // per expert) arrives back-to-back, and parking between them would cost
      // more than the work itself.
      bool got = false;
      for (int i = 0; i < 4096; ++i) {
        if (generation_.load(std::memory_order_acquire) != seen) {
          got = true;
          break;
        }
        SB_SPIN_HINT();
      }
      if (!got) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, seen] {
          return !running_ || generation_ != seen;
        });
        if (!running_ && generation_ == seen) return;
      }
      if (!running_.load(std::memory_order_acquire) &&
          generation_.load(std::memory_order_acquire) == seen) {
        return;
      }
      seen = generation_.load(std::memory_order_acquire);
      run_chunk(id, total_.load(std::memory_order_relaxed));
      pending_.fetch_sub(1, std::memory_order_release);
    }
  }

  const size_t nthreads_;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> running_{true};
  std::atomic<uint64_t> generation_{0};
  std::atomic<size_t> total_{0};
  std::atomic<size_t> pending_{0};
  const std::function<void(size_t, size_t)>* task_ = nullptr;
};

// ---------------------------------------------------------------------------
// Projection kernels
// ---------------------------------------------------------------------------
//
// out[m][n] = sum_k in[m][k] * W[n][k]
//
// W is [N, K] row-major (PyTorch nn.Linear), so each output n walks one
// contiguous weight row -- the layout that makes the M=1 decode case a clean
// sequential stream. [n_begin, n_end) is this thread's slice of the output.
//
// Weights stay packed NVFP4, exactly as the checkpoint carries them: two
// 4-bit elements per byte (low nibble first) plus one float8_e4m3fn scale per
// 16 elements, times one global scale per plane. So
//
//     W[n][k] = e2m1(nibble) * e4m3(blockscale) * gscale
//
// which is the expression vLLM's dequantize_to_dtype evaluates. That
// equivalence is the entire correctness argument and is deliberately not
// re-derived: the reference quantiser in the same vLLM module folds the
// global scale in the opposite direction, and matching the wrong one yields
// plausible weights rather than an error.
//
// Packed costs arithmetic and saves 3.56x the DRAM (0.5625 vs 2 bytes per
// weight). For a tier whose whole cost model is bytes-read that is why it can
// hold a 60 GiB expert bank at all -- dequantized, the same bank would need
// 216 GiB.

// Decoding is hoisted out of the token loop. With bf16 the "decode" was a
// single shift, so repeating it per token cost nothing and the natural
// token-outer order was fine. An NVFP4 element costs ~8 integer ops, so
// decoding once per (row, token) rather than once per row multiplies this
// tier's real work by M -- measured, that alone made packed storage 2x
// slower than bf16 at M=128 despite reading 3.56x fewer bytes.
//
// So: row outer, decode once into scratch, then every token dots against it.

// One decoded weight row, in floats. 4096 covers hidden=3072 (the 122B
// shape) with room to spare, and 16 KiB of stack stays in L1 next to the
// activation. Anything larger falls back to the heap rather than smashing
// the stack.
constexpr size_t kRowScratch = 4096;

inline void decode_row(const uint8_t* __restrict q,
                       const uint8_t* __restrict s, float gscale,
                       const float* __restrict sf_tab, float* __restrict w,
                       size_t kblocks) noexcept {
  for (size_t b = 0; b < kblocks; ++b) {
    // The 16 weights of a block share one scale, so it lifts out of the
    // element loop, leaving 16 independent decodes.
    const float bs = sf_tab[s[b]] * gscale;
    const uint8_t* qb = q + b * 8;
    float* wb = w + b * 16;
#if defined(__AVX2__)
    // The whole E2M1 magnitude ladder doubled -- {0,1,2,3,4,6,8,12} and its
    // negatives -- fits in int8, so a single vpshufb decodes all 16 nibbles
    // of the block against a 16-byte table, sign included. Halving is folded
    // into the scale, so nothing has to undo the doubling.
    //
    // This matters more than it looks: scalar, the decode costs ~8 integer
    // ops per weight and leaves the tier 20x above its DRAM floor, which is
    // the opposite of what packing was for.
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12,
                                      0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i bytes = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(qb));
    const __m128i lo = _mm_and_si128(bytes, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    // Low nibble is the even element, high the odd one; unpacklo restores
    // that order. Swapping them transposes weight pairs undetectably.
    const __m128i codes = _mm_unpacklo_epi8(_mm_shuffle_epi8(lut, lo),
                                            _mm_shuffle_epi8(lut, hi));
    const __m256 scale = _mm256_set1_ps(bs * 0.5f);
    _mm256_storeu_ps(wb, _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)), scale));
    _mm256_storeu_ps(wb + 8, _mm256_mul_ps(
        _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(codes, 8))),
        scale));
#else
    for (size_t j = 0; j < 8; ++j) {
      const uint8_t byte = qb[j];
      wb[2 * j] = e2m1_to_f32(byte & 0x0Fu) * bs;
      wb[2 * j + 1] = e2m1_to_f32(byte >> 4) * bs;
    }
#endif
  }
}

class RowScratch {
 public:
  explicit RowScratch(size_t K) : ptr_(stack_) {
    if (K > kRowScratch) {
      heap_.resize(K);
      ptr_ = heap_.data();
    }
  }
  float* get() noexcept { return ptr_; }

 private:
  float stack_[kRowScratch];
  std::vector<float> heap_;
  float* ptr_;
};

void matmul_nvfp4_bf16in(const uint8_t* __restrict Wq,
                         const uint8_t* __restrict Wsf, float gscale,
                         const float* __restrict sf_tab,
                         const uint16_t* __restrict in, float* __restrict out,
                         size_t M, size_t N, size_t K, size_t n_begin,
                         size_t n_end) {
  const size_t knib = K / 2;
  const size_t kblocks = K / 16;
  RowScratch scratch(K);
  float* __restrict w = scratch.get();

  for (size_t n = n_begin; n < n_end; ++n) {
    decode_row(Wq + n * knib, Wsf + n * kblocks, gscale, sf_tab, w, kblocks);
    for (size_t m = 0; m < M; ++m) {
      const uint16_t* xrow = in + m * K;
      // Four accumulators: breaks the loop-carried dependency so the FMA
      // pipeline stays fed instead of stalling on latency.
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      size_t k = 0;
      for (; k + 4 <= K; k += 4) {
        a0 += bf16_to_f32(xrow[k + 0]) * w[k + 0];
        a1 += bf16_to_f32(xrow[k + 1]) * w[k + 1];
        a2 += bf16_to_f32(xrow[k + 2]) * w[k + 2];
        a3 += bf16_to_f32(xrow[k + 3]) * w[k + 3];
      }
      for (; k < K; ++k) a0 += bf16_to_f32(xrow[k]) * w[k];
      out[m * N + n] = (a0 + a1) + (a2 + a3);
    }
  }
}

// Same shape, fp32 activations: the down projection consumes the SwiGLU
// product, which is computed rather than loaded.
void matmul_nvfp4_f32in(const uint8_t* __restrict Wq,
                        const uint8_t* __restrict Wsf, float gscale,
                        const float* __restrict sf_tab,
                        const float* __restrict in, float* __restrict out,
                        size_t M, size_t N, size_t K, size_t n_begin,
                        size_t n_end) {
  const size_t knib = K / 2;
  const size_t kblocks = K / 16;
  RowScratch scratch(K);
  float* __restrict w = scratch.get();

  for (size_t n = n_begin; n < n_end; ++n) {
    decode_row(Wq + n * knib, Wsf + n * kblocks, gscale, sf_tab, w, kblocks);
    for (size_t m = 0; m < M; ++m) {
      const float* xrow = in + m * K;
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      size_t k = 0;
      for (; k + 4 <= K; k += 4) {
        a0 += xrow[k + 0] * w[k + 0];
        a1 += xrow[k + 1] * w[k + 1];
        a2 += xrow[k + 2] * w[k + 2];
        a3 += xrow[k + 3] * w[k + 3];
      }
      for (; k < K; ++k) a0 += xrow[k] * w[k];
      out[m * N + n] = (a0 + a1) + (a2 + a3);
    }
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

// One expert's packed planes. Each plane is nibbles followed by its block
// scales; the three planes plus their scales form one contiguous extent so
// the prefill path can DMA a whole expert to the 5090 in a single transfer.
struct CpuExpert {
  const uint8_t* gate_q;   // nvfp4 [intermediate, hidden/2]
  const uint8_t* gate_sf;  // e4m3  [intermediate, hidden/16]
  const uint8_t* up_q;
  const uint8_t* up_sf;
  const uint8_t* down_q;   // nvfp4 [hidden, intermediate/2]
  const uint8_t* down_sf;  // e4m3  [hidden, intermediate/16]
  float gs_gate;
  float gs_up;
  float gs_down;
};

class CpuExpertHost {
 public:
  CpuExpertHost(size_t num_layers, size_t num_experts, size_t hidden,
                size_t intermediate, size_t max_experts, size_t num_threads)
      : num_layers_(num_layers),
        num_experts_(num_experts),
        hidden_(hidden),
        intermediate_(intermediate),
        arena_(expert_bytes(hidden, intermediate) * max_experts +
               kHugePage /* slack for alignment */),
        pool_(num_threads) {
    slots_.assign(num_layers * num_experts, -1);
    for (size_t i = 0; i < 256; ++i) {
      sf_tab_[i] = e4m3_to_f32(static_cast<uint8_t>(i));
    }
  }

  bool valid() const noexcept { return arena_.valid(); }

  // Bytes one expert occupies: 3 planes of hidden*intermediate weights, each
  // half a byte, plus one e4m3 scale per 16 weights. 0.5625 B/weight against
  // bf16's 2.
  static size_t expert_bytes(size_t hidden, size_t intermediate) noexcept {
    const size_t elems = hidden * intermediate;
    return 3 * (elems / 2 + elems / 16);
  }

  int load_expert(size_t layer, size_t expert, const void* gate_q,
                  const void* gate_sf, float gs_gate, const void* up_q,
                  const void* up_sf, float gs_up, const void* down_q,
                  const void* down_sf, float gs_down) {
    if (layer >= num_layers_ || expert >= num_experts_) return -1;
    const size_t elems = hidden_ * intermediate_;
    if (elems % 16 != 0) return -3;  // block scales assume 16 | K
    const size_t nq = elems / 2;
    const size_t nsf = elems / 16;
    const size_t key = layer * num_experts_ + expert;

    CpuExpert* slot;
    if (slots_[key] >= 0) {
      slot = &experts_[static_cast<size_t>(slots_[key])];
    } else {
      // One allocation for everything. Contiguity is a requirement, not a
      // convenience: the prefill path DMAs an expert as a single transfer,
      // and separate bump allocations are only adjacent while every plane
      // size happens to be a multiple of the arena's alignment.
      char* blk = static_cast<char*>(arena_.alloc(3 * (nq + nsf)));
      if (!blk) return -2;  // arena exhausted
      auto* p = reinterpret_cast<uint8_t*>(blk);
      experts_.push_back(CpuExpert{
          p, p + nq,
          p + (nq + nsf), p + (nq + nsf) + nq,
          p + 2 * (nq + nsf), p + 2 * (nq + nsf) + nq,
          0.0f, 0.0f, 0.0f});
      slots_[key] = static_cast<int32_t>(experts_.size() - 1);
      slot = &experts_.back();
    }
    std::memcpy(const_cast<uint8_t*>(slot->gate_q), gate_q, nq);
    std::memcpy(const_cast<uint8_t*>(slot->gate_sf), gate_sf, nsf);
    std::memcpy(const_cast<uint8_t*>(slot->up_q), up_q, nq);
    std::memcpy(const_cast<uint8_t*>(slot->up_sf), up_sf, nsf);
    std::memcpy(const_cast<uint8_t*>(slot->down_q), down_q, nq);
    std::memcpy(const_cast<uint8_t*>(slot->down_sf), down_sf, nsf);
    slot->gs_gate = gs_gate;
    slot->gs_up = gs_up;
    slot->gs_down = gs_down;
    return 0;
  }

  const CpuExpert* lookup(size_t layer, size_t expert) const noexcept {
    if (layer >= num_layers_ || expert >= num_experts_) return nullptr;
    const int32_t slot = slots_[layer * num_experts_ + expert];
    return slot < 0 ? nullptr : &experts_[static_cast<size_t>(slot)];
  }

  // y = (silu(x @ gate^T) * (x @ up^T)) @ down^T
  void ffn(const CpuExpert& e, const uint16_t* input_bf16, float* output,
           size_t M) {
    ensure_scratch(M);
    float* g = scratch_gate_.data();
    float* u = scratch_up_.data();
    const size_t inter = intermediate_;
    const size_t hid = hidden_;

    pool_.parallel_for(inter, [&](size_t b, size_t en) {
      matmul_nvfp4_bf16in(e.gate_q, e.gate_sf, e.gs_gate, sf_tab_,
                          input_bf16, g, M, inter, hid, b, en);
    });
    pool_.parallel_for(inter, [&](size_t b, size_t en) {
      matmul_nvfp4_bf16in(e.up_q, e.up_sf, e.gs_up, sf_tab_,
                          input_bf16, u, M, inter, hid, b, en);
    });
    // Fuse the activation into gate's buffer; down then reads one array.
    for (size_t i = 0, n = M * inter; i < n; ++i) g[i] = silu(g[i]) * u[i];
    pool_.parallel_for(hid, [&](size_t b, size_t en) {
      matmul_nvfp4_f32in(e.down_q, e.down_sf, e.gs_down, sf_tab_,
                         g, output, M, hid, inter, b, en);
    });
  }

  int expert_forward(size_t layer, size_t expert, const void* input,
                     float* output, size_t M) {
    const CpuExpert* e = lookup(layer, expert);
    if (!e) return -1;
    ffn(*e, static_cast<const uint16_t*>(input), output, M);
    return 0;
  }

  int moe_forward(size_t layer, const uint16_t* hidden, const int32_t* ids,
                  const float* weights, size_t M, size_t topk, float* output) {
    if (layer >= num_layers_) return -1;
    std::fill(output, output + M * hidden_, 0.0f);
    if (M == 0 || topk == 0) return 0;

    // Bucket routes by expert so each weight set is streamed from DRAM once
    // per call instead of once per token. At the cold tier the whole cost is
    // that stream, so re-reading it per token would multiply the dominant
    // term by the number of tokens sharing the expert.
    bucket_counts_.assign(num_experts_, 0);
    for (size_t r = 0, n = M * topk; r < n; ++r) {
      const int32_t eid = ids[r];
      if (eid < 0) continue;
      if (static_cast<size_t>(eid) >= num_experts_) continue;
      if (!lookup(layer, static_cast<size_t>(eid))) {
        // The caller's partition should never route a non-resident expert
        // here. Count it rather than dropping it silently: a lost route is a
        // wrong answer, not a slow one.
        skipped_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      ++bucket_counts_[static_cast<size_t>(eid)];
    }

    bucket_offsets_.assign(num_experts_ + 1, 0);
    size_t max_bucket = 0;
    for (size_t e = 0; e < num_experts_; ++e) {
      bucket_offsets_[e + 1] = bucket_offsets_[e] + bucket_counts_[e];
      max_bucket = std::max(max_bucket, bucket_counts_[e]);
    }
    const size_t active = bucket_offsets_[num_experts_];
    if (active == 0) return 0;

    bucket_tokens_.resize(active);
    bucket_weights_.resize(active);
    bucket_fill_ = bucket_offsets_;  // running cursor per expert
    for (size_t m = 0; m < M; ++m) {
      for (size_t k = 0; k < topk; ++k) {
        const int32_t eid = ids[m * topk + k];
        if (eid < 0 || static_cast<size_t>(eid) >= num_experts_) continue;
        if (!lookup(layer, static_cast<size_t>(eid))) continue;
        const size_t cursor = bucket_fill_[static_cast<size_t>(eid)]++;
        bucket_tokens_[cursor] = m;
        bucket_weights_[cursor] = weights[m * topk + k];
      }
    }

    // Sized by the largest bucket, not by M: an expert repeated across a
    // token's top-k slots takes more rows than there are tokens.
    ensure_gather(max_bucket);
    for (size_t e = 0; e < num_experts_; ++e) {
      const size_t begin = bucket_offsets_[e];
      const size_t end = bucket_offsets_[e + 1];
      if (begin == end) continue;
      const CpuExpert* expert = lookup(layer, e);
      const size_t rows = end - begin;

      // Gather this expert's token rows into a dense batch.
      for (size_t i = 0; i < rows; ++i) {
        std::memcpy(gather_in_.data() + i * hidden_,
                    hidden + bucket_tokens_[begin + i] * hidden_,
                    hidden_ * sizeof(uint16_t));
      }
      ffn(*expert, gather_in_.data(), gather_out_.data(), rows);

      // Scatter-accumulate with routing weights.
      for (size_t i = 0; i < rows; ++i) {
        const float w = bucket_weights_[begin + i];
        const float* src = gather_out_.data() + i * hidden_;
        float* dst = output + bucket_tokens_[begin + i] * hidden_;
        for (size_t h = 0; h < hidden_; ++h) dst[h] += w * src[h];
      }
    }
    return 0;
  }

  size_t arena_used() const noexcept { return arena_.used(); }
  size_t arena_capacity() const noexcept { return arena_.capacity(); }
  size_t resident_count() const noexcept { return experts_.size(); }
  uint64_t skipped() const noexcept { return skipped_.load(); }

  // Size of one expert's contiguous gate|up|down block, and the arena it
  // lives in. Together these let the caller pin the arena once and then
  // DMA any resident expert straight out of it, with no staging copy.
  size_t expert_block_bytes() const noexcept {
    return expert_bytes(hidden_, intermediate_);
  }
  const void* arena_base() const noexcept { return arena_.base(); }

  // The three per-plane global scales. They live outside the DMA block
  // because the streamed copy carries only the packed bytes; the GPU-side
  // dequant needs these separately and they are 12 bytes, not worth
  // complicating the block layout or its alignment for.
  int expert_gscales(size_t layer, size_t expert, float out[3]) const {
    const CpuExpert* e = lookup(layer, expert);
    if (!e) return -2;
    out[0] = e->gs_gate;
    out[1] = e->gs_up;
    out[2] = e->gs_down;
    return 0;
  }

 private:
  void ensure_scratch(size_t M) {
    const size_t need = M * intermediate_;
    if (scratch_gate_.size() < need) {
      scratch_gate_.resize(need);
      scratch_up_.resize(need);
    }
  }

  void ensure_gather(size_t M) {
    if (gather_in_.size() < M * hidden_) {
      gather_in_.resize(M * hidden_);
      gather_out_.resize(M * hidden_);
    }
  }

  const size_t num_layers_;
  const size_t num_experts_;
  const size_t hidden_;
  const size_t intermediate_;
  HugeArena arena_;
  ThreadPool pool_;

  // float8_e4m3fn -> fp32, built once. Read once per 16 weights, so a table
  // here costs nothing while keeping the element decode branch-light.
  float sf_tab_[256];

  std::vector<int32_t> slots_;      // (layer, expert) -> index into experts_
  std::vector<CpuExpert> experts_;  // arena-backed weight pointers

  std::vector<float> scratch_gate_;
  std::vector<float> scratch_up_;
  std::vector<uint16_t> gather_in_;
  std::vector<float> gather_out_;
  std::vector<size_t> bucket_counts_;
  std::vector<size_t> bucket_offsets_;
  std::vector<size_t> bucket_fill_;
  std::vector<size_t> bucket_tokens_;
  std::vector<float> bucket_weights_;

  std::atomic<uint64_t> skipped_{0};
};

// One registered layer's flags and pinned buffers. Buffers are owned by the
// caller (CUDA pinned allocations) and must outlive the poller.
struct PollLayer {
  size_t layer;
  volatile uint32_t* signal;
  volatile uint32_t* completion;
  const uint16_t* hidden;  // bf16
  const int32_t* ids;
  const float* weights;
  float* output;
  size_t topk;
};

// Host-side watcher for graph-captured dispatch. Structurally identical to
// the B70 poller; see the header for why it cannot live in Python.
class CpuPoller {
 public:
  explicit CpuPoller(CpuExpertHost* host) : host_(host) {}

  ~CpuPoller() { stop(); }

  CpuPoller(const CpuPoller&) = delete;
  CpuPoller& operator=(const CpuPoller&) = delete;

  // Safe to call while running: the sweep re-snapshots under the same mutex,
  // and layers are only ever appended.
  void add(const PollLayer& layer) {
    std::lock_guard<std::mutex> guard(mutex_);
    layers_.push_back(layer);
    layer_count_.store(layers_.size(), std::memory_order_release);
  }

  int start() {
    if (running_.exchange(true)) return 0;  // already running
    try {
      thread_ = std::thread([this] { loop(); });
    } catch (...) {
      running_ = false;
      return -1;
    }
    return 0;
  }

  void stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
  }

  uint64_t dispatch_count() const { return dispatches_.load(); }
  uint64_t error_count() const { return errors_.load(); }
  uint64_t service_ns() const { return service_ns_.load(); }

 private:
  void loop() {
    std::vector<PollLayer> snapshot;
    size_t known = 0;
    int idle = 0;

    while (running_.load(std::memory_order_relaxed)) {
      if (layer_count_.load(std::memory_order_acquire) != known) {
        std::lock_guard<std::mutex> guard(mutex_);
        snapshot = layers_;
        known = snapshot.size();
      }

      bool worked = false;
      for (const PollLayer& entry : snapshot) {
        // The signal's VALUE is the batch size M; 0 means idle.
        const uint32_t M = entry.signal[0];
        if (M == 0) continue;

        // Clear before dispatching so the next graph replay can signal this
        // layer again while we work.
        entry.signal[0] = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        worked = true;

        const auto t0 = std::chrono::steady_clock::now();
        const int rc = host_->moe_forward(entry.layer, entry.hidden, entry.ids,
                                          entry.weights, M, entry.topk,
                                          entry.output);
        if (rc != 0) errors_.fetch_add(1);

        service_ns_.fetch_add(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count()));

        // Always release the waiter, even on failure: the CUDA side is parked
        // in cuStreamWaitValue32, which has no timeout.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        entry.completion[0] = 1;
        dispatches_.fetch_add(1);
      }

      // Back off when idle rather than spinning hot. A dispatch costs ~195us
      // here, so sleep granularity is noise against it, and a spinning
      // watcher would take a core from the FFN pool that needs it.
      if (worked) {
        idle = 0;
      } else if (++idle < 512) {
        SB_SPIN_HINT();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
    }
  }

  CpuExpertHost* host_;
  std::vector<PollLayer> layers_;
  std::mutex mutex_;
  std::atomic<size_t> layer_count_{0};
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> dispatches_{0};
  std::atomic<uint64_t> errors_{0};
  std::atomic<uint64_t> service_ns_{0};
  std::thread thread_;
};

inline CpuPoller* as_poller(sb_cpu_poller_t* p) {
  return static_cast<CpuPoller*>(p);
}

inline CpuExpertHost* as_host(sb_cpu_host_t* h) {
  return static_cast<CpuExpertHost*>(h);
}

}  // namespace

extern "C" {

sb_cpu_host_t* sb_cpu_create(size_t num_layers, size_t num_experts,
                             size_t hidden, size_t intermediate,
                             size_t max_experts, size_t num_threads) {
  if (num_layers == 0 || num_experts == 0 || hidden == 0 ||
      intermediate == 0 || max_experts == 0) {
    return nullptr;
  }
  if (num_threads == 0) {
    const unsigned hw = std::thread::hardware_concurrency();
    num_threads = hw ? std::max(1u, hw / 4u) : 4;
  }
  auto* host = new (std::nothrow) CpuExpertHost(
      num_layers, num_experts, hidden, intermediate, max_experts, num_threads);
  if (!host) return nullptr;
  if (!host->valid()) {
    delete host;
    return nullptr;
  }
  return host;
}

int sb_cpu_load_expert(sb_cpu_host_t* host, size_t layer, size_t expert,
                       const void* gate_q, const void* gate_sf, float gs_gate,
                       const void* up_q, const void* up_sf, float gs_up,
                       const void* down_q, const void* down_sf,
                       float gs_down) {
  if (!host || !gate_q || !gate_sf || !up_q || !up_sf || !down_q || !down_sf) {
    return -1;
  }
  return as_host(host)->load_expert(layer, expert, gate_q, gate_sf, gs_gate,
                                    up_q, up_sf, gs_up, down_q, down_sf,
                                    gs_down);
}

int sb_cpu_expert_gscales(sb_cpu_host_t* host, size_t layer, size_t expert,
                          float* out3) {
  if (!host || !out3) return -1;
  return as_host(host)->expert_gscales(layer, expert, out3);
}

int sb_cpu_has_expert(sb_cpu_host_t* host, size_t layer, size_t expert) {
  if (!host) return 0;
  return as_host(host)->lookup(layer, expert) != nullptr ? 1 : 0;
}

int sb_cpu_expert_forward(sb_cpu_host_t* host, size_t layer, size_t expert,
                          const void* input_bf16, float* output_f32, size_t M) {
  if (!host || !input_bf16 || !output_f32) return -1;
  return as_host(host)->expert_forward(layer, expert, input_bf16, output_f32, M);
}

int sb_cpu_moe_forward(sb_cpu_host_t* host, size_t layer,
                       const void* hidden_bf16, const int32_t* expert_ids,
                       const float* weights, size_t M, size_t topk,
                       float* output_f32) {
  if (!host || !hidden_bf16 || !expert_ids || !weights || !output_f32) {
    return -1;
  }
  return as_host(host)->moe_forward(
      layer, static_cast<const uint16_t*>(hidden_bf16), expert_ids, weights, M,
      topk, output_f32);
}

int sb_cpu_expert_block(sb_cpu_host_t* host, size_t layer, size_t expert,
                        const void** out_ptr, size_t* out_bytes) {
  if (!host || !out_ptr || !out_bytes) return -1;
  CpuExpertHost* h = as_host(host);
  const CpuExpert* e = h->lookup(layer, expert);
  if (!e) return -2;  // not resident
  *out_ptr = e->gate_q;
  *out_bytes = h->expert_block_bytes();
  return 0;
}

const void* sb_cpu_arena_base(sb_cpu_host_t* host) {
  return host ? as_host(host)->arena_base() : nullptr;
}

size_t sb_cpu_arena_used(sb_cpu_host_t* host) {
  return host ? as_host(host)->arena_used() : 0;
}

size_t sb_cpu_arena_capacity(sb_cpu_host_t* host) {
  return host ? as_host(host)->arena_capacity() : 0;
}

size_t sb_cpu_resident_count(sb_cpu_host_t* host) {
  return host ? as_host(host)->resident_count() : 0;
}

uint64_t sb_cpu_skipped_routes(sb_cpu_host_t* host) {
  return host ? as_host(host)->skipped() : 0;
}

sb_cpu_poller_t* sb_cpu_poll_create(sb_cpu_host_t* host) {
  if (!host) return nullptr;
  return new (std::nothrow) CpuPoller(as_host(host));
}

int sb_cpu_poll_register(sb_cpu_poller_t* poller, size_t layer,
                         volatile uint32_t* signal,
                         volatile uint32_t* completion, const void* hidden,
                         const int32_t* ids, const float* weights,
                         float* output, size_t topk) {
  if (!poller || !signal || !completion || !hidden || !ids || !weights ||
      !output || topk == 0) {
    return -1;
  }
  as_poller(poller)->add(PollLayer{layer, signal, completion,
                                   static_cast<const uint16_t*>(hidden), ids,
                                   weights, output, topk});
  return 0;
}

int sb_cpu_poll_start(sb_cpu_poller_t* poller) {
  return poller ? as_poller(poller)->start() : -1;
}

void sb_cpu_poll_stop(sb_cpu_poller_t* poller) {
  if (poller) as_poller(poller)->stop();
}

uint64_t sb_cpu_poll_dispatch_count(sb_cpu_poller_t* poller) {
  return poller ? as_poller(poller)->dispatch_count() : 0;
}

uint64_t sb_cpu_poll_error_count(sb_cpu_poller_t* poller) {
  return poller ? as_poller(poller)->error_count() : 0;
}

uint64_t sb_cpu_poll_service_ns(sb_cpu_poller_t* poller) {
  return poller ? as_poller(poller)->service_ns() : 0;
}

void sb_cpu_poll_destroy(sb_cpu_poller_t* poller) {
  delete as_poller(poller);
}

void sb_cpu_destroy(sb_cpu_host_t* host) {
  delete as_host(host);
}

}  // extern "C"
