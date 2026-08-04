/*
 * B70 routed-expert worker using llm-scaler's IPEX K-major SLM ESIMD path.
 *
 * Source pattern:
 *   intel-xpu/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl
 *   - moe_up_routed_int4_slm_kernel
 *   - moe_down_routed_int4_slm_kernel
 *
 * Colibri uses group_size=64. The upstream source hard-codes 128, so only the
 * group count/index is parameterized; packed offset-binary marlin weights,
 * FP16 scale loads, SLM reduction, and ESIMD launch geometry are preserved.
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

using fp16 = sycl::half;
using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

namespace {

constexpr int kSlmGroup = 16;
constexpr int kAccumulateWidth = 64;
constexpr int kMarlinUnshuffle[8] = {0, 2, 4, 6, 1, 3, 5, 7};

struct B70State {
    sycl::queue *queue = nullptr;
    int experts = 0;
    int hidden = 0;
    int intermediate = 0;
    int topk = 0;
    int group_size = 0;

    uint32_t *gate_up = nullptr;
    fp16 *gate_up_scales = nullptr;
    uint32_t *down = nullptr;
    fp16 *down_scales = nullptr;

    fp16 *activation = nullptr;
    fp16 *intermediates = nullptr;
    fp16 *route_output = nullptr;
    fp16 *output = nullptr;
    int *expert_ids = nullptr;
    fp16 *routing_weights = nullptr;

    fp16 *host_activation = nullptr;
    fp16 *host_output = nullptr;
    fp16 *host_routing_weights = nullptr;
    int *host_expert_ids = nullptr;
    sycl::event pending_event;
    std::chrono::steady_clock::time_point issue_start;
    bool pending = false;
    bool profile = false;
    uint64_t dispatches = 0;
    double dispatch_us = 0.0;
};

B70State S;

static sycl::device find_b70() {
    for (const auto &platform : sycl::platform::get_platforms()) {
        for (const auto &device : platform.get_devices()) {
            if (!device.is_gpu() ||
                device.get_info<sycl::info::device::vendor_id>() != 0x8086)
                continue;
            const auto memory =
                device.get_info<sycl::info::device::global_mem_size>();
            if (memory > 8'000'000'000ULL && memory < 40'000'000'000ULL)
                return device;
        }
    }
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Intel Arc Pro B70 not found");
}

class B70MoeUpSlm;
class B70MoeDownSlm;
class B70MoeAccumulate;

static sycl::event submit_up(sycl::queue &queue, int routes) {
    const int hidden = S.hidden;
    const int intermediate = S.intermediate;
    const int group_size = S.group_size;
    const int packed_k = hidden / 8;
    const int scale_groups = hidden / group_size;
    const int gate_up_rows = 2 * intermediate;
    const int output_tiles = intermediate / 16;
    const int packed_per_group = group_size / 8;
    const fp16 *activation = S.activation;
    const uint32_t *gate_up = S.gate_up;
    const fp16 *scales = S.gate_up_scales;
    const int *expert_ids = S.expert_ids;
    fp16 *intermediates = S.intermediates;

    return queue.submit([=](sycl::handler &handler) {
        handler.parallel_for<B70MoeUpSlm>(
            sycl::nd_range<2>(
                sycl::range<2>(routes, output_tiles * kSlmGroup),
                sycl::range<2>(1, kSlmGroup)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<kSlmGroup * 2 * 16 * sizeof(float)>();

                const int route = (int)item.get_global_id(0);
                const int tile = (int)item.get_group(1);
                const int lane = (int)item.get_local_id(1);
                const int output_start = tile * 16;
                const int expert = expert_ids[route];

                const uint32_t *weight_base = gate_up +
                    (size_t)expert * packed_k * gate_up_rows;
                const fp16 *scale_base = scales +
                    (size_t)expert * scale_groups * gate_up_rows;
                const int gate_offset = output_start;
                const int up_offset = intermediate + output_start;

                simd<float, 16> gate_acc(0.0f);
                simd<float, 16> up_acc(0.0f);
                for (int packed = lane; packed < packed_k;
                     packed += kSlmGroup) {
                    simd<fp16, 8> input =
                        block_load<fp16, 8>(activation + packed * 8);
                    simd<uint32_t, 16> gate_words =
                        block_load<uint32_t, 16>(
                            weight_base + (size_t)packed * gate_up_rows +
                            gate_offset);
                    simd<uint32_t, 16> up_words =
                        block_load<uint32_t, 16>(
                            weight_base + (size_t)packed * gate_up_rows +
                            up_offset);

                    const int scale_group = packed / packed_per_group;
                    simd<fp16, 16> gate_scale = block_load<fp16, 16>(
                        scale_base + (size_t)scale_group * gate_up_rows +
                        gate_offset);
                    simd<fp16, 16> up_scale = block_load<fp16, 16>(
                        scale_base + (size_t)scale_group * gate_up_rows +
                        up_offset);

                    #pragma unroll
                    for (int element = 0; element < 8; element++) {
                        const int nibble_position = kMarlinUnshuffle[element];
                        simd<uint32_t, 16> gate_nibble =
                            (gate_words >> (nibble_position * 4)) & 0x0f;
                        simd<uint32_t, 16> up_nibble =
                            (up_words >> (nibble_position * 4)) & 0x0f;
                        simd<float, 16> gate_values =
                            (convert<float>(gate_nibble) - 8.0f) *
                            convert<float>(gate_scale);
                        simd<float, 16> up_values =
                            (convert<float>(up_nibble) - 8.0f) *
                            convert<float>(up_scale);
                        const float value = (float)(fp16)input[element];
                        gate_acc += gate_values * value;
                        up_acc += up_values * value;
                    }
                }

                const uint32_t slm_offset =
                    (uint32_t)(lane * 2 * 16) * sizeof(float);
                slm_block_store<float, 16>(slm_offset, gate_acc);
                slm_block_store<float, 16>(slm_offset + 64, up_acc);
                barrier();

                if (lane == 0) {
                    simd<float, 16> gate_sum(0.0f);
                    simd<float, 16> up_sum(0.0f);
                    #pragma unroll
                    for (int source = 0; source < kSlmGroup; source++) {
                        const uint32_t offset =
                            (uint32_t)(source * 2 * 16) * sizeof(float);
                        gate_sum += slm_block_load<float, 16>(offset);
                        up_sum += slm_block_load<float, 16>(offset + 64);
                    }
                    simd<float, 16> result =
                        (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
                    block_store<fp16, 16>(
                        intermediates + (size_t)route * intermediate +
                        output_start,
                        convert<fp16>(result));
                }
            });
    });
}

static sycl::event submit_down(sycl::queue &queue, int routes) {
    const int hidden = S.hidden;
    const int intermediate = S.intermediate;
    const int group_size = S.group_size;
    const int packed_k = intermediate / 8;
    const int scale_groups = intermediate / group_size;
    const int output_tiles = hidden / 16;
    const int packed_per_group = group_size / 8;
    const fp16 *intermediates = S.intermediates;
    const uint32_t *down = S.down;
    const fp16 *scales = S.down_scales;
    const fp16 *routing_weights = S.routing_weights;
    const int *expert_ids = S.expert_ids;
    fp16 *route_output = S.route_output;

    return queue.submit([=](sycl::handler &handler) {
        handler.parallel_for<B70MoeDownSlm>(
            sycl::nd_range<2>(
                sycl::range<2>(routes, output_tiles * kSlmGroup),
                sycl::range<2>(1, kSlmGroup)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<kSlmGroup * 16 * sizeof(float)>();

                const int route = (int)item.get_global_id(0);
                const int tile = (int)item.get_group(1);
                const int lane = (int)item.get_local_id(1);
                const int output_start = tile * 16;
                const int expert = expert_ids[route];
                const fp16 *input =
                    intermediates + (size_t)route * intermediate;
                const uint32_t *weight_base =
                    down + (size_t)expert * packed_k * hidden;
                const fp16 *scale_base =
                    scales + (size_t)expert * scale_groups * hidden;

                simd<float, 16> acc(0.0f);
                for (int packed = lane; packed < packed_k;
                     packed += kSlmGroup) {
                    simd<fp16, 8> input_values =
                        block_load<fp16, 8>(input + packed * 8);
                    simd<uint32_t, 16> words = block_load<uint32_t, 16>(
                        weight_base + (size_t)packed * hidden + output_start);
                    const int scale_group = packed / packed_per_group;
                    simd<fp16, 16> scale = block_load<fp16, 16>(
                        scale_base + (size_t)scale_group * hidden +
                        output_start);

                    #pragma unroll
                    for (int element = 0; element < 8; element++) {
                        const int nibble_position = kMarlinUnshuffle[element];
                        simd<uint32_t, 16> nibble =
                            (words >> (nibble_position * 4)) & 0x0f;
                        simd<float, 16> weight =
                            (convert<float>(nibble) - 8.0f) *
                            convert<float>(scale);
                        acc += weight * (float)(fp16)input_values[element];
                    }
                }

                const uint32_t slm_offset =
                    (uint32_t)(lane * 16) * sizeof(float);
                slm_block_store<float, 16>(slm_offset, acc);
                barrier();

                if (lane == 0) {
                    simd<float, 16> sum(0.0f);
                    #pragma unroll
                    for (int source = 0; source < kSlmGroup; source++)
                        sum += slm_block_load<float, 16>(
                            (uint32_t)(source * 16) * sizeof(float));
                    const float route_weight = (float)routing_weights[route];
                    block_store<fp16, 16>(
                        route_output + (size_t)route * hidden + output_start,
                        convert<fp16>(sum * route_weight));
                }
            });
    });
}

static sycl::event submit_accumulate(sycl::queue &queue, int routes) {
    const int hidden = S.hidden;
    const fp16 *route_output = S.route_output;
    fp16 *output = S.output;
    return queue.submit([=](sycl::handler &handler) {
        handler.parallel_for<B70MoeAccumulate>(
            sycl::range<1>(hidden / kAccumulateWidth),
            [=](sycl::id<1> index) SYCL_ESIMD_KERNEL {
                const int start = (int)index[0] * kAccumulateWidth;
                simd<float, kAccumulateWidth> sum(0.0f);
                for (int route = 0; route < routes; route++)
                    sum += convert<float>(block_load<fp16, kAccumulateWidth>(
                        route_output + (size_t)route * hidden + start));
                block_store<fp16, kAccumulateWidth>(output + start,
                                                    convert<fp16>(sum));
            });
    });
}

static void release_state() {
    if (!S.queue)
        return;
    for (void *pointer : {static_cast<void *>(S.gate_up),
                          static_cast<void *>(S.gate_up_scales),
                          static_cast<void *>(S.down),
                          static_cast<void *>(S.down_scales),
                          static_cast<void *>(S.activation),
                          static_cast<void *>(S.intermediates),
                          static_cast<void *>(S.route_output),
                          static_cast<void *>(S.output),
                          static_cast<void *>(S.expert_ids),
                          static_cast<void *>(S.routing_weights),
                          static_cast<void *>(S.host_activation),
                          static_cast<void *>(S.host_output),
                          static_cast<void *>(S.host_routing_weights),
                          static_cast<void *>(S.host_expert_ids)}) {
        if (pointer)
            sycl::free(pointer, *S.queue);
    }
    delete S.queue;
    S = B70State{};
}

}  // namespace

extern "C" {

int b70_moe_init(int experts, int hidden, int intermediate, int topk,
                 int group_size) {
    if (experts <= 0 || hidden <= 0 || intermediate <= 0 || topk <= 0 ||
        group_size < 16 || group_size % 16 || hidden % group_size ||
        intermediate % group_size || hidden % 64 || intermediate % 16)
        return -1;

    try {
        S.experts = experts;
        S.hidden = hidden;
        S.intermediate = intermediate;
        S.topk = topk;
        S.group_size = group_size;
        S.profile = std::getenv("B70_PROFILE") != nullptr;

        const sycl::device device = find_b70();
        sycl::async_handler async_handler = [](sycl::exception_list errors) {
            for (const auto &error : errors)
                std::rethrow_exception(error);
        };
        S.queue = new sycl::queue(
            device, async_handler,
            sycl::property_list{sycl::property::queue::in_order()});

        const size_t gate_up_words =
            (size_t)experts * (hidden / 8) * (2 * intermediate);
        const size_t gate_up_scale_count =
            (size_t)experts * (hidden / group_size) * (2 * intermediate);
        const size_t down_words =
            (size_t)experts * (intermediate / 8) * hidden;
        const size_t down_scale_count =
            (size_t)experts * (intermediate / group_size) * hidden;

        S.gate_up = sycl::malloc_device<uint32_t>(gate_up_words, *S.queue);
        S.gate_up_scales =
            sycl::malloc_device<fp16>(gate_up_scale_count, *S.queue);
        S.down = sycl::malloc_device<uint32_t>(down_words, *S.queue);
        S.down_scales =
            sycl::malloc_device<fp16>(down_scale_count, *S.queue);
        S.activation = sycl::malloc_device<fp16>(hidden, *S.queue);
        S.intermediates =
            sycl::malloc_device<fp16>((size_t)topk * intermediate, *S.queue);
        S.route_output =
            sycl::malloc_device<fp16>((size_t)topk * hidden, *S.queue);
        S.output = sycl::malloc_device<fp16>(hidden, *S.queue);
        S.expert_ids = sycl::malloc_device<int>(topk, *S.queue);
        S.routing_weights = sycl::malloc_device<fp16>(topk, *S.queue);
        S.host_activation = sycl::malloc_host<fp16>(hidden, *S.queue);
        S.host_output = sycl::malloc_host<fp16>(hidden, *S.queue);
        S.host_routing_weights = sycl::malloc_host<fp16>(topk, *S.queue);
        S.host_expert_ids = sycl::malloc_host<int>(topk, *S.queue);

        if (!S.gate_up || !S.gate_up_scales || !S.down || !S.down_scales ||
            !S.activation || !S.intermediates || !S.route_output ||
            !S.output || !S.expert_ids || !S.routing_weights ||
            !S.host_activation || !S.host_output ||
            !S.host_routing_weights || !S.host_expert_ids) {
            std::fprintf(stderr, "[b70_moe] device allocation failed\n");
            release_state();
            return -1;
        }

        const double weight_gib =
            (gate_up_words * sizeof(uint32_t) +
             gate_up_scale_count * sizeof(fp16) +
             down_words * sizeof(uint32_t) +
             down_scale_count * sizeof(fp16)) /
            (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr,
            "[b70_moe] %s: K-major SLM ESIMD, %d slots, gs=%d, %.2f GiB\n",
            device.get_info<sycl::info::device::name>().c_str(), experts,
            group_size, weight_gib);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[b70_moe] init failed: %s\n", error.what());
        release_state();
        return -1;
    }
}

int b70_moe_upload(int expert, const uint32_t *gate_up,
                   const uint16_t *gate_up_scales, const uint32_t *down,
                   const uint16_t *down_scales) {
    if (!S.queue || expert < 0 || expert >= S.experts || !gate_up ||
        !gate_up_scales || !down || !down_scales)
        return -1;

    try {
        const size_t gate_up_words =
            (size_t)(S.hidden / 8) * (2 * S.intermediate);
        const size_t gate_up_scale_count =
            (size_t)(S.hidden / S.group_size) * (2 * S.intermediate);
        const size_t down_words =
            (size_t)(S.intermediate / 8) * S.hidden;
        const size_t down_scale_count =
            (size_t)(S.intermediate / S.group_size) * S.hidden;

        S.queue->memcpy(S.gate_up + (size_t)expert * gate_up_words,
                        gate_up, gate_up_words * sizeof(uint32_t));
        S.queue->memcpy(
            S.gate_up_scales + (size_t)expert * gate_up_scale_count,
            gate_up_scales, gate_up_scale_count * sizeof(fp16));
        S.queue->memcpy(S.down + (size_t)expert * down_words,
                        down, down_words * sizeof(uint32_t));
        S.queue->memcpy(
            S.down_scales + (size_t)expert * down_scale_count,
            down_scales,
            down_scale_count * sizeof(fp16)).wait_and_throw();
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[b70_moe] upload failed: %s\n", error.what());
        return -1;
    }
}

int b70_moe_issue(const float *activation, const int *expert_ids,
                  const float *weights, int routes) {
    if (!S.queue || S.pending || !activation || !expert_ids || !weights ||
        routes <= 0 || routes > S.topk)
        return -1;
    for (int route = 0; route < routes; route++)
        if (expert_ids[route] < 0 || expert_ids[route] >= S.experts)
            return -1;

    try {
        for (int index = 0; index < S.hidden; index++)
            S.host_activation[index] = fp16(activation[index]);
        for (int route = 0; route < routes; route++) {
            S.host_expert_ids[route] = expert_ids[route];
            S.host_routing_weights[route] = fp16(weights[route]);
        }
        S.issue_start = std::chrono::steady_clock::now();
        S.queue->memcpy(S.activation, S.host_activation,
                        (size_t)S.hidden * sizeof(fp16));
        S.queue->memcpy(S.expert_ids, S.host_expert_ids,
                        (size_t)routes * sizeof(int));
        S.queue->memcpy(S.routing_weights, S.host_routing_weights,
                        (size_t)routes * sizeof(fp16));
        submit_up(*S.queue, routes);
        submit_down(*S.queue, routes);
        submit_accumulate(*S.queue, routes);
        S.pending_event = S.queue->memcpy(
            S.host_output, S.output, (size_t)S.hidden * sizeof(fp16));
        S.pending = true;
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[b70_moe] issue failed: %s\n", error.what());
        S.pending = false;
        return -1;
    }
}

int b70_moe_take(float *partial) {
    if (!S.queue || !S.pending || !partial)
        return -1;
    try {
        S.pending_event.wait_and_throw();
        for (int index = 0; index < S.hidden; index++)
            partial[index] = (float)S.host_output[index];
        S.pending = false;

        if (S.profile) {
            const auto end = std::chrono::steady_clock::now();
            S.dispatch_us += std::chrono::duration<double, std::micro>(
                end - S.issue_start).count();
            S.dispatches++;
            if (S.dispatches % 100 == 0)
                std::fprintf(stderr,
                    "[b70_moe] issue-to-take avg %.1f us (%llu calls)\n",
                    S.dispatch_us / S.dispatches,
                    (unsigned long long)S.dispatches);
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[b70_moe] take failed: %s\n", error.what());
        S.pending = false;
        return -1;
    }
}

int b70_moe_dispatch(const float *activation, float *partial,
                     const int *expert_ids, const float *weights, int routes) {
    if (b70_moe_issue(activation, expert_ids, weights, routes) != 0)
        return -1;
    return b70_moe_take(partial);
}

void b70_moe_shutdown(void) {
    if (S.profile && S.dispatches)
        std::fprintf(stderr,
                     "[b70_moe] final dispatch avg %.1f us (%llu calls)\n",
                     S.dispatch_us / S.dispatches,
                     (unsigned long long)S.dispatches);
    release_state();
    std::fprintf(stderr, "[b70_moe] shutdown\n");
}

}  // extern "C"
