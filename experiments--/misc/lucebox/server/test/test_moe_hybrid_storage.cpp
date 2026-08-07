#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_storage.h"

#include <cstdint>
#include <vector>

using namespace dflash::common;

namespace {
struct MoeHybridStorageFixture {};
}

TEST_CASE(MoeHybridStorageFixture, expert_residency_tracks_model_sized_expert_sets) {
    MoeHybridLayerStorage storage;
    storage.reset_expert_vram_mask(320);

    storage.set_expert_hot(0);
    storage.set_expert_hot(255);
    storage.set_expert_hot(256);
    storage.set_expert_hot(319);

    REQUIRE(storage.is_expert_hot(0));
    REQUIRE(storage.is_expert_hot(255));
    REQUIRE(storage.is_expert_hot(256));
    REQUIRE(storage.is_expert_hot(319));
    REQUIRE(!storage.is_expert_hot(320));

    const std::vector<int32_t> all_hot = {0, 256, 319, -1};
    REQUIRE(storage.all_routed_are_hot(all_hot.data(), (int)all_hot.size()));

    const std::vector<int32_t> includes_cold = {0, 257};
    REQUIRE(!storage.all_routed_are_hot(includes_cold.data(), (int)includes_cold.size()));

    storage.clear_expert_hot(256);
    REQUIRE(!storage.is_expert_hot(256));
    REQUIRE(!storage.all_routed_are_hot(all_hot.data(), (int)all_hot.size()));
}
