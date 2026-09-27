from experiments.analyze_campaign_profile import (
    duration,
    intersection_duration,
    merge_intervals,
)


def test_union_does_not_double_count_concurrent_engines():
    assert merge_intervals([(20, 30), (0, 10), (4, 6), (10, 20)]) == [(0, 30)]
    assert duration([(0, 10), (4, 12), (20, 25)]) == 17


def test_overlap_and_exposed_service_partition_native_time():
    cuda = [(0, 10), (4, 12), (20, 25)]
    native = [(5, 15), (14, 22), (30, 35)]
    overlap = intersection_duration(cuda, native)
    assert overlap == 9
    assert overlap == intersection_duration(native, cuda)
    assert duration(native) - overlap == 13
    assert duration(cuda + native) == duration(cuda) + duration(native) - overlap


def test_empty_and_touching_intervals_have_no_overlap():
    assert duration([]) == 0
    assert intersection_duration([], [(1, 2)]) == 0
    assert intersection_duration([(0, 1)], [(1, 2)]) == 0


def test_invalid_interval_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="negative"):
        merge_intervals([(4, 3)])
