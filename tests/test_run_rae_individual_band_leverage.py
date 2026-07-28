from experiments.run_rae_individual_band_leverage import individual_schedules


def test_individual_schedules_cover_each_band_once():
    schedules = individual_schedules(4)
    assert schedules == (
        "baseline",
        "partial_high_all",
        "partial_high_band0",
        "partial_high_band1",
        "partial_high_band2",
        "partial_high_band3",
    )
