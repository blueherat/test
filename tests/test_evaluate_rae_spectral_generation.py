from experiments.evaluate_rae_spectral_generation import sample_folder_name


def test_sample_folder_name_preserves_existing_50_step_contract():
    assert sample_folder_name(5000, 10000, 50) == "fixed_seed20260715_5000_step10000"
    assert sample_folder_name(5000, 10000, 25) == "fixed_seed20260715_5000_step10000_25steps"
