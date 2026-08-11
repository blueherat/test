from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_imagenet100_sit_flow import (
    MOMENT_SHAPE,
    NpyMomentsDataset,
    ModelEMA,
    linear_flow_state_target,
    load_official_sit_module,
    sample_sdvae_posterior,
)


def test_linear_flow_has_official_sit_orientation() -> None:
    data = torch.randn(3, 4, 2, 2)
    noise = torch.randn_like(data)
    state, target = linear_flow_state_target(
        data, noise, torch.tensor([0.0, 0.5, 1.0])
    )
    assert torch.equal(state[0], noise[0])
    assert torch.equal(state[2], data[2])
    assert torch.allclose(state[1], (noise[1] + data[1]) / 2)
    assert torch.equal(target, data - noise)


def test_linear_flow_matches_official_icplan() -> None:
    import importlib.util

    path = Path("/home/zhoushunyu/data/research_repos/SiT/transport/path.py")
    if not path.is_file():
        pytest.skip("official SiT checkout is not present")
    spec = importlib.util.spec_from_file_location("eqvae_official_sit_path", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = torch.randn(7, 4, 5, 5)
    noise = torch.randn_like(data)
    time_value = torch.rand(7)
    ours = linear_flow_state_target(data, noise, time_value)
    _, official_state, official_target = module.ICPlan().plan(
        time_value, noise, data
    )
    assert torch.equal(ours[0], official_state)
    assert torch.equal(ours[1], official_target)


def test_sdvae_posterior_uses_mean_std_and_official_scale() -> None:
    mean = torch.full((2, 4, 32, 32), 2.0)
    std = torch.full_like(mean, 0.25)
    posterior_noise = torch.full_like(mean, -2.0)
    moments = torch.cat([mean, std], dim=1)
    result = sample_sdvae_posterior(moments, posterior_noise, scaling_factor=0.5)
    assert torch.equal(result, torch.full_like(mean, 0.75))


def test_memmap_dataset_returns_exact_moments_and_labels(tmp_path: Path) -> None:
    moments = np.random.default_rng(2).normal(size=(5, *MOMENT_SHAPE)).astype(np.float32)
    moments[:, 4:] = np.abs(moments[:, 4:])
    labels = np.asarray([4, 3, 2, 1, 0], dtype=np.int16)
    np.save(tmp_path / "train_moments.npy", moments, allow_pickle=False)
    np.save(tmp_path / "train_labels.npy", labels, allow_pickle=False)
    dataset = NpyMomentsDataset(tmp_path, "train")
    recovered, label = dataset[3]
    assert torch.equal(recovered, torch.from_numpy(moments[3]))
    assert label == 1


def test_foreach_ema_matches_scalar_definition() -> None:
    torch.manual_seed(5)
    model = torch.nn.Linear(4, 3)
    ema = ModelEMA(model)
    before = {name: value.detach().clone() for name, value in ema.module.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))
    ema.update(0.75)
    for name, value in ema.module.named_parameters():
        expected = before[name] * 0.75 + dict(model.named_parameters())[name] * 0.25
        assert torch.allclose(value, expected)


def test_official_sit_source_hash_and_small_forward() -> None:
    repo = Path("/home/zhoushunyu/data/research_repos/SiT")
    if not repo.is_dir():
        pytest.skip("official SiT checkout is not present")
    module, metadata = load_official_sit_module(repo, verify_source=True)
    model = module.SiT(
        input_size=4,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=1,
        num_heads=4,
        num_classes=100,
        learn_sigma=True,
    ).eval()
    output = model(torch.randn(2, 4, 4, 4), torch.rand(2), torch.tensor([0, 1]))
    assert output.shape == (2, 4, 4, 4)
    assert torch.count_nonzero(output) == 0
    assert metadata["git_commit"]
