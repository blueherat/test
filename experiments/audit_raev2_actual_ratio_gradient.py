"""Verify the new actual-law critic gradient on held-out native query states."""
import copy
import json
import numpy as np
import torch
from experiments.raev2_actual_ratio_data import ActualPairs
from experiments.raev2_paired_ratio_model import PairedRatioCritic
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = DATA / 'actual_ratio_fit/critic.pt'
    trained = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert trained['validation']['entry_condition_passed']
    model = PairedRatioCritic(trained['state_dict']['class_features']).cuda().eval().requires_grad_(False)
    model.load_state_dict(trained['state_dict'])
    reference = copy.deepcopy(model).double()
    manifest = json.loads((DATA / 'paired_ratio_data/manifest.json').read_text())
    bank = ActualPairs(DATA / 'actual_ratio_bank/validation', manifest['splits']['validation'], 1000)
    request = json.loads((DATA / 'actual_ratio_bank/validation/shard0/request.json').read_text())
    rows = []
    for index in (0, 1, 25, 75, 99):
        value = request['query_grid'][index]
        if index == 0:
            ids = np.arange(8)
            state = torch.from_numpy(np.array(bank.noise[0][:8])).cuda()
            labels = torch.from_numpy(ids).cuda()
        else:
            ids = np.flatnonzero(bank.times == np.float32(value))[:8]
            assert len(ids) == 8
            _, state, _, labels = bank.batch(ids, np.zeros(8, dtype=int), 'cuda')
        times = torch.full((8,), value, device='cuda')
        correction = model.clean_correction(state, times, labels)
        gradient = correction/value**2
        assert torch.isfinite(gradient).all()
        x64 = state.double().requires_grad_(True)
        g64, = torch.autograd.grad(reference(x64, times.double(), labels).sum(), x64)
        norms = g64.flatten(1).norm(dim=1)
        assert (norms > 0).all()
        error = ((gradient.double()-g64).flatten(1).norm(dim=1)/norms).max().item()
        direction = g64/norms[:, None, None, None]
        with torch.no_grad():
            fd = (reference(x64+.001*direction, times.double(), labels)-
                  reference(x64-.001*direction, times.double(), labels))/.002
        fd_error = ((fd-norms).abs()/norms).max().item()
        assert error < .002 and fd_error < .002, (index, error, fd_error)
        rows.append({'query_index': index, 'time': value, 'ids': ids.tolist(),
                     'fp32_vs_fp64_gradient_relative_error': error,
                     'fp64_finite_difference_relative_error': fd_error,
                     'correction_rms': correction.square().mean().sqrt().item()})
    result = {'complete': True, 'checkpoint_sha256': sha(checkpoint), 'rows': rows,
              'actual_heldout_states_and_pure_noise_checked': True, 'fid_used': False}
    path = ROOT / 'experiments/results/raev2_guidance_20260907/actual_ratio_gradient_audit.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
