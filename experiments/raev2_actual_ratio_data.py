"""Read actual native query states paired with independent real endpoints."""
from pathlib import Path
import numpy as np
import torch


class ActualPairs:
    def __init__(self, actual_root, real_split, count):
        self.count = count
        self.real = np.load(real_split['real_path'], mmap_mode='r')
        self.real_lookup = np.load(real_split['real_lookup'])
        self.states, self.noise = [], []
        self.mapping = np.full((count, 2), -1, dtype=np.int64)
        self.times = np.empty(count, dtype=np.float32)
        for rank in range(4):
            folder = Path(actual_root) / f'shard{rank}'
            states = np.load(folder / 'states.npy', mmap_mode='r')
            noise = np.load(folder / 'noise.npy', mmap_mode='r')
            meta = np.load(folder / 'metadata.npz')['records']
            ids = meta[:, 0].astype(np.int64)
            assert np.all(self.mapping[ids, 0] == -1)
            assert states.shape == noise.shape == (len(ids), 1024, 16, 16)
            assert states.dtype == noise.dtype == np.float32
            assert np.array_equal(ids // 8 % 4, np.full(len(ids), rank))
            assert np.array_equal(meta[:, 1], ids % 1000)
            self.mapping[ids] = np.stack((np.full(len(ids), rank), np.arange(len(ids))), axis=1)
            self.times[ids] = meta[:, 3]
            self.states.append(states)
            self.noise.append(noise)
        assert np.all(self.mapping >= 0) and np.all((self.times > 0) & (self.times < 1))

    def batch(self, ids, real_k, device):
        ids = np.asarray(ids, dtype=np.int64)
        labels_np = ids % 1000
        real = np.array(self.real[self.real_lookup[labels_np, real_k]], copy=True)
        state, noise = np.empty_like(real, dtype=np.float32), np.empty_like(real, dtype=np.float32)
        location = self.mapping[ids]
        for rank in range(4):
            mask = location[:, 0] == rank
            rows = location[mask, 1]
            state[mask] = self.states[rank][rows]
            noise[mask] = self.noise[rank][rows]
        p = torch.from_numpy(real).to(device=device, dtype=torch.float32)
        q = torch.from_numpy(state).to(device=device)
        eps = torch.from_numpy(noise).to(device=device)
        t = torch.from_numpy(self.times[ids]).to(device=device)
        a = 1 - t
        p = a[:, None, None, None]*p + t[:, None, None, None]*eps
        labels = torch.from_numpy(labels_np).to(device=device)
        return p, q, t, labels
