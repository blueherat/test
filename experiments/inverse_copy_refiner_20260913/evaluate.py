"""Postprocess frozen CFG/APG endpoints with a trained inverse-copy refiner.

Run with ``python -m experiments.inverse_copy_refiner_20260913.evaluate infer``
or replace ``infer`` with ``rounds``. This loads only the refiner and SD-VAE;
there is no generation-model call, new noise, classifier, or FID execution.
The rounds phase repeats P_beta on latent outputs; decoding is visualization
only, not a VAE encode/decode feedback loop or the original copy operator C.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913')
BASELINES = ROOT.parent / 'cfg_transport_search_20260913/baseline_1k'
ARMS = ('cfg_a1.25_s64', 'apg_a2_s64')
BETAS = (.5, 1.)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def npz(path, **values):
    temporary = Path(path).with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **values)
    temporary.replace(path)


def save_images(pixels, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for index, pixel in enumerate(pixels):
        Image.fromarray(pixel).save(directory / f'{index:06d}.png')


def gallery(pixels, path, captions=None, columns=8):
    side = 128
    canvas = Image.new('RGB', (columns * side, ((len(pixels)+columns-1)//columns)*148), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, pixel in enumerate(pixels):
        left, top = index % columns * side, index // columns * 148
        canvas.paste(Image.fromarray(pixel).resize((side, side)), (left, top))
        if captions is not None:
            draw.text((left+2, top+side), str(captions[index]), fill='black')
    canvas.save(path)


def load_frozen_bank(root, arm):
    """Verify original batch identities, endpoint order, and cached pixels."""
    request = read(root/'request.json')
    digest = sha(root/'request.json')
    assert request['samples'] == 1000 and request['batch'] == 16
    assert sha(root/'inputs.npz') == request['inputs_sha256']
    with np.load(root/'inputs.npz', allow_pickle=False) as data:
        labels, noise = data['labels'], data['noise']
    out = root/arm
    summary = read(out/'summary.json')
    assert summary['complete'] and summary['request_sha256'] == digest
    assert summary['config'] in request['configs']
    assert summary['config']['kind'] in ('cfg', 'apg')
    with np.load(out/'endpoints.npz', allow_pickle=False) as data:
        latents = data['latents']
        np.testing.assert_array_equal(data['labels'], labels)
    assert latents.dtype == np.float32 and latents.shape == (1000, 4, 32, 32)
    assert np.isfinite(latents).all()
    assert sha(out/'samples.npz') == summary['samples_sha256']
    with np.load(out/'samples.npz', allow_pickle=False) as data:
        pixels = data['arr_0']
    assert pixels.dtype == np.uint8 and pixels.shape == (1000, 256, 256, 3)
    records = {str(Path(record['path']).resolve()): record['sha256'] for record in summary['records']}
    expected = [out/f'batch{start:06d}.npz' for start in range(0, 1000, 16)]
    assert set(records) == {str(path.resolve()) for path in expected}
    for start, path in zip(range(0, 1000, 16), expected):
        stop = min(start+16, 1000)
        assert sha(path) == records[str(path.resolve())]
        with np.load(path, allow_pickle=False) as data:
            assert int(data['start']) == start and str(data['request_sha256']) == digest
            assert str(data['noise_sha256']) == array_sha(noise[start:stop])
            np.testing.assert_array_equal(data['labels'], labels[start:stop])
            np.testing.assert_array_equal(data['latents'], latents[start:stop])
            np.testing.assert_array_equal(data['pixels'], pixels[start:stop])
    metadata = dict(arm=arm, config=summary['config'], samples=1000,
                    baseline_request_sha256=digest, baseline_summary_sha256=sha(out/'summary.json'),
                    endpoints_sha256=sha(out/'endpoints.npz'), samples_path=str(out/'samples.npz'),
                    samples_sha256=sha(out/'samples.npz'), labels_sha256=array_sha(labels),
                    inputs_sha256=request['inputs_sha256'], noise_sha256=request['noise_sha256'],
                    original_sampling_seconds=summary['seconds'],
                    original_full_calls_per_output=summary['full_calls_per_output'])
    return latents, labels, pixels, metadata


@contextmanager
def precision(decoder=False):
    """FP32 refiner matches training; decoder matches frozen Runtime('sit_small')."""
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark)
    torch.set_float32_matmul_precision('high' if decoder else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = decoder
    torch.backends.cudnn.allow_tf32 = decoder
    torch.backends.cudnn.deterministic = not decoder
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
        torch.backends.cudnn.deterministic = previous[3]
        torch.backends.cudnn.benchmark = previous[4]


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def apply_refiner(model, z, labels, beta):
    # A zero beta is an exact no-op, including no unnecessary network evaluation.
    if beta == 0:
        return z.clone()
    with precision():
        corrected = model(z, labels)
        # Preserve the trained P exactly at beta=1 (avoid an extra FP32 subtract/add).
        result = corrected if beta == 1 else z + beta*(corrected-z)
    if result.shape != z.shape or not torch.isfinite(result).all():
        raise FloatingPointError('Invalid refiner output')
    return result


def decoder(args):
    from diffusers.models import AutoencoderKL
    from huggingface_hub import snapshot_download
    from experiments.sample_imagenet100_sit_fid import decode_latents_in_chunks, official_pixel_quantization
    snapshot = Path(snapshot_download('stabilityai/sd-vae-ft-mse', local_files_only=True))
    vae = AutoencoderKL.from_pretrained(str(snapshot), local_files_only=True, use_safetensors=True)
    vae = vae.to(device=args.device, dtype=torch.float32).eval().requires_grad_(False)

    def decode(z):
        with precision(decoder=True):
            return official_pixel_quantization(decode_latents_in_chunks(
                vae, z, scaling_factor=.18215, chunk_size=2))
    decode.metadata = dict(snapshot=str(snapshot), sha256={str(snapshot/name): sha(snapshot/name)
        for name in ('config.json', 'diffusion_pytorch_model.safetensors')})
    return decode


def load_model(checkpoint, device):
    from experiments.inverse_copy_refiner_20260913 import model as module
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    source = str(Path(module.__file__).resolve())
    assert payload['run_metadata']['source_sha256'][source] == sha(source), 'Trained model source changed'
    model = module.Refiner(**payload['model_config'])
    model.load_state_dict(payload['model'], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), step=payload['step'],
                    model_config=payload['model_config'], heldout=payload['heldout'],
                    parameters=sum(p.numel() for p in model.parameters()),
                    training_pairs_sha256=payload['run_metadata']['pairs_sha256'])
    return model, metadata


def check_zero(model, decode, latents, labels, device, expected=None):
    z = torch.from_numpy(latents[:8]).to(device)
    c = torch.from_numpy(labels[:8]).to(device)
    zero = apply_refiner(model, z, c, 0.)
    assert torch.equal(zero, z), 'beta=0 changed latent values'
    original, reproduced = decode(z), decode(zero)
    np.testing.assert_array_equal(original, reproduced, err_msg='beta=0 changed decoded pixels')
    if expected is not None:
        np.testing.assert_array_equal(original, expected[:8], err_msg='Frozen baseline decode mismatch')
    return dict(samples=len(z), latent_bitwise_equal=True, decoded_pixels_bitwise_equal=True,
                cached_pixels_bitwise_equal=True if expected is not None else None,
                zero_beta_refiner_calls=0)


def infer(args, model, decode, checkpoint_metadata):
    all_results = []
    for arm in ARMS:
        latents, labels, pixels, baseline = load_frozen_bank(args.baselines, arm)
        zero = check_zero(model, decode, latents, labels, args.device, expected=pixels)
        original = args.destination/f'{arm}_original'
        original.mkdir()
        validation_seconds = 0.
        for start in range(0, len(latents), args.batch):
            z = torch.from_numpy(latents[start:start+args.batch]).to(args.device)
            sync(args.device); began = time.perf_counter()
            reproduced = decode(z)
            sync(args.device); validation_seconds += time.perf_counter()-began
            np.testing.assert_array_equal(reproduced, pixels[start:start+args.batch],
                                          err_msg=f'Frozen pixels differ at {arm}, start={start}')
        gallery(pixels[:32], original/'grid.png', labels[:32])
        atomic(original/'summary.json', dict(**baseline, zero_beta_check=zero,
            all_cached_pixels_reproduced=True, decode_validation_seconds=validation_seconds))
        for beta in BETAS:
            out = args.destination/f'{arm}_beta{beta:g}'
            out.mkdir()
            images, endpoints, model_seconds, decode_seconds = [], [], 0., 0.
            for start in range(0, len(latents), args.batch):
                z = torch.from_numpy(latents[start:start+args.batch]).to(args.device)
                c = torch.from_numpy(labels[start:start+args.batch]).to(args.device)
                sync(args.device); began = time.perf_counter()
                corrected = apply_refiner(model, z, c, beta)
                sync(args.device); model_seconds += time.perf_counter()-began
                began = time.perf_counter()
                images.append(decode(corrected))
                sync(args.device); decode_seconds += time.perf_counter()-began
                endpoints.append(corrected.cpu().numpy())
            images, endpoints = np.concatenate(images), np.concatenate(endpoints)
            displacement_mse = np.square(endpoints.astype(np.float64)-latents).mean(axis=(1, 2, 3))
            npz(out/'samples.npz', arr_0=images)
            npz(out/'endpoints.npz', latents=endpoints, labels=labels,
                sample_indices=np.arange(len(labels)), displacement_mse=displacement_mse)
            save_images(images, out/'png')
            gallery(images[:32], out/'grid.png', labels[:32])
            comparison = np.stack((pixels[:8], images[:8]), axis=1).reshape(-1, 256, 256, 3)
            gallery(comparison, out/'comparison.png', ['original', f'beta={beta:g}']*8, columns=8)
            summary = dict(complete=True, baseline=baseline, beta=beta, samples=len(images),
                checkpoint=checkpoint_metadata, original_and_zero_beta_checks=zero,
                original_full_decode_exact=True, refiner_calls_per_output=1,
                extra_generator_calls=0, refiner_seconds=model_seconds, decode_seconds=decode_seconds,
                displacement_mse_mean=float(displacement_mse.mean()),
                samples_sha256=sha(out/'samples.npz'), endpoints_sha256=sha(out/'endpoints.npz'),
                interpretation='Paired reuse of the previous 1K search bank; no independent confirmation or FID claim.')
            atomic(out/'summary.json', summary)
            all_results.append(dict(arm=arm, beta=beta, directory=str(out)))
            print(json.dumps(all_results[-1]), flush=True)
    return dict(arms=all_results)


def rounds(args, model, decode, checkpoint_metadata):
    assert sha(args.pairs) == checkpoint_metadata['training_pairs_sha256']
    with np.load(args.pairs, allow_pickle=False) as data:
        ids = data['source_ids'] if 'source_ids' in data else data['source_id']
        split = data['split']
        assert len(set(ids.tolist())) == len(ids)
        assert not (set(ids[split == 0].tolist()) & set(ids[split == 1].tolist()))
        selected = np.flatnonzero(split == 1)[:8]
        assert len(selected) == 8
        real = data['clean'][selected], data['labels'][selected], ids[selected]
    cfg, cfg_labels, cfg_pixels, baseline = load_frozen_bank(args.baselines, ARMS[0])
    groups = [('real_holdout', *real, None),
              ('cfg_generated', cfg[:8], cfg_labels[:8], np.arange(8), cfg_pixels[:8])]
    completed = []
    for name, initial, labels, identities, expected in groups:
        zero = check_zero(model, decode, initial, labels, args.device, expected=expected)
        for beta in BETAS:
            out = args.destination/f'{name}_beta{beta:g}'
            out.mkdir()
            z = torch.from_numpy(initial).to(args.device)
            c = torch.from_numpy(labels).to(args.device)
            frames, stats = [], []
            for index in range(6):
                previous = z
                sync(args.device); began = time.perf_counter()
                if index:
                    z = apply_refiner(model, z, c, beta)
                sync(args.device); model_seconds = time.perf_counter()-began if index else 0.
                began = time.perf_counter()
                pixels = decode(z)
                sync(args.device); decode_seconds = time.perf_counter()-began
                latent = z.cpu().numpy()
                frame = out/f'round{index:02d}'
                frame.mkdir()
                npz(frame/'state.npz', latents=latent, pixels=pixels, labels=labels,
                    source_ids=identities, round_index=index, beta=beta)
                save_images(pixels, frame/'png')
                gallery(pixels, frame/'grid.png', labels)
                frames.append(pixels)
                stats.append(dict(round=index, refiner_seconds=model_seconds, decode_seconds=decode_seconds,
                    step_mse=float((z-previous).double().square().mean()),
                    initial_mse=float(np.square(latent.astype(np.float64)-initial).mean()),
                    latent_sha256=array_sha(latent), pixels_sha256=array_sha(pixels)))
            # One row per source, six columns for R0..R5; no image truncation.
            grid = np.stack(frames, axis=1).reshape(-1, 256, 256, 3)
            captions = [f'ID={identity} R{r}' for identity in identities for r in range(6)]
            gallery(grid, out/'rounds.png', captions, columns=6)
            summary = dict(complete=True, source=name, samples=8, rounds=5, beta=beta,
                checkpoint=checkpoint_metadata, baseline=baseline if expected is not None else None,
                pairs_sha256=sha(args.pairs) if expected is None else None,
                zero_beta_check=zero, per_round=stats, refiner_calls_per_source=5,
                extra_generator_calls=0, fresh_noise=False, decoded_images_reencoded=False,
                interpretation='Iterated latent P_beta; R0 is input and each R1..R5 is separately applied/saved. '
                    'Real holdout also selected the checkpoint; these are diagnostic images, not an independent test.')
            atomic(out/'summary.json', summary)
            completed.append(str(out))
            print(json.dumps(dict(completed=str(out))), flush=True)
    return dict(directories=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['infer', 'rounds'])
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'training/best.pt')
    parser.add_argument('--pairs', type=Path, default=ROOT/'pairs.npz')
    parser.add_argument('--baselines', type=Path, default=BASELINES)
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    if args.batch < 2 or args.batch % 2:
        raise ValueError('Use a positive even batch to preserve baseline VAE chunk pairing')
    args.device = torch.device(args.device)
    args.destination = args.output/args.phase
    if args.destination.exists():
        raise FileExistsError(f'Preserving existing output: {args.destination}')
    torch.set_num_threads(2)
    model, metadata = load_model(args.checkpoint, args.device)
    decode = decoder(args)
    args.destination.mkdir(parents=True)
    import experiments.sample_imagenet100_sit_fid as decoding_source
    sources = [Path(__file__).resolve(), Path(__file__).with_name('model.py').resolve(),
               Path(decoding_source.__file__).resolve()]
    request = dict(phase=args.phase, checkpoint=metadata, beta=list(BETAS), batch=args.batch,
        device=str(args.device), source_sha256={str(p): sha(p) for p in sources},
        vae='stabilityai/sd-vae-ft-mse; local_files_only=True; FP32', vae_scale=.18215,
        vae_chunk_size=2, quantization='clamp(127.5*x+128,0,255), NHWC uint8 truncation',
        refiner_precision='FP32, TF32 disabled, cudnn deterministic',
        decoder_precision='FP32, TF32 enabled/high, cudnn deterministic=False, benchmark=False',
        vae_files=decode.metadata, torch_version=str(torch.__version__), numpy_version=np.__version__,
        baseline_root=str(args.baselines), generator_loaded=False, fresh_noise=False)
    atomic(args.destination/'request.json', request)
    began = time.perf_counter()
    with torch.inference_mode():
        result = (infer if args.phase == 'infer' else rounds)(args, model, decode, metadata)
    assert sha(args.checkpoint) == metadata['checkpoint_sha256'], 'Checkpoint changed during evaluation'
    atomic(args.destination/'summary.json', dict(complete=True, phase=args.phase,
        wall_seconds=time.perf_counter()-began, **result))


if __name__ == '__main__':
    main()
