"""CPU-only feedback drift measurements and fixed-index sample galleries.

Pixel MSE uses the saved uint8 values (range 0..255), and PSNR uses peak 255.
These are fidelity/drift diagnostics, not image-quality scores. Latent distances
use the model's native latent coordinates without VAE scaling or normalization.
Galleries always show the earliest sample indices; no score selects an image.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _array(value, key):
    """Accept an ndarray, an .npy path, or a raw batch .npz path."""
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix == '.npy':
            return np.load(path, mmap_mode='r', allow_pickle=False)
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise ValueError(f'{path}: expected array {key!r}')
            return np.array(data[key])
    return np.asarray(value)


def _flatten(value):
    return np.asarray(value, dtype=np.float64).reshape(len(value), -1)


def _cosine(current, reference):
    """Finite extension: two zero vectors have cosine 1; exactly one has 0."""
    a2 = np.einsum('ij,ij->i', current, current)
    b2 = np.einsum('ij,ij->i', reference, reference)
    product = np.sqrt(a2 * b2)
    value = np.divide(np.einsum('ij,ij->i', current, reference), product,
                      out=np.zeros(len(current), dtype=np.float64), where=product > 0)
    value[(a2 == 0) & (b2 == 0)] = 1.
    return value.clip(-1., 1.)


def compare_round(current_latents, current_pixels, previous_latents,
                  previous_pixels, initial_latents, initial_pixels):
    """Return float64, per-image drift arrays in matching input order.

    Arguments may independently be arrays or .npy/.npz paths. For .npz paths,
    pixels are read from ``arr_0`` and latent states from ``latents``. Exact
    pixel identity produces +inf PSNR, which is retained in the array output;
    use ``summarize_comparison`` before writing strict JSON. Cosine follows the
    explicitly documented zero-vector convention in ``_cosine``.
    """
    latents = [_array(value, 'latents') for value in
               (current_latents, previous_latents, initial_latents)]
    pixels = [_array(value, 'arr_0') for value in
              (current_pixels, previous_pixels, initial_pixels)]
    n = len(latents[0])
    if n == 0 or latents[0].ndim < 2:
        raise ValueError('A nonempty batch of latent arrays is required')
    for value in latents:
        if value.shape != latents[0].shape or not np.isfinite(value).all():
            raise ValueError('Latent batches must have equal shapes and finite values')
    for value in pixels:
        if value.shape != pixels[0].shape or len(value) != n or value.dtype != np.uint8:
            raise ValueError('Pixel batches must be equally shaped uint8 arrays matching the latents')
        if value.ndim != 4 or value.shape[-1] != 3:
            raise ValueError('Pixel batches must be NHWC RGB images')
    current = _flatten(latents[0])
    current_images = _flatten(pixels[0])
    result = dict(latent_rms_current=np.sqrt(np.mean(current * current, axis=1)))
    for index, name in enumerate(('previous', 'initial'), start=1):
        reference = _flatten(latents[index])
        pixel_delta = current_images - _flatten(pixels[index])
        mse = np.mean(pixel_delta * pixel_delta, axis=1)
        psnr = np.full(n, np.inf, dtype=np.float64)
        positive = mse > 0
        psnr[positive] = 10. * np.log10(255. ** 2 / mse[positive])
        latent_delta = current - reference
        result.update({
            f'pixel_mse_{name}': mse,
            f'pixel_psnr_{name}_db': psnr,
            f'latent_delta_rms_{name}': np.sqrt(np.mean(latent_delta * latent_delta, axis=1)),
            f'latent_cosine_{name}': _cosine(current, reference),
        })
    return result


def summarize_comparison(metrics):
    """JSON-safe per-column summaries; preserve infinity counts explicitly.

    The mean is null if any entry is nonfinite. Quartiles are order statistics
    (nearest observed values), avoiding interpolation between finite and
    infinite PSNR. ``finite_mean`` is labeled separately so exact identities
    cannot be silently discarded. ``psnr_from_mean_mse_db`` summarizes each
    complete image batch; null with zero MSE denotes perfect identity.
    """
    columns = {}
    sizes = set()
    for key, raw in metrics.items():
        value = np.asarray(raw, dtype=np.float64)
        if value.ndim != 1 or not len(value):
            raise ValueError(f'{key}: expected a nonempty per-image vector')
        sizes.add(len(value))
        finite = np.isfinite(value)
        # Direct order statistics work even when some PSNR values are +inf.
        ordered = np.sort(value)
        selected = ordered[np.rint(np.array((0., .25, .5, .75, 1.)) * (len(value)-1)).astype(int)]
        safe = lambda x: float(x) if np.isfinite(x) else None
        columns[key] = dict(count=len(value), finite_count=int(finite.sum()),
            positive_infinity_count=int(np.isposinf(value).sum()),
            negative_infinity_count=int(np.isneginf(value).sum()),
            nan_count=int(np.isnan(value).sum()),
            mean=float(value.mean()) if finite.all() else None,
            finite_mean=float(value[finite].mean()) if finite.any() else None,
            **{label: safe(number) for label, number in
               zip(('minimum', 'p25', 'p50', 'p75', 'maximum'), selected)})
    if len(sizes) != 1:
        raise ValueError('Every drift column must cover the same image count')
    aggregate_psnr = {}
    for name in ('previous', 'initial'):
        key = f'pixel_mse_{name}'
        if key in metrics:
            value = np.asarray(metrics[key], dtype=np.float64)
            mean = float(value.mean())
            aggregate_psnr[name] = dict(
                psnr_from_mean_mse_db=float(10*np.log10(255.**2/mean)) if mean > 0 else None,
                exact_pixel_identity_count=int((value == 0).sum()),
                all_images_identical_to_reference=bool((value == 0).all()))
    result = dict(samples=sizes.pop(), pixel_mse_units='squared uint8 intensity (0..255)',
        pixel_psnr_peak=255, latent_units='native model latent',
        zero_vector_cosine='both zero: 1; exactly one zero: 0',
        interpretation='Image fidelity/drift diagnostics; not image-quality scores',
        quantile_method='nearest observed order statistic', columns=columns,
        aggregate_psnr=aggregate_psnr)
    json.dumps(result, allow_nan=False)
    return result


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            value.update(part)
    return value.hexdigest()


def _batch_key(path):
    match = re.fullmatch(r'batch([0-9]+)\.npz', Path(path).name)
    if match is None:
        raise ValueError(f'Unrecognized raw batch filename: {path}')
    return int(match.group(1))


def _first_images(batch_paths, count):
    if count < 1:
        raise ValueError('Gallery image count must be positive')
    paths = sorted(map(Path, batch_paths), key=_batch_key)
    if not paths or len({_batch_key(p) for p in paths}) != len(paths):
        raise ValueError('Gallery requires raw batches with unique start indices')
    images, labels, sources, indices = [], [], [], []
    expected = 0
    for path in paths:
        if expected >= count:
            break
        with np.load(path, allow_pickle=False) as data:
            start = int(data['start'])
            if start != _batch_key(path) or start != expected:
                raise ValueError(f'{path}: missing, reordered, or mismatched leading sample indices')
            batch = data['arr_0']
            target = data['labels']
            if batch.ndim != 4 or batch.shape[-1] != 3 or batch.dtype != np.uint8:
                raise ValueError(f'{path}: expected uint8 NHWC RGB images')
            if len(batch) == 0 or len(batch) != len(target):
                raise ValueError(f'{path}: invalid batch coverage or labels')
            take = min(count-expected, len(batch))
            images.extend(np.array(batch[:take]))
            labels.extend(int(label) for label in target[:take])
            indices.extend(range(start, start+take))
            expected += take
        sources.append(dict(path=str(path), sha256=_sha(path), first_index=start, selected_count=take))
    if expected != count:
        raise ValueError(f'Only {expected} leading samples are available; requested {count}')
    return np.stack(images), labels, indices, sources


def _write_image(canvas, output_path, metadata):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name+'.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        canvas.save(temporary, format='PNG')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(metadata, output=str(path), output_sha256=_sha(path),
                  width=canvas.width, height=canvas.height,
                  selection='Earliest sample indices, fixed across rounds; no score-based selection',
                  interpretation='Visual fidelity/drift inspection; not an independent quality assessment')
    sidecar = path.with_suffix(path.suffix+'.json')
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.'+sidecar.name+'.',
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    try:
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def render_round_gallery(batch_paths, output_path, count=16, columns=4):
    """Save a fixed first-16 grid from raw batches; never load all round images."""
    if columns < 1:
        raise ValueError('Gallery columns must be positive')
    pixels, labels, indices, sources = _first_images(batch_paths, count)
    height, width = pixels.shape[1:3]
    gap, label_height, heading_height = 8, 20, 28
    rows = (count+columns-1)//columns
    canvas = Image.new('RGB', (gap+columns*(width+gap), heading_height+gap+rows*(height+label_height+gap)), 'white')
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    draw.text((gap, 8), f'Fixed first {count} samples | order: sample index', fill='black', font=font)
    for offset, (pixels_i, label, index) in enumerate(zip(pixels, labels, indices)):
        x = gap+(offset % columns)*(width+gap)
        y = heading_height+gap+(offset//columns)*(height+label_height+gap)
        draw.text((x, y), f'ID {index:04d} | label {label}', fill='black', font=font)
        canvas.paste(Image.fromarray(pixels_i), (x, y+label_height))
    return _write_image(canvas, output_path, dict(kind='per_round_grid', samples=count,
        columns=columns, selected_indices=indices, labels=labels, sources=sources,
        sample_pixels_resized=False))


def render_round_contact_sheet(round_batch_paths, output_path, count=8):
    """Save matching first-eight images as rows; round0..latest are columns.

    ``round_batch_paths`` maps integer round indices to lists of raw batch paths.
    A sequence of lists is also accepted and interpreted as rounds 0..N-1.
    The image is updated after each completed round, retaining every earlier
    round at the same sample indices and their native saved pixel resolution.
    """
    if isinstance(round_batch_paths, Mapping):
        batches = {int(key): value for key, value in round_batch_paths.items()}
        if len(batches) != len(round_batch_paths):
            raise ValueError('Duplicate round indices after integer conversion')
    else:
        batches = dict(enumerate(round_batch_paths))
    rounds = sorted(batches)
    if not rounds or rounds != list(range(rounds[-1]+1)):
        raise ValueError('Contact sheet rounds must be contiguous from round0')
    images, sources = {}, {}
    expected_labels = expected_indices = expected_shape = None
    for round_index in rounds:
        pixels, labels, indices, records = _first_images(batches[round_index], count)
        if expected_labels is None:
            expected_labels, expected_indices, expected_shape = labels, indices, pixels.shape
        elif labels != expected_labels or indices != expected_indices or pixels.shape != expected_shape:
            raise ValueError('Every displayed round must use matching sample indices, labels, and resolution')
        images[round_index], sources[round_index] = pixels, records
    height, width = expected_shape[1:3]
    gap, label_width, heading_height = 8, 90, 30
    canvas = Image.new('RGB', (label_width+gap+len(rounds)*(width+gap), heading_height+gap+count*(height+gap)), 'white')
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for column, round_index in enumerate(rounds):
        x = label_width+gap+column*(width+gap)
        draw.text((x, 10), f'Round {round_index}', fill='black', font=font)
        for row, pixels_i in enumerate(images[round_index]):
            y = heading_height+gap+row*(height+gap)
            canvas.paste(Image.fromarray(pixels_i), (x, y))
    for row, (index, label) in enumerate(zip(expected_indices, expected_labels)):
        y = heading_height+gap+row*(height+gap)
        draw.text((gap, y+8), f'ID {index:04d}\nlabel {label}', fill='black', font=font)
    return _write_image(canvas, output_path, dict(kind='matching_round_contact_sheet',
        samples=count, rounds=rounds, selected_indices=expected_indices, labels=expected_labels,
        sources=sources, sample_pixels_resized=False))
