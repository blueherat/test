"""Verify completed artifacts and collate metrics without overwriting costs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):digest.update(part)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def main():
    verified={}
    rows=[]
    reference=None
    with np.load(ROOT/'images/screen_bank.npz') as bank:
        noise=bank['noise']
        labels=bank['labels']
    assert len(labels)==400
    np.testing.assert_array_equal(np.bincount(labels,minlength=100),np.full(100,4))
    for family,folder in [('main',ROOT/'images'),('K_0p03',ROOT/'images/small_k_0p03')]:
        status=read(folder/'status.json')
        assert status['complete'],folder
        request=read(folder/'request.json')
        request_hash=sha(folder/'request.json')
        for category in ('sources','assets','inputs'):
            for path,expected in request[category].items():
                if path not in verified:verified[path]=sha(path)
                assert verified[path]==expected,(category,path)
        with np.load(folder/'screen_bank.npz') as bank:
            np.testing.assert_array_equal(bank['noise'],noise)
            np.testing.assert_array_equal(bank['labels'],labels)
        for arm in request['arms']:
            path=folder/arm
            summary=read(path/'summary.json')
            fid=read(path/'fid.json')
            classifier=read(path/'classifier.json')
            assert summary['complete'] and summary['n']==fid['sample_count']==400
            assert summary['request_sha256']==request_hash
            assert sha(path/'samples.npz')==summary['samples_sha256']
            assert Path(fid['samples']).resolve()==(path/'samples.npz').resolve()
            if reference is None:reference=fid['reference']
            assert fid['reference']==reference
            with np.load(path/'samples.npz') as data:
                pixels=data['arr_0']
                assert pixels.shape==(400,256,256,3) and pixels.dtype==np.uint8
            with np.load(path/'diagnostics.npz') as data:
                np.testing.assert_array_equal(data['labels'],labels)
                assert data['latents'].shape==(400,4,32,32)
                assert np.isfinite(data['latents']).all() and np.isfinite(data['trace']).all()
            with np.load(path/'trajectory_5_times.npz') as data:
                np.testing.assert_array_equal(data['times'],[0.,.25,.5,.75,1.])
                assert data['states'].shape[:2]==(8,5)
                assert data['pixels'].shape==(8,5,256,256,3)
            assert summary['full_calls_per_output']==224
            rows.append(dict(family=family,arm=arm,n=400,fid400=fid['fid'],sfid400=fid['sfid'],
                convnext_top1=classifier['target_top1'],convnext_target_probability=classifier['target_probability'],
                sample_seconds=summary['seconds'],classifier_seconds=classifier['seconds'],
                fid_seconds=None,mean_extra=summary['mean_parallel_extra'],
                extra_norm_ratio=summary['mean_extra_to_native_norm'],off_gap_fraction=summary['mean_off_gap_fraction'],
                proxy_reads=summary['proxy_extra_vae_decodes_per_output'],path=str(path)))
    assert len(rows)==16
    with (ROOT/'all_image_results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (ROOT/'oracle/metrics.csv').open() as stream:oracle_rows=list(csv.DictReader(stream))
    assert len(oracle_rows)==40
    diagnostic=read(ROOT/'diagnostic/summary.json')
    metadata=read(ROOT/'diagnostic/metadata.json')
    assert diagnostic['complete'] and metadata['samples']==128 and metadata['data']['real_data']
    assert len(diagnostic['noise_strata'])==5
    with np.load(ROOT/'diagnostic/results.npz') as data:
        assert all(np.isfinite(data[name]).all() for name in data.files)
    independent=read(ROOT/'independent_classifier/summary.json')
    assert independent['complete'] and independent['cpu_only']
    for arm,value in independent['arms'].items():
        assert value['input_sha256']==read(ROOT/'images'/arm/'summary.json')['samples_sha256']
    contracts=read(ROOT/'contracts/summary.json')
    assert contracts['complete'] and not contracts['quality_experiment']
    result=dict(passed=True,image_configurations=len(rows),primary_images=sum(r['n'] for r in rows),
        common_noise_and_labels=True,balanced_classes=100,images_per_class_per_arm=4,
        all_image_fid_sample_counts=400,all_image_trajectories_saved_at_five_times=True,
        oracle_rows=40,real_diagnostic_images=128,real_diagnostic_time_strata=5,
        independent_classifier_arms=len(independent['arms']),
        sample_seconds_summed_over_devices=sum(r['sample_seconds'] for r in rows),
        summed_sample_seconds_is_not_elapsed_wall_time=True,
        fid_seconds_recorded=False,sources_assets_inputs_verified=len(verified),
        image_table_sha256=sha(ROOT/'all_image_results.csv'),
        script_sha256=sha(Path(__file__)),quality_success_asserted=False)
    (ROOT/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
