"""Read-only raw-result audit and final presentation; never changes samplers."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from experiments.cfg_invariants_20260913 import image_screen as main

HERE=Path(__file__).resolve()
ROOT=main.ROOT


def read(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def csv_text(rows):
    fields=['arm','K','fid','sfid','inception_score','target_top1','target_top5','target_probability',
        'mean_parallel_extra','mean_extra_to_native_norm','mean_off_gap_fraction',
        'latent_rms','latent_variance','saturation_fraction','sample_seconds','classifier_seconds',
        'fid_seconds','full_calls_per_output','proxy_extra_vae_decodes_per_output','result_directory']
    out=io.StringIO();writer=csv.DictWriter(out,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return out.getvalue()


def load_rows(root):
    request=read(root/'request.json');rows=[];inputs={}
    for arm in request['arms']:
        path=root/arm
        summary=read(path/'summary.json');classifier=read(path/'classifier.json');fid=read(path/'fid.json')
        assert summary['n']==400 and fid['sample_count']==400
        assert summary['full_calls_per_output']==224
        assert summary['request_sha256']==main.c.sha(root/'request.json')
        assert summary['samples_sha256']==main.c.sha(path/'samples.npz')
        row=dict(summary)
        row.pop('seconds',None)
        row.update({k:v for k,v in classifier.items() if k!='seconds'})
        row.update({k:v for k,v in fid.items() if k!='seconds'})
        row.update(K=request['ctrl_K'] if arm.startswith('ctrl_') or arm=='fixedset_projection' else None,
            sample_seconds=summary['seconds'],classifier_seconds=classifier.get('seconds'),
            fid_seconds=fid.get('seconds'),result_directory=str(path))
        rows.append(row)
        for name in ('summary','classifier','fid'):
            source=path/(name+'.json');inputs[str(source)]=main.c.sha(source)
    (root/'results.csv').write_text(csv_text(rows))
    return rows,inputs


def add_note(path,note):
    path=Path(path)
    text=path.read_text().split('\n<!-- FINAL_RESULT_AUDIT -->')[0]
    path.write_text(text+'\n<!-- FINAL_RESULT_AUDIT -->\n'+note)


def main_run():
    low=ROOT/'small_k_0p03'
    assert read(ROOT/'status.json')['complete'] and read(low/'status.json')['complete']
    rows,inputs=load_rows(ROOT);low_rows,low_inputs=load_rows(low);inputs.update(low_inputs)
    all_rows=rows+low_rows
    (ROOT/'all_image_results.csv').write_text(csv_text(all_rows))
    calibration=read(ROOT/'calibration.json')
    costs=dict(evaluated_images=sum(r['n'] for r in all_rows),
        calibration_generated_paths=calibration['generated_paths'],
        total_generated_paths=sum(r['n'] for r in all_rows)+calibration['generated_paths'],
        sample_gpu_seconds_sum=sum(r['sample_seconds'] for r in all_rows),
        calibration_gpu_seconds_sum=sum(row['seconds'] for row in calibration['costs']),
        endpoint_classifier_seconds_sum=sum(r['classifier_seconds'] for r in all_rows),
        fid_seconds=None,fid_seconds_note='The original ADM evaluator did not record duration; unavailable, not replaced with classifier time.',
        model_branch_evaluations=sum(r['n']*r['full_calls_per_output'] for r in all_rows)+calibration['generated_paths']*224,
        elapsed_wall_time_is_not_sum_of_gpu_seconds=True,
        sampling_seconds_include_final_decoding_and_40_gallery_decodes_per_arm=True,
        evidence_feedback_decoding_and_classification_already_included_in_its_sampling_seconds=True)
    write_json(ROOT/'evaluation_costs.json',costs)
    cost_lines=['', '最终复核：', '',
       'CSV 将 sample_seconds、classifier_seconds、fid_seconds 分开。采样时间包含最终解码及每组40张轨迹可视化解码；证据反馈的3次额外读取已包含在该组采样时间中。ADM 工具没有记录耗时，fid_seconds 留空。',
       f'正式生成并评估 {costs["evaluated_images"]} 张；另有 {costs["calibration_generated_paths"]} 条校准轨迹。采样设备秒合计 {costs["sample_gpu_seconds_sum"]:.2f}s、校准 {costs["calibration_gpu_seconds_sum"]:.2f}s；这是并行各配置时间之和，非墙钟耗时。',
       'fixedset_projection 与 ctrl_direction_matched 是作者控制器输出上的包装，保留作者原始 modified-gap proposal 为历史；ctrl_gap_project 将投影后 gap 提交为历史。ctrl_physical 保存 raw gap，且使用物理时间导数。',
       'velocity/clean/epsilon 单位运输的等价核查在可逆中间时刻成立；epsilon 的 t=0 和 clean 的 t=1 为奇异端点，不要求逆变换。',
       '', '|arm|K|sample s|classifier s|', '|---|---:|---:|---:|']
    for row in all_rows:
        cost_lines.append(f'|{row["arm"]}|{row["K"] if row["K"] is not None else "—"}|{row["sample_seconds"]:.2f}|{row["classifier_seconds"]:.2f}|')
    cost_lines+=['',f'[16组总CSV](<{ROOT}/all_image_results.csv>)；[实际预算](<{ROOT}/evaluation_costs.json>)；[小K三组](<{low}/image_results.md>)。','']
    note='\n'.join(cost_lines)
    add_note(main.REPORT,note);add_note(ROOT/'image_results.md',note)
    low_note='\nCSV 的 sample_seconds 与 classifier_seconds 分开，FID 耗时未记录，因此 fid_seconds 留空；采样成本逐项取原始 summary.json，未使用分类器耗时替代。\n'
    add_note(HERE.with_name('small_k_results.md'),low_note);add_note(low/'image_results.md',low_note)
    write_json(ROOT/'final_audit.json',dict(complete=True,rows=len(all_rows),paired_noise_shared=True,
        parent_bank_sha256=main.c.sha(ROOT/'screen_bank.npz'),small_k_bank_sha256=main.c.sha(low/'screen_bank.npz'),
        source_sha256=main.c.sha(HERE),raw_inputs=inputs,costs=costs))
    assert main.c.sha(ROOT/'screen_bank.npz')==main.c.sha(low/'screen_bank.npz')
    print(json.dumps(dict(complete=True,rows=len(all_rows),costs=costs)),flush=True)


if __name__=='__main__':main_run()
