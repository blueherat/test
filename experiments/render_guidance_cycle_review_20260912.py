"""Render committed results without changing any frozen experiment source."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

WORK=Path('/home/zhoushunyu/eqvae')
BASE=Path('/home/zhoushunyu/data/eqvae/experiments')
REPORT=WORK/'docs/GUIDANCE_RESEARCH_CYCLES_20260912_ZH.md'
STAGES=(('cycle01','sit_measure_guidance_20260912','measure_screen_1k'),
    ('cycle02','sit_posterior_guidance_20260912','routing_screen_1k'),
    ('cycle03','sit_internal_coarse_20260912','internal_screen_1k'),
    ('cycle04','sit_prefix_coarse_20260912','prefix_screen_1k'),
    ('cycle05','sit_prefix_confirmation_20260912','prefix_confirm_5k'),
    ('cycle06','sit_prefix_incumbent_20260912','prefix_incumbent_5k'))


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    lines=[f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}。FID越低越好；前四轮为同bank 1K筛选，第五轮为独立5K，第六轮使用相同5K补充一个旧对照；不可将1K与5K直接相减。\n']
    evidence=[]
    for cycle,root_name,stage in STAGES:
        root=BASE/root_name;base=root/stage
        if not (base/'request.json').exists():
            state=read(root/'status.json') if (root/'status.json').exists() else {}
            lines.append(f'**{cycle}：{state.get("phase","尚未开始GPU生成")}。**\n')
            continue
        request=read(base/'request.json')
        rows=read(base/'results.json') if (base/'results.json').exists() else []
        for row in rows:
            commit=read(base/row['arm']/'commit.json')
            assert commit['request_sha256']==sha(base/'request.json')
            assert commit['files']['result.json']==sha(base/row['arm']/'result.json')
            assert row['fid']==read(base/row['arm']/'result.json')['fid']
        state=read(base/'status.json')
        audit=read(base/'analysis_audit.json') if (base/'analysis_audit.json').exists() else {}
        lines.append(f'**{cycle}：{len(rows)}/{len(request["configs"])}组，状态 {state["phase"]}；'
            f'最终审计 {"通过" if audit.get("passed") else "待完成"}。**\n')
        groups=defaultdict(list)
        for row in rows:groups[row['family']].append(row)
        lines+=['|方法|已提交/计划|当前最低FID|该点强度|Full/prefix调用|采样及解码GPU秒|',
            '|---|--:|--:|--:|--:|--:|']
        for family,pool in groups.items():
            good=[r for r in pool if r['complete']]
            if not good:continue
            best=min(good,key=lambda r:r['fid'])
            total=sum(c['family']==family for c in request['configs'])
            lines.append(f'|{family}|{len(pool)}/{total}|{best["fid"]:.4f}|{best["strength"]:g}|'
                f'{best["full_calls_per_image"]:.0f}/{best["prefix_calls_per_image"]:.0f}|{best["sum_batch_gpu_seconds"]:.2f}|')
        lines.append('')
        evidence.append(dict(cycle=cycle,root=str(root),request_sha256=sha(base/'request.json'),
            results_sha256=sha(base/'results.json'),committed=len(rows),planned=len(request['configs']),
            bank=request['bank'],samples=request['samples'],independent_confirmation=request.get('independent_confirmation',False),
            same_bank_incumbent_supplement=request.get('same_bank_incumbent_supplement',False),
            final_audit_passed=bool(audit.get('passed'))))
    screen=[e for e in evidence if not e['independent_confirmation']]
    if len(screen)>1:
        assert len({e['bank']['noise_sha256'] for e in screen})==1
        assert len({e['bank']['label_sha256'] for e in screen})==1
    for e in evidence:
        if not e['independent_confirmation']:continue
        assert e['samples']==5000
        assert all(e['bank']['noise_sha256']!=r['bank']['noise_sha256'] for r in screen)
    supplements=[e for e in evidence if e['same_bank_incumbent_supplement']]
    for e in supplements:
        peers=[r for r in evidence if r['independent_confirmation'] and not r['same_bank_incumbent_supplement']]
        assert any(e['bank']['noise_sha256']==r['bank']['noise_sha256'] and e['bank']['label_sha256']==r['bank']['label_sha256'] for r in peers)
    output=REPORT.read_text();before,rest=output.split('<!-- RESULTS_BEGIN -->')
    _,after=rest.split('<!-- RESULTS_END -->')
    REPORT.write_text(before+'<!-- RESULTS_BEGIN -->\n'+'\n'.join(lines)+'\n<!-- RESULTS_END -->'+after)
    folder=WORK/'docs/data/guidance_research_cycles_20260912';folder.mkdir(parents=True,exist_ok=True)
    (folder/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in e.items() if k not in ('bank',)} for e in evidence],indent=2))


if __name__=='__main__':run()
