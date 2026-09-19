"""Publish a clearly provisional, source-backed table as complete5K metrics arrive."""
import datetime,json,time
from pathlib import Path
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments')
WORK=Path('/home/zhoushunyu/eqvae')
OUT=WORK/'docs/GUIDANCE_5K_LIVE_20260914_ZH.md'
HEAD=ROOT/'raev2_shallow_ig_20260914'
SG=ROOT/'ig_sg_5k_20260914'
KINDS=('incumbent','native4','mlp4','mlp8')
NAMES=dict(incumbent='原生第8层',native4='第4层DDT 50K',mlp4='第4层MLP 50K',mlp8='第8层MLP 50K')


def read(path):return json.loads(path.read_text())


def cell(out,batch=16):
    if (out/'skipped.json').exists():return '用户取消（不计FID）'
    if (out/'invalid.json').exists():return '数值无效'
    if (out/'metrics.json').exists():
        metric=read(out/'metrics.json');metric=metric[0] if isinstance(metric,list) else metric
        summary=read(out/'summary.json');assert summary.get('samples',summary.get('primary_samples'))==5000
        return f"**{metric['fid']:.4f}**"
    count=sum(min(batch,5000-int(p.stem[5:])) for p in out.glob('batch*.npz'))
    return f'{count}/5000' if count else '待采样'


def write():
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8')
    paused=(SG/'pause_requested.json').exists()
    lines=['# IG 与 SG：全5K扫描实时记录','',f'更新于 {now}。','',
        '保留的粗扫与细化点均使用完整5K；1K排名不参与候选淘汰。**用户已取消独立5K验证，最终直接汇总调参结果。当前表不能代替完成扫描后的结论。**','',
        '单元格粗体数字为已完成整组5K的FID（越低越好）；进度表示尚未计算完整FID。所有当前指标均来自对应的原始metrics.json。','',
        '## RAEv2 后训练头','',
        '三个后训练头均已完成总计50K，强主干冻结。原粗扫已停止，第4层后续搜索放弃。用户要求第8层MLP只补完w=1.35；保留w=1.25、1.30、1.35三组完整5K，其余细扫全部取消。第8层MLP w=1.8已完成（FID=7.3673），w=2已停止。默认原生w=1.78复用同一bank已有5K作对照（FID=6.9208）。','',
        '| 外推w | 原生第8层 | 第4层DDT 50K | 第4层MLP 50K | 第8层MLP 50K |','|---|---:|---:|---:|---:|']
    if paused:lines[4:4]=['**SG／log-SG 扫描已按用户要求暂停。已保存批次、原始请求和队列保留，等待明确恢复指令。**','']
    for w in tuple(i/100 for i in range(100,201,20)):
        values=[]
        for kind in KINDS:
            arm=f'{kind if w!=1 else "incumbent"}_w{w:.4f}'.replace('.','p')
            if kind=='incumbent' and w==1.78:
                out=ROOT/'raev2_context_5k_20260914/raev2/confirm_5000/native_base';values.append(cell(out)+'†')
            else:values.append(cell(HEAD/'grid_5k'/arm))
        lines.append('| '+f'{w:g}'+' | '+' | '.join(values)+' |')
    for phase,label in [('mlp8_refine_5k','仅第8层MLP：5K细扫')]:
        path=HEAD/phase/'request.json'
        if path.exists():
            lines+=['',f'### {label}','','| 设置 | w | 5K FID或进度 |','|---|---:|---:|']
            for cfg in read(path)['configs']:lines.append(f"| {NAMES[cfg['kind']]} | {cfg['w']:.4f} | {cell(path.parent/cfg['arm'])} |")
    lines+=['','[当前执行修订](RAEV2_MLP8_FINISH_135_20260914_ZH.md) · [头实验结果页](RAEV2_SHALLOW_IG_5K_SELECTION_RESULTS_20260914_ZH.md)。','',
        '## IG + SG','',
        '沿用此前冻结IG；RAEv2为原生第8层、w=1.78。纸面原版SG与log-score版本均扫描SG外推强度，保留粗扫／细化点及同一bank成本对照，每点5K；取消独立验证。三个模型的绝对FID不跨模型比较。','']
    started=False
    for phase,label in [('grid_tune','5K粗扫'),('grid_refine','5K细化'),('cost_5k','同一bank的NFE成本对照')]:
        for name in ('sit_small','jit','raev2'):
            path=SG/phase/name/'request.json'
            if not path.exists():continue
            req=read(path)
            if not any(any((path.parent/cfg['arm']).glob('batch*.npz')) for cfg in req['configs']):continue
            started=True;lines += [f'### {name}：{label}','','| 设置 | w_SG | 5K FID或进度 |','|---|---:|---:|']
            for cfg in req['configs']:lines.append(f"| {cfg['arm']} | {1+cfg.get('omega',0):.4f} | {cell(path.parent/cfg['arm'],req['batch'])} |")
            lines.append('')
    if not started:lines+=['已准备三个模型各11组粗扫、每组5K；按用户要求等待后训练头任务完成后开始生成。','']
    lines+=['[SG冻结方案](GUIDANCE_5K_GRID_20260914_ZH.md)。','',
        '原始目录：`/home/zhoushunyu/data/eqvae/experiments/raev2_shallow_ig_20260914/` 与 `ig_sg_5k_20260914/`。']
    tmp=OUT.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(OUT)


def main():
    while True:
        write()
        if (SG/'report_complete.json').exists():break
        time.sleep(45)

if __name__=='__main__':main()
