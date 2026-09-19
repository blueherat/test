"""Build a reproducible navigation catalog without reading bulk experiment data."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae')
OUT = ROOT / 'archive/manifests/20260919'


def category(name):
    name = name.lower()
    if any(s in name for s in ('classifier_guidance', 'adversarial_weak', 'adversarial_guidance_endpoint', 'binary_endpoint')):
        return 'classifier'
    if any(s in name for s in ('guidance_dynamic', 'guidance_loss_50k', 'weak_reference_loss', 'shallow_ig', 'guidance_complete', 'local_head', 'guidance_distribution', 'guidance_pasted')):
        return 'baselines_and_dependencies'
    if any(s in name for s in ('self_guidance', 'ig_sg', 'ig_plus_sg')):
        return 'sg_paused'
    if 'jit' in name:
        return 'jit_history'
    if 'rae' in name:
        return 'rae_history'
    if 'sit' in name:
        return 'sit_history'
    return 'other_history'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for base, kind in [(ROOT/'experiments', 'code'), (ROOT/'docs', 'report'), (DATA/'experiments', 'external_data')]:
        for p in sorted(base.iterdir()):
            if p.name.startswith('.') or p.name == '__pycache__':
                continue
            items.append(dict(kind=kind, category=category(p.name), path=str(p), directory=p.is_dir()))
    (OUT/'catalog.json').write_text(json.dumps(items, ensure_ascii=False, indent=2)+'\n')
    groups = defaultdict(list)
    for item in items:
        groups[item['category']].append(item)
    text = ['# 工作区分类目录', '', '当前主线：`classifier_guidance/`。历史源码/报告保留原路径；数据条目仅索引顶层实验目录，不暗示每个目录都成功完成。', '']
    for group, rows in sorted(groups.items()):
        text += ['## '+group, '', '| 类型 | 路径 |', '|---|---|']
        for row in rows:
            p = Path(row['path'])
            link = os.path.relpath(p, OUT) if p.is_relative_to(ROOT) else str(p)
            text.append(f"| {row['kind']} | [{p.name}]({link}) |")
        text.append('')
    (OUT/'catalog.md').write_text('\n'.join(text).rstrip()+'\n')
    inventory=[]
    excluded={'.git','__pycache__','.pytest_cache','external','research_repos','readings','archive'}
    for current, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in excluded and not Path(current,d).is_symlink())
        for name in sorted(files):
            p=Path(current,name)
            if p.suffix in ('.pyc','.log','.pt','.pth','.ckpt','.npz','.npy','.safetensors'):
                continue
            row=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size)
            if p.is_symlink():
                row['link']=os.readlink(p)
            elif p.stat().st_size <= 2*1024**2:
                row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            inventory.append(row)
    (OUT/'source_and_evidence.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in inventory))
    links={
        'runs': DATA/'experiments/adversarial_weak_training_20260915',
        'baseline_50k': DATA/'experiments/guidance_dynamic_50k_20260915',
        'raev2_heads': DATA/'experiments/raev2_shallow_ig_20260914',
        'sit_training_data': DATA/'imagenet_sit_flow/imagenet100_cmc_sdvae',
        'imagenet_rgb': Path('/data/shared/imagenet-1k/random_access_v1'),
        'reports': ROOT/'docs/classifier_guidance',
        'source': ROOT/'classifier_guidance',
    }
    project=DATA/'projects/classifier_guidance'
    project.mkdir(parents=True,exist_ok=True)
    for name,target in links.items():
        assert target.exists(), target
        p=project/name
        if not p.is_symlink(): p.symlink_to(target,target_is_directory=True)
        assert p.resolve()==target.resolve()
    (OUT/'data_locations.json').write_text(json.dumps({k:str(v) for k,v in links.items()},indent=2)+'\n')
    (project/'README.md').write_text('# 分类器方法数据入口\n\n这些链接指向已有数据原件，没有复制或删除。首轮优化基准在 `performance_20260919/`；理论审计、进一步优化探测和短训验证在 `refinement_20260919/`。\n')
    print(json.dumps(dict(catalog_entries=len(items),inventory_entries=len(inventory),external_data_entries=sum(i['kind']=='external_data' for i in items))))


if __name__=='__main__':
    main()
