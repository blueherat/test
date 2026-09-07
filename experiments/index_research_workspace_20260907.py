"""Inventory all Git-owned research, including historical root-level experiments.

This reads working-tree contents without moving assets, changing experiments, or
equating document enumeration with paper reading or scientific validation.
The JSON excludes itself to avoid a circular digest; the generated navigation is
written first and then included in the digest. Large ignored assets stay external.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote, unquote


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'experiments/results/raev2_guidance_20260907/workspace_research_inventory.json'
NAVIGATION = ROOT / 'docs/RESEARCH_WORKSPACE_INVENTORY_20260907_ZH.md'
FAMILIES = {
    'raev2': 'RAEv2 guidance、理论与迁移',
    'rae_decoder': 'RAE、LPL、decoder 与 latent 几何',
    'pfr': 'PFR、反事实参考与半群 guidance',
    'sit': 'SiT / ImageNet-100 与内部头',
    'dit': 'DiT bad/good、事件与统计审核',
    'advfd': 'AdvFD、Fréchet 与 score 后训练',
    'toy': '预测目标、频谱与低维机制实验',
    'generation': '生成训练、评估与其他数据集',
    'general': '通用理论、协议、环境与总入口',
}
CODE_EXT = {'.py', '.sh', '.ipynb'}
CONFIG_EXT = {'.yaml', '.yml', '.toml'}
DATA_EXT = {'.json', '.jsonl', '.csv', '.tsv', '.npz', '.npy', '.pt'}
FIGURE_EXT = {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}


def git_paths(*args):
    output = subprocess.check_output(['git', 'ls-files', '-z', *args], cwd=ROOT)
    return {part.decode() for part in output.split(b'\0') if part}


def family(path):
    s = path.lower()
    if 'raev2' in s:
        return 'raev2'
    if any(k in s for k in ['pfr', 'projected_future_reference', 'autoguidance_foresight',
                             'affine_counterfactual', 'semigroup_consistent']):
        return 'pfr'
    if any(k in s for k in ['advfd', 'frechet', 'fréchet', 'residual_score_posttrain']):
        return 'advfd'
    if re.search(r'(^|[/_])dit([/_\.]|$)', s):
        return 'dit'
    if any(k in s for k in ['rae_', '/rae/', 'lpl', 'latent_trust', 'prior_decoder']):
        return 'rae_decoder'
    if any(k in s for k in ['sit', 'imagenet100', 'v800_', 'internal_head', 'ig_ablation',
                             'external_v180', 'checkpoint_reference', 'spectral_ig_mechanism']):
        return 'sit'
    if any(k in s for k in ['toy', 'prediction_target', 'frequency_', 'fractal',
                             'closed_loop_spiral', 'discriminator_ag_transport']):
        return 'toy'
    if any(k in s for k in ['imagenette', 'cifar', 'imagenet', 'train_gen/', 'train_eqvae/',
                             'evaluation/', 'baselines/', 'models/']):
        return 'generation'
    return 'general'


def kind(path):
    suffix = Path(path).suffix.lower()
    if suffix == '.md':
        return 'document'
    if suffix in CODE_EXT:
        return 'code_or_notebook'
    if suffix in CONFIG_EXT:
        return 'configuration'
    if suffix in DATA_EXT:
        return 'structured_data_or_evidence'
    if suffix in FIGURE_EXT:
        return 'figure_or_pdf'
    if suffix in {'.patch', '.pending'}:
        return 'historical_patch_or_locked_source'
    return 'other'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def title(path):
    content = path.read_text(errors='replace')
    match = re.search(r'^#\s+(.+)$', content, re.M)
    return match.group(1).strip() if match else path.name


def link(path):
    relative = os.path.relpath(ROOT / path, NAVIGATION.parent)
    return quote(relative, safe='/._-')


def count_table(paths):
    rows = ['| 家族 | 全部文件 | 文档 | 代码/笔记本 | 结构化数据/证据 |',
            '|---|---:|---:|---:|---:|']
    for group, label in FAMILIES.items():
        members = [p for p in paths if family(p) == group]
        kinds = collections.Counter(kind(p) for p in members)
        rows.append(f"| {label} | {len(members)} | {kinds['document']} | "
                    f"{kinds['code_or_notebook']} | {kinds['structured_data_or_evidence']} |")
    return '\n'.join(rows)


def write_navigation(paths):
    lines = [
        '# 研究工作区完整索引：理论、实验、代码与数据', '',
        '2026-09-07。此索引覆盖整个 Git 已跟踪工作区，以及本次可见且未被忽略的新文件，'
        '包括根目录下的历史实验；没有只按 RAEv2 文件名前缀筛选。原文件保留原址和原结论。'
        '这是归档导航，枚举文档不等于重新读完或验证其中的论文、证明和实验。', '',
        '当前 RAEv2 3% 目标的结论及可比质量表见'
        '[当前研究总入口](RAEV2_RESEARCH_ARCHIVE_INDEX_20260907_ZH.md)，'
        '剩余次数受[最多八轮台账](RAEV2_FINAL_EIGHT_ROUNDS_20260907_ZH.md)约束。'
        '旧 SiT/PFR 正结果不作为 RAEv2 成果；未采样候选不记为 FID 失败。', '',
        '## 如何找回研究', '',
        '- [2026-09-06 各理论家族与失败边界](RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md)。',
        '- [52 篇一手文献阅读总索引](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md)：'
        '原文版本、实际阅读范围和外部 PDF/代码身份；本索引不增加已读篇数。',
        '- [较早实验归档入口](EXPERIMENT_ARCHIVE_INDEX_ZH.md)：PFR、SiT、预测目标、'
        'latent/decoder、AdvFD 等旧结果的语境与限制。',
        '- [RAEv2 当前机器证据索引](../experiments/results/raev2_guidance_20260907/research_evidence_index.json)：'
        '当前配对 1K/5K、独立审核、成本、样本路径与身份。',
        '- [全工作区机器清单](../experiments/results/raev2_guidance_20260907/workspace_research_inventory.json)：'
        '逐文件大小、SHA256、类型、家族，以及 Markdown 的本地引用检查。',
        '- [逐项引用复核与旧资产缺口](RESEARCH_WORKSPACE_REFERENCE_REVIEW_20260907_ZH.md)：'
        '区分外部仓库的相对路径、冻结副本链接和确实不在本机的原始资产。',
        '- [最后八轮的理论结论](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md)：'
        'Gaussian 风险与 FID 的严格反例、实际轨迹比值及 1K/5K 的解释边界。',
        '- [构建脚本](../experiments/index_research_workspace_20260907.py)：'
        '运行 `python experiments/index_research_workspace_20260907.py` 更新清单。', '',
        '## 覆盖范围与可验证边界', '',
        f'本快照覆盖 **{len(paths):,} 个文件**；其中文档 '
        f'**{sum(p.endswith(".md") for p in paths)} 份**。'
        '家族按路径关键词划分，只用于导航，不推断科学状态。一个文件可涉及多个方向，'
        '这里按构建器中固定优先序归入一个家族；全部路径仍可在 JSON 中检索。', '',
        count_table(paths), '',
        '所有普通文件本轮读取全部字节计算 SHA256。机器清单自身排除，避免循环摘要；'
        '本文先生成再纳入摘要。Git HEAD 记录的是构建时的父提交，文件摘要反映当时工作区，'
        '不把未提交修改冒称为该父提交的内容。若存在符号链接，只记录链接文本及目标存在性。', '',
        '大模型、latent、原始样本、完整特征和外部代码库仍保留在原记录位置，'
        '不会被本脚本复制进 Git，也不宣称本轮重算了全部外部资产的 SHA。'
        '当前 RAEv2 数据根为 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/`；'
        '各 execution、request、summary、audit 提供冻结源码、输入、样本和成本身份。', '',
        'Markdown 引用检查只解析明确的链接和单个行内本地路径，记录存在、缺失、'
        '歧义或模板表达式；历史绝对路径失效不会自动改写为另一个实验。'
        '原文引用中的 URL 只登记，不表示重新抓取或读过；动态路径、命令和 glob 不冒报为缺失文件。'
        '引用存在也不等于数据内容正确，质量结论仍以相应独立审核为准。', '',
        '## 全部理论与说明文档', '',
        '以下逐项连接工作区中的 Markdown，包括实验目录的 README、冻结协议和历史审核。'
        '文档标题为原首个一级标题，没有用自动生成摘要覆盖原论证。', '',
    ]
    for group, label in FAMILIES.items():
        lines += [f'### {label}', '']
        for path in sorted(paths):
            if path.endswith('.md') and family(path) == group:
                text = title(ROOT / path) if ROOT / path != NAVIGATION else '研究工作区完整索引：理论、实验、代码与数据'
                text = text.replace('[', '\\[').replace(']', '\\]').replace('|', '\\|')
                lines.append(f'- [{text}]({link(path)}) — `{path}`')
        lines.append('')
    NAVIGATION.write_text('\n'.join(lines))


def references(source):
    content = (ROOT / source).read_text(errors='replace')
    # A mathematical [r](y) or a fenced example is not a navigation link.
    content = re.sub(r'^```[^\n]*\n.*?^```[^\n]*$', '', content, flags=re.M | re.S)
    content = re.sub(r'\$\$.*?\$\$|\\\[.*?\\\]', '', content, flags=re.S)
    candidates = []
    # This intentionally does not attempt to interpret arbitrary code or shell.
    for match in re.finditer(r'!?\[[^\]\n]*\]\(([^\)\n]+)\)', content):
        candidates.append(('markdown_link', match.group(1).strip()))
    for match in re.finditer(r'(?<!`)`([^`\n]+)`(?!`)', content):
        value = match.group(1).strip()
        if re.match(r'(?:/home/|/data/|\.\.?/|docs/|experiments/|tests/|configs/|research_repos/|external/)', value):
            candidates.append(('inline_local_path', value))
    result = []
    for syntax, value in sorted(set(candidates)):
        row = {'source': source, 'syntax': syntax, 'literal': value}
        target = value.strip('<>')
        if re.match(r'(?:https?|mailto|app|data):', target):
            row['status'] = 'external_url_not_fetched'
        elif target.startswith('#'):
            row['status'] = 'same_document_anchor_not_checked'
        elif re.search(r'[\s*{}\[\]<>$|…]', target) or '\\' in target or '...' in target:
            row['status'] = 'expression_or_unsupported_syntax_not_resolved'
        else:
            target = unquote(target.split('#', 1)[0])
            line_suffix = re.search(r':(\d+)(?:-(\d+))?$', target)
            if line_suffix:
                row['line_reference_not_content_validated'] = line_suffix.group(0)
                target = target[:line_suffix.start()]
            if target.startswith('/'):
                possible = [Path(target)]
            elif syntax == 'markdown_link':
                possible = [ROOT / source / '..' / target]
            elif target.startswith(('docs/', 'experiments/', 'tests/', 'configs/', 'research_repos/', 'external/')):
                possible = [ROOT / target, ROOT / source / '..' / target]
            else:
                possible = [ROOT / source / '..' / target, ROOT / target]
            # abspath normalizes document/../ without requiring it to be a directory.
            normalized = sorted({os.path.abspath(p) for p in possible})
            row['candidate_paths'] = normalized
            existing = [p for p in normalized if Path(p).exists()]
            row['existing_paths'] = existing
            row['status'] = ('generated_inventory_destination' if normalized == [str(OUTPUT)] else
                             'exists' if len(existing) == 1 else
                             'ambiguous_multiple_existing' if existing else 'missing_literal_path')
        result.append(row)
    return result


def main():
    started = time.time()
    tracked = git_paths()
    new = git_paths('--others', '--exclude-standard')
    output_relative = OUTPUT.relative_to(ROOT).as_posix()
    paths = (tracked | new | {NAVIGATION.relative_to(ROOT).as_posix()}) - {output_relative}
    missing = [p for p in sorted(paths) if not (ROOT / p).exists() and not (ROOT / p).is_symlink()
               and ROOT / p != NAVIGATION]
    assert not missing, f'Listed working-tree files are missing: {missing}'
    write_navigation(paths)
    records, refs = [], []
    for name in sorted(paths):
        path = ROOT / name
        item = {'path': name, 'family': family(name), 'kind': kind(name),
                'git_tracked_at_snapshot': name in tracked, 'bytes': path.lstat().st_size}
        if path.is_symlink():
            target = os.readlink(path)
            item.update({'symlink_text': target, 'symlink_text_sha256': hashlib.sha256(target.encode()).hexdigest(),
                         'target_exists': path.exists(), 'target_bytes_not_hashed': True})
        else:
            assert path.is_file(), name
            item['sha256'] = sha(path)
        records.append(item)
        if name.endswith('.md'):
            refs.extend(references(name))
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    output = {
        'complete': True, 'goal_achieved': False, 'created_unix': time.time(),
        'scope': 'All Git-tracked working-tree files and nonignored untracked files, excluding this JSON itself',
        'repository': str(ROOT), 'git_head_before_snapshot': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_bytes_are_authoritative_not_parent_commit_contents': True,
        'file_count': len(records), 'total_bytes': sum(row['bytes'] for row in records),
        'document_count': sum(row['kind'] == 'document' for row in records),
        'type_counts': dict(collections.Counter(row['kind'] for row in records)),
        'family_counts': dict(collections.Counter(row['family'] for row in records)),
        'family_rule': 'Fixed ordered path-keyword heuristic in source, not scientific classification',
        'families': FAMILIES, 'file_records_sha256': digest, 'files': records,
        'reference_status_counts': dict(collections.Counter(row['status'] for row in refs)),
        'markdown_references': refs,
        'all_regular_listed_files_fully_hashed': True,
        'ignored_external_models_samples_and_features_not_copied_or_comprehensively_rehashed': True,
        'external_urls_not_fetched_and_not_counted_as_papers_read': True,
        'navigation': str(NAVIGATION.relative_to(ROOT)),
        'builder_sha256': sha(Path(__file__).resolve()), 'cpu_wall_seconds': time.time() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['file_count', 'total_bytes', 'document_count',
                                          'type_counts', 'reference_status_counts', 'cpu_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
