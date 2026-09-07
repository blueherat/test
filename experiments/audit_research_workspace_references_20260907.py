"""Review literal reference warnings against explicit historical document context."""
import json
from pathlib import Path
import subprocess
from experiments.index_research_workspace_20260907 import ROOT, git_paths, references, sha


def main():
    path = ROOT / 'experiments/results/raev2_guidance_20260907/workspace_reference_review.json'
    warning_rows = []
    paths = git_paths() | git_paths('--others', '--exclude-standard')
    for name in sorted(paths):
        if name.endswith('.md'):
            warning_rows.extend(row for row in references(name)
                                if row['status'] == 'missing_literal_path'
                                and row['candidate_paths'] != [str(path)])
    rows = []
    for warning in warning_rows:
        source, literal = warning['source'], warning['literal']
        row = {**warning, 'source_sha256': sha(ROOT / source), 'original_source_not_modified': True}
        alternative = None
        if source == 'docs/ADVFD_OFFICIAL_IMPLEMENTATION_AUDIT_ZH.md':
            assert literal == 'docs/plans/2026-07-28-fd-adv-shared-residual-rms.md'
            alternative = Path('/data/users/zhoushunyu/research_repos/AdvFD') / literal
            row['disposition'] = 'Path is relative to the official AdvFD repository explicitly named in the document'
            row['external_head_checked'] = '4e4cfed944e4fc38a75fae3ea7701ae9e5587060'
            row['cited_commit_checked'] = 'c9480401062bc0042405ec5c3591163ec3db0947'
            external_repo = '/data/users/zhoushunyu/research_repos/AdvFD'
            for revision, expected in [('HEAD', row['external_head_checked']), ('c948040', row['cited_commit_checked'])]:
                actual = subprocess.check_output(['git', '-C', external_repo, 'rev-parse', revision], text=True).strip()
                assert actual == expected
        elif source == 'docs/RAEV2_FINAL_ARCHIVE_VALIDATION_20260906_ZH.md':
            assert literal == 'docs/y'
            row['disposition'] = 'The document explicitly describes this as a prior mathematical regex false positive, not an asset citation'
        elif source == 'docs/RAEV2_GUIDANCE_READING_AUTOENCODER_SCORE_20260906_ZH.md':
            assert literal == 'configs/stage1/training/dinov3l-k7-imagenet.yaml'
            alternative = ROOT / 'external/RAEv2' / literal
            row['disposition'] = 'Quoted configuration is relative to the RAEv2 official repository being reviewed'
        elif source == 'docs/RAEV2_GUIDANCE_READING_FLOW_MATCHING_20260906_ZH.md':
            assert literal == 'experiments/raev2_guidance_restart_20260906/reading_flow_matching_v1'
            alternative = Path('/home/zhoushunyu/data/eqvae') / literal
            row['disposition'] = 'Same document later states the complete existing external data path'
        elif source == 'docs/data/raev2_guidance_final_20260906/files/query_mean_error_compatibility_v1/frozen_protocol.md':
            assert literal == 'RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md'
            alternative = ROOT / 'docs' / literal
            row['disposition'] = 'Immutable copied protocol retains its old relative link base; repository document exists'
        elif source == 'experiments/locks/dit_blur_focused_eprocess_protocol_lock_v1/theory_zh.md':
            assert literal.startswith(('../experiments/', '../tests/'))
            alternative = ROOT / literal[3:]
            row['disposition'] = 'Immutable locked theory retains a docs-relative link; exact named implementation exists at repository root'
        elif source in {
            'docs/RAEV2_LPL_STRICT_CONTINUATION_ZH.md',
            'docs/RAE_DECODER_RISK_PHASE0_RESULTS_ZH.md',
            'docs/RAE_DETERMINISTIC_LPL_REPRODUCTION_ZH.md',
            'docs/RAE_LPL_EXPERIMENT_LEDGER_ZH.md',
        }:
            assert literal.startswith('/home/zhoushunyu/data/eqvae/')
            assert not Path(literal).exists()
            row['disposition'] = 'Historical external asset absent at recorded path; original scientific result remains documented but raw asset is not certified available'
            row['nearby_smoke_directories_not_accepted_as_replacement'] = True
        else:
            raise AssertionError(f'New warning needs explicit review: {source}: {literal}')
        if alternative is not None:
            assert alternative.exists(), alternative
            row['context_resolved_path'] = str(alternative)
            row['context_resolved_path_exists'] = True
            if alternative.is_file():
                row['context_resolved_file_sha256'] = sha(alternative)
            else:
                row['directory_contents_not_comprehensively_hashed'] = True
        rows.append(row)
    output = {'complete': True, 'goal_achieved': False, 'warning_count': len(rows), 'rows': rows,
              'unreviewed_warnings': 0, 'missing_raw_assets_not_recreated_or_replaced': True,
              'literal_parser_warnings_are_not_all_missing_assets': True,
              'source_sha256': sha(Path(__file__).resolve())}
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'reviewed_warnings': len(rows),
                      'context_resolved_existing': sum('context_resolved_path' in r for r in rows),
                      'historical_absent': sum('nearby_smoke_directories_not_accepted_as_replacement' in r for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
