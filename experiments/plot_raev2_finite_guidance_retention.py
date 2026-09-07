"""Standalone descriptive figures for the paired native finite impulse pilot."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    result = json.loads((root/'analysis.json').read_text())
    assert json.loads((root/'audit.json').read_text())['complete']
    grid = json.loads((root/'cohort.json').read_text())['time_grid']
    groups = result['groups']
    x = np.array([g['write_query'] for g in groups])
    q = np.array([(grid[k]-grid[k+1])/grid[k] for k in x])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for field, label, color in [('first_read_projection', 'First Full read', '#1676a8'),
                                 ('terminal_projection', 'Full-only endpoint', '#d17425')]:
        axes[0].plot(x, np.array([g[field]['mean'] for g in groups])/q, 'o-', color=color, label=label)
    axes[0].axhline(1, color='#777777', linewidth=.8, linestyle='--')
    axes[0].set_ylabel('Message-aligned finite response / write coefficient')
    axes[0].set_title('Response relative to the original write')
    for field, label, color in [('first_read_cosine', 'First Full read vs. message', '#1676a8'),
                                 ('terminal_cosine', 'Endpoint vs. message', '#d17425'),
                                 ('one_relative_to_all_cosine', 'Endpoint vs. all remaining IG', '#568842')]:
        axes[1].plot(x, [g[field]['mean'] for g in groups], 'o-', color=color, label=label)
    axes[1].set_ylabel('Cosine similarity')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Direction of the finite response')
    for axis in axes:
        axis.set_xlabel('Write query index (native 100-step, shift-8 grid)')
        axis.set_xlim(0, 100)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, frameon=False)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('One native IG write, followed by Full alone\nTwo B8 batches per query; cohorts differ across queries; no FID evaluation', fontsize=11)
    for extension in ('png', 'pdf'):
        fig.savefig(root/f'finite_retention.{extension}', dpi=180)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {'analysis_sha256': sha(root/'analysis.json'),
                'source_sha256': sha(Path(__file__).resolve()),
                'files': {p.name: sha(p) for p in (root/'finite_retention.png', root/'finite_retention.pdf')}}
    (root/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
