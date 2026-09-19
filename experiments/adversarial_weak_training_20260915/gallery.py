"""Unselected paired image evidence: always the first eight quality inputs."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import common as c


def main(args):
    candidate = args.point / "batches/00000.npz"
    baseline = ((args.baseline_point / "batches/00000.npz") if args.baseline_point else
                c.original.model_root("sit_small") / "points/real__c0040/batches/00000.npz")
    with np.load(candidate) as data:
        current, labels = data["arr_0"], data["labels"]
    with np.load(baseline) as data:
        previous = data["arr_0"]
        np.testing.assert_array_equal(labels, data["labels"])
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.7))
    for index in range(8):
        group, column = divmod(index, 4)
        for offset, images, title in ((0, previous, args.baseline_title),
                                      (1, current, args.title)):
            axis = axes[2 * group + offset, column]
            axis.imshow(images[index], interpolation="nearest")
            axis.set_title(f"{title}\ninput {index}, class {labels[index]}", fontsize=8)
            axis.axis("off")
    fig.suptitle("Paired fixed first eight inputs; no image selection", fontsize=12)
    fig.tight_layout()
    target = args.point / "paired_first8.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    c.atomic(args.point / "paired_first8.json", dict(input_ids=list(range(8)),
        baseline=str(baseline), baseline_sha256=c.sha(baseline),
        candidate=str(candidate), candidate_sha256=c.sha(candidate), selection="fixed first 8"))
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", type=Path, required=True)
    parser.add_argument("--title", default="Endpoint-trained head, a=1")
    parser.add_argument("--baseline-point", type=Path)
    parser.add_argument("--baseline-title", default="50K diffusion head, a=1")
    main(parser.parse_args())
