"""Read-only snapshot and CPU comparison of existing signed guidance schedules."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eqvae_self_guidance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/self_guidance_schedule_shape_20260923"
DATA = Path("/home/zhoushunyu/data/eqvae/projects/classifier_guidance")
RUNS = {
    "sit_native": DATA/"sit_native_signed_schedule_20260919",
    "jit": DATA/"jit_block1_gan_schedule_20260922",
    "sit_joint": DATA/"sit_joint_gan_20260922",
}


def rows_from(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def merged(root, filename):
    rows = {}
    for path in sorted(root.glob("*/"+filename)):
        for row in rows_from(path):
            if "step" in row and "coefficients" in row:
                row = {**row, "source_file": str(path)}
                rows[row["step"]] = row
    return [rows[step] for step in sorted(rows)]


def diagnostics(root):
    rows = {}
    for path in sorted(root.glob("*/diagnostics.jsonl")):
        for row in rows_from(path):
            if "head_gradient" in row:
                rows[row["step"]] = row
    return [rows[step] for step in sorted(rows)]


def stats(row):
    a = np.asarray(row["coefficients"])
    n = len(a)
    t = (np.arange(n)+.5)/n
    low, high = int(a.argmin()), int(a.argmax())
    negative_bins = np.flatnonzero(a < 0)
    blocks = np.split(negative_bins, np.flatnonzero(np.diff(negative_bins) != 1)+1)
    ranges = [dict(left=float(block[0]/n), right=float((block[-1]+1)/n),
                   count=len(block)) for block in blocks if len(block)]
    return dict(step=row["step"], updates=row.get("coefficient_updates",row.get("joint_updates")),
                recorded_utc=row.get("updated_utc"), source_file=row["source_file"],
                a=a.tolist(), time_midpoints=t.tolist(),
                minimum=dict(a=float(a[low]),total_w=float(1+a[low]),index=low,time=float(t[low])),
                maximum=dict(a=float(a[high]),index=high,time=float(t[high])),
                first=float(a[0]),last=float(a[-1]),negative_intervals=ranges,
                net_time_integral=float(a.mean()),
                positive_time_integral=float(np.maximum(a,0).mean()),
                negative_time_integral=float(np.minimum(a,0).mean()),
                total_below_zero_count=int((a < -1).sum()),
                temporal_first_moment=float(np.mean(t*a)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"captured_utc":datetime.now(timezone.utc).isoformat(),"runs":{}}
    histories = {}
    for name, root in RUNS.items():
        history = merged(root, "train.jsonl")
        histories[name] = history
        result["runs"][name] = stats(history[-1])
        probes = diagnostics(root)[-1000:]
        if probes:
            matrix = np.array([r["head_gradient"] for r in probes])
            matrix = matrix[np.linalg.norm(matrix,axis=1)>0]
            normalized = matrix/np.linalg.norm(matrix,axis=1,keepdims=True)
            singular = np.linalg.svd(normalized,compute_uv=False)
            fractions = singular**2/np.sum(singular**2)
            result["runs"][name]["normalized_gradient_history"] = dict(
                samples=len(normalized), first_step=probes[0]["step"], last_step=probes[-1]["step"],
                energy_first_1=float(fractions[:1].sum()),
                energy_first_3=float(fractions[:3].sum()),
                energy_first_5=float(fractions[:5].sum()),
                meaning="Uncentered SVD of unit-normalized historical scalar-loss batch gradients; not a controllability rank or Fisher estimate.")
        chosen = []
        for updates in [512,1000,3000,10000,20000,30000]:
            target = updates + 128
            if target <= history[-1]["step"]:
                chosen.append(min(history,key=lambda r:abs(r["step"]-target)))
        result["runs"][name]["history_landmarks"] = [
            dict(step=r["step"],minimum=float(min(r["coefficients"])),maximum=float(max(r["coefficients"])))
            for r in chosen]
    quality = json.loads((RUNS["sit_native"]/"pipeline_status.json").read_text())
    result["sit_native_quality"] = quality.get("quality")
    time = np.linspace(.01,.99,400)
    curves = {key:np.interp(time,value["time_midpoints"],value["a"])
              for key,value in result["runs"].items()}
    result["shape_comparison"] = dict(
        interpolation="Linear interpolation of bin midpoint coefficients; descriptive only.",
        native_sit_jit_full_correlation=float(np.corrcoef(curves["sit_native"],curves["jit"])[0,1]),
        native_sit_jit_after_0p7_correlation=float(np.corrcoef(
            curves["sit_native"][time>.7],curves["jit"][time>.7])[0,1]),
        warning="Different weak heads, initial schedules, training ages, class sets, prediction spaces and solvers.")
    (OUT/"schedule_snapshot.json").write_text(json.dumps(result,indent=2)+"\n")
    fig, axes = plt.subplots(1,3,figsize=(15,4.1),constrained_layout=True)
    colors = {"sit_native":"#1764ab","jit":"#da7616","sit_joint":"#498b49"}
    labels = {"sit_native":"SiT native: 30K scale updates",
              "jit":f"JiT: {result['runs']['jit']['updates']} scale updates",
              "sit_joint":f"SiT joint: {result['runs']['sit_joint']['updates']} updates"}
    for key, value in result["runs"].items():
        a = np.array(value["a"])
        edges = np.linspace(0,1,len(a)+1)
        axes[0].stairs(a,edges,baseline=None,label=labels[key],color=colors[key],linewidth=1.6)
        axes[1].stairs(1+a,edges,baseline=None,color=colors[key],linewidth=1.6)
        if key != "sit_joint":
            tail = (np.arange(len(a))+.5)/len(a) > .7
            tail_range = a[tail].max()-a[tail].min()
            scaled = (a-a[tail].min())/tail_range
            axes[2].stairs(scaled,edges,baseline=None,color=colors[key],linewidth=1.6)
    axes[0].set(title="Observed extra coefficient a",xlabel="Noise 0 → image 1",ylabel="a")
    axes[0].axhline(0,color="black",linewidth=.7)
    axes[0].legend(fontsize=8)
    axes[1].set(title="Total strong coefficient w = 1 + a",xlabel="Noise 0 → image 1",ylabel="w")
    for y in [0,1]:
        axes[1].axhline(y,color="gray",linewidth=.7,linestyle="--")
    axes[2].set(title="Late trough / rebound (range normalized)",
                xlabel="Noise 0 → image 1",ylabel="(a − late minimum) / late range",
                xlim=(.7,1),ylim=(-.05,1.05))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT/"schedule_comparison.png",dpi=180)
    fig.savefig(OUT/"schedule_comparison.pdf")
    plt.close(fig)
    compact = {key:{k:v for k,v in value.items() if k not in ("a","time_midpoints")}
               for key,value in result["runs"].items()}
    print(json.dumps({"runs":compact,"shape_comparison":result["shape_comparison"]},indent=2))


if __name__ == "__main__":
    main()
