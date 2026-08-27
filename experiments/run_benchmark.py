"""Full color-transfer benchmark on one image pair.

Runs every OT solver (emd, sinkhorn, unbalanced, partial, sliced) plus the three
classical baselines (reinhard, idt, mkl) on a source/target pair, writes each
recolored image to ``results/``, prints a metric table, and saves a Pareto plot
(x = W2 color distance to target, y = SSIM to source) to
``results/pareto_plot.png``.

Usage
-----
    python experiments/run_benchmark.py                     # synthetic pair
    python experiments/run_benchmark.py SRC.png TGT.png     # your own images
    python experiments/run_benchmark.py SRC.png TGT.png --k 512
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from barycentric import transfer  # noqa: E402
from baselines import (  # noqa: E402
    iterative_distribution_transfer,
    mkl_transfer,
    reinhard_transfer,
)
from color_pipeline import build_ot_problem, make_synthetic_pair, to_float_rgb  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from solvers import solve_ot  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")

OT_METHODS = ["emd", "sinkhorn", "unbalanced", "partial", "sliced"]
BASELINES = ["reinhard", "idt", "mkl"]


# --------------------------------------------------------------------------- #
def _load(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _save(img_rgb01: np.ndarray, name: str) -> str:
    from PIL import Image

    path = os.path.join(RESULTS, name)
    Image.fromarray((np.clip(img_rgb01, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


def load_pair(src_path: str | None, tgt_path: str | None):
    """User paths -> else data/{source,target}.* -> else synthetic."""
    if src_path and tgt_path:
        return _load(src_path), _load(tgt_path), "custom"

    for ext in ("png", "jpg", "jpeg"):
        s = os.path.join(DATA, f"source.{ext}")
        t = os.path.join(DATA, f"target.{ext}")
        if os.path.exists(s) and os.path.exists(t):
            return _load(s), _load(t), "data/"

    src, tgt = make_synthetic_pair(size=192, seed=0)
    return src, tgt, "synthetic"


# --------------------------------------------------------------------------- #
def run(src, tgt, k: int):
    prob = build_ot_problem(src, tgt, k=k, random_state=0)
    print(f"OT problem: K_s={len(prob.a)}  K_t={len(prob.b)}  "
          f"cost median={np.median(prob.C[prob.C > 0]):.0f}\n")

    results = {}          # name -> rgb float image
    timings = {}

    ot_kwargs = {
        "partial": {"m": 0.7},
        "sliced": {"Xs": prob.Xs, "Xt": prob.Xt, "n_projections": 100},
    }
    for method in OT_METHODS:
        t0 = time.perf_counter()
        P = solve_ot(prob.a, prob.b, prob.C, method, **ot_kwargs.get(method, {}))
        results[method] = transfer(prob, P)
        timings[method] = time.perf_counter() - t0

    baseline_fns = {
        "reinhard": lambda: reinhard_transfer(src, tgt),
        "idt": lambda: iterative_distribution_transfer(src, tgt, n_iter=20, seed=0),
        "mkl": lambda: mkl_transfer(src, tgt),
    }
    for name, fn in baseline_fns.items():
        t0 = time.perf_counter()
        results[name] = fn()
        timings[name] = time.perf_counter() - t0

    # ---- metrics -------------------------------------------------------- #
    rows = {}
    for name, img in results.items():
        rows[name] = compute_all_metrics(img, src, tgt)
        _save(img, f"transfer_{name}.png")

    _save(to_float_rgb(src), "input_source.png")
    _save(to_float_rgb(tgt), "input_target.png")

    return prob, results, rows, timings


# --------------------------------------------------------------------------- #
def print_table(rows, timings):
    print(f"{'method':<12} {'type':<9} {'W2_color':>9} {'SSIM_src':>9} "
          f"{'hist_int':>9} {'time(s)':>8}")
    print("-" * 62)
    for name, m in rows.items():
        kind = "OT" if name in OT_METHODS else "baseline"
        print(f"{name:<12} {kind:<9} {m['w2_color_distance']:9.2f} "
              f"{m['ssim_with_source']:9.3f} "
              f"{m['color_histogram_intersection']:9.3f} "
              f"{timings[name]:8.2f}")


def pareto_plot(rows, path: str):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, m in rows.items():
        is_ot = name in OT_METHODS
        x, y = m["w2_color_distance"], m["ssim_with_source"]
        ax.scatter(x, y, s=90, marker="o" if is_ot else "s",
                   color="tab:blue" if is_ot else "tab:orange",
                   edgecolor="black", zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 5),
                    fontsize=9)

    # Pareto frontier (minimize W2, maximize SSIM)
    pts = sorted((m["w2_color_distance"], m["ssim_with_source"])
                 for m in rows.values())
    front, best_y = [], -np.inf
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.plot(fx, fy, "--", color="gray", zorder=2, label="Pareto frontier")

    ax.set_xlabel("W2 color distance to target  (lower = closer palette)")
    ax.set_ylabel("SSIM to source  (higher = structure kept)")
    ax.set_title("Color transfer: palette match vs structure preservation")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="tab:blue",
                   markeredgecolor="black", label="OT solver"),
        plt.Line2D([], [], marker="s", ls="", color="tab:orange",
                   markeredgecolor="black", label="classical baseline"),
    ]
    ax.legend(handles=handles, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def montage(results, src, tgt, path: str):
    names = list(results.keys())
    n = len(names) + 2
    cols = 5
    r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(3 * cols, 3 * r))
    axes = np.atleast_2d(axes).ravel()
    panels = [("source", to_float_rgb(src)), ("target", to_float_rgb(tgt))]
    panels += [(nm, results[nm]) for nm in names]
    for ax, (nm, img) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(nm, fontsize=10)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None)
    ap.add_argument("target", nargs="?", default=None)
    ap.add_argument("--k", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    src, tgt, origin = load_pair(args.source, args.target)
    print(f"image pair source: {origin}   "
          f"source {np.asarray(src).shape}  target {np.asarray(tgt).shape}\n")

    _, results, rows, timings = run(src, tgt, args.k)

    print_table(rows, timings)

    pareto_path = os.path.join(RESULTS, "pareto_plot.png")
    pareto_plot(rows, pareto_path)
    montage_path = os.path.join(RESULTS, "montage.png")
    montage(results, src, tgt, montage_path)

    print(f"\nsaved: {os.path.relpath(pareto_path, ROOT)}, "
          f"{os.path.relpath(montage_path, ROOT)}, "
          f"and results/transfer_*.png")


if __name__ == "__main__":
    main()
