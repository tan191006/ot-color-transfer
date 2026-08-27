"""Core experiment: balanced OT fails under a semantic distribution mismatch,
unbalanced / partial OT recover.

Scenario (synthetic, or your own images via CLI):

    A (source): ~70% green foliage  + ~30% brown soil
    B (target): ~70% orange sunset  + ~30% blue sky

Green has *no* colour neighbour in B.  Balanced OT (EMD) is forced to move every
green pixel onto red/orange anyway.  Unbalanced OT and Partial OT are allowed to
leave mass in place, so the green survives.

Outputs
-------
    results/mismatch_comparison.png     grid of all configs, titled
    printed table                       w2_color_distance / ssim_with_source

Usage
-----
    python experiments/mismatch_demo.py                    # synthetic
    python experiments/mismatch_demo.py A.png B.png        # your images
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from barycentric import transfer  # noqa: E402
from color_pipeline import build_ot_problem, to_float_rgb  # noqa: E402
from metrics import ssim_with_source, w2_color_distance  # noqa: E402
from solvers import solve_ot  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")

REG_M_VALUES = [0.1, 1.0, 10.0]     # unbalanced: marginal-relaxation weight
MASS_VALUES = [0.5, 0.7, 0.9]       # partial: fraction of mass to transport
ENTROPIC_REG = 0.02                 # entropic reg on the *normalized* cost


# --------------------------------------------------------------------------- #
def make_mismatch_pair(size: int = 200, seed: int = 0):
    """A: 70% green / 30% brown.   B: 70% orange / 30% blue.  uint8 RGB."""
    rng = np.random.default_rng(seed)
    h = w = size
    cut = int(round(0.70 * h))

    A = np.zeros((h, w, 3))
    A[:cut] = [0.20, 0.55, 0.15]              # green canopy (top 70%)
    A[cut:] = [0.40, 0.26, 0.12]              # brown soil  (bottom 30%)
    # a little vertical shading so the image has structure for SSIM
    A *= (0.85 + 0.15 * np.linspace(1.0, 0.7, h))[:, None, None]
    A += rng.normal(0.0, 0.015, A.shape)

    B = np.zeros((h, w, 3))
    B[:cut] = [0.95, 0.45, 0.15]              # orange sunset (top 70%)
    B[cut:] = [0.15, 0.30, 0.75]              # blue sky-ish  (bottom 30%)
    B += rng.normal(0.0, 0.015, B.shape)

    A = np.clip(A, 0, 1)
    B = np.clip(B, 0, 1)
    return (A * 255).astype(np.uint8), (B * 255).astype(np.uint8)


def _load(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


# --------------------------------------------------------------------------- #
def build_configs(prob):
    """name -> recolored RGB image."""
    Cn = prob.C / prob.C.max()               # normalize cost to [0, 1]
    a, b = prob.a, prob.b
    out = {}

    # balanced reference
    P = solve_ot(a, b, Cn, "emd")
    out["EMD (balanced)"] = transfer(prob, P)

    # unbalanced, sweeping reg_m
    for rm in REG_M_VALUES:
        P = solve_ot(a, b, Cn, "unbalanced", reg=ENTROPIC_REG, reg_m=rm)
        out[f"Unbalanced  reg_m={rm:g}"] = transfer(prob, P, keep_unmatched=True)

    # partial, sweeping transported mass
    for m in MASS_VALUES:
        P = solve_ot(a, b, Cn, "partial", m=m)
        out[f"Partial  mass={m:g}"] = transfer(prob, P, keep_unmatched=True)

    return out


def evaluate(configs, src, tgt):
    rows = {"source (identity)": (0.0, 1.0)}
    rows["source (identity)"] = (
        w2_color_distance(to_float_rgb(src), tgt),
        ssim_with_source(to_float_rgb(src), src),
    )
    for name, img in configs.items():
        rows[name] = (w2_color_distance(img, tgt), ssim_with_source(img, src))
    return rows


def print_table(rows):
    print(f"\n{'config':<24} {'W2_color_distance':>18} {'ssim_with_source':>18}")
    print("-" * 62)
    for name, (w2, ss) in rows.items():
        print(f"{name:<24} {w2:18.2f} {ss:18.3f}")
    print()


def make_figure(configs, src, tgt, rows, path):
    panels = [("A: source (70% green / 30% brown)", to_float_rgb(src)),
              ("B: target (70% orange / 30% blue)", to_float_rgb(tgt))]
    panels += [(n, img) for n, img in configs.items()]

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, img) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1))
        title = name
        if name in rows:
            w2, ss = rows[name]
            title = f"{name}\nW2={w2:.1f}  SSIM={ss:.2f}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Balanced OT vs Unbalanced / Partial OT under color-distribution mismatch",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None)
    ap.add_argument("target", nargs="?", default=None)
    ap.add_argument("--k", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.source and args.target:
        src, tgt = _load(args.source), _load(args.target)
        origin = "custom"
    else:
        src, tgt = make_mismatch_pair(size=200, seed=0)
        origin = "synthetic"

    print(f"pair: {origin}   source {src.shape}  target {tgt.shape}")
    prob = build_ot_problem(src, tgt, k=args.k, random_state=0)
    print(f"K_s={len(prob.a)}  K_t={len(prob.b)}  "
          f"cost median (Lab^2)={np.median(prob.C[prob.C > 0]):.0f}")

    configs = build_configs(prob)
    rows = evaluate(configs, src, tgt)
    print_table(rows)

    out = os.path.join(RESULTS, "mismatch_comparison.png")
    make_figure(configs, src, tgt, rows, out)
    print(f"saved {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
