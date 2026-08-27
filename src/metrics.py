"""Evaluation metrics for color transfer.

    w2_color_distance(result, target)      Wasserstein-2 between the two color
                                           distributions (empirical, in Lab).
                                           Lower = closer to target palette.
    ssim_with_source(result, source)       structural similarity to the ORIGINAL
                                           source image. Higher = structure kept.
    color_histogram_intersection(r, t)     3-D histogram intersection in [0, 1].
                                           Higher = better palette overlap.

``compute_all_metrics`` bundles the three into a dict.
"""

from __future__ import annotations

import numpy as np
import ot
from skimage import color as skcolor
from skimage.metrics import structural_similarity

from color_pipeline import to_float_rgb


def _lab_pixels(rgb: np.ndarray) -> np.ndarray:
    return skcolor.rgb2lab(to_float_rgb(rgb)).reshape(-1, 3)


def w2_color_distance(
    result_rgb: np.ndarray,
    target_rgb: np.ndarray,
    n_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Empirical Wasserstein-2 distance between result and target colors (Lab).

    Pixels are subsampled to ``n_samples`` from each image; the exact discrete
    OT cost (``ot.emd2``) is taken between the two uniform point clouds and
    square-rooted.
    """
    rng = np.random.default_rng(seed)
    r = _lab_pixels(result_rgb)
    t = _lab_pixels(target_rgb)

    ri = rng.choice(len(r), size=min(n_samples, len(r)), replace=False)
    ti = rng.choice(len(t), size=min(n_samples, len(t)), replace=False)
    r, t = r[ri], t[ti]

    a = np.full(len(r), 1.0 / len(r))
    b = np.full(len(t), 1.0 / len(t))
    C = ot.dist(r, t, metric="sqeuclidean")
    cost = ot.emd2(a, b, C, numItermax=200_000)
    return float(np.sqrt(max(cost, 0.0)))


def ssim_with_source(result_rgb: np.ndarray, source_rgb: np.ndarray) -> float:
    """Structural similarity between the result and the original source image."""
    r = to_float_rgb(result_rgb)
    s = to_float_rgb(source_rgb)
    return float(
        structural_similarity(s, r, channel_axis=-1, data_range=1.0)
    )


def color_histogram_intersection(
    result_rgb: np.ndarray,
    target_rgb: np.ndarray,
    bins: int = 16,
) -> float:
    """3-D RGB histogram intersection, normalized to [0, 1]."""
    r = to_float_rgb(result_rgb).reshape(-1, 3)
    t = to_float_rgb(target_rgb).reshape(-1, 3)
    rng = [[0.0, 1.0]] * 3
    hr, _ = np.histogramdd(r, bins=bins, range=rng)
    ht, _ = np.histogramdd(t, bins=bins, range=rng)
    hr /= hr.sum()
    ht /= ht.sum()
    return float(np.minimum(hr, ht).sum())


def compute_all_metrics(
    result_rgb: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    **kwargs,
) -> dict:
    return {
        "w2_color_distance": w2_color_distance(result_rgb, target_rgb, **kwargs),
        "ssim_with_source": ssim_with_source(result_rgb, source_rgb),
        "color_histogram_intersection": color_histogram_intersection(
            result_rgb, target_rgb
        ),
    }


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from barycentric import transfer
    from baselines import mkl_transfer, reinhard_transfer
    from color_pipeline import build_ot_problem, make_synthetic_pair
    from solvers import solve_ot

    src, tgt = make_synthetic_pair(size=160, seed=0)
    prob = build_ot_problem(src, tgt, k=256, random_state=0)

    candidates = {
        "source (identity)": to_float_rgb(src),
        "target (oracle)": to_float_rgb(tgt),
        "emd": transfer(prob, solve_ot(prob.a, prob.b, prob.C, "emd")),
        "partial(m=.6)": transfer(
            prob, solve_ot(prob.a, prob.b, prob.C, "partial", m=0.6)
        ),
        "reinhard": reinhard_transfer(src, tgt),
        "mkl": mkl_transfer(src, tgt),
    }

    print(f"{'candidate':<20} {'W2_color':>10} {'SSIM_src':>10} {'hist_inter':>11}")
    print("-" * 54)
    for name, img in candidates.items():
        m = compute_all_metrics(img, src, tgt)
        print(f"{name:<20} {m['w2_color_distance']:10.2f} "
              f"{m['ssim_with_source']:10.3f} "
              f"{m['color_histogram_intersection']:11.3f}")
    print("\nOK")
