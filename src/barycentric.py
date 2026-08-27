"""Barycentric mapping: turn a transport plan into a recolored image.

Given a plan ``P`` (K_s, K_t), each source cluster ``i`` is recolored to the
``P``-weighted average of the target cluster centers::

    new_lab[i] = sum_j P[i, j] * Xt[j]  /  sum_j P[i, j]

Rows of ``P`` that carry (almost) no mass -- which happens with ``partial`` and
``unbalanced`` OT, where some source colors are intentionally left untouched --
fall back to the original source center ``Xs[i]``.

Finally every pixel takes the new color of its source cluster
(``new_lab[source_labels]``), the image is reshaped and converted back to RGB.
"""

from __future__ import annotations

import numpy as np

from color_pipeline import lab_to_rgb


def barycentric_centers(
    P: np.ndarray,
    Xt: np.ndarray,
    Xs: np.ndarray,
    mass_eps: float = 1e-9,
) -> np.ndarray:
    """Recolored Lab position for every source cluster, shape (K_s, 3)."""
    P = np.asarray(P, dtype=np.float64)
    row_mass = P.sum(axis=1)
    safe = np.maximum(row_mass, mass_eps)
    new_lab = (P @ Xt) / safe[:, None]
    # keep original color where (almost) no mass was transported
    dropped = row_mass <= mass_eps
    new_lab[dropped] = Xs[dropped]
    return new_lab


def map_colors(
    P: np.ndarray,
    X_target_centers: np.ndarray,
    source_labels: np.ndarray,
    X_source_centers: np.ndarray,
    source_shape: tuple | None = None,
    return_lab: bool = False,
) -> np.ndarray:
    """Apply barycentric mapping and return the recolored image.

    Parameters
    ----------
    P : (K_s, K_t) transport plan.
    X_target_centers : (K_t, 3) target cluster centers in Lab.
    source_labels : (H*W,) source pixel -> source cluster id.
    X_source_centers : (K_s, 3) source cluster centers in Lab.
    source_shape : (H, W, 3); if None the result is returned flat as (H*W, 3).
    return_lab : if True, return Lab instead of RGB.

    Returns
    -------
    ndarray  recolored image, RGB in [0, 1] (or Lab if ``return_lab``).
    """
    new_lab_centers = barycentric_centers(P, X_target_centers, X_source_centers)
    pixels_lab = new_lab_centers[np.asarray(source_labels)]

    if source_shape is not None:
        pixels_lab = pixels_lab.reshape(source_shape)

    if return_lab:
        return pixels_lab
    return lab_to_rgb(pixels_lab)


def transfer(prob, P, return_lab: bool = False) -> np.ndarray:
    """Convenience wrapper around :func:`map_colors` using an ``OTProblem``."""
    return map_colors(
        P,
        prob.Xt,
        prob.source_labels,
        prob.Xs,
        source_shape=prob.source_shape,
        return_lab=return_lab,
    )


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    import numpy as np
    from skimage import color as skcolor

    from color_pipeline import build_ot_problem, make_synthetic_pair, to_float_rgb
    from solvers import solve_ot

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    src, tgt = make_synthetic_pair(size=160, seed=0)
    prob = build_ot_problem(src, tgt, k=256, random_state=0)

    tgt_mean = skcolor.rgb2lab(to_float_rgb(tgt)).reshape(-1, 3).mean(0)
    src_mean = skcolor.rgb2lab(to_float_rgb(src)).reshape(-1, 3).mean(0)
    print(f"source mean Lab : {np.round(src_mean, 2)}")
    print(f"target mean Lab : {np.round(tgt_mean, 2)}\n")

    try:
        from PIL import Image
        Image.fromarray(src).save(os.path.join(results_dir, "demo_source.png"))
        Image.fromarray(tgt).save(os.path.join(results_dir, "demo_target.png"))
    except Exception:
        Image = None

    configs = {
        "emd": {},
        "sinkhorn": {},
        "unbalanced": {},
        "partial": {"m": 0.6},
        "sliced": {"Xs": prob.Xs, "Xt": prob.Xt, "n_projections": 100},
    }

    print(f"{'method':<12} {'result mean Lab':>26} {'dLab->target':>13} "
          f"{'dLab->source':>13}")
    print("-" * 68)
    for method, kw in configs.items():
        P = solve_ot(prob.a, prob.b, prob.C, method, **kw)
        out_lab = transfer(prob, P, return_lab=True)
        out_rgb = lab_to_rgb(out_lab)
        m = out_lab.reshape(-1, 3).mean(0)
        d_t = np.linalg.norm(m - tgt_mean)
        d_s = np.linalg.norm(m - src_mean)
        print(f"{method:<12} {np.array2string(np.round(m, 2)):>26} "
              f"{d_t:13.2f} {d_s:13.2f}")
        if Image is not None:
            Image.fromarray((out_rgb * 255).astype(np.uint8)).save(
                os.path.join(results_dir, f"demo_{method}.png")
            )

    print(f"\nsaved demo images to {os.path.normpath(results_dir)}/")
    print("OK")
