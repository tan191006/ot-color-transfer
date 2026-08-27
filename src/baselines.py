"""Classical (non-OT) color-transfer baselines.

All three operate on the *full* pixel set (no clustering) and in CIELAB:

    reinhard_transfer               per-channel mean/std matching   (Reinhard 2001)
    iterative_distribution_transfer  random-rotation 1-D histogram   (Pitie IDT 2007)
                                     matching, iterated
    mkl_transfer                     closed-form Gaussian Monge map  (Pitie MKL 2007)

Each takes ``source_rgb`` / ``target_rgb`` (any dtype/scale accepted) and returns
an RGB float image in [0, 1] with the source's spatial shape.
"""

from __future__ import annotations

import numpy as np
from skimage import color as skcolor

from color_pipeline import lab_to_rgb, to_float_rgb


def _to_lab_pixels(rgb: np.ndarray):
    """RGB image -> (lab_image, flat_pixels (N, 3))."""
    lab = skcolor.rgb2lab(to_float_rgb(rgb))
    return lab, lab.reshape(-1, 3)


# --------------------------------------------------------------------------- #
# Reinhard 2001
# --------------------------------------------------------------------------- #
def reinhard_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    """Match per-channel mean and std of the source to the target in Lab."""
    src_lab, src_px = _to_lab_pixels(source_rgb)
    _, tgt_px = _to_lab_pixels(target_rgb)

    mu_s, sd_s = src_px.mean(0), src_px.std(0)
    mu_t, sd_t = tgt_px.mean(0), tgt_px.std(0)
    sd_s = np.where(sd_s < 1e-6, 1.0, sd_s)

    out = (src_lab - mu_s) * (sd_t / sd_s) + mu_t
    return lab_to_rgb(out)


# --------------------------------------------------------------------------- #
# Pitie IDT 2007
# --------------------------------------------------------------------------- #
def _hist_match_1d(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Remap 1-D ``source`` values so their distribution matches ``template``."""
    n, m = len(source), len(template)
    ranks = source.argsort().argsort()          # 0 .. n-1
    quantiles = (ranks + 0.5) / n
    tmpl_sorted = np.sort(template)
    tmpl_q = (np.arange(m) + 0.5) / m
    return np.interp(quantiles, tmpl_q, tmpl_sorted)


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    q *= np.sign(np.diag(r))                    # fix signs
    if np.linalg.det(q) < 0:                    # ensure a proper rotation
        q[:, 0] *= -1
    return q


def iterative_distribution_transfer(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    n_iter: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Pitie's Iterative Distribution Transfer (IDT) in Lab.

    Each iteration: pick a random rotation, project both point clouds onto its
    axes, 1-D histogram-match the source to the target along every axis, rotate
    back.  Converges to matching the full 3-D distribution.
    """
    src_lab, src_px = _to_lab_pixels(source_rgb)
    _, tgt_px = _to_lab_pixels(target_rgb)
    rng = np.random.default_rng(seed)

    x = src_px.copy()
    for _ in range(n_iter):
        R = _random_rotation(3, rng)
        xr, tr = x @ R, tgt_px @ R
        for c in range(3):
            xr[:, c] = _hist_match_1d(xr[:, c], tr[:, c])
        x = xr @ R.T

    return lab_to_rgb(x.reshape(src_lab.shape))


# --------------------------------------------------------------------------- #
# Pitie MKL 2007  (linear Monge-Kantorovitch / Gaussian OT closed form)
# --------------------------------------------------------------------------- #
def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _psd_pow(M: np.ndarray, p: float, eps: float = 1e-10) -> np.ndarray:
    """Symmetric-PSD matrix power via eigendecomposition."""
    vals, vecs = np.linalg.eigh(_sym(M))
    vals = np.clip(vals, eps, None)
    return (vecs * (vals ** p)) @ vecs.T


def mkl_transfer(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    """Closed-form Monge map between the two images' Lab Gaussians.

        T(x) = mu_t + A (x - mu_s)
        A    = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}
    """
    src_lab, src_px = _to_lab_pixels(source_rgb)
    _, tgt_px = _to_lab_pixels(target_rgb)

    mu_s, mu_t = src_px.mean(0), tgt_px.mean(0)
    Sig_s = np.cov(src_px, rowvar=False) + 1e-6 * np.eye(3)
    Sig_t = np.cov(tgt_px, rowvar=False) + 1e-6 * np.eye(3)

    Ss_half = _psd_pow(Sig_s, 0.5)
    Ss_mhalf = _psd_pow(Sig_s, -0.5)
    inner = _psd_pow(_sym(Ss_half @ Sig_t @ Ss_half), 0.5)
    A = Ss_mhalf @ inner @ Ss_mhalf

    out = (src_px - mu_s) @ A.T + mu_t
    return lab_to_rgb(out.reshape(src_lab.shape))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    from color_pipeline import make_synthetic_pair

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    src, tgt = make_synthetic_pair(size=192, seed=0)
    tgt_px = skcolor.rgb2lab(to_float_rgb(tgt)).reshape(-1, 3)
    src_px = skcolor.rgb2lab(to_float_rgb(src)).reshape(-1, 3)
    print(f"source Lab  mean {np.round(src_px.mean(0), 2)}  std {np.round(src_px.std(0), 2)}")
    print(f"target Lab  mean {np.round(tgt_px.mean(0), 2)}  std {np.round(tgt_px.std(0), 2)}\n")

    methods = {
        "reinhard": lambda: reinhard_transfer(src, tgt),
        "idt": lambda: iterative_distribution_transfer(src, tgt, n_iter=20, seed=0),
        "mkl": lambda: mkl_transfer(src, tgt),
    }

    try:
        from PIL import Image
    except Exception:
        Image = None

    print(f"{'method':<10} {'result Lab mean':>24} {'result Lab std':>24} "
          f"{'mean err vs target':>19}")
    print("-" * 80)
    for name, fn in methods.items():
        out = fn()
        out_px = skcolor.rgb2lab(out).reshape(-1, 3)
        mean_err = np.linalg.norm(out_px.mean(0) - tgt_px.mean(0))
        print(f"{name:<10} {np.array2string(np.round(out_px.mean(0), 2)):>24} "
              f"{np.array2string(np.round(out_px.std(0), 2)):>24} {mean_err:19.2f}")
        if Image is not None:
            Image.fromarray((out * 255).astype(np.uint8)).save(
                os.path.join(results_dir, f"baseline_{name}.png")
            )

    print(f"\nsaved baseline images to {os.path.normpath(results_dir)}/")
    print("OK")
