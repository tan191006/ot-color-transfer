"""Color pipeline: RGB image -> Lab -> KMeans clusters -> OT problem (a, b, C).

The core entry point is :func:`build_ot_problem`, which takes a source and a
target RGB image and returns everything the OT solvers and the barycentric
mapping need:

    a, b            normalized cluster weights (histograms over Lab clusters)
    C               squared-euclidean cost matrix in Lab between source and
                    target cluster centers, shape (K_s, K_t)
    Xs, Xt          cluster centers in Lab space, shape (K_s, 3) / (K_t, 3)
    source_labels   per-pixel cluster id for the source image, shape (H*W,)
    target_labels   per-pixel cluster id for the target image, shape (H*W,)

A small synthetic-image generator (:func:`make_synthetic_pair`) is included so
the pipeline can be exercised without any real photographs: the source is a
green foliage-like image and the target is a red sunset-sky-like image, which is
exactly the "semantic distribution mismatch" situation the benchmark targets.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import ot
from skimage import color as skcolor
from sklearn.cluster import KMeans


class OTProblem(NamedTuple):
    """Container for a discrete OT color-transfer problem."""

    a: np.ndarray            # (K_s,) source cluster weights, sums to 1
    b: np.ndarray            # (K_t,) target cluster weights, sums to 1
    C: np.ndarray            # (K_s, K_t) squared L2 cost in Lab
    Xs: np.ndarray           # (K_s, 3) source cluster centers in Lab
    Xt: np.ndarray           # (K_t, 3) target cluster centers in Lab
    source_labels: np.ndarray  # (H*W,) source pixel -> source cluster id
    target_labels: np.ndarray  # (H*W,) target pixel -> target cluster id
    source_shape: tuple      # (H, W, 3) original source image shape
    target_shape: tuple      # (H, W, 3) original target image shape


# --------------------------------------------------------------------------- #
# Color-space helpers
# --------------------------------------------------------------------------- #
def to_float_rgb(img: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) float RGB image in [0, 1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img = img / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 4:  # drop alpha
        img = img[:, :, :3]
    return np.clip(img, 0.0, 1.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB in [0, 1]  ->  (H, W, 3) CIELAB."""
    return skcolor.rgb2lab(to_float_rgb(rgb))


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(H, W, 3) CIELAB  ->  (H, W, 3) RGB in [0, 1] (clipped)."""
    return np.clip(skcolor.lab2rgb(lab), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def _cluster_image(lab: np.ndarray, k: int, random_state: int):
    """KMeans over the pixels of one Lab image.

    Returns (centers (k', 3), labels (H*W,), weights (k',)) where k' <= k when
    the image has fewer distinct pixels than requested clusters.
    """
    pixels = lab.reshape(-1, 3)
    k_eff = int(min(k, pixels.shape[0]))
    km = KMeans(n_clusters=k_eff, random_state=random_state, n_init="auto")
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_
    weights = np.bincount(labels, minlength=k_eff).astype(np.float64)
    weights /= weights.sum()
    return centers, labels, weights


def build_ot_problem(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    k: int = 512,
    random_state: int = 0,
) -> OTProblem:
    """Build the discrete OT color-transfer problem for a source/target pair."""
    source_rgb = to_float_rgb(source_rgb)
    target_rgb = to_float_rgb(target_rgb)

    source_lab = skcolor.rgb2lab(source_rgb)
    target_lab = skcolor.rgb2lab(target_rgb)

    Xs, source_labels, a = _cluster_image(source_lab, k, random_state)
    Xt, target_labels, b = _cluster_image(target_lab, k, random_state)

    # squared euclidean cost in Lab between every source/target center pair
    C = ot.dist(Xs, Xt, metric="sqeuclidean")

    return OTProblem(
        a=a,
        b=b,
        C=np.ascontiguousarray(C),
        Xs=Xs,
        Xt=Xt,
        source_labels=source_labels,
        target_labels=target_labels,
        source_shape=source_rgb.shape,
        target_shape=target_rgb.shape,
    )


# --------------------------------------------------------------------------- #
# Synthetic test images
# --------------------------------------------------------------------------- #
def make_synthetic_pair(size: int = 256, seed: int = 0):
    """Two synthetic RGB uint8 images with mismatched color semantics.

    source: green "foliage" — vertical green gradient with leafy noise + a
            brown trunk band.
    target: red "sunset sky" — horizontal orange->deep-red gradient with a
            bright sun disc and a dark foreground strip.
    """
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny, nx = yy / (h - 1), xx / (w - 1)

    # ---- source: greens ---------------------------------------------------- #
    src = np.zeros((h, w, 3))
    src[..., 0] = 0.10 + 0.25 * ny                     # low red
    src[..., 1] = 0.30 + 0.55 * (1.0 - ny)             # strong green, brighter up
    src[..., 2] = 0.08 + 0.15 * ny                     # low blue
    src += rng.normal(0.0, 0.04, src.shape)            # leafy texture
    trunk = (np.abs(nx - 0.5) < 0.08)                  # central brown trunk
    src[trunk] = [0.35, 0.22, 0.10]
    src = np.clip(src, 0.0, 1.0)

    # ---- target: red sunset --------------------------------------------- #
    tgt = np.zeros((h, w, 3))
    tgt[..., 0] = 0.85 - 0.35 * ny                     # lots of red
    tgt[..., 1] = 0.35 - 0.30 * ny                     # some green up high
    tgt[..., 2] = 0.20 + 0.10 * (1.0 - ny)             # a little blue up high
    sun = np.sqrt((nx - 0.7) ** 2 + (ny - 0.35) ** 2) < 0.12
    tgt[sun] = [1.0, 0.9, 0.55]                        # bright sun disc
    ground = ny > 0.8
    tgt[ground] = [0.12, 0.06, 0.08]                   # dark foreground
    tgt += rng.normal(0.0, 0.02, tgt.shape)
    tgt = np.clip(tgt, 0.0, 1.0)

    return (src * 255).astype(np.uint8), (tgt * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    src, tgt = make_synthetic_pair(size=192, seed=0)

    try:
        from PIL import Image

        Image.fromarray(src).save(os.path.join(out_dir, "synthetic_source.png"))
        Image.fromarray(tgt).save(os.path.join(out_dir, "synthetic_target.png"))
        print(f"saved synthetic images to {os.path.normpath(out_dir)}/")
    except Exception as exc:  # pragma: no cover
        print(f"(could not save PNGs: {exc})")

    prob = build_ot_problem(src, tgt, k=512, random_state=0)

    print("\n=== OT problem built ===")
    print(f"source image      : {prob.source_shape}")
    print(f"target image      : {prob.target_shape}")
    print(f"a (source weights): shape {prob.a.shape}, sum {prob.a.sum():.6f}, "
          f"min {prob.a.min():.2e}, max {prob.a.max():.2e}")
    print(f"b (target weights): shape {prob.b.shape}, sum {prob.b.sum():.6f}")
    print(f"Xs (source Lab)   : shape {prob.Xs.shape}, "
          f"L range [{prob.Xs[:,0].min():.1f}, {prob.Xs[:,0].max():.1f}]")
    print(f"Xt (target Lab)   : shape {prob.Xt.shape}")
    print(f"C (cost matrix)   : shape {prob.C.shape}, "
          f"min {prob.C.min():.2f}, max {prob.C.max():.2f}, mean {prob.C.mean():.2f}")
    print(f"source_labels     : shape {prob.source_labels.shape}, "
          f"{prob.source_labels.min()}..{prob.source_labels.max()}")
    print(f"target_labels     : shape {prob.target_labels.shape}")

    # sanity: mean Lab of each cluster set vs raw pixels
    src_lab = skcolor.rgb2lab(to_float_rgb(src)).reshape(-1, 3)
    tgt_lab = skcolor.rgb2lab(to_float_rgb(tgt)).reshape(-1, 3)
    print("\nmean Lab  source pixels :", np.round(src_lab.mean(0), 2),
          " | source centers (weighted):",
          np.round((prob.a[:, None] * prob.Xs).sum(0), 2))
    print("mean Lab  target pixels :", np.round(tgt_lab.mean(0), 2),
          " | target centers (weighted):",
          np.round((prob.b[:, None] * prob.Xt).sum(0), 2))
    print("\nOK")
