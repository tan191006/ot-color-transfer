"""OT solvers wrapper.

A single entry point, :func:`solve_ot`, returns a transport plan ``P`` of shape
``(K_s, K_t)`` for every supported method so the downstream barycentric mapping
can treat them uniformly:

    method       POT routine                                    notes
    ---------    -------------------------------------------     ----------------
    "emd"        ot.emd                                          exact balanced
    "sinkhorn"   ot.sinkhorn                                     entropic balanced
    "unbalanced" ot.unbalanced.sinkhorn_unbalanced              KL-relaxed marginals
    "partial"    ot.partial.partial_wasserstein                  transports mass m<=1
    "sliced"     averaged 1-D OT couplings over random slices    needs Xs, Xt

For "sliced", POT's ``sliced_wasserstein_distance`` only returns a scalar, so we
build an approximate coupling ourselves: project the cluster centers onto many
random directions, solve the exact 1-D OT problem on each projection
(``ot.emd_1d``), and average the resulting couplings.  :func:`sliced_wasserstein`
exposes the plain POT scalar for the metrics module.
"""

from __future__ import annotations

import numpy as np
import ot

METHODS = ["emd", "sinkhorn", "unbalanced", "partial", "sliced"]


def _default_reg(C: np.ndarray) -> float:
    """Entropic regularization in absolute (Lab^2) cost units."""
    return 0.05 * float(np.median(C[C > 0]))


def _sliced_plan(
    a: np.ndarray,
    b: np.ndarray,
    Xs: np.ndarray,
    Xt: np.ndarray,
    n_projections: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Approximate coupling as the mean of exact 1-D OT couplings over slices."""
    rng = np.random.default_rng(seed)
    d = Xs.shape[1]
    P = np.zeros((Xs.shape[0], Xt.shape[0]), dtype=np.float64)
    for _ in range(n_projections):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        ps, pt = Xs @ v, Xt @ v
        # exact 1-D OT plan for the projected weighted point clouds
        G = ot.emd_1d(ps, pt, a, b, metric="sqeuclidean", dense=True)
        P += G
    P /= n_projections
    return P


def solve_ot(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    method: str,
    **kwargs,
) -> np.ndarray:
    """Solve a discrete OT problem and return a transport plan ``P`` (K_s, K_t).

    Parameters
    ----------
    a, b : (K_s,), (K_t,) source / target weights (each sums to 1).
    C    : (K_s, K_t) cost matrix.
    method : one of :data:`METHODS`.

    Common kwargs
    -------------
    num_iter_max : int              iteration cap for iterative solvers.
    reg : float                     entropic reg for "sinkhorn" / "unbalanced".
    reg_m : float                   marginal KL relaxation for "unbalanced".
    m : float in (0, 1]             mass to transport for "partial".
    Xs, Xt : (K, 3)                 cluster centers, required for "sliced".
    n_projections, seed : int       for "sliced".
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    method = method.lower()

    if method == "emd":
        return ot.emd(a, b, C, numItermax=kwargs.get("num_iter_max", 200_000))

    if method == "sinkhorn":
        reg = float(kwargs.get("reg", _default_reg(C)))
        return ot.sinkhorn(
            a, b, C, reg,
            numItermax=kwargs.get("num_iter_max", 2000),
            stopThr=kwargs.get("stop_thr", 1e-9),
        )

    if method == "unbalanced":
        reg = float(kwargs.get("reg", _default_reg(C)))
        reg_m = float(kwargs.get("reg_m", 0.5 * float(np.median(C[C > 0]))))
        return ot.unbalanced.sinkhorn_unbalanced(
            a, b, C, reg, reg_m,
            numItermax=kwargs.get("num_iter_max", 2000),
            stopThr=kwargs.get("stop_thr", 1e-9),
        )

    if method == "partial":
        m = float(kwargs.get("m", 0.7))
        m = min(m, float(a.sum()), float(b.sum()))
        return ot.partial.partial_wasserstein(
            a, b, C, m=m,
            numItermax=kwargs.get("num_iter_max", 100_000),
        )

    if method == "sliced":
        Xs = kwargs.get("Xs")
        Xt = kwargs.get("Xt")
        if Xs is None or Xt is None:
            raise ValueError("method='sliced' requires Xs and Xt keyword args")
        return _sliced_plan(
            a, b,
            np.asarray(Xs, dtype=np.float64),
            np.asarray(Xt, dtype=np.float64),
            n_projections=int(kwargs.get("n_projections", 100)),
            seed=int(kwargs.get("seed", 0)),
        )

    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def sliced_wasserstein(
    Xs: np.ndarray,
    Xt: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    n_projections: int = 200,
    seed: int = 0,
) -> float:
    """POT's sliced Wasserstein-2 distance between two Lab point clouds."""
    return float(
        ot.sliced_wasserstein_distance(
            np.asarray(Xs, dtype=np.float64),
            np.asarray(Xt, dtype=np.float64),
            a=a, b=b,
            n_projections=n_projections,
            p=2,
            seed=seed,
        )
    )


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from color_pipeline import build_ot_problem, make_synthetic_pair

    src, tgt = make_synthetic_pair(size=160, seed=0)
    prob = build_ot_problem(src, tgt, k=256, random_state=0)
    a, b, C = prob.a, prob.b, prob.C

    print(f"problem: K_s={len(a)}  K_t={len(b)}  "
          f"median(C)={np.median(C[C > 0]):.1f}  default reg={_default_reg(C):.1f}\n")

    configs = {
        "emd": {},
        "sinkhorn": {},
        "unbalanced": {},
        "partial": {"m": 0.6},
        "sliced": {"Xs": prob.Xs, "Xt": prob.Xt, "n_projections": 100},
    }

    print(f"{'method':<12} {'plan sum':>9} {'row-marg L1':>12} "
          f"{'col-marg L1':>12} {'<C,P> cost':>12}")
    print("-" * 62)
    for method, kw in configs.items():
        P = solve_ot(a, b, C, method, **kw)
        mass = P.sum()
        row_err = np.abs(P.sum(1) - a).sum()
        col_err = np.abs(P.sum(0) - b).sum()
        cost = float((P * C).sum()) / max(mass, 1e-12)
        print(f"{method:<12} {mass:9.4f} {row_err:12.4e} "
              f"{col_err:12.4e} {cost:12.2f}")

    sw = sliced_wasserstein(prob.Xs, prob.Xt, a, b)
    print(f"\not.sliced_wasserstein_distance (p=2): {sw:.3f}")
    print("\nOK")
