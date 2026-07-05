"""PCA projection and fixed-point helpers for the animated state-space figure."""

from __future__ import annotations

import numpy as np

from .constants import _N_PCA_COMPONENTS


def _compute_pca_projection(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (n_neurons, n_steps) activity matrix onto its first 3 PCs.

    Returns
    -------
    coords   : (n_steps, 3)  PC coordinates for every time step
    components : (3, n_neurons)  the three principal directions
    explained  : (3,)  fraction of variance explained by each PC
    """
    X = activity - activity.mean(axis=1, keepdims=True)   # (n, T)
    n, T = activity.shape
    n_comp = _N_PCA_COMPONENTS

    U, s, Vt = np.linalg.svd(X.T, full_matrices=False)    # U:(T,k), s:(k,), Vt:(k,n)
    k = s.shape[0]
    total_var = (s ** 2).sum() or 1.0

    coords = U * s  # (T, k)

    if k < n_comp:
        coords = np.hstack([coords, np.zeros((T, n_comp - k))])
        components = np.vstack([Vt, np.zeros((n_comp - k, n))])
        explained = np.concatenate([(s ** 2) / total_var, np.zeros(n_comp - k)])
    else:
        components = Vt[:n_comp]
        explained = (s[:n_comp] ** 2) / total_var

    return coords[:, :n_comp], components[:n_comp], explained[:n_comp]


def _find_fixed_points_pca(
    weight_matrix: np.ndarray,
    activity: np.ndarray,
    components: np.ndarray,
    n_candidates: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate fixed points and project them into PCA space.

    Strategy
    --------
    We take several candidate states from the activity trajectory (evenly
    spaced), then refine each one via a short fixed-point iteration:
        x <- tanh-inverse of the effective drive  (Newton-like relaxation)
    In practice we use a simple gradient-descent relaxation:
        x <- x - alpha * (-x + W @ tanh(x))
    until convergence, which is fast for well-conditioned networks.

    Each converged point is then classified as stable or unstable by the
    spectral radius of its Jacobian  J = -I + W * diag(sech²(x*)).
    If all eigenvalues of J have Re < 0 the point is stable; otherwise unstable.

    Returns
    -------
    fp_pca    : (m, k)  unique fixed points in PC space, k = components.shape[0]
    stability : (m,)  bool array, True = stable
    """
    n, T = activity.shape
    W = weight_matrix
    alpha = 0.05
    max_iter = 400
    tol = 1e-6

    # Candidate starting points: evenly sampled from trajectory.
    cand_indices = np.linspace(0, T - 1, n_candidates, dtype=int)
    candidates = [activity[:, i].copy() for i in cand_indices]

    fps_raw: list[np.ndarray] = []
    stabilities: list[bool] = []

    for x0 in candidates:
        x = x0.copy()
        for _ in range(max_iter):
            dx = -x + W @ np.tanh(x)
            x = x + alpha * dx
            if np.linalg.norm(dx) < tol:
                break
        # Deduplicate: skip if too close to an existing fixed point.
        is_dup = any(np.linalg.norm(x - fp) < 0.1 for fp in fps_raw)
        if not is_dup:
            fps_raw.append(x.copy())
            # Stability: Jacobian eigenvalues.
            sech2 = 1.0 / (np.cosh(x) ** 2)          # (n,)
            J = -np.eye(n) + W * sech2[np.newaxis, :]  # (n, n)
            eigvals = np.linalg.eigvals(J)
            stable = bool(np.all(eigvals.real < 0))
            stabilities.append(stable)

    if not fps_raw:
        return np.empty((0, components.shape[0])), np.empty(0, dtype=bool)

    # Project fixed points into PCA space using the same components.
    X_mean = activity.mean(axis=1)                         # (n,)
    fp_matrix = np.array(fps_raw)                          # (m, n)
    fp_centred = fp_matrix - X_mean[np.newaxis, :]         # (m, n)
    fp_pca = fp_centred @ components.T                     # (m, k)

    return fp_pca, np.array(stabilities)
