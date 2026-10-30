#!/usr/bin/env python3
"""
lift_verify.py -- minimal verification script for the LIFT-R project.

Follows the notation of "Optimizing Lyapunov Certificates via Stability-Preserving
Quadratization for Polynomial Systems" (ODQ, arXiv 2609.18871):

    w' = A(L) w + H(L) (w kron w),   A(L) = A0 - sum_i l_i e_{n+i} e_{n+i}^T,
    H(L) = H0 + sum_i l_i e_{n+i} s_i^T,           stabilizers h_i = y_i - a_i b_i.

Parts
  A  Manifold invariance for the scalar cubic  x' = -x - x^3,  y = x^2
     (on-manifold residual, plus off-manifold decay vs the closed form h' = -(2+l+2y) h).
  B  Lyapunov solve and certificate radius for the cubic, sweeping the gain l:
     rho_an (Prop. 2.7, default representation) vs rho* = 1/phi (gauge-optimised, via SDP).
  C  Planar quintic (paper Sec. 5.1): rho*(Lambda) at Lambda=0 and on a small grid.

Run:   python lift_verify.py            (saves figures/ and prints a summary)
       python lift_verify.py --no-quintic   (skip Part C)

Requires: numpy, scipy, matplotlib, cvxpy.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp

import cvxpy as cp

FIG_DIR = "figures"


# --------------------------------------------------------------------------
# Generic helpers (any lifted system  w' = A w + H (w kron w))
# --------------------------------------------------------------------------
def kron_ij(N, i, j):
    """Row vector in R^{N^2} that picks w_i * w_j out of (w kron w)."""
    v = np.zeros(N * N)
    v[i * N + j] = 1.0
    return v


def sym_pair(N, i, j):
    """Symmetric Kronecker encoding of the monomial w_i w_j."""
    return 0.5 * (kron_ij(N, i, j) + kron_ij(N, j, i))


def rhs(A, H):
    return lambda t, w: A @ w + H @ np.kron(w, w)


def lyapunov_P(A, Q=None):
    """Solve A^T P + P A = -Q  (Q = I by default)."""
    N = A.shape[0]
    Q = np.eye(N) if Q is None else Q
    return sla.solve_continuous_lyapunov(A.T, -Q)


def rho_an(A, H, Q=None):
    """Analytic radius of Prop. 2.7:  lam_min(Q) / (2 ||H|| sqrt(||P||))."""
    N = A.shape[0]
    Q = np.eye(N) if Q is None else Q
    P = lyapunov_P(A, Q)
    nH = np.linalg.norm(H, 2)
    if nH == 0:
        return np.inf
    return np.linalg.eigvalsh(Q).min() / (2 * nH * np.sqrt(np.linalg.norm(P, 2)))


def gauge_basis(N):
    """Basis of {K : K (w kron w) = 0}: each row of K reshapes to a skew-symmetric matrix."""
    basis = []
    for r in range(N):
        for i in range(N):
            for j in range(i + 1, N):
                K = np.zeros((N, N * N))
                K[r, i * N + j] = 1.0
                K[r, j * N + i] = -1.0
                basis.append(K)
    return basis  # length N^2 (N-1) / 2


def G_matrix(H, P):
    """G = col_i (M_i^T P + P M_i), with (M_i)_{rj} = H[r, i*N + j]  (eq. 3.12)."""
    N = P.shape[0]
    blocks = []
    for i in range(N):
        Mi = H[:, i * N:(i + 1) * N]
        blocks.append(Mi.T @ P + P @ Mi)
    return np.vstack(blocks)  # N^2 x N


def J_matrix(H, P):
    """J = (P_f^T kron I)^{-1} G with P = P_f^T P_f (eq. 3.12)."""
    N = P.shape[0]
    Pf = sla.cholesky(P, lower=False)  # upper, Pf^T Pf = P
    X = np.kron(Pf.T, np.eye(N))
    return np.linalg.solve(X, G_matrix(H, P))


def phi_gauge_opt(A, H):
    """phi = min_mu || J(H + K(mu)) ||_2   (inner SDP, eq. 3.15). Returns (phi, J_opt)."""
    N = A.shape[0]
    P = lyapunov_P(A)
    J0 = J_matrix(H, P)
    Js = [J_matrix(K, P) for K in gauge_basis(N)]  # J is linear in H
    mu = cp.Variable(len(Js))
    J = J0 + sum(mu[k] * Js[k] for k in range(len(Js)))
    prob = cp.Problem(cp.Minimize(cp.sigma_max(J)))
    prob.solve(solver=cp.CLARABEL)
    phi = float(prob.value)
    J_opt = J0 + sum(float(mu.value[k]) * Js[k] for k in range(len(Js)))
    return phi, J_opt


# --------------------------------------------------------------------------
# Scalar cubic:  x' = -x - x^3,  y = x^2,  stabiliser h = y - x^2
#   w = (x, y):   x' = -x - x y,   y' = -(2+l) y + l x^2 - 2 y^2
# --------------------------------------------------------------------------
def cubic_lift(lam):
    A = np.diag([-1.0, -(2.0 + lam)])
    H = np.zeros((2, 4))
    H[0] += -sym_pair(2, 0, 1)          # -x y
    H[1] += lam * sym_pair(2, 0, 0)     # +l x^2
    H[1] += -2.0 * sym_pair(2, 1, 1)    # -2 y^2
    return A, H


def part_A(lam=1.0, x0=0.5, T=8.0):
    print(f"\n=== Part A: manifold invariance (cubic, lambda={lam}, x0={x0}) ===")
    t_eval = np.linspace(0, T, 8001)
    opts = dict(rtol=1e-11, atol=1e-13, dense_output=True)
    orig = solve_ivp(lambda t, x: -x - x ** 3, [0, T], [x0], **opts)
    A, H = cubic_lift(lam)
    lift = solve_ivp(rhs(A, H), [0, T], [x0, x0 ** 2], **opts)
    xs = lift.sol(t_eval)
    res_on = np.abs(xs[1] - xs[0] ** 2)
    diff = np.abs(xs[0] - orig.sol(t_eval)[0])
    print(f"  max |y - x^2| on manifold     = {res_on.max():.2e}")
    print(f"  max |x_lift - x_original|     = {diff.max():.2e}")

    # Start OFF the manifold: h = y - x^2 obeys  h' = -(2 + lam + 2y) h  exactly.
    h0 = 0.3
    off = solve_ivp(rhs(A, H), [0, T], [x0, x0 ** 2 + h0], **opts)
    xo = off.sol(t_eval)
    h_num = xo[1] - xo[0] ** 2
    integral = np.concatenate(([0.0], np.cumsum(
        0.5 * (2 + lam + 2 * xo[1][1:] + 2 + lam + 2 * xo[1][:-1]) * np.diff(t_eval))))
    h_pred = h0 * np.exp(-integral)
    err = np.abs(h_num - h_pred).max()
    print(f"  off-manifold: max |h_num - h_closed_form| = {err:.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    ax[0].semilogy(t_eval, res_on + 1e-17)
    ax[0].set(xlabel="t", ylabel=r"$|y-x^2|$", title="On-manifold residual (numerical noise)")
    ax[1].semilogy(t_eval, np.abs(h_num), label="simulation")
    ax[1].semilogy(t_eval, np.abs(h_pred), "--", label=r"$h_0 e^{-\int(2+\lambda+2y)}$")
    ax[1].set(xlabel="t", ylabel=r"$|h|$", title="Off-manifold decay of $h=y-x^2$")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "A_manifold.png"), dpi=200)
    plt.close(fig)
    return res_on.max(), err


def part_B(lams=None):
    print("\n=== Part B: certificate radius vs gain (cubic) ===")
    lams = np.linspace(0.0, 10.0, 21) if lams is None else lams
    rows = []
    for lam in lams:
        A, H = cubic_lift(lam)
        P = lyapunov_P(A)
        phi, _ = phi_gauge_opt(A, H)
        s = lam + 2.0
        rows.append((lam, np.linalg.norm(P, 2), np.linalg.norm(H, 2),
                     rho_an(A, H), 1.0 / phi, np.sqrt(s) / (2 * np.sqrt(2))))
    rows = np.array(rows)
    print("   lam   ||P||   ||H||   rho_an   rho*(SDP)   rho*(paper Ex.3.11)")
    for r in rows[::4]:
        print(f"  {r[0]:5.2f}  {r[1]:.3f}  {r[2]:6.3f}  {r[3]:.4f}   {r[4]:.4f}      {r[5]:.4f}")
    print(f"  max |rho*(SDP) - closed form| = {np.abs(rows[:, 4] - rows[:, 5]).max():.2e}")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(rows[:, 0], rows[:, 3], "o-", label=r"$\rho_{an}$ (default representation)")
    ax.plot(rows[:, 0], rows[:, 4], "s-", label=r"$\rho^\star=1/\phi$ (gauge-optimised SDP)")
    ax.plot(rows[:, 0], rows[:, 5], "k--", lw=1, label=r"$\sqrt{\lambda+2}/(2\sqrt{2})$")
    ax.set(xlabel=r"stabiliser gain $\lambda$", ylabel="certified radius (P-norm)",
           title=r"Cubic: $\dot x=-x-x^3$, $y=x^2$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "B_radius_vs_gain.png"), dpi=200)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
# Planar quintic (paper Sec. 5.1):  x1' = -x1 + x2,  x2' = -2 x2 - x1^5
#   y1 = x1^2, y2 = x1^3;  h1 = y1 - x1 x1,  h2 = y2 - x1 y1;   w = (x1, x2, y1, y2)
# --------------------------------------------------------------------------
def quintic_lift(l1, l2):
    N, n = 4, 2
    A0 = np.array([[-1, 1, 0, 0],
                   [0, -2, 0, 0],
                   [0, 0, -2, 0],
                   [0, 0, 0, -3]], dtype=float)
    H0 = np.zeros((N, N * N))
    H0[1] += -sym_pair(N, 2, 3)          # x2' : -y1 y2
    H0[2] += 2.0 * sym_pair(N, 0, 1)     # y1' : +2 x1 x2
    H0[3] += 3.0 * sym_pair(N, 1, 2)     # y2' : +3 x2 y1
    A, H = A0.copy(), H0.copy()
    s = [sym_pair(N, 0, 0), sym_pair(N, 0, 2)]   # a1 b1 = x1 x1,  a2 b2 = x1 y1
    for i, lam in enumerate((l1, l2)):
        A[n + i, n + i] -= lam
        H[n + i] += lam * s[i]
    return A, H


def part_C(grid=7, hi=3.0):
    print("\n=== Part C: planar quintic (paper Sec. 5.1) ===")
    for name, lam in [("Lambda = (0, 0)", (0.0, 0.0)),
                      ("Lambda = (0.7337, 1.5003)  [paper's ODQ gain]", (0.7337, 1.5003))]:
        A, H = quintic_lift(*lam)
        assert np.linalg.eigvals(A).real.max() < 0, "A must be Hurwitz"
        phi, _ = phi_gauge_opt(A, H)
        print(f"  {name}: rho* = 1/phi = {1 / phi:.4f}   rho_an = {rho_an(A, H):.4f}")
    print("  paper reports rho_safe = 1.832534 (Lambda=0) and 3.040679 (ODQ gain); "
          "rho_safe is slightly below rho*.")

    ls = np.linspace(0, hi, grid)
    Z = np.zeros((grid, grid))
    for a, l1 in enumerate(ls):
        for b, l2 in enumerate(ls):
            A, H = quintic_lift(l1, l2)
            Z[b, a] = 1.0 / phi_gauge_opt(A, H)[0]
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    cs = ax.contourf(ls, ls, Z, levels=14)
    fig.colorbar(cs, label=r"$\rho^\star(\Lambda)$")
    ax.plot([0.7337], [1.5003], "r*", ms=12, label="paper's ODQ gain")
    ax.set(xlabel=r"$\lambda_1$", ylabel=r"$\lambda_2$", title="Planar quintic: gauge-optimised radius")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "C_quintic_grid.png"), dpi=200)
    plt.close(fig)
    i = np.unravel_index(Z.argmax(), Z.shape)
    print(f"  grid max rho* = {Z.max():.4f} at (l1, l2) = ({ls[i[1]]:.2f}, {ls[i[0]]:.2f})")
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-quintic", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    part_A()
    part_B()
    if not args.no_quintic:
        part_C()
    print(f"\nFigures saved in ./{FIG_DIR}/")


if __name__ == "__main__":
    main()