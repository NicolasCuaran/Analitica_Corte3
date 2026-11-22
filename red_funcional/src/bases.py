"""Bases B-spline, matrices de Gram por Gauss-Legendre, MCO enmascarado y elección de K por VCG."""
import numpy as np
from scipy.interpolate import BSpline

from .comun import N_DIAS

GRADO = 3


def nodos(K):
    return np.r_[[1.0] * (GRADO + 1), np.linspace(1, N_DIAS, K - GRADO + 1)[1:-1], [float(N_DIAS)] * (GRADO + 1)]


def base(t, K, deriv=0):
    B = BSpline(nodos(K), np.eye(K), GRADO, extrapolate=False)
    B = B.derivative(deriv) if deriv else B
    return np.nan_to_num(B(np.clip(np.asarray(t, float), 1, N_DIAS)))       # Dim: (len(t), K)


def gram(K, deriv=0, n_min=64):
    # W = int phi^(d) phi^(d)' dt con Gauss-Legendre compuesto entre nodos (exacto para productos de polinomios por tramos)
    br = np.unique(nodos(K)); n = int(np.ceil(n_min / (len(br) - 1)))
    x, w = np.polynomial.legendre.leggauss(n)
    a, b = br[:-1, None], br[1:, None]
    tq = ((b - a) / 2 * x + (a + b) / 2).ravel(); wq = ((b - a) / 2 * w).ravel()
    P = base(tq, K, deriv)
    return (P * wq[:, None]).T @ P, tq.size


def mco_enmascarado(Phi, Y, R=None, lam=0.0):
    # c_p = (Phi_o' Phi_o + lam R)^-1 Phi_o' y_o para todas las curvas a la vez; Y (365, P) con NaN donde no hay dato
    M = ~np.isnan(Y); Y0 = np.where(M, Y, 0.0)
    G = np.einsum("tp,tk,tl->pkl", M.astype(float), Phi, Phi)                       # Dim: (P, K, K)
    A = G + lam * R if R is not None else G
    b = Y0.T @ Phi                                                                  # Dim: (P, K)
    ok = np.linalg.cond(A) < 1e10
    C = np.full(b.shape, np.nan)
    C[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    nu = np.full(len(b), np.nan)
    nu[ok] = np.trace(np.linalg.solve(A[ok], G[ok]), axis1=1, axis2=2)
    E = (Y0 - Phi @ np.nan_to_num(C).T) * M
    return C.T, (E ** 2).sum(0), M.sum(0), nu                                       # C: (K, P)


def k_por_vcg(Y, t_obs, grilla=range(4, 21)):
    # K de la base que minimiza la VCG media de las curvas
    def vcg(K):
        C, sce, n, _ = mco_enmascarado(base(t_obs, K), Y)
        v = (sce / n) / (1 - K / n) ** 2
        v[~np.isfinite(C).all(0)] = np.nan
        return v
    V = np.array([vcg(K) for K in grilla])
    return list(grilla)[int(np.argmin(V[:, np.isfinite(V).all(0)].mean(1)))]


def raiz_curva(Phi, C):
    # Raíz de la curva de intensidad media (no de cada día), proyectada por MCO en la base Phi
    return np.linalg.lstsq(Phi, np.sqrt(np.maximum(Phi @ np.nan_to_num(C), 0)), rcond=None)[0]
