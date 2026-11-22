"""Curvas funcionales de estaciones y satélite, FPCA del decodificador, pliegues de validación cruzada y referencia de picos."""
import numpy as np
import pandas as pd
import torch

from .bases import base, gram, k_por_vcg, mco_enmascarado, raiz_curva
from .comun import N_DIAS, PLIEGUES_CV
from .datos import ANIOS_ENT, anota, col_pixel

M_FPCA = 5                                  # componentes FPCA (4 <= M <= 6)
K_PEN = 55
t_dia = np.arange(1, N_DIAS + 1, dtype=float)


def construir(ctx):
    # K por VCG con el entrenamiento, coeficientes MCO, raíz de las curvas, verdad y satélite en mm/día y matrices de Gram
    meta, uso, Yd, Xp, anios, t_sat, Xs = ctx.meta, ctx.uso, ctx.Yd, ctx.Xp, ctx.anios, ctx.t_sat, ctx.Xs
    K_Y = k_por_vcg(Yd[:, uso["entrenamiento"]], t_dia)
    pix_ciegos = col_pixel(ctx, meta.pixel[meta.ciega.astype(str).eq("True")].astype(int).unique())
    K_X = k_por_vcg(np.delete(Xp[anios <= ANIOS_ENT[1]], pix_ciegos, axis=2).transpose(1, 0, 2).reshape(72, -1), t_sat)
    Phi_y, Phi_x = base(t_dia, K_Y), base(t_dia, K_X)
    Cy = mco_enmascarado(Phi_y, Yd)[0]
    Cx = mco_enmascarado(base(t_sat, K_X), Xs)[0]
    Cys, Cxs = raiz_curva(Phi_y, Cy), raiz_curva(Phi_x, Cx)
    valida = np.isfinite(Cy).all(0) & np.isfinite(Cx).all(0)
    idx = {k: np.flatnonzero(v & valida) for k, v in uso.items()}
    Y_true = np.maximum(Phi_y @ np.nan_to_num(Cy), 0)                                    # verdad en mm/día (curva MCO de la estación)
    X_mm = np.maximum(Phi_x @ np.nan_to_num(Cx), 0)                                      # satélite sin corregir en mm/día
    W_x, nq = gram(K_X); W_y, _ = gram(K_Y); R_x, _ = gram(K_X, deriv=2)
    t_fino = np.linspace(1, N_DIAS, 36401); w_fino = np.r_[0.5, np.ones(36399), 0.5] * (t_fino[1] - t_fino[0])
    Pf = base(t_fino, K_X)
    anota(ctx, "Gram W (Gauss-Legendre) frente al trapecio fino, máx. relativo", np.abs(W_x - (Pf * w_fino[:, None]).T @ Pf).max() / np.abs(W_x).max(), 1e-6)

    tr = idx["entrenamiento"]
    anio_m = meta.year.to_numpy()
    ev, U = np.linalg.eigh(W_y); W_h = U @ np.diag(np.sqrt(ev)) @ U.T; W_hi = U @ np.diag(1 / np.sqrt(ev)) @ U.T
    vars(ctx).update(K_Y=K_Y, K_X=K_X, Phi_y=Phi_y, Phi_x=Phi_x, Cy=Cy, Cx=Cx, Cys=Cys, Cxs=Cxs, idx=idx, Y_true=Y_true,
                     X_mm=X_mm, W_x=W_x, W_y=W_y, R_x=R_x, nq=nq, w_fino=w_fino, Pf=Pf, tr=tr, anio_m=anio_m, W_h=W_h, W_hi=W_hi)
    ctx.Wy_t = torch.tensor(W_y)
    ctx.ESC_R = np.trace(W_x) / np.trace(R_x)
    return pd.DataFrame({"vista": list(idx), "curvas": [len(v) for v in idx.values()],
                         "estaciones": [meta.station_id[v].nunique() for v in idx.values()]})


def fpca(ctx, i):
    # Media y M autofunciones (ortonormales en W_y) de las curvas sqrt de i: autovectores de W^1/2 Cov(C) W^1/2
    mu = ctx.Cys[:, i].mean(1)
    lam, V = np.linalg.eigh(ctx.W_h @ np.cov(ctx.Cys[:, i]) @ ctx.W_h)
    lam, V = lam[::-1], V[:, ::-1]
    Ps = (ctx.W_hi @ V[:, :M_FPCA]).T                                                    # Dim: (M, K_Y)
    return mu, Ps * np.sign(Ps.sum(1, keepdims=True)), lam


def ajustar_fpca(ctx):
    # FPCA de las curvas de entrenamiento (base común del decodificador) y pliegues cronológicos de ventana creciente:
    # el pliegue f entrena con los años anteriores a su bloque y valida en el bloque
    tr, anio_m = ctx.tr, ctx.anio_m
    mu_c, Psi, lam_fp = fpca(ctx, tr)
    var_exp = np.cumsum(lam_fp) / lam_fp.sum()
    anota(ctx, "Ortonormalidad de las autofunciones, máx|Psi W Psi' - I|", np.abs(Psi @ ctx.W_y @ Psi.T - np.eye(M_FPCA)).max(), 1e-10)
    pliegues = [(tr[anio_m[tr] < a0], tr[(anio_m[tr] >= a0) & (anio_m[tr] <= a1)]) for a0, a1 in PLIEGUES_CV]
    varianza = pd.DataFrame({"M": np.arange(1, ctx.K_Y + 1), "autovalor": lam_fp, "varianza_explicada_acumulada": var_exp})
    ctx.tablas["fpca_varianza"] = varianza
    vars(ctx).update(mu_c=mu_c, Psi=Psi, lam_fp=lam_fp, pliegues=pliegues)
    folds = pd.DataFrame([dict(pliegue=f + 1, entrena=f"{ANIOS_ENT[0]}-{a0 - 1}", curvas_entrenamiento=len(i_),
                               valida=f"{a0}-{a1}", curvas_validacion=len(v_))
                          for f, ((a0, a1), (i_, v_)) in enumerate(zip(PLIEGUES_CV, pliegues))])
    return varianza, folds


def referencia_picos(ctx):
    # Curva de referencia para las fechas de los picos: suavizamiento penalizado (K = 55, lambda por VCG con entrenamiento)
    tr, idx, Yd = ctx.tr, ctx.idx, ctx.Yd
    Phi_p = base(t_dia, K_PEN); R_p = gram(K_PEN, deriv=2)[0]; esc_p = np.trace(Phi_p.T @ Phi_p) / np.trace(R_p)
    sub = np.random.default_rng(42).choice(tr, size=min(400, len(tr)), replace=False)
    vcg_p = []
    for lr_ in np.arange(-3, 3.01, 0.5):
        C_, sce, n, nu = mco_enmascarado(Phi_p, Yd[:, sub], R_p, 10 ** lr_ * esc_p)
        vcg_p.append(np.nanmean((sce / n) / (1 - nu / n) ** 2))
    lam_p = 10 ** np.arange(-3, 3.01, 0.5)[int(np.argmin(vcg_p))] * esc_p
    ev_idx = np.r_[idx["prueba"], idx["ciegas"]]
    Ref = np.full((N_DIAS, len(ctx.meta)), np.nan)
    Ref[:, ev_idx] = Phi_p @ mco_enmascarado(Phi_p, Yd[:, ev_idx], R_p, lam_p)[0]
    ctx.Ref = Ref
    return float(np.log10(lam_p / esc_p))
