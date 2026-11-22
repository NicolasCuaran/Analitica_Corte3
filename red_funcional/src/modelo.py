"""Red funcional: capa KAN, cabezas lineal, MLP y KAN, codificador z_k = a_k' W c y decodificador FPCA congelado."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .curvas import M_FPCA, fpca
from .datos import FASES

K_NEU = 6                                   # neuronas funcionales (6 <= K <= 8)
UMBRAL_PODA = 0.10                          # arista activa: amplitud > 10 % de la mayor
COVARIABLES = ["altitud", "oni"]
CABEZAS = ("lineal", "mlp", "kan")
NOMBRES = {"satelite": "Satélite sin corregir", "lineal": "Cabeza lineal", "mlp": "Cabeza MLP", "kan": "Cabeza KAN"}
COMBOS = {"satélite": [], "satélite + altitud": ["altitud"], "satélite + ONI": ["oni"], "satélite + altitud + ONI": COVARIABLES}


class KANLayer(nn.Module):
    # Capa KAN estilo FastKAN: phi_oi(x) = w_b[o,i] SiLU(x) + sum_g c[o,i,g] exp(-((x - mu_g)/h)^2).
    # norm="capa": LayerNorm por muestra (phi_oi deja de ser función de x_i sola); norm="variable": BatchNorm sin afinidad,
    # cada entrada con su media y varianza del entrenamiento (phi_oi univariada, forma de Kolmogorov-Arnold).
    def __init__(self, in_features, out_features, grid_size=8, grid_range=(-2.0, 2.0), norm="capa", spline_init_std=0.1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("grid", grid)
        self.h = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        if norm == "capa":
            self.norm = nn.LayerNorm(in_features)
        elif norm == "variable":
            self.norm = nn.BatchNorm1d(in_features, affine=False)
        else:
            self.norm = nn.Identity()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) * spline_init_std)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.register_buffer("mascara", torch.ones(out_features, in_features))      # poda: 0 = arista retirada

    def basis(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.h) ** 2)              # Dim: (B, in) -> (B, in, G)

    def forward(self, x):
        x = self.norm(x)
        return (F.silu(x) @ (self.base_weight * self.mascara).T
                + torch.einsum("big,oig->bo", self.basis(x), self.spline_weight * self.mascara[..., None]))   # Dim: (B, out)

    @torch.no_grad()
    def edge_function(self, out_idx, in_idx, x_grid):
        b = self.basis(x_grid[:, None])[:, 0, :]
        return self.mascara[out_idx, in_idx] * (self.base_weight[out_idx, in_idx] * F.silu(x_grid) + b @ self.spline_weight[out_idx, in_idx])

    def spline_l1(self):
        # L1 sobre los coeficientes: favorece aristas simples
        return self.spline_weight.abs().mean()

    def entropia(self):
        # Entropía de la distribución de magnitudes de las aristas: baja cuando pocas aristas concentran la señal
        a = (self.spline_weight.abs().mean(-1) + self.base_weight.abs()) * self.mascara
        p = a / (a.sum() + 1e-12)
        return -(p * torch.log(p + 1e-12)).sum()

    @torch.no_grad()
    def podar(self, amp):
        # Retira las aristas con amplitud < UMBRAL_PODA de la mayor
        self.mascara[torch.tensor(amp < UMBRAL_PODA * amp.max())] = 0.0


def make_head(cfg, in_features=None, out_features=M_FPCA):
    # Cabeza sobre x = [z, covariables]: "lineal", "mlp" o KAN con forma "aditiva" (s_m = sum_i phi_mi(x_i); con phi_mi rectas
    # es la lineal), "ka" (s_m = sum_q Phi_mq(sum_p phi_qp(x_p)), con interacciones) o "proyeccion" (Linear-Dropout-KANLayer)
    in_features = in_features or K_NEU + len(cfg.get("cov", COVARIABLES))
    if cfg["cabeza"] == "lineal":
        return nn.Linear(in_features, out_features)
    if cfg["cabeza"] == "mlp":
        return nn.Sequential(nn.Linear(in_features, cfg["ancho"]), nn.GELU(), nn.Dropout(cfg["drop"]), nn.Linear(cfg["ancho"], out_features))
    if cfg["forma"] == "proyeccion":
        return nn.Sequential(nn.Linear(in_features, cfg["ancho"]), nn.Dropout(cfg["drop"]), KANLayer(cfg["ancho"], out_features, cfg["G"]))
    if cfg["forma"] == "aditiva":
        return nn.Sequential(KANLayer(in_features, out_features, cfg["G"], norm="variable"))
    if cfg["forma"] == "ka":
        return nn.Sequential(KANLayer(in_features, cfg["ancho"], cfg["G"], norm="variable"), KANLayer(cfg["ancho"], out_features, cfg["G"], norm="variable"))
    raise ValueError(cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def entrada(ctx, i):
    # Coeficientes sqrt del satélite y escalares [altitud normalizada, ONI medio del año, fase ENSO (0 Niño, 1 Niña, 2 neutro)];
    # la cabeza solo ve las covariables de cfg["cov"] y la fase solo la usa el decodificador por fase
    return (torch.tensor(ctx.Cxs[:, i].T), torch.tensor(np.c_[ctx.alt_n[i], ctx.oni_a[i], ctx.fase_i[i]]))


def objetivo(ctx, i):
    return torch.tensor(ctx.Cys[:, i].T)


def perdida_l2(c_hat, c, W):
    # Pérdida L2 funcional (c_hat - c)' W (c_hat - c), media del lote
    return torch.einsum("bk,kl,bl->b", c_hat - c, W, c_hat - c).mean()


def preparar(ctx, i):
    # Lo que se ajusta con las curvas i: media y autofunciones FPCA comunes y por fase ENSO, escala inicial del codificador y
    # escala de la L1 (varianza funcional del entrenamiento); incluye las matrices W y R del satélite
    mu, Ps, _ = fpca(ctx, i)
    por_fase = [fpca(ctx, i[ctx.fase_i[i] == f]) for f in range(len(FASES))]
    Dc = ctx.Cys[:, i] - mu[:, None]
    return dict(mu=mu, Psi=Ps, mu_f=np.stack([p[0] for p in por_fase]), Psi_f=np.stack([p[1] for p in por_fase]),
                U=np.abs(ctx.Cxs[:, i].T @ ctx.W_x).mean(), esc_l1=float(np.einsum("kb,kl,lb->b", Dc, ctx.W_y, Dc).mean()),
                W=ctx.W_x, R=ctx.R_x)


class RedFuncional(nn.Module):
    # Codificador z_k = a_k' W c (w_k(t) = phi(t)' a_k), cabeza y decodificador FPCA congelado (común o por fase ENSO)
    def __init__(self, cfg, base):
        super().__init__()
        self.cfg = dict(cfg)
        K_X = base["W"].shape[0]
        self.register_buffer("W", torch.tensor(base["W"]))                          # Dim: (K_X, K_X), Gram congelada
        self.register_buffer("R", torch.tensor(base["R"]))                          # Dim: (K_X, K_X), rugosidad int phi'' phi''
        self.register_buffer("mu", torch.tensor(base["mu"]))                        # Dim: (K_Y,), media FPCA congelada
        self.register_buffer("Psi", torch.tensor(base["Psi"]))                      # Dim: (M, K_Y), autofunciones congeladas
        self.register_buffer("mu_f", torch.tensor(base["mu_f"]))                    # Dim: (3, K_Y), por fase ENSO
        self.register_buffer("Psi_f", torch.tensor(base["Psi_f"]))                  # Dim: (3, M, K_Y), por fase ENSO
        self.register_buffer("cols", torch.tensor([COVARIABLES.index(c) for c in cfg.get("cov", COVARIABLES)], dtype=torch.long))
        self.A = nn.Parameter(torch.randn(K_NEU, K_X) / (base["U"] * math.sqrt(K_X)))  # w_k(t) = phi(t)' a_k
        self.head = make_head(cfg)

    def forward(self, cx, esc, scores=False):
        z = cx @ self.W @ self.A.T                                                  # Dim: (B, K_X) -> (B, K_NEU), z_k = a_k' W c
        s = self.head(torch.cat([z, esc[:, self.cols]], 1))                         # Dim: (B, K_NEU + n_cov) -> (B, M)
        if self.cfg.get("decod", "comun") == "por_fase":
            f = esc[:, 2].long()
            c = self.mu_f[f] + torch.einsum("bm,bmk->bk", s, self.Psi_f[f])
        else:
            c = self.mu + s @ self.Psi                                              # Dim: (B, M) -> (B, K_Y), coeficientes de sqrt(y)
        return (c, s, z) if scores else c

    def rugosidad(self):
        return torch.einsum("kj,jl,kl->", self.A, self.R, self.A)                   # sum_k a_k' R a_k

    def l1(self):
        return sum(c.spline_l1() for c in self.head.modules() if isinstance(c, KANLayer))

    def entropia(self):
        return sum(c.entropia() for c in self.head.modules() if isinstance(c, KANLayer))
