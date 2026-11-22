"""Interpretación de la red: ventanas w_k, aristas de la KAN, ajuste simbólico, dependencia parcial, aporte por fase ENSO
y peores casos en las estaciones ciegas."""
import math

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit

from . import figuras
from .datos import ANIOS_ENT, FASES
from .entrenamiento import SEMILLAS, amplitudes, entrada_kan
from .evaluacion import a_mm
from .modelo import CABEZAS, K_NEU, KANLayer, entrada

CANDIDATAS = {
    "a x + b": lambda ys: (lambda x, a, b: a * x + b, [1, 0]),
    "a x^2 + b x + c": lambda ys: (lambda x, a, b, c: a * x ** 2 + b * x + c, [1, 1, 0]),
    "a x^3 + b x^2 + c x + d": lambda ys: (lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d, [1, 1, 1, 0]),
    "a sin(b x + c) + d": lambda ys: (lambda x, a, b, c, d: a * np.sin(b * x + c) + d, [np.ptp(ys) / 2, 1.5, 0, ys.mean()]),
    "a tanh(b x + c) + d": lambda ys: (lambda x, a, b, c, d: a * np.tanh(b * x + c) + d, [np.ptp(ys) / 2, 1, 0, ys.mean()]),
    "a exp(b x) + c": lambda ys: (lambda x, a, b, c: a * np.exp(b * x) + c, [1, 0.5, 0]),
}


def total_medio(ctx, kind, alt=None, oni=None):
    # Total anual medio predicho (mm) con la altitud o el ONI fijos en cada valor de la rejilla para las curvas de
    # entrenamiento (satélite y demás entradas observados); media de las 3 semillas
    tr, esc_tr, cx_tr = ctx.tr, ctx.esc_tr, ctx.cx_tr
    G_ = len(alt if alt is not None else oni); n = len(tr)
    E = esc_tr.repeat(G_, 1)                                                        # Dim: (G * n, 2)
    if alt is not None: E[:, 0] = torch.tensor(np.repeat((np.asarray(alt) - ctx.ALT_MU) / ctx.ALT_SD, n))
    if oni is not None: E[:, 1] = torch.tensor(np.repeat(np.broadcast_to(oni, G_), n))
    tot = []
    for s in SEMILLAS:
        with torch.no_grad():
            ys = ctx.Phi_y @ ctx.modelos[(kind, s)](cx_tr.repeat(G_, 1), E).numpy().T  # Dim: (365, G * n)
        tot.append(a_mm(ctx, ys, (kind, s)).sum(0).reshape(G_, n).mean(1))
    return np.mean(tot, 0)                                                          # Dim: (G,), mm/año


def curva_media(ctx, kind, oni_v, f):
    # Curva media predicha (mm/día) de las curvas de entrenamiento con el ONI y la fase fijos; media de las 3 semillas
    E = ctx.esc_tr.clone(); E[:, 1] = oni_v; E[:, 2] = f
    cs = []
    for s in SEMILLAS:
        with torch.no_grad():
            ys = ctx.Phi_y @ ctx.modelos[(kind, s)](ctx.cx_tr, E).numpy().T         # Dim: (365, n)
        cs.append(a_mm(ctx, ys, (kind, s)).mean(1))
    return np.mean(cs, 0)                                                          # Dim: (365,), mm/día


def interpretar(ctx):
    # Funciones de peso w_k(t), aristas de la primera capa KAN (con la recta de la cabeza lineal), aristas que sobreviven a la
    # poda, ajuste simbólico, dependencia parcial frente a la altitud y al ONI, y aporte o atenuación por fase ENSO
    modelos, tr, alt, oni_a, oni_t, ALT_MU, ALT_SD = ctx.modelos, ctx.tr, ctx.alt, ctx.oni_a, ctx.oni_t, ctx.ALT_MU, ctx.ALT_SD
    mk, ml = modelos[("kan", 42)], modelos[("lineal", 42)]
    figuras.ventanas_wk(ctx.Phi_x, ml, mk)

    # Aristas en el rango 1 %-99 % de cada entrada del entrenamiento; con "aditiva" y "ka" el eje va en unidades físicas y
    # con "aditiva" phi_mi es el efecto parcial exacto de x_i sobre s_m, comparable con la recta B_mi x_i de la cabeza lineal
    kan_l, _, h_in = entrada_kan(ctx, mk, tr)
    FISICA, CONTRASTE = mk.cfg["forma"] != "proyeccion", mk.cfg["forma"] == "aditiva"
    n_o, n_i = kan_l.out_features, kan_l.in_features
    nom_in = ([f"z{k + 1}" for k in range(K_NEU)] + ["altitud (m)", "ONI (°C)"]) if FISICA else [f"h{i + 1}" for i in range(n_i)]
    nom_out = [f"oculta q{o + 1}" for o in range(n_o)] if mk.cfg["forma"] == "ka" else [f"score s{o + 1}" for o in range(n_o)]
    lo, hi = np.quantile(h_in.numpy(), [0.01, 0.99], axis=0)
    B = ml.head.weight.detach().numpy()

    def a_fisico(i_, x):
        if not FISICA:
            return x
        u = kan_l.norm.running_mean[i_].item() + math.sqrt(kan_l.norm.running_var[i_].item() + kan_l.norm.eps) * x
        return ALT_MU + ALT_SD * u if i_ == K_NEU else u

    no_lin, curvas = np.zeros((n_o, n_i)), {}
    activa = kan_l.mascara.numpy() > 0                                                  # Dim: (n_o, n_i); False = arista podada
    for i_ in range(n_i):
        xg = torch.linspace(lo[i_], hi[i_], 300); xf = a_fisico(i_, xg.numpy())
        for o in range(n_o):
            y_ = kan_l.edge_function(o, i_, xg).numpy(); curvas[(o, i_)] = (xg.numpy(), xf, y_)
            if activa[o, i_] and y_.var() > 0:
                no_lin[o, i_] = y_.var() * (1 - np.corrcoef(xg.numpy(), y_)[0, 1] ** 2)   # varianza que una recta no explica
    figuras.activaciones_kan(curvas, activa, nom_in, nom_out, B, CONTRASTE, mk.cfg["forma"], ALT_MU, ALT_SD)

    # Aristas que sobreviven a la poda en las 3 semillas del ajuste final
    ar_filas = []
    for s in SEMILLAS:
        capa_s = next(c for c in modelos[("kan", s)].head if isinstance(c, KANLayer))
        amp_s = amplitudes(ctx, modelos[("kan", s)], tr)
        ar_filas += [dict(semilla=s, entrada=nom_in[i_], salida=nom_out[o], amplitud=amp_s[o, i_], activa=bool(capa_s.mascara[o, i_] > 0))
                     for o in range(n_o) for i_ in range(n_i)]
    aristas_kan = pd.DataFrame(ar_filas)
    ctx.tablas["aristas_kan"] = aristas_kan.round(4)

    # Ajuste simbólico: con entrada física, la arista más amplia del ONI, la de la altitud y la más no lineal de las z_k; con
    # "proyeccion", la más no lineal de todas
    amp_k = np.array([[np.ptp(curvas[(o, i_)][2]) for i_ in range(n_i)] for o in range(n_o)])
    amp_k = np.where(activa, amp_k, 0.0)
    if FISICA:
        aristas = {"ONI": (int(np.argmax(amp_k[:, K_NEU + 1])), K_NEU + 1), "altitud": (int(np.argmax(amp_k[:, K_NEU])), K_NEU),
                   "más no lineal de z_k": np.unravel_index(np.argmax(no_lin[:, :K_NEU]), (n_o, K_NEU))}
    else:
        aristas = {"más no lineal": np.unravel_index(np.argmax(no_lin), no_lin.shape)}
    for tipo in [k for k, (o_s, i_s) in aristas.items() if amp_k[o_s, i_s] == 0]:
        print(f"Ajuste simbólico: sin arista activa para «{tipo}» (todas podadas)"); del aristas[tipo]
    simb = []
    for tipo, (o_s, i_s) in aristas.items():
        xs, _, ys = curvas[(o_s, i_s)]
        for nombre, gen_ in CANDIDATAS.items():
            fn, p0 = gen_(ys)
            try:
                p, _ = curve_fit(fn, xs, ys, p0=p0, maxfev=20000)
                r2 = 1 - ((ys - fn(xs, *p)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
                simb.append(dict(arista=f"{nom_out[o_s].split()[-1]} <- {nom_in[i_s]}", criterio=tipo, expresion=nombre, R2=r2,
                                 parametros=np.round(p, 4).tolist(), n_parametros=len(p), amplitud=np.ptp(ys)))
            except RuntimeError:
                pass
    simb = pd.DataFrame(simb).sort_values(["criterio", "R2"], ascending=[True, False])
    ctx.tablas["ajuste_simbolico"] = simb
    figuras.ajuste_simbolico(aristas, curvas, simb, CANDIDATAS, nom_in, FISICA)

    # Dependencia parcial del total anual predicho frente a la altitud y al ONI; con la cabeza lineal la respuesta en sqrt es
    # una recta y no cambia con la otra variable, en la KAN puede curvarse e interactuar
    ctx.cx_tr, ctx.esc_tr = entrada(ctx, tr)
    alt_g = np.linspace(*np.quantile(alt[tr], [0.02, 0.98]), 40)
    oni_g = np.linspace(*np.quantile(oni_a[tr], [0.02, 0.98]), 40)
    oni_lo, oni_hi = np.quantile(oni_a[tr], [0.1, 0.9])
    dep, dep_curvas = [], {}
    for kind in CABEZAS:
        r_alt, r_oni = total_medio(ctx, kind, alt=alt_g), total_medio(ctx, kind, oni=oni_g)
        d_oni = total_medio(ctx, kind, alt=alt_g, oni=oni_hi) - total_medio(ctx, kind, alt=alt_g, oni=oni_lo)
        dep_curvas[kind] = (r_alt, r_oni, d_oni)
        dep += [dict(cabeza=kind, variable="altitud_m", x=x_, total_mm=y_) for x_, y_ in zip(alt_g, r_alt)]
        dep += [dict(cabeza=kind, variable="oni", x=x_, total_mm=y_) for x_, y_ in zip(oni_g, r_oni)]
        dep += [dict(cabeza=kind, variable=f"efecto_oni_{oni_hi:.2f}_menos_{oni_lo:.2f}_segun_altitud_m", x=x_, total_mm=y_) for x_, y_ in zip(alt_g, d_oni)]
    figuras.dependencia_parcial(dep_curvas, alt_g, oni_g, oni_lo, oni_hi, alt[tr], oni_a[tr], len(tr))
    dep = pd.DataFrame(dep)
    ctx.tablas["dependencia_parcial"] = dep.round(3)

    # Aporte y atenuación por fase ENSO: ONI fijado en el de un año Niña, uno neutro y uno Niño típicos (mediana del ONI medio
    # de los años de entrenamiento de cada fase), con el satélite y la altitud observados
    oni_tip = {f: float(oni_t.oni_medio[(oni_t.fase_codigo == f) & (oni_t.index <= ANIOS_ENT[1])].median()) for f in FASES}
    ap_filas, ap_curvas = [], {}
    for kind in CABEZAS:
        base_n = curva_media(ctx, kind, oni_tip["neutro"], FASES.index("neutro"))
        for f_ in ("nina", "nino"):
            dc = curva_media(ctx, kind, oni_tip[f_], FASES.index(f_)) - base_n
            ap_curvas[(kind, f_)] = dc
            ap_filas.append(dict(cabeza=kind, fase=f_, oni=oni_tip[f_], oni_neutro=oni_tip["neutro"], cambio_total_anual_mm=dc.sum(),
                                 cambio_pct=100 * dc.sum() / base_n.sum(), dias_con_aporte=int((dc > 0.05).sum()),
                                 dias_con_atenuacion=int((dc < -0.05).sum()), cambio_max_mm_dia=dc.max(), cambio_min_mm_dia=dc.min()))
    aporte = pd.DataFrame(ap_filas)
    ctx.tablas["aporte_atenuacion_enso"] = aporte.round(3)
    figuras.aporte_atenuacion_enso(FISICA, curvas, activa, nom_out, oni_tip, ap_curvas)
    ctx.B_lineal = B
    return dict(aristas=aristas_kan, simbolico=simb, dependencia=dep, aporte=aporte)


def complementos(ctx):
    # Coeficientes de la cabeza lineal, autofunciones del decodificador común y por fase, y peores casos de la KAN en las ciegas
    figuras.cabeza_lineal(ctx.B_lineal)
    figuras.fpca_autofunciones(ctx.Psi_t, ctx.FP_FASE, ctx.Phi_y, ctx.lam_fp)
    meta, res, fase, Y_true, X_mm = ctx.meta, ctx.res, ctx.fase, ctx.Y_true, ctx.X_mm
    i = ctx.idx["ciegas"]; peor = np.argsort(res[("ciegas", "kan")]["L2_rel"])[-4:][::-1]
    figuras.peores_casos(ctx, i, peor)
    peores = pd.DataFrame({"station_id": meta.station_id[i[peor]].to_numpy(), "year": meta.year[i[peor]].to_numpy(),
                           "altitud_m": meta.altitud_m[i[peor]].to_numpy(), "fase": fase[i[peor]],
                           "total_estacion_mm": Y_true[:, i[peor]].sum(0), "total_satelite_mm": X_mm[:, i[peor]].sum(0),
                           "L2_rel_kan": res[("ciegas", "kan")]["L2_rel"][peor]})
    ctx.tablas["peores_casos_ciegas"] = peores.round(2)
    return peores
