"""Evaluación en la prueba y en las estaciones ciegas: equivalencia de la cabeza lineal, métricas, bootstrap por estación,
fases ENSO, años de prueba, aporte de las covariables, CSV de predicciones y verificaciones finales."""
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from . import figuras
from .bases import base, mco_enmascarado, raiz_curva
from .comun import CSV_BUSQUEDA, CSV_PRED, N_DIAS, PROC
from .datos import ANIOS_PRU, FASES, TRAMOS, anota, satelite_de
from .entrenamiento import SEMILLAS, ajustar_final, probar
from .modelo import CABEZAS, COMBOS, COVARIABLES, K_NEU, NOMBRES, count_params, entrada

B_BOOT = 1000
VENT_P1, VENT_P2 = (60, 181), (244, 349)   # ventanas de los picos (días)
w_trap = np.r_[0.5, np.ones(N_DIAS - 2), 0.5]


def equivalencia_lineal(ctx):
    # La cabeza lineal equivale a la regresión funcional de rango bajo beta(s,t) = sum_k sum_m w_k(s) B_km psi_m(t)
    Phi_y, Pf, w_fino, alt_n, oni_a = ctx.Phi_y, ctx.Pf, ctx.w_fino, ctx.alt_n, ctx.oni_a
    te = ctx.idx["prueba"]
    Psi_t = ctx.Psi @ Phi_y.T                                                           # Dim: (M, 365), psi_m(t)
    Xf = Pf @ ctx.Cxs[:, te]                                                            # Dim: (36401, n), X(s) en la malla fina
    dif_lin = []
    for s in SEMILLAS:
        m = ctx.modelos[("lineal", s)]
        with torch.no_grad():
            y_red = Phi_y @ m(*entrada(ctx, te)).numpy().T                              # Dim: (365, n), salida de la red
        A = m.A.detach().numpy(); B = m.head.weight.detach().numpy(); b0 = m.head.bias.detach().numpy()
        beta = (Pf @ A.T) @ B[:, :K_NEU].T @ Psi_t                                      # Dim: (36401, 365), beta(s, t)
        integral = beta.T @ (w_fino[:, None] * Xf)                                      # Dim: (365, n), int beta(s,t) X(s) ds
        escalar = Psi_t.T @ (b0[:, None] + B[:, K_NEU:] @ np.vstack([alt_n[te], oni_a[te]]))   # gamma(t) alt + delta(t) ONI + término libre
        y_freg = (Phi_y @ ctx.mu_c)[:, None] + escalar + integral
        dif_lin.append(np.abs(y_red - y_freg).max())
    anota(ctx, "max_t |y_red_lineal - y_regresion_funcional| (prueba, 3 semillas)", max(dif_lin), 1e-4)
    ctx.Psi_t = Psi_t
    return max(dif_lin)


def predecir(ctx, m, i, Cx_=None, alt_=None, oni_=None, fase_=None):
    # sqrt de la curva predicha (365, n) para las curvas i o para entradas dadas
    cx = torch.tensor((ctx.Cxs[:, i] if Cx_ is None else Cx_).T)
    esc = torch.tensor(np.c_[ctx.alt_n[i], ctx.oni_a[i], ctx.fase_i[i]] if alt_ is None else np.c_[alt_, oni_, fase_])
    with torch.no_grad():
        return ctx.Phi_y @ m(cx, esc).numpy().T                                     # Dim: (365, n), sqrt de la curva


def sig2(ctx, m):
    # Varianza residual sigma^2(t) en la escala sqrt con el entrenamiento
    return ((predecir(ctx, m, ctx.tr) - ctx.Phi_y @ ctx.Cys[:, ctx.tr]) ** 2).mean(1)


def a_mm(ctx, ys, k):
    # Retransformación a mm/día: E[y] = m^2 + sigma^2(t), con recorte en cero
    return np.maximum(ys, 0) ** 2 + ctx.sigma2[k][:, None]


def kge(P, Y):
    r = ((P - P.mean(0)) * (Y - Y.mean(0))).sum(0) / np.sqrt(((P - P.mean(0)) ** 2).sum(0) * ((Y - Y.mean(0)) ** 2).sum(0))
    return 1 - np.sqrt((r - 1) ** 2 + (P.std(0) / Y.std(0) - 1) ** 2 + (P.mean(0) / Y.mean(0) - 1) ** 2)


def picos(Y):
    return np.c_[VENT_P1[0] + np.argmax(Y[VENT_P1[0] - 1:VENT_P1[1]], 0), VENT_P2[0] + np.argmax(Y[VENT_P2[0] - 1:VENT_P2[1]], 0)]   # Dim: (n, 2)


def por_curva(ctx, P, i):
    # Métricas por curva: L2 relativo, KGE, diferencia y total (para el PBIAS) y error en días de las fechas de los picos
    Y = ctx.Y_true[:, i]; pr, pv = picos(P), picos(ctx.Ref[:, i])
    return dict(L2_rel=np.sqrt((w_trap[:, None] * (P - Y) ** 2).sum(0) / (w_trap[:, None] * Y ** 2).sum(0)), KGE=kge(P, Y),
                dif=(P - Y).sum(0), tot=Y.sum(0), err_pico1=np.abs(pr[:, 0] - pv[:, 0]), err_pico2=np.abs(pr[:, 1] - pv[:, 1]))


def bootstrap(ctx, tramo, modelo, rng, ref=None):
    # IC 95 % por conglomerado: se remuestrean estaciones (con todas sus curvas) B_BOOT veces. Con ref, también el valor p
    # bootstrap de la diferencia (dos colas: 2 x la fracción de remuestreos del lado contrario al cero)
    i = ctx.idx[tramo]; est = ctx.meta.station_id.to_numpy()[i]
    u, inv = np.unique(est, return_inverse=True)
    rem = rng.integers(0, len(u), size=(B_BOOT, len(u)))
    cnt = np.bincount(inv)
    def stat(r, k):
        if k == "PBIAS":
            return 100 * np.bincount(inv, weights=r["dif"])[rem].sum(1) / np.bincount(inv, weights=r["tot"])[rem].sum(1)
        return np.bincount(inv, weights=r[k])[rem].sum(1) / cnt[rem].sum(1)
    out = {}
    for k in ("L2_rel", "KGE", "PBIAS", "err_pico1", "err_pico2"):
        v = stat(ctx.res[(tramo, modelo)], k) - (stat(ctx.res[(tramo, ref)], k) if ref else 0)
        out[k] = (np.quantile(v, 0.025), np.quantile(v, 0.975))
        out[k + "_p"] = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
    return out


def por_estacion(r, est, k):
    # Métrica por estación: PBIAS = 100 sum(P - Y) / sum(Y) de sus curvas (la razón que remuestrea bootstrap()); las demás,
    # media de sus curvas
    if k == "PBIAS":
        return 100 * pd.Series(r["dif"]).groupby(est).sum() / pd.Series(r["tot"]).groupby(est).sum()
    return pd.Series(r[k]).groupby(est).mean()


def comparar_pares(ctx):
    # Diferencias pareadas por estación (IC y valor p bootstrap; Wilcoxon como complemento) en L2, KGE, PBIAS y error en
    # días de cada pico. Las cinco salen de la misma llamada a bootstrap() y no consumen otro estado aleatorio; las filas
    # de PBIAS y de los picos van después de las de L2 y KGE
    res, meta, idx = ctx.res, ctx.meta, ctx.idx
    filas, extra = [], []
    for tramo in ("prueba", "ciegas"):
        est = meta.station_id.to_numpy()[idx[tramo]]
        # MLP - lineal mide la no linealidad del sesgo; KAN - MLP, el costo de la interpretabilidad
        for a, b in (("kan", "lineal"), ("kan", "mlp"), ("kan", "satelite"), ("lineal", "satelite"), ("mlp", "lineal"),
                     ("mlp", "satelite")):
            ic = bootstrap(ctx, tramo, a, np.random.default_rng(42), ref=b)
            for k in ("L2_rel", "KGE", "PBIAS", "err_pico1", "err_pico2"):
                ma, mb = por_estacion(res[(tramo, a)], est, k), por_estacion(res[(tramo, b)], est, k)
                d = ma - mb
                if k.startswith("err_pico"):
                    d = d.round(9)   # medias de errores enteros: empates y ceros exactos en los rangos del Wilcoxon
                (filas if k in ("L2_rel", "KGE") else extra).append(
                    dict(tramo=tramo, comparacion=f"{NOMBRES[a]} - {NOMBRES[b]}", metrica=k, diferencia=(ma - mb).mean(),
                         IC95_inf=ic[k][0], IC95_sup=ic[k][1], IC_excluye_cero=bool(ic[k][0] > 0 or ic[k][1] < 0),
                         p_bootstrap=ic[k + "_p"], wilcoxon_p=wilcoxon(d).pvalue, estaciones=len(ma)))
    return pd.DataFrame(filas + extra)


def evaluar(ctx):
    # Tabla maestra de la ablación (satélite y tres cabezas, media de 3 semillas, IC bootstrap por estación), diferencias
    # pareadas (IC y valor p bootstrap; Wilcoxon por estación como complemento) en L2, KGE, PBIAS y fechas de los picos,
    # métricas por fase ENSO y por año de prueba
    idx, meta, modelos, params, fase, anio_m = ctx.idx, ctx.meta, ctx.modelos, ctx.params, ctx.fase, ctx.anio_m
    ctx.sigma2 = {k: sig2(ctx, m) for k, m in modelos.items()}                       # E[y] = m^2 + Var(e)
    res, pct_trunc, preds_mm = {}, {}, {}
    ctx.res, ctx.preds_mm = res, preds_mm
    for tramo in TRAMOS:
        i = idx[tramo]
        res[(tramo, "satelite")] = por_curva(ctx, ctx.X_mm[:, i], i)
        for kind in ("lineal", "mlp", "kan"):
            sem = []
            for s in SEMILLAS:
                ys = predecir(ctx, modelos[(kind, s)], i)
                pct_trunc[(tramo, kind, s)] = 100 * (ys < 0).mean()
                P = a_mm(ctx, ys, (kind, s)); preds_mm[(tramo, kind, s)] = P
                sem.append(por_curva(ctx, P, i))
            res[(tramo, kind)] = {k: np.mean([x[k] for x in sem], 0) for k in sem[0]}          # métrica por curva, media de las 3 semillas
            res[(tramo, kind)]["sd_semillas_L2"] = np.std([x["L2_rel"].mean() for x in sem])

    filas = []
    for tramo in TRAMOS:
        for modelo in ("satelite", "lineal", "mlp", "kan"):
            r = res[(tramo, modelo)]; ic = bootstrap(ctx, tramo, modelo, np.random.default_rng(42))
            fila = dict(tramo=tramo, modelo=NOMBRES[modelo], curvas=len(idx[tramo]), estaciones=meta.station_id[idx[tramo]].nunique(),
                        L2_rel=r["L2_rel"].mean(), L2_IC=f"[{ic['L2_rel'][0]:.3f}, {ic['L2_rel'][1]:.3f}]",
                        KGE=r["KGE"].mean(), KGE_IC=f"[{ic['KGE'][0]:.3f}, {ic['KGE'][1]:.3f}]",
                        PBIAS=100 * r["dif"].sum() / r["tot"].sum(), PBIAS_IC=f"[{ic['PBIAS'][0]:.1f}, {ic['PBIAS'][1]:.1f}]",
                        err_pico1_dias=r["err_pico1"].mean(), err_pico1_IC=f"[{ic['err_pico1'][0]:.1f}, {ic['err_pico1'][1]:.1f}]",
                        err_pico2_dias=r["err_pico2"].mean(), err_pico2_IC=f"[{ic['err_pico2'][0]:.1f}, {ic['err_pico2'][1]:.1f}]",
                        pct_truncado=np.mean([pct_trunc[(tramo, modelo, s)] for s in SEMILLAS]) if modelo != "satelite" else 0.0,
                        sd_semillas_L2=r.get("sd_semillas_L2", 0.0), parametros=params.get(modelo, 0))
            filas.append(fila)
    maestra = pd.DataFrame(filas)
    ctx.tablas["tabla_maestra_ablacion"] = maestra.round(4)

    pareadas = comparar_pares(ctx)
    ctx.tablas["comparaciones_pareadas"] = pareadas.round(4)

    filas = []
    for tramo in ("prueba", "ciegas"):
        i = idx[tramo]
        for f_ in ("nino", "nina", "neutro"):
            sel = fase[i] == f_
            if sel.sum() == 0:
                continue
            for modelo in ("satelite", "lineal", "mlp", "kan"):
                r = res[(tramo, modelo)]
                filas.append(dict(tramo=tramo, fase=f_, curvas=int(sel.sum()), modelo=NOMBRES[modelo], L2_rel=r["L2_rel"][sel].mean(),
                                  KGE=r["KGE"][sel].mean(), PBIAS=100 * r["dif"][sel].sum() / r["tot"][sel].sum()))
    enso = pd.DataFrame(filas)
    ctx.tablas["metricas_por_fase_enso"] = enso.round(4)

    # Año por año de la prueba: total anual y sesgo de cada modelo
    i = idx["prueba"]
    filas = []
    for a in range(ANIOS_PRU[0], ANIOS_PRU[1] + 1):
        sel = anio_m[i] == a
        if sel.sum() == 0:
            continue
        fila = dict(anio=a, fase=ctx.oni_t.fase_codigo[a], oni_medio=ctx.oni_t.oni_medio[a], curvas=int(sel.sum()),
                    total_estacion_mm=ctx.Y_true[:, i[sel]].sum(0).mean(), total_satelite_mm=ctx.X_mm[:, i[sel]].sum(0).mean())
        for kind in CABEZAS:
            fila[f"total_{kind}_mm"] = np.mean([preds_mm[("prueba", kind, s)][:, sel].sum(0).mean() for s in SEMILLAS])
        for modelo in ("satelite", "lineal", "mlp", "kan"):
            r = res[("prueba", modelo)]
            fila[f"L2_{modelo}"] = r["L2_rel"][sel].mean(); fila[f"PBIAS_{modelo}"] = 100 * r["dif"][sel].sum() / r["tot"][sel].sum()
        filas.append(fila)
    por_anio = pd.DataFrame(filas)
    ctx.tablas["metricas_por_anio_prueba"] = por_anio.round(4)
    figuras.prueba_por_anio_enso(por_anio)
    figuras.ablacion(maestra)
    return dict(maestra=maestra, pareadas=pareadas, enso=enso, por_anio=por_anio)


def predicciones(ctx):
    # CSV de predicciones de las estaciones ciegas con la cabeza elegida (media de 3 semillas), año por año y día por día
    anios = ctx.anios
    emp = pd.read_csv(PROC / "metadatos/emparejamiento_estacion_pixel.csv")
    ciegas = emp[emp.ciega.astype(str).eq("True") & emp.pixel.notna() & emp.altitud_m.notna()]
    sin_pred = emp[emp.ciega.astype(str).eq("True") & ~(emp.pixel.notna() & emp.altitud_m.notna())].station_id.tolist()
    st = np.repeat(ciegas.station_id.to_numpy(), len(anios)); yr = np.tile(anios, len(ciegas))
    Cx_c = mco_enmascarado(base(ctx.t_sat, ctx.K_X), satelite_de(ctx, np.repeat(ciegas.pixel.to_numpy(), len(anios)), yr))[0]
    Cxs_c = raiz_curva(ctx.Phi_x, Cx_c)
    alt_c = (np.repeat(ciegas.altitud_m.to_numpy(), len(anios)) - ctx.ALT_MU) / ctx.ALT_SD
    oni_c = ctx.oni_t.oni_medio.reindex(yr).to_numpy()
    fase_c = pd.Index(FASES).get_indexer(ctx.oni_t.fase_codigo.reindex(yr))
    CABEZA_CSV = ctx.CABEZA_CSV
    P_ens = np.mean([a_mm(ctx, predecir(ctx, ctx.modelos[(CABEZA_CSV, s)], None, Cxs_c, alt_c, oni_c, fase_c), (CABEZA_CSV, s)) for s in SEMILLAS], 0)   # Dim: (365, n)
    pred = pd.DataFrame({"station_id": np.repeat(st, N_DIAS), "year": np.repeat(yr, N_DIAS), "day": np.tile(np.arange(1, N_DIAS + 1), len(st)),
                         "value_mm": np.round(P_ens.T.ravel(), 3)})
    pred.to_csv(CSV_PRED, index=False)
    anota(ctx, "Valores negativos en el CSV de predicciones", int((pred.value_mm < 0).sum()), 0)
    return pred, sin_pred


def ablacion_covariables(ctx):
    # Aporte de cada covariable: cada cabeza con el satélite solo, con la altitud, con el ONI y con las tres; misma
    # configuración, validación cruzada y épocas, 3 semillas, evaluado en la prueba y en las ciegas
    FINAL, modelos, idx, res = ctx.FINAL, ctx.modelos, ctx.idx, ctx.res
    ab_filas = []
    for kind in CABEZAS:
        for nombre, cov in COMBOS.items():
            cfg = FINAL[kind] if cov == COVARIABLES else {**FINAL[kind], "cov": cov}
            v = probar(ctx, cfg)
            ms = [modelos[(kind, s)] for s in SEMILLAS] if cov == COVARIABLES else [m for m, _ in ajustar_final(ctx, cfg)[0].values()]
            fila = dict(cabeza=kind, entradas=nombre, parametros=count_params(ms[0]), cv_rel_media=v.mean(), cv_rel_sd=v.std(ddof=1))
            for tramo in TRAMOS:
                i = idx[tramo]
                met = [por_curva(ctx, np.maximum(predecir(ctx, m, i), 0) ** 2 + sig2(ctx, m)[:, None], i) for m in ms]
                r = {k: np.mean([x[k] for x in met], 0) for k in met[0]}
                key = (tramo, f"abl|{kind}|{nombre}"); res[key] = r
                ic = bootstrap(ctx, tramo, key[1], np.random.default_rng(42))
                fila.update({f"{tramo}_L2": r["L2_rel"].mean(), f"{tramo}_L2_IC": f"[{ic['L2_rel'][0]:.3f}, {ic['L2_rel'][1]:.3f}]",
                             f"{tramo}_KGE": r["KGE"].mean(), f"{tramo}_PBIAS": 100 * r["dif"].sum() / r["tot"].sum()})
            ab_filas.append(fila)
            print(f"{kind} con {nombre}: listo", flush=True)
    abl = pd.DataFrame(ab_filas)
    ctx.tablas["ablacion_covariables"] = abl.round(4)
    pd.DataFrame(ctx.registro.values()).sort_values(["cabeza", "config", "pliegue"]).to_csv(CSV_BUSQUEDA, index=False)
    figuras.ablacion_covariables(abl, res)
    return abl


def verificaciones(ctx):
    # Tabla de verificaciones de la red; falla si alguna no se cumple
    v = pd.DataFrame(ctx.verif); ctx.tablas["verificaciones_item4"] = v
    print(f"Tiempo total: {(time.time() - ctx.t_ini) / 60:.1f} min")
    if not v.cumple.all():
        raise RuntimeError("Hay verificaciones que no se cumplen:\n" + v[~v.cumple].to_string(index=False))
    return v
