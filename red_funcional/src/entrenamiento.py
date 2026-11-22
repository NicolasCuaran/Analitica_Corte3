"""Entrenamiento de la red funcional: prueba sintética de la KAN, búsqueda de hiperparámetros, ajuste final y diagnósticos."""
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import figuras
from .comun import CSV_BUSQUEDA, PLIEGUES_CV
from .curvas import M_FPCA, fpca
from .datos import ETQ_FASE, FASES, anota
from .modelo import (CABEZAS, K_NEU, UMBRAL_PODA, KANLayer, RedFuncional, count_params, entrada, make_head, objetivo,
                     perdida_l2, preparar)

SEMILLAS = [42, 123, 999]
EPOCAS, PACIENCIA, LOTE, CLIP = 600, 60, 128, 1.0
SEMILLA_CV = 42                             # una semilla por pliegue: la variabilidad de la validación cruzada la dan los años
ENT_ESPARSIDAD = 2.0                        # peso de la entropía frente a la L1
INICIAL = {"lineal": dict(cabeza="lineal", lr=3e-3, lam_w=1e-4, wd=1e-4, l1=0.0),
           "mlp": dict(cabeza="mlp", lr=3e-3, lam_w=1e-4, wd=1e-4, l1=0.0, ancho=32, drop=0.1),
           "kan": dict(cabeza="kan", lr=3e-3, lam_w=1e-4, wd=1e-4, l1=0.0, ent=0.0, forma="aditiva", ancho=0, G=10, drop=0.0)}
# Formas de la KAN con las entradas físicas [z, altitud, ONI] en las aristas de la primera capa
FORMAS_KAN = (dict(forma="aditiva", ancho=0, G=10, drop=0.0),               # KANLayer(8,5): 440 parámetros
              dict(forma="ka", ancho=4, G=8, drop=0.0))                     # KANLayer(8,4)-KANLayer(4,5): 468
# Etapas de la búsqueda por coordenadas; lambda_w es relativo a tr W / tr R y la rejilla 10^x no contiene el 0
ETAPAS = (("A", "estructura de la cabeza", {"lineal": [dict(wd=v) for v in (0.0, 1e-4, 1e-3)],
                                            "mlp": [dict(drop=v) for v in (0.0, 0.1, 0.2)], "kan": list(FORMAS_KAN)}),
          ("B", "rugosidad de w_k (lambda_w relativo)", {k: [dict(lam_w=v) for v in (1e-4, 1e-3, 1e-2, 1e-1)] for k in CABEZAS}),
          ("C", "tasa de aprendizaje máxima", {k: [dict(lr=v) for v in (1e-3, 3e-3, 1e-2, 3e-2)] for k in CABEZAS}),
          ("D", "regularización de la cabeza (KAN: L1 + entropía y poda)",
           {"lineal": [dict(wd=v) for v in (0.0, 1e-4, 1e-3)], "mlp": [dict(wd=v) for v in (0.0, 1e-4, 1e-3)],
            "kan": [dict(l1=0.0, ent=0.0)] + [dict(l1=v, ent=ENT_ESPARSIDAD) for v in (1e-3, 1e-2, 1e-1)]}))
PASADAS = 2


def entrenar(ctx, cfg, semilla, i_tr, i_va=None, epocas=None):
    # AdamW + OneCycleLR + recorte de gradiente. Con i_va (pliegue de la validación cruzada): parada temprana con esa
    # validación. Sin i_va (ajuste final): "epocas" fijas con el calendario OneCycleLR ajustado a ellas. Con L1 > 0 se podan
    # al final las aristas débiles de la primera capa KAN. Devuelve (modelo, historia, pérdida de validación, mejor época).
    torch.manual_seed(semilla); np.random.seed(semilla)
    b = preparar(ctx, i_tr)
    m = RedFuncional(cfg, b)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    pasos = math.ceil(len(i_tr) / LOTE)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], epochs=EPOCAS if epocas is None else epocas,
                                              steps_per_epoch=pasos, pct_start=0.3)
    xtr, etr = entrada(ctx, i_tr); ytr = objetivo(ctx, i_tr)
    if i_va is not None:
        xva, eva = entrada(ctx, i_va); yva = objetivo(ctx, i_va)
    gen = torch.Generator().manual_seed(semilla)
    mejor, estado, sin_mejora, hist, ep_mejor = np.inf, None, 0, [], 0
    for ep in range(EPOCAS if epocas is None else epocas):
        m.train(); suma = 0.0
        for bt in torch.randperm(len(i_tr), generator=gen).split(LOTE):
            opt.zero_grad()
            L = perdida_l2(m(xtr[bt], etr[bt]), ytr[bt], ctx.Wy_t)
            reg = cfg["lam_w"] * ctx.ESC_R * m.rugosidad() + cfg["l1"] * b["esc_l1"] * (m.l1() + cfg.get("ent", 0.0) * m.entropia())
            (L + reg).backward(); nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step(); sch.step()
            suma += L.item() * len(bt)
        if i_va is None:
            hist.append(suma / len(i_tr))
            continue
        m.eval()
        with torch.no_grad():
            lv = perdida_l2(m(xva, eva), yva, ctx.Wy_t).item()
        hist.append(lv)
        if lv < mejor - 1e-9:
            mejor, estado, sin_mejora, ep_mejor = lv, {k: v.clone() for k, v in m.state_dict().items()}, 0, ep + 1
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    if estado is not None:
        m.load_state_dict(estado)
    m.eval()
    if cfg["cabeza"] == "kan" and cfg["l1"] > 0:
        next(c for c in m.head if isinstance(c, KANLayer)).podar(amplitudes(ctx, m, i_tr))
        if i_va is not None:
            with torch.no_grad():
                mejor = perdida_l2(m(xva, eva), yva, ctx.Wy_t).item()
    return m, hist, mejor, ep_mejor


def entrada_kan(ctx, m, i):
    # Entrada de la primera capa KAN ya normalizada; con "aditiva" y "ka" es [z, covariables], con "proyeccion" la mezcla u = Wx
    capa = next(c for c in m.head if isinstance(c, KANLayer))
    with torch.no_grad():
        cx, esc = entrada(ctx, i)
        _, _, z = m(cx, esc, scores=True)
        x = torch.cat([z, esc[:, m.cols]], 1)                                       # Dim: (n, K_NEU + n_cov)
        u = m.head[0](x) if m.cfg["forma"] == "proyeccion" else x
        return capa, u, capa.norm(u)                                                # Dim: (n, in), (n, in)


def amplitudes(ctx, m, i=None):
    # Amplitud de cada arista phi_oi en el rango 1 %-99 % de su entrada en el entrenamiento
    capa, _, h = entrada_kan(ctx, m, ctx.tr if i is None else i)
    lo, hi = np.quantile(h.numpy(), [0.01, 0.99], axis=0)
    amp = np.zeros((capa.out_features, capa.in_features))
    for i_ in range(capa.in_features):
        xg = torch.linspace(lo[i_], hi[i_], 200)
        Y = torch.stack([capa.edge_function(o, i_, xg) for o in range(capa.out_features)])
        amp[:, i_] = (Y.max(1).values - Y.min(1).values).numpy()
    return amp


def prueba_sintetica(ctx):
    # Capa KAN de una entrada ajustada a f(x) = sin(pi x) + exp(x/2); sin RMSE < 1e-2 no se procesa lluvia real
    torch.manual_seed(42)
    kan_1d = KANLayer(1, 1, grid_size=20, grid_range=(-1.0, 1.0), norm=None)
    f_sint = lambda x: torch.sin(math.pi * x) + torch.exp(x / 2)
    x_s = torch.linspace(-1, 1, 2001)[:, None]; x_v = torch.rand(1000, 1, generator=torch.Generator().manual_seed(7)) * 2 - 1
    opt = torch.optim.AdamW(kan_1d.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(4000):
        opt.zero_grad(); L = ((kan_1d(x_s) - f_sint(x_s)) ** 2).mean(); L.backward(); opt.step()
    with torch.no_grad():
        rmse_sint = ((kan_1d(x_v) - f_sint(x_v)) ** 2).mean().sqrt().item()
    anota(ctx, "RMSE de la KAN sintética en 1 000 puntos de validación", rmse_sint, 1e-2)
    if rmse_sint > 1e-2:
        raise RuntimeError("La KAN no pasa la prueba sintética: no se procesa lluvia real")
    figuras.kan_sintetica(x_s, f_sint, kan_1d, rmse_sint)
    return rmse_sint


def clave_cfg(cfg):
    return json.dumps(cfg, sort_keys=True)


def etiqueta(cambio):
    return cambio["forma"] if "forma" in cambio else ", ".join(f"{k}={v:g}" for k, v in cambio.items())


def probar(ctx, cfg):
    # Pérdida de validación relativa a la de la climatología en cada pliegue; reutiliza las corridas de ctx.registro
    vals = []
    for f, (i_tr, i_va) in enumerate(ctx.pliegues):
        k = (clave_cfg(cfg), f + 1)
        if k not in ctx.registro:
            m, h, mejor, ep = entrenar(ctx, cfg, SEMILLA_CV, i_tr, i_va)
            ctx.en_memoria[k] = (m, h)
            amp = amplitudes(ctx, m, i_tr) if cfg["cabeza"] == "kan" else np.zeros(1)
            act = int((amp > UMBRAL_PODA * amp.max()).sum()) if cfg["cabeza"] == "kan" else -1
            # Parámetros efectivos: sin los de las aristas podadas (1 peso base + G coeficientes cada una), que quedan en cero
            podadas = sum(int((c.mascara == 0).sum()) * (1 + c.spline_weight.shape[-1]) for c in m.head.modules() if isinstance(c, KANLayer))
            ctx.registro[k] = dict(cabeza=cfg["cabeza"], config=k[0], parametros=count_params(m), parametros_efectivos=count_params(m) - podadas, pliegue=f + 1,
                                   validacion="%d-%d" % PLIEGUES_CV[f], semilla=SEMILLA_CV, epocas=len(h), epoca_mejor=ep,
                                   perdida_val=mejor, perdida_climatologia=ctx.perdida_clima[f], perdida_rel=mejor / ctx.perdida_clima[f],
                                   aristas_activas=act)
        vals.append(ctx.registro[k]["perdida_rel"])
    return np.array(vals)


def elegir(V, actual, parametros, efectivos, suavidad=None):
    # V: (candidatos, pliegues). Empatan con la mejor las que no la superan por más del error estándar de su diferencia
    # pareada; entre ellas, la de menos parámetros, luego la de menos parámetros efectivos y, a igualdad, la que ya estaba.
    # Con suavidad (lambda_w de cada candidato), tras los parámetros nominales gana el lambda_w mayor.
    j = int(V.mean(1).argmin())
    dif = V - V[j]
    emp = np.flatnonzero(dif.mean(1) <= dif.std(1, ddof=1) / np.sqrt(V.shape[1]))
    emp = emp[parametros[emp] == parametros[emp].min()]
    if suavidad is not None:
        return int(emp[suavidad[emp].argmax()])
    emp = emp[efectivos[emp] <= efectivos[emp].min() + 0.5]
    ya = [e for e in emp if actual[e]]
    return int(ya[0]) if ya else int(emp[V[emp].mean(1).argmin()])


def epocas_cv(ctx, cfg):
    # Épocas del ajuste final: mediana de la mejor época de los pliegues
    return max(1, int(np.median([ctx.registro[(clave_cfg(cfg), f + 1)]["epoca_mejor"] for f in range(len(ctx.pliegues))])))


def ajustar_final(ctx, cfg):
    # 3 semillas con todo el entrenamiento y las épocas de la validación cruzada
    E = epocas_cv(ctx, cfg)
    return {s: entrenar(ctx, cfg, s, ctx.tr, None, epocas=E)[:2] for s in SEMILLAS}, E


def buscar(ctx, buscar=False):
    # Búsqueda por coordenadas en cuatro etapas y dos pasadas con el mismo presupuesto por cabeza; cada configuración se
    # entrena en los 4 pliegues y se puntúa con la pérdida de validación relativa a la de la climatología del pliegue.
    # Con buscar=False se reutilizan las corridas guardadas en busqueda_hiperparametros.csv.
    ctx.registro, ctx.en_memoria = {}, {}
    if CSV_BUSQUEDA.exists() and not buscar:
        _b = pd.read_csv(CSV_BUSQUEDA)
        if "pliegue" in _b.columns:
            ctx.registro = {(r["config"], int(r["pliegue"])): r for r in _b.to_dict("records")}
    ctx.perdida_clima = [perdida_l2(torch.tensor(fpca(ctx, i_)[0])[None], objetivo(ctx, v_), ctx.Wy_t).item() for i_, v_ in ctx.pliegues]
    registro, pliegues = ctx.registro, ctx.pliegues

    actual, resumen, probadas = {k: dict(v) for k, v in INICIAL.items()}, [], {k: set() for k in CABEZAS}
    for pasada in range(1, PASADAS + 1):
        for etapa, factor, grilla in ETAPAS:
            for cab in CABEZAS:
                cands = [{**actual[cab], **c} for c in grilla[cab]]
                probadas[cab] |= {clave_cfg(c) for c in cands}
                V = np.array([probar(ctx, c) for c in cands])                               # Dim: (candidatos, pliegues)
                par = np.array([registro[(clave_cfg(c), 1)]["parametros"] for c in cands])
                par_ef = np.array([np.mean([registro[(clave_cfg(c), f + 1)]["parametros_efectivos"] for f in range(len(pliegues))]) for c in cands])
                act = np.array([c == actual[cab] for c in cands])
                j = elegir(V, act, par, par_ef, np.array([c["lam_w"] for c in cands]) if etapa == "B" else None)
                resumen += [dict(pasada=pasada, etapa=etapa, factor=factor, cabeza=cab, valor=etiqueta(c), parametros=int(par[n]),
                                 parametros_efectivos=int(round(par_ef[n])),
                                 media_val=V[n].mean(), sd_val=V[n].std(ddof=1), val_pliegues=np.round(V[n], 4).tolist(), partida=bool(act[n]),
                                 elegida=n == j) for n, c in enumerate(grilla[cab])]
                actual[cab] = cands[j]
                if buscar:
                    print(f"pasada {pasada}, etapa {etapa}, {cab}: {etiqueta(grilla[cab][j])}", flush=True)
    FINAL = actual
    busqueda = pd.DataFrame(resumen)
    pd.DataFrame(registro.values()).sort_values(["cabeza", "config", "pliegue"]).to_csv(CSV_BUSQUEDA, index=False)
    ctx.tablas["busqueda_resumen"] = busqueda

    params = {k: count_params(RedFuncional(FINAL[k], preparar(ctx, ctx.tr))) for k in CABEZAS}
    params_cab = {k: count_params(make_head(FINAL[k])) for k in CABEZAS}
    disc = abs(params["kan"] - params["mlp"]) / params["mlp"]
    anota(ctx, "Discrepancia de parámetros KAN frente a MLP (modelo completo)", disc, 0.10)
    tab_par = pd.DataFrame({"cabeza": list(params), "parametros_cabeza": [params_cab[k] for k in params],
                            "parametros_codificador": [K_NEU * ctx.K_X] * 3, "parametros_totales": list(params.values()),
                            "candidatos_evaluados": [int((busqueda.cabeza == k).sum()) for k in params],
                            "configuraciones_distintas": [len(probadas[k]) for k in params], "corridas_busqueda": [len(pliegues) * len(probadas[k]) for k in params],
                            "epocas_ajuste_final": [epocas_cv(ctx, FINAL[k]) for k in params],
                            "configuracion": [clave_cfg(FINAL[k]) for k in params]})
    ctx.tablas["parametros"] = tab_par
    vars(ctx).update(FINAL=FINAL, busqueda=busqueda, params=params)
    return tab_par


def ajuste_final(ctx):
    # Tres cabezas elegidas con 3 semillas, puntaje de cada pliegue y cabeza del CSV de predicciones, elegida entre las tres
    # con la regla de la búsqueda y solo con la validación cruzada
    FINAL, registro, pliegues, params = ctx.FINAL, ctx.registro, ctx.pliegues, ctx.params
    modelos, historias = {}, {}
    for kind in CABEZAS:
        fits, E = ajustar_final(ctx, FINAL[kind])
        for s, (m, h) in fits.items():
            modelos[(kind, s)], historias[(kind, s)] = m, h

    cv_final = pd.DataFrame([dict(cabeza=k, pliegue=f + 1, validacion="%d-%d" % PLIEGUES_CV[f],
                                  fases_validacion=" / ".join(ETQ_FASE[x] for x in ctx.oni_t.fase_codigo.reindex(range(PLIEGUES_CV[f][0], PLIEGUES_CV[f][1] + 1))),
                                  **{c: registro[(clave_cfg(FINAL[k]), f + 1)][c] for c in ("perdida_val", "perdida_climatologia", "perdida_rel", "epoca_mejor")})
                             for k in CABEZAS for f in range(len(pliegues))])
    ctx.tablas["validacion_cruzada_pliegues"] = cv_final.round(4)

    V_cab = np.array([[registro[(clave_cfg(FINAL[k]), f + 1)]["perdida_rel"] for f in range(len(pliegues))] for k in CABEZAS])
    ef_cab = np.array([np.mean([registro[(clave_cfg(FINAL[k]), f + 1)]["parametros_efectivos"] for f in range(len(pliegues))]) for k in CABEZAS])
    par_cab = np.array([params[k] for k in CABEZAS])
    CABEZA_CSV = CABEZAS[elegir(V_cab, np.zeros(len(CABEZAS), bool), par_cab, ef_cab)]
    dif_cab = V_cab - V_cab[int(V_cab.mean(1).argmin())]
    ee_cab = dif_cab.std(1, ddof=1) / np.sqrt(V_cab.shape[1])
    seleccion = pd.DataFrame({"cabeza": CABEZAS, "cv_rel_media": V_cab.mean(1), "cv_rel_sd": V_cab.std(1, ddof=1),
                              "dif_media_con_la_mejor": dif_cab.mean(1), "ee_dif": ee_cab, "empata_con_la_mejor": dif_cab.mean(1) <= ee_cab,
                              "parametros": par_cab, "parametros_efectivos": ef_cab, "elegida_csv": [k == CABEZA_CSV for k in CABEZAS]})
    ctx.tablas["seleccion_cabeza_csv"] = seleccion.round(4)
    vars(ctx).update(modelos=modelos, historias=historias, cv_final=cv_final, CABEZA_CSV=CABEZA_CSV)
    return cv_final, seleccion


def historia_cv(ctx, cfg, f):
    # Curva de validación del pliegue f (se reentrena si la corrida vino de busqueda_hiperparametros.csv)
    k = (clave_cfg(cfg), f + 1)
    if k not in ctx.en_memoria:
        ctx.en_memoria[k] = entrenar(ctx, cfg, SEMILLA_CV, *ctx.pliegues[f])[:2]
    return ctx.en_memoria[k][1]


def diagnosticos(ctx):
    # Diagnósticos que no deciden la configuración: capacidad a parámetros igualados (±10 %), esparsidad de la KAN elegida
    # (rejilla de L1 de la etapa D) y FPCA por fase ENSO frente a la base común; figuras de la búsqueda
    FINAL, registro, pliegues, tr = ctx.FINAL, ctx.registro, ctx.pliegues, ctx.tr
    NIVELES = {"mlp": [dict(ancho=v) for v in (16, 32, 64)],
               "kan": {"proyeccion": [dict(ancho=v) for v in (4, 8, 16)], "aditiva": [dict(G=v) for v in (5, 10, 22)],
                       "ka": [dict(ancho=v) for v in (2, 4, 8)]}[FINAL["kan"]["forma"]]}
    capac = []
    for nivel in range(3):
        fila = {}
        for cab in ("mlp", "kan"):
            cfg = {**FINAL[cab], **NIVELES[cab][nivel]}; v = probar(ctx, cfg)
            fila[cab] = dict(nivel=nivel + 1, cabeza=cab, cambio=etiqueta(NIVELES[cab][nivel]), parametros=count_params(RedFuncional(cfg, preparar(ctx, tr))),
                             media_val=v.mean(), sd_val=v.std(ddof=1))
        fila["kan"]["discrepancia"] = fila["mlp"]["discrepancia"] = abs(fila["kan"]["parametros"] - fila["mlp"]["parametros"]) / fila["mlp"]["parametros"]
        capac += list(fila.values())
    capac = pd.DataFrame(capac)
    ctx.tablas["curva_capacidad"] = capac.round(4)

    esp, capa0 = [], next(c for c in ctx.modelos[("kan", 42)].head if isinstance(c, KANLayer))
    for l1 in (0.0, 1e-3, 1e-2, 1e-1):
        cfg = {**FINAL["kan"], "l1": l1, "ent": ENT_ESPARSIDAD if l1 > 0 else 0.0}; v = probar(ctx, cfg)
        act = [registro[(clave_cfg(cfg), f + 1)]["aristas_activas"] for f in range(len(pliegues))]
        esp.append(dict(lambda_L1=l1, peso_entropia=cfg["ent"], media_val=v.mean(), sd_val=v.std(ddof=1), aristas_activas_media=np.mean(act),
                        aristas_totales=capa0.in_features * capa0.out_features))
    esp = pd.DataFrame(esp)
    ctx.tablas["esparsidad_kan"] = esp.round(4)

    # Varianza de cada fase captada por la base común frente a su propia base, ángulos entre subespacios y validación
    # cruzada de la KAN con decodificador por fase
    Psi, W_h, W_y, Cys, fase_i, mu_c = ctx.Psi, ctx.W_h, ctx.W_y, ctx.Cys, ctx.fase_i, ctx.mu_c
    fp_filas, Qc = [], W_h @ Psi.T                                                     # columnas ortonormales (Psi W Psi' = I)
    for f, nom in enumerate(FASES):
        i_f = tr[fase_i[tr] == f]
        mu_f, Ps_f, lam_f = fpca(ctx, i_f)
        Cv = np.cov(Cys[:, i_f])
        cos = np.linalg.svd(Qc.T @ (W_h @ Ps_f.T), compute_uv=False)
        fp_filas.append(dict(fase=nom, curvas=len(i_f), varianza_captada_base_propia=lam_f[:M_FPCA].sum() / lam_f.sum(),
                             varianza_captada_base_comun=np.trace(Psi @ W_y @ Cv @ W_y @ Psi.T) / lam_f.sum(),
                             coseno_angulo_principal_minimo=cos.min(), coseno_angulo_principal_medio=cos.mean(),
                             distancia_media_fase_a_media_comun_rel=np.sqrt((mu_f - mu_c) @ W_y @ (mu_f - mu_c) / (mu_c @ W_y @ mu_c))))
    v_comun, v_fase = probar(ctx, FINAL["kan"]), probar(ctx, {**FINAL["kan"], "decod": "por_fase"})
    d_fase = v_fase - v_comun
    fp = pd.DataFrame(fp_filas)
    fp["cv_kan_decodificador_comun"] = v_comun.mean(); fp["cv_kan_decodificador_por_fase"] = v_fase.mean()
    fp["cv_diferencia_pareada_media"] = d_fase.mean(); fp["cv_diferencia_pareada_sd"] = d_fase.std(ddof=1)
    fp["por_fase_mejora_concluyente"] = bool(d_fase.mean() < -d_fase.std(ddof=1))
    ctx.tablas["fpca_por_fase"] = fp.round(4)
    ctx.FP_FASE = {f: fpca(ctx, tr[fase_i[tr] == f]) for f in range(len(FASES))}
    pd.DataFrame(registro.values()).sort_values(["cabeza", "config", "pliegue"]).to_csv(CSV_BUSQUEDA, index=False)

    figuras.busqueda_hiperparametros(ctx.busqueda, len(pliegues))
    figuras.capacidad_esparsidad(capac, esp, ctx.busqueda)
    hist_cv = {(kind, f): historia_cv(ctx, FINAL[kind], f) for kind in CABEZAS for f in range(len(pliegues))}
    figuras.curvas_validacion(hist_cv, ctx.perdida_clima, ctx.cv_final, len(pliegues))
    return capac, esp, fp
