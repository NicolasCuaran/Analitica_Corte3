"""Validación del preprocesamiento: diagnósticos de las reglas, enmascaramiento, acumulados, sensibilidad, fuga y auditoría."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline

from . import comun
from .comun import ANIO_BASE, ANIO_FIN, DIR_IMAGENES, N_DIAS, PERIODO_PROYECTO, RAIZ, SEMILLA, guardar, ruta, subcarpeta
from .depuracion import (A2_RETIRAR_PRIMERO, ANIO_EST_MAX, COB_MIN, CTX_RAZON, GAP_MAX, GAP_MIN_ACUM, GRADO_VIF, HUECO_MAX,
                         IMP_K, MIN_ANIOS_CV, N_CASOS, N_T, Q_DIARIO, REGLA_E, REGLAS_ACUM, REP_MIN_MM, VENTANA_EST, VIF_B,
                         VIF_C, VIF_MAX, a_markdown, base_bspline, construir_matriz, depurar, desplazar, distancias_donantes,
                         inflacion_varianza, leer_fuentes, t_idx, vecinas_mismo_dia, vecinos_por_ocurrencia)
from .satelite import tabla_pentadas

N_CASOS_ACUM, K_VAL = 1000, 20


def cargar_registros():
    # Registros crudos con la ventana (días desde el registro anterior de la estación) y el destino de la depuración
    d = leer_fuentes()
    d = d[d.v.notna()].copy()
    d["fecha"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day), errors="coerce")
    d = d[d.fecha.notna()].drop_duplicates(["station_id", "fecha"]).sort_values(["station_id", "fecha"])
    d["gap"] = d.groupby("station_id").fecha.diff().dt.days.fillna(9999).astype(int)
    d = d[~((d.month == 2) & (d.day == 29))].copy()
    d["doy"] = d.fecha.dt.dayofyear - (d.fecha.dt.is_leap_year & (d.month > 2)).astype(int)
    ev = pd.read_csv(ruta("eventos_depuracion.csv"), usecols=["station_id", "year", "month", "day", "regla", "destino"])
    d = d.merge(ev, on=["station_id", "year", "month", "day"], how="left")
    d["regla"], d["destino"] = d.regla.fillna(""), d.destino.fillna("")
    return d.reset_index(drop=True)


def cargar_diario(columnas=("station_id", "year", "month", "doy", "value_mm")):
    return pd.read_csv(ruta("precip_diaria_depurada.csv"), usecols=list(columnas))


def ids_ciegas():
    return pd.read_csv(ruta("estaciones_ciegas_propias.csv")).station_id.to_numpy()


def matriz_diaria(d):
    # Matriz (tiempo x estación) de lluvia diaria válida a partir de una tabla con station_id, year, doy, value_mm
    est = np.sort(d.station_id.unique())
    x = d.rename(columns={"value_mm": "v"}).assign(j=np.searchsorted(est, d.station_id.to_numpy()),
                                                   t=t_idx(d.year.to_numpy(), d.doy.to_numpy()))
    return construir_matriz(x, len(est)), est


def contexto_maximo(A, desplazamientos):
    return np.maximum.reduce([np.nan_to_num(desplazar(A, o), nan=-1.0) for o in desplazamientos])


def cargar_satelite():
    s = pd.read_csv(ruta("satelite_pentadal_mm.csv"), na_values=["NA"])
    anios = np.sort(s.anio.unique())
    px = [c for c in s.columns if c.startswith("px")]
    S = np.full((len(anios), 72, len(px)), np.nan)
    S[np.searchsorted(anios, s.anio.to_numpy()), s.pentada.to_numpy() - 1] = s[px].to_numpy()
    return S, anios


def diag_criterio_v1():
    # Lo que retiraría el criterio anterior (cobertura >= 90 %, hueco <= 30 d, VIF <= 1.21, estación con >= n_min años)
    meta = pd.read_csv(ruta("metadata_depuracion.csv"))
    umb = json.loads(ruta("umbrales_depuracion.json").read_text(encoding="utf-8"))
    n_min = umb["cobertura"]["anios_min_estacion"]
    p = meta.copy()
    ok_cob = p.cobertura >= COB_MIN
    ok_hueco = ok_cob & (p.hueco_max_dias <= HUECO_MAX)
    ok_vif = ok_hueco & np.isfinite(p.vif_k20) & (p.vif_k20 <= VIF_MAX)
    n_anios = p[ok_vif & p.year.between(*VENTANA_EST)].groupby("station_id").size()
    apta = p.station_id.isin(n_anios[n_anios >= n_min].index)
    p["motivo"] = np.select([ok_vif & apta, ok_vif, ok_hueco, ok_cob],
                            ["conservado", "estacion_con_pocos_anios", "ajuste_singular_o_vif_alto", "hueco_continuo_excesivo"],
                            "anio_incompleto_cobertura")
    d = cargar_diario(("station_id", "year", "value_mm")).merge(p[["station_id", "year", "motivo"]], on=["station_id", "year"])
    llu = d[d.value_mm > 0]
    t = (d.groupby("motivo").value_mm.agg(registros="size", p_lluvia=lambda s: (s > 0).mean())
         .join(llu.groupby("motivo").value_mm.agg(dia_lluvioso_medio="mean", dia_lluvioso_p95=lambda s: s.quantile(0.95)))
         .join(p.groupby("motivo").agg(pares=("year", "size"), cobertura_media=("cobertura", "mean"))))
    t["pct_registros"] = 100 * t.registros / t.registros.sum()
    t = t.reset_index()[["motivo", "pares", "registros", "pct_registros", "cobertura_media", "p_lluvia", "dia_lluvioso_medio", "dia_lluvioso_p95"]]
    guardar("diag_criterio_v1", t)
    d["excluido"] = d.motivo != "conservado"
    por_anio = d.groupby("year").excluido.mean().mul(100)
    est = d.groupby("station_id").agg(media=("value_mm", "mean"), conservado=("excluido", lambda s: (~s).any()))
    tot = est[~est.conservado]
    r = [("registros_excluidos_pct", 100 * d.excluido.mean()),
         ("estaciones_totalmente_excluidas", int(len(tot))),
         ("mediana_lluvia_diaria_estaciones_excluidas_mm", float(tot.media.median())),
         ("mediana_lluvia_diaria_estaciones_conservadas_mm", float(est[est.conservado].media.median())),
         ("pct_excluido_1981", float(por_anio.get(1981, np.nan))), ("pct_excluido_1982", float(por_anio.get(1982, np.nan))),
         ("pct_excluido_mediana_demas_anios", float(por_anio.drop([1981, 1982], errors="ignore").median()))]
    res = pd.DataFrame(r, columns=["indicador", "valor"])
    guardar("diag_criterio_v1_resumen", res)
    return t, res


def diag_acumulados_razon():
    # Razón r = v / (g mu): vale ~1 si el registro es el total de las g jornadas de su ventana
    reg = cargar_registros()
    dia = cargar_diario(("station_id", "year", "month", "value_mm"))
    mu = dia[dia.year.between(*VENTANA_EST)].groupby(["station_id", "month"]).value_mm.mean().rename("mu").reset_index()
    sel = reg.merge(mu, on=["station_id", "month"], how="left")
    sel = sel[sel.gap.between(2, GAP_MAX) & sel.mu.notna()].copy()
    sel["razon"] = sel.v / (sel.gap * sel.mu.clip(lower=0.5))
    sel["razon_un_dia"] = sel.v / sel.mu.clip(lower=0.5)
    sel["origen"] = np.where(sel.destino == "acumulado_ventana", "acumulado_" + sel.regla,
                             np.where(sel.regla == "", "normal_con_hueco", "otro_" + sel.regla))
    t = sel.groupby("origen").agg(registros=("v", "size"), ventana_mediana=("gap", "median"), r_mediana=("razon", "median"),
                                  r_p25=("razon", lambda s: s.quantile(0.25)), r_p75=("razon", lambda s: s.quantile(0.75)),
                                  r_un_dia_mediana=("razon_un_dia", "median")).reset_index()
    guardar("diag_acumulados_razon", t)
    return t


def diag_impacto_reglas(n_max=300):
    # Cambio de la curva MCO (K=20) de cada par al conservar los valores marcados frente a retirarlos
    reg = cargar_registros()
    reg = reg[reg.year.between(*PERIODO_PROYECTO)]
    grupos = {"A1 desorbitado": reg.regla == "A1_desorbitado", "A2 repetición": reg.regla == "A2_repeticion",
              "C ceros falsos": reg.regla == "C_cero_falso", "D lectura periódica": reg.regla == "D_gauge_periodico",
              "B1/B2 acumulados": reg.regla.isin(["B1_total_mensual_ultimo_dia", "B2_registro_unico_mes"]) & (reg.destino != ""),
              "A3/B3 picos": reg.regla.isin(["B3_pico_aislado", "A3_pico_desproporcionado"])}
    Phi = base_bspline(K_VAL)
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for nombre, m in grupos.items():
        pares = reg[m][["station_id", "year"]].drop_duplicates()
        if len(pares) > n_max:
            pares = pares.sample(n_max, random_state=SEMILLA)
        sub = reg.merge(pares, on=["station_id", "year"])
        res = []
        for _, g in sub.groupby(["station_id", "year"]):
            todos, limpios = np.zeros(N_DIAS, bool), np.zeros(N_DIAS, bool)
            y = np.zeros(N_DIAS)
            y[g.doy.to_numpy() - 1] = g.v.to_numpy()
            todos[g.doy.to_numpy() - 1] = True
            limpios[g.doy[g.regla == ""].to_numpy() - 1] = True
            if todos.sum() < 30 or not np.isfinite(inflacion_varianza(Phi, todos[None, :])[0]) \
                    or not np.isfinite(inflacion_varianza(Phi, limpios[None, :])[0]):
                continue
            xr = mco(Phi, y[None, :], todos[None, :])[0][0]
            xl = mco(Phi, y[None, :], limpios[None, :])[0][0]
            res.append((np.linalg.norm(xr - xl) / np.linalg.norm(xl), 100 * (xr.sum() - xl.sum()) / xl.sum()))
        r = np.array(res)
        if len(r):
            filas.append(dict(grupo=nombre, pares=len(r), cambio_l2_mediana=np.median(r[:, 0]), cambio_l2_p90=np.quantile(r[:, 0], 0.9),
                              cambio_total_pct_mediana=np.median(r[:, 1]), cambio_total_pct_p10=np.quantile(r[:, 1], 0.1),
                              cambio_total_pct_p90=np.quantile(r[:, 1], 0.9)))
    t = pd.DataFrame(filas)
    guardar("diag_impacto_reglas", t)
    return t


def emparejar_pixeles(S, d):
    # Píxel de mayor correlación de anomalías de péntada con cada estación (solo diagnóstico)
    pen = tabla_pentadas()
    n_dias = pen.n_dias.to_numpy()
    doy2pen = np.repeat(pen.pentada.to_numpy() - 1, pen.n_dias.to_numpy())
    x = d.assign(pen=doy2pen[d.doy.to_numpy() - 1]).groupby(["station_id", "year", "pen"]).value_mm.agg(n="size", tot="sum").reset_index()
    x = x[x.n == n_dias[x.pen.to_numpy()]]
    ny = S.shape[0]
    Pix = S.reshape(ny * 72, -1)
    anom = np.nan_to_num(Pix - np.tile(np.nanmean(S, axis=0), (ny, 1)))
    mejor = {}
    for sid, g in x.groupby("station_id"):
        if len(g) < 150:
            continue
        idx = ((g.year.to_numpy() - 1981) * 72 + g.pen.to_numpy())
        ya = g.tot.to_numpy() - g.groupby("pen").tot.transform("mean").to_numpy()
        X = anom[idx] - anom[idx].mean(0)
        yc = ya - ya.mean()
        cor = (X * yc[:, None]).sum(0) / np.sqrt((X ** 2).sum(0) * (yc ** 2).sum() + 1e-12)
        j = int(np.argmax(cor))
        mejor[sid] = (j, float(cor[j]))
    return mejor, doy2pen


def diag_ceros_satelite():
    # Contraste de la regla C con el satélite: lluvia del píxel relativa a su climatología en los días marcados como ceros
    # falsos frente a las rachas secas conservadas
    S, anios = cargar_satelite()
    S = S[(anios >= 1981) & (anios <= 2003)]
    pen = tabla_pentadas()
    d = cargar_diario(("station_id", "year", "doy", "value_mm"))
    d = d[d.year <= 2003]
    mejor, doy2pen = emparejar_pixeles(S, d)
    tasa = S / pen.n_dias.to_numpy()[None, :, None]
    clim = tasa.mean(0)
    ev = pd.read_csv(ruta("eventos_depuracion.csv"), usecols=["station_id", "year", "doy", "regla"])
    marc = ev[(ev.regla == "C_cero_falso") & (ev.year <= 2003) & (ev.year >= 1981)][["station_id", "year", "doy"]]
    d = d.sort_values(["station_id", "year", "doy"])
    t = ((d.year - 1981) * 365 + d.doy).to_numpy()
    z = (d.value_mm == 0).to_numpy()
    st = d.station_id.to_numpy()
    nuevo = np.r_[True, (st[1:] != st[:-1]) | (np.diff(t) != 1) | (~z[1:]) | (~z[:-1])]
    rid = np.cumsum(nuevo)
    cons = d[z & (np.bincount(rid)[rid] >= 7)][["station_id", "year", "doy"]]

    def razon(df):
        df = df[df.station_id.isin(mejor)].copy()
        pix = np.array([mejor[s][0] for s in df.station_id])
        p = doy2pen[df.doy.to_numpy() - 1]
        y = df.year.to_numpy() - 1981
        df["razon"] = tasa[y, p, pix] / clim[p, pix]
        df["seca"] = tasa[y, p, pix] < 1.0
        return df

    marc, cons = razon(marc), razon(cons)
    a = marc.groupby("station_id").agg(r=("razon", "mean"), s=("seca", "mean"), n=("razon", "size"))
    b = cons.groupby("station_id").agg(r=("razon", "mean"), s=("seca", "mean"))
    j = a.join(b, lsuffix="_marcados", rsuffix="_conservados", how="inner")
    j = j[j.n >= 20]
    filas = [("estaciones_emparejadas", len(mejor)), ("correlacion_anomalias_mediana", float(np.median([v[1] for v in mejor.values()]))),
             ("estaciones_comparadas", len(j)), ("razon_marcados", float(j.r_marcados.mean())),
             ("razon_conservados", float(j.r_conservados.mean())),
             ("estaciones_marcados_mayor", int((j.r_marcados > j.r_conservados).sum())),
             ("pct_pentadas_secas_marcados", float(100 * j.s_marcados.mean())),
             ("pct_pentadas_secas_conservados", float(100 * j.s_conservados.mean()))]
    res = pd.DataFrame(filas, columns=["indicador", "valor"])
    guardar("diag_ceros_satelite", res)
    return res


def diag_a2_primer_dia():
    # Primer día de cada racha de repetición frente a las copias (valores redondos) y a un día normal
    reg = cargar_registros()
    marc = reg[reg.regla == "A2_repeticion"].sort_values(["station_id", "fecha"]).reset_index(drop=True)
    inicio = (marc.station_id != marc.station_id.shift()) | (marc.fecha.diff().dt.days != 1)
    if A2_RETIRAR_PRIMERO:
        primero, copias = marc[inicio], marc[~inicio]
    else:
        previo = marc[inicio][["station_id", "fecha"]].assign(fecha=lambda x: x.fecha - pd.Timedelta(days=1))
        primero, copias = reg.merge(previo, on=["station_id", "fecha"]), marc
    base = reg[(reg.regla == "") & (reg.v >= REP_MIN_MM)]
    filas = [dict(grupo=n, registros=len(x), pct_igual_10=100 * (x.v == 10).mean(), pct_igual_5=100 * (x.v == 5).mean(),
                  pct_multiplo_5=100 * (x.v % 5 == 0).mean())
             for n, x in (("copias", copias), ("primer_dia", primero), ("dias_normales_>=5mm", base))]
    t = pd.DataFrame(filas)
    guardar("diag_a2_primer_dia", t)
    return t


def diag_x01():
    # Picos con el día previo presente: dígito de más (x10) frente a tormentas reales
    cand = pd.read_csv(ruta("correcciones_candidatas_x0.1.csv"))
    d = cargar_diario(("station_id", "year", "doy", "value_mm"))
    A, est = matriz_diaria(d)
    ctx2 = contexto_maximo(A, (-2, -1, 1, 2))
    fecha = pd.to_datetime(dict(year=cand.year, month=cand.month, day=cand.day))
    doy = fecha.dt.dayofyear - ((fecha.dt.is_leap_year) & (cand.month > 2)).astype(int)
    j = np.searchsorted(est, cand.station_id.to_numpy())
    ok = np.isin(cand.station_id.to_numpy(), est)
    t = t_idx(cand.year.to_numpy(), doy.to_numpy())
    c = np.full(len(cand), np.nan)
    c[ok] = ctx2[t[ok], j[ok]]
    ratio = cand.v_crudo.to_numpy() / np.where(c > 0, c, np.nan)
    ratio10 = ratio / 10
    ref1 = (A / np.where(ctx2 > 0, ctx2, np.nan))[(A >= 20) & (A <= 60) & (ctx2 > 0)]
    ref2 = (A / np.where(ctx2 > 0, ctx2, np.nan))[(A >= 100) & (A < 140) & (ctx2 > 0)]
    r10 = ratio10[np.isfinite(ratio10)]
    r1 = ratio[np.isfinite(ratio)]
    umb = json.loads(ruta("umbrales_depuracion.json").read_text(encoding="utf-8"))
    U = umb["regla_B"]["pico_umbral_mm"]
    rng = np.random.default_rng(SEMILLA)
    prev, sig = np.nan_to_num(desplazar(A, 1), nan=-1.0), np.nan_to_num(desplazar(A, -1), nan=-1.0)
    pos = np.argwhere((A >= np.ceil(U / 10)) & (A <= 60) & (ctx2 > 0) & (prev >= 0) & (sig >= 0))
    pos = pos[rng.choice(len(pos), size=min(3000, len(pos)), replace=False)]
    ctx3 = contexto_maximo(A, (-3, -2, -1, 1, 2, 3))[pos[:, 0], pos[:, 1]]
    v10 = 10 * A[pos[:, 0], pos[:, 1]]
    rel = v10 / 10 / ctx2[pos[:, 0], pos[:, 1]]
    marcado = (v10 >= U) & (v10 > CTX_RAZON * np.maximum(ctx3, 0))
    coherente = (rel >= 0.5) & (rel <= 3.0)
    n_c = len(cand)
    fin0 = (cand.v_crudo % 10 == 0)
    reales = d.value_mm[(d.value_mm >= 100)]
    base_fin0 = float((reales % 10 == 0).mean())
    e = (fin0.mean() - base_fin0) / (1 - base_fin0)
    filas = [("picos_con_dia_previo", n_c), ("con_contexto_positivo", int(len(r10))),
             ("log10_razon_v10_mediana", float(np.median(np.log10(r10)))),
             ("log10_razon_dias_normales_20_60_mediana", float(np.median(np.log10(ref1)))),
             ("log10_razon_v_mediana", float(np.median(np.log10(r1)))),
             ("log10_razon_tormentas_100_140_mediana", float(np.median(np.log10(ref2)))),
             ("pct_termina_en_cero_candidatos", float(100 * fin0.mean())),
             ("pct_termina_en_cero_valores_ge_100", 100 * base_fin0),
             ("fraccion_estimada_digito_de_mas", float(e)),
             ("simulacion_x10_casos", int(len(pos))), ("simulacion_marcados_pct", float(100 * marcado.mean())),
             ("simulacion_marcados_y_coherentes_pct", float(100 * (marcado & coherente).mean()))]
    res = pd.DataFrame(filas, columns=["indicador", "valor"])
    guardar("diag_x01", res)
    return res


def diag_dominio_pentadas():
    # Tasa = total / días reales de la péntada en su baricentro (decisión D3) frente a v/5 en t = 5p - 2
    S, anios = cargar_satelite()
    pen = tabla_pentadas()
    rng = np.random.default_rng(SEMILLA)
    ok = [i for i, a in enumerate(anios) if a <= PERIODO_PROYECTO[1] and a % 4 != 0]
    n_px = S.shape[2]
    sel = np.column_stack([rng.choice(ok, 1000), rng.integers(0, n_px, 1000)])
    V = S[sel[:, 0], :, sel[:, 1]].T
    Phi365 = base_bspline(K_VAL)
    agg = np.zeros((72, N_DIAS))
    for p, r in enumerate(pen.itertuples()):
        agg[p, r.doy_inicio - 1:r.doy_fin] = 1.0

    def ajustar(t, y):
        B = BSpline.design_matrix(np.clip(t, 1, 365), np.r_[[1.0] * 4, np.linspace(1, N_DIAS, K_VAL - 2)[1:-1], [365.0] * 4], 3).toarray()
        C = np.linalg.solve(B.T @ B, B.T @ y)
        return Phi365 @ C

    xa = ajustar(pen.t_baricentro.to_numpy(), V / pen.n_dias.to_numpy()[:, None])
    xb = ajustar(5.0 * pen.pentada.to_numpy() - 2, V / 5.0)
    filas = []
    for nombre, x in (("decision_D3", xa), ("v/5 en 5p-2", xb)):
        tot = agg @ x
        err = tot - V
        filas.append(dict(regla=nombre, rmse_total_pentada_mm=float(np.sqrt((err ** 2).mean())),
                          error_abs_rel_pentada_pct=float(100 * (np.abs(err).sum(0) / V.sum(0)).mean()),
                          error_total_anual_pct_mediana=float(np.median(100 * (tot.sum(0) - V.sum(0)) / V.sum(0)))))
    t = pd.DataFrame(filas)
    t["desplazamiento_max_baricentro_dias"] = [0.0, float(np.abs(5.0 * pen.pentada - 2 - pen.t_baricentro).max())]
    guardar("diag_dominio_pentadas", t)
    return t


def diag_exceso_tras_hueco():
    # Exceso de registros tras un hueco sobre el cuantil diario de su estación y mes: estima la fracción de acumulados
    reg = cargar_registros()
    dia = cargar_diario(("station_id", "year", "month", "value_mm"))
    b = dia[dia.year.between(*VENTANA_EST)].groupby(["station_id", "month"]).value_mm.agg(mu="mean", q=lambda s: s.quantile(Q_DIARIO)).reset_index()
    x = reg[reg.gap.between(GAP_MIN_ACUM, GAP_MAX) & ~reg.regla.isin(["D_gauge_periodico", "A1_desorbitado", "A2_repeticion", "C_cero_falso"])]
    x = x.merge(b, on=["station_id", "month"], how="left")
    x = x[x.q.notna()]
    sobre = x.v > x.q
    esperada = 1 - Q_DIARIO
    filas = [("registros_tras_hueco_8_dias_o_mas", len(x)), ("superan_el_cuantil_diario", int(sobre.sum())),
             ("tasa_observada_pct", 100 * float(sobre.mean())), ("tasa_esperada_sin_acumulados_pct", 100 * esperada),
             ("acumulados_estimados", float(sobre.sum() - esperada * len(x))),
             ("fraccion_de_acumulados_entre_los_que_superan_pct", 100 * float(1 - esperada * len(x) / max(sobre.sum(), 1))),
             ("marcados_por_regla_E", int((x.regla == REGLA_E).sum())),
             ("superan_y_no_marcados", int((sobre & ~x.regla.isin(REGLAS_ACUM + (REGLA_E,))).sum()))]
    res = pd.DataFrame(filas, columns=["indicador", "valor"])
    guardar("diag_exceso_tras_hueco", res)
    return res


def diagnosticos():
    # Todos los diagnósticos de la depuración y su resumen en Markdown
    partes = {"Criterio de cobertura anterior aplicado a los datos actuales": diag_criterio_v1,
              "Razón de ventana de los acumulados": diag_acumulados_razon,
              "Impacto de conservar los valores marcados": diag_impacto_reglas,
              "Ceros falsos frente al satélite": diag_ceros_satelite,
              "Primer día de las rachas A2": diag_a2_primer_dia,
              "Picos y dígito de más (x10)": diag_x01,
              "Dominio común de las péntadas": diag_dominio_pentadas,
              "Exceso de valores altos tras un hueco": diag_exceso_tras_hueco}
    md = ["# Diagnósticos de la depuración (Ítem 4.1)", "", "Cada tabla la genera red_funcional/src/validacion.py.", ""]
    for titulo, f in partes.items():
        r = f()
        tablas = list(r) if isinstance(r, tuple) else [r]
        md += [f"## {titulo}", ""]
        for tb in tablas:
            md += [a_markdown(tb.round(4)), ""]
    guardar("diag_resumen_md", "\n".join(md))


COB_REF = 0.99


def estaciones_con_historia(meta, est):
    # Estaciones con al menos MIN_ANIOS_CV años de nivel A en el entrenamiento (climatologías de dejar-un-año-fuera)
    n = meta[(meta.nivel == "A") & meta.year.between(*VENTANA_EST) & meta.station_id.isin(est)].groupby("station_id").size()
    return n.index[n >= MIN_ANIOS_CV]


LAMBDAS_REL = np.logspace(-4, 2, 10)

PATRONES = {
    "aleatorio_20": ("aleatorio", 0.20), "aleatorio_40": ("aleatorio", 0.40),
    "aleatorio_60": ("aleatorio", 0.60), "aleatorio_80": ("aleatorio", 0.80),
    "bloque_30": ("bloque", 30), "bloque_60": ("bloque", 60), "bloque_120": ("bloque", 120),
    "bloque_180": ("bloque", 180),
    "bloque_60_lluvias_abr_may": ("bloque_en", (90, 60)), "bloque_60_seca_jul_ago": ("bloque_en", (180, 60)),
    "mascara_real": ("real", None),
}


def rugosidad(K_VAL, grado=GRADO_VIF):
    # Matriz R = int phi'' phi'' dt por la regla del trapecio
    t = np.arange(1, N_DIAS + 1, dtype=float)
    interiores = np.linspace(1, N_DIAS, K_VAL - grado + 1)[1:-1]
    nodos = np.r_[[1.0] * (grado + 1), interiores, [float(N_DIAS)] * (grado + 1)]
    malla = np.linspace(1, N_DIAS, 4 * N_DIAS)
    d2 = np.column_stack([BSpline(nodos, np.eye(K_VAL)[k], grado).derivative(2)(malla) for k in range(K_VAL)])
    return trapezoid(d2[:, :, None] * d2[:, None, :], malla, axis=0)


def mascaras(rng, n, patron, pool=None):
    # Días ocultos de n casos según el patrón (aleatorio, bloque, bloque en una temporada o máscara real)
    tipo, par = PATRONES[patron]
    M = np.zeros((n, N_DIAS), dtype=bool)
    if tipo == "real":
        return ~pool[rng.integers(0, len(pool), n)]
    if tipo == "aleatorio":
        return rng.random((n, N_DIAS)) < par
    if tipo == "bloque":
        ini = rng.integers(0, N_DIAS - par + 1, n)
    else:
        centro, ancho = par
        ini = rng.integers(centro, centro + 31, n)
        par = ancho
    k = np.arange(N_DIAS)
    return (k[None, :] >= ini[:, None]) & (k[None, :] < (ini + par)[:, None])


def mco(Phi, Y, obs, R=None, lam=0.0):
    # Y (n,365) con NaN permitido en ~obs; c = (Phi_o' Phi_o + lam R)^-1 Phi_o' y_o  -> curva (n,365)
    Yz = np.where(obs, Y, 0.0)
    Go = np.einsum("nt,tk,tl->nkl", obs.astype(float), Phi, Phi)
    b = np.einsum("nt,tk->nk", Yz, Phi)
    if R is not None:
        Go = Go + lam * R
    Go = Go + 1e-9 * np.trace(Phi.T @ Phi) / K_VAL * np.eye(K_VAL)
    c = np.linalg.solve(Go, b[:, :, None])[:, :, 0]
    return c @ Phi.T, c


def mco_gcv(Phi, Y, obs, R):
    # MCO penalizado con lambda elegido por VCG en la rejilla LAMBDAS_REL, caso por caso
    escala = np.trace(Phi.T @ Phi) / np.trace(R)
    n_obs = obs.sum(axis=1)
    mejor = np.full(len(Y), np.inf)
    curva = np.zeros_like(Y, dtype=float)
    lam_sel = np.zeros(len(Y))
    Go0 = np.einsum("nt,tk,tl->nkl", obs.astype(float), Phi, Phi)
    for lr in LAMBDAS_REL:
        lam = lr * escala
        yhat, _ = mco(Phi, Y, obs, R, lam)
        A = Go0 + lam * R + 1e-9 * np.trace(Phi.T @ Phi) / K_VAL * np.eye(K_VAL)
        tr = np.trace(np.linalg.solve(A, Go0), axis1=1, axis2=2)
        rss = (np.where(obs, Y - yhat, 0.0) ** 2).sum(axis=1)
        gcv = (rss / n_obs) / np.clip(1 - tr / n_obs, 1e-6, None) ** 2
        mej = gcv < mejor
        mejor[mej], curva[mej], lam_sel[mej] = gcv[mej], yhat[mej], lr
    return curva, lam_sel


def escribir_md():
    # Resumen en Markdown del experimento de enmascaramiento
    r = pd.read_csv(ruta("validacion_faltantes_metodos.csv"))
    v = pd.read_csv(ruta("validacion_faltantes_vif.csv"))
    res = json.loads((ruta("validacion_faltantes_resumen.json")).read_text(encoding="utf-8"))
    nombres = {"suma_observada": "Suma de lo observado (curva acumulada ingenua)", "suma_escalada_1/cobertura": "Suma escalada por 1/cobertura",
               "mco_sin_penalizar": "MCO sin penalizar (R2), K_VAL=20", "mco_penalizado_gcv": "MCO penalizado, λ por VCG",
               "media_global_estacion": "Imputación por la media de la estación", "media_por_dia_del_anio": "Imputación por la media del día del año",
               "regresion_vecinos": "Imputación por regresión con vecinas",
               "knn_mismo_dia": f"KNN del mismo día ({IMP_K} vecinas, media, solo distancia horizontal)",
               "idw_altura": f"IDW del mismo día con distancia horizontal y altura ({IMP_K} vecinas; paso P6)"}
    cols = {"abs_error_total_pct_mediana": "|error| total anual, mediana (%)", "error_l2_rel_mediana": "error L2 relativo de la curva, mediana",
            "razon_varianza_interanual": "varianza interanual retenida (1 = sin pérdida)",
            "rmse_dias_ocultos_mediana": "RMSE en los días ocultos, mediana (mm/día)"}
    md = ["# Experimento de enmascaramiento (Ítem 4.1)", "",
          f"{res['casos']} estación-años de nivel A con cobertura ≥ {res['cobertura_ref']:.0%} (1981–{res['anio_max_usado']}, {res['estaciones']} estaciones, semilla {res['semilla']}). "
          f"Se ocultan días con {len(PATRONES)} patrones y cada estrategia se compara con la curva MCO del año completo. Solo años de entrenamiento (≤ {ANIO_EST_MAX}) y sin estaciones ciegas.", ""]
    orden = list(nombres)
    for col, titulo in cols.items():
        t = r.pivot(index="patron", columns="metodo", values=col).reindex(columns=orden)
        t = t.loc[r.patron.unique()]
        t.columns = [nombres[c] for c in t.columns]
        t.insert(0, "fracción oculta", r.groupby("patron").frac_oculta_media.first().loc[t.index])
        md += [f"## {titulo}", "", t.reset_index().rename(columns={"patron": "patrón"}).pipe(
            lambda x: a_markdown(x.round(3))), ""]
    md += ["## Error del MCO sin penalizar según el VIF del patrón de datos observados", "",
           a_markdown(v.round(3)), ""]
    guardar("validacion_faltantes_md", chr(10).join(md).replace("| nan ", "| — "))


def validar_faltantes():
    # Experimento de enmascaramiento: estrategias para los días faltantes frente a la curva MCO del año completo
    rng = np.random.default_rng(SEMILLA)
    d = pd.read_csv(ruta("precip_diaria_depurada.csv"), usecols=["station_id", "year", "doy", "value_mm"])
    d = d[d.year.between(*VENTANA_EST) & ~d.station_id.isin(ids_ciegas())].copy()
    est = np.sort(d.station_id.unique())
    n_s = len(est)
    d["j"] = np.searchsorted(est, d.station_id.to_numpy())
    d["t"] = t_idx(d.year.to_numpy(), d.doy.to_numpy())
    d = d.rename(columns={"value_mm": "v"})
    A = construir_matriz(d, n_s)
    n_anios = N_T // N_DIAS
    vecinos = vecinos_por_ocurrencia(A)

    NB = np.full(A.shape, np.nan, dtype=np.float32)
    for j, ix in enumerate(vecinos):
        if ix.size:
            b = A[:, ix]
            nv = (~np.isnan(b)).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.nansum(b, axis=1) / nv
            m[nv < 2] = np.nan
            NB[:, j] = m

    A3, NB3 = A.reshape(n_anios, N_DIAS, n_s), NB.reshape(n_anios, N_DIAS, n_s)
    ok2 = ~np.isnan(A3) & ~np.isnan(NB3)
    x, y = np.where(ok2, NB3, 0.0).astype(float), np.where(ok2, A3, 0.0).astype(float)
    estad = dict(n=ok2.sum(1).astype(float), sx=x.sum(1), sy=y.sum(1), sxx=(x * x).sum(1), sxy=(x * y).sum(1))
    tot = {k: v.sum(0) for k, v in estad.items()}
    ok1 = ~np.isnan(A3)
    s_doy, n_doy = np.where(ok1, A3, 0.0).sum(0), ok1.sum(0)
    s_glob, n_glob = s_doy.sum(0), n_doy.sum(0)

    meta = pd.read_csv(ruta("metadata_depuracion.csv"), usecols=["station_id", "year", "cobertura", "nivel"])
    cand = meta[(meta.nivel == "A") & (meta.cobertura >= COB_REF) & meta.year.between(*VENTANA_EST)
                & meta.station_id.isin(estaciones_con_historia(meta, est))]
    cand = cand.sample(n=min(N_CASOS, len(cand)), random_state=SEMILLA).reset_index(drop=True)
    jp = np.searchsorted(est, cand.station_id.to_numpy())
    yp = cand.year.to_numpy() - ANIO_BASE
    idx = yp[:, None] * N_DIAS + np.arange(N_DIAS)[None, :]
    Y0 = A[idx, jp[:, None]].astype(float)
    obs0 = ~np.isnan(Y0)
    Phi = base_bspline(K_VAL)
    R = rugosidad(K_VAL)
    X_true, _ = mco(Phi, Y0, obs0)
    tot_true = X_true.sum(axis=1)

    def loyo_param(j, y):
        n, sx, sy, sxx, sxy = (tot[k][j] - estad[k][y, j] for k in ("n", "sx", "sy", "sxx", "sxy"))
        den = n * sxx - sx ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where((n >= 30) & (den > 1e-9), (n * sxy - sx * sy) / den, np.nan)
            alfa = (sy - beta * sx) / n
        return alfa, beta

    alfa, beta = loyo_param(jp, yp)
    clim_doy = np.full((len(cand), N_DIAS), np.nan)
    ndo = n_doy[:, jp].T - obs0
    sdo = s_doy[:, jp].T - np.where(obs0, Y0, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        clim_doy = np.where(ndo >= 3, sdo / ndo, np.nan)
        media_glob = (s_glob[jp] - np.where(obs0, Y0, 0.0).sum(1)) / (n_glob[jp] - obs0.sum(1))
    X_clim = mco(Phi, np.where(np.isnan(clim_doy), media_glob[:, None], clim_doy), np.ones_like(obs0))[0]
    nb_caso = NB3[yp[:, None], np.arange(N_DIAS)[None, :], jp[:, None]].astype(float)
    pred_vec = np.clip(alfa[:, None] + beta[:, None] * nb_caso, 0, None)
    emp_v = pd.read_csv(ruta("emparejamiento_estacion_pixel.csv"))
    D_knn = distancias_donantes(emp_v, est, ())[0]
    pred_knn = np.vstack([vecinas_mismo_dia(A, D_knn, j, y * N_DIAS + np.arange(N_DIAS), idw=False)[0] for j, y in zip(jp, yp)])
    alfa_imp = json.loads(ruta("umbrales_depuracion.json").read_text(encoding="utf-8"))["imputacion"]["alfa_m_por_m"]
    D_idw = distancias_donantes(emp_v, est, (), alfa_imp)[0]
    pred_idw = np.vstack([vecinas_mismo_dia(A, D_idw, j, y * N_DIAS + np.arange(N_DIAS))[0] for j, y in zip(jp, yp)])
    reales = meta[(meta.cobertura >= 0.4) & (meta.cobertura < 0.9) & meta.year.between(*VENTANA_EST) & meta.station_id.isin(est)]
    jr, yr = np.searchsorted(est, reales.station_id.to_numpy()), reales.year.to_numpy() - ANIO_BASE
    pool = ~np.isnan(A[yr[:, None] * N_DIAS + np.arange(N_DIAS)[None, :], jr[:, None]])

    filas, filas_vif, por_estacion = [], [], []
    for patron in PATRONES:
        oculto = mascaras(rng, len(cand), patron, pool) & obs0
        obs = obs0 & ~oculto
        vif = inflacion_varianza(Phi, obs)
        Yobs = np.where(obs, Y0, np.nan)
        n_obs = obs.sum(1)
        frac = oculto.sum(1) / obs0.sum(1)

        def llenar(rell):
            Yf = np.where(oculto & ~np.isnan(rell), rell, Yobs)
            return Yf, ~np.isnan(Yf)

        resultados = {}
        resultados["suma_observada"] = (None, np.nansum(Yobs, axis=1))
        resultados["suma_escalada_1/cobertura"] = (None, np.nansum(Yobs, axis=1) * obs0.sum(1) / n_obs)
        xm, _ = mco(Phi, Y0, obs)
        resultados["mco_sin_penalizar"] = (xm, xm.sum(1))
        xp, lam_sel = mco_gcv(Phi, Y0, obs, R)
        resultados["mco_penalizado_gcv"] = (xp, xp.sum(1))
        for nombre, rell in (("media_global_estacion", np.repeat(media_glob[:, None], N_DIAS, 1)),
                             ("media_por_dia_del_anio", np.where(np.isnan(clim_doy), media_glob[:, None], clim_doy)),
                             ("regresion_vecinos", np.where(np.isnan(pred_vec), np.nan, pred_vec)),
                             ("knn_mismo_dia", pred_knn), ("idw_altura", pred_idw)):
            Yf, of = llenar(rell)
            xc = mco_gcv(Phi, Yf, of, R)[0] if nombre == "regresion_vecinos" else mco(Phi, Yf, of)[0]
            resultados[nombre] = (xc, xc.sum(1))

        for nombre, (curva, total) in resultados.items():
            err_tot = 100 * (total - tot_true) / tot_true
            fila = dict(patron=patron, metodo=nombre, casos=len(cand), frac_oculta_media=float(frac.mean()),
                        sesgo_total_pct_mediana=float(np.median(err_tot)),
                        abs_error_total_pct_mediana=float(np.median(np.abs(err_tot))),
                        abs_error_total_pct_p90=float(np.quantile(np.abs(err_tot), 0.9)))
            if curva is not None:
                e = np.linalg.norm(curva - X_true, axis=1) / np.linalg.norm(X_true, axis=1)
                habilidad = 1 - ((curva - X_true) ** 2).sum(1) / ((X_clim - X_true) ** 2).sum(1)
                rmse = np.sqrt((np.where(oculto, curva - np.nan_to_num(Y0), 0.0) ** 2).sum(1) / np.maximum(oculto.sum(1), 1))
                fila["rmse_dias_ocultos_mediana"] = float(np.median(rmse[oculto.sum(1) > 0]))
                fila.update(error_l2_rel_mediana=float(np.median(e)), error_l2_rel_p90=float(np.quantile(e, 0.9)),
                            habilidad_vs_climatologia_mediana=float(np.median(habilidad)))
            filas.append(fila)
            g = pd.DataFrame(dict(est=cand.station_id, tot_hat=total, tot_true=tot_true))
            g = g.groupby("est").filter(lambda s: len(s) >= 4)
            if len(g):
                v = g.groupby("est").agg(vh=("tot_hat", "var"), vt=("tot_true", "var"))
                por_estacion.append(dict(patron=patron, metodo=nombre, razon_varianza_interanual=float((v.vh / v.vt).median()),
                                         estaciones=len(v)))
            if nombre == "mco_sin_penalizar":
                e = np.linalg.norm(curva - X_true, axis=1) / np.linalg.norm(X_true, axis=1)
                filas_vif.append(pd.DataFrame(dict(patron=patron, vif=vif, error_l2_rel=e, frac_oculta=frac)))

    res = pd.DataFrame(filas)
    res = res.merge(pd.DataFrame(por_estacion), on=["patron", "metodo"], how="left")
    res.to_csv(ruta("validacion_faltantes_metodos.csv"), index=False, float_format="%.4f")
    f = res[(res.metodo == "mco_penalizado_gcv") & res.patron.str.startswith("aleatorio")].copy()
    f["cobertura"] = 1 - f.frac_oculta_media
    pasa = f[(f.abs_error_total_pct_mediana <= 10) & (f.razon_varianza_interanual <= 1.5)]
    cobertura_min = round(float(pasa.cobertura.min()), 2) if len(pasa) else None

    fv = pd.concat(filas_vif)
    cortes = [0, VIF_MAX, 2.0, VIF_B, VIF_C, np.inf]
    etiq = [f"<= {VIF_MAX:.2f} (nivel A)", f"{VIF_MAX:.2f}-2", f"2-{VIF_B:g} (nivel B)", f"{VIF_B:g}-{VIF_C:g} (nivel C)", f"> {VIF_C:g} (nivel D)"]
    fv["banda_vif"] = pd.cut(fv.vif.where(np.isfinite(fv.vif), 1e9), cortes, labels=etiq, include_lowest=True).astype(object)
    fv["singular"] = ~np.isfinite(fv.vif)
    fv.loc[fv.singular, "banda_vif"] = "singular (nivel D)"
    tabla_vif = fv.groupby("banda_vif", observed=True).agg(casos=("vif", "size"), error_l2_rel_mediana=("error_l2_rel", "median"),
                                                          error_l2_rel_p90=("error_l2_rel", lambda s: s.quantile(0.9)),
                                                          frac_oculta_media=("frac_oculta", "mean")).reset_index()
    tabla_vif.to_csv(ruta("validacion_faltantes_vif.csv"), index=False, float_format="%.4f")
    resumen = dict(semilla=SEMILLA, casos=len(cand), cobertura_ref=COB_REF, anio_max_usado=int(cand.year.max()),
                   estaciones=int(cand.station_id.nunique()), k=K_VAL,
                   cobertura_minima_penalizado=cobertura_min, mascaras_reales=int(len(pool)))
    (ruta("validacion_faltantes_resumen.json")).write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    escribir_md()


ESCENARIOS = {
    "un_mes_acumulado": ("bloque", 30), "una_semana_acumulada": ("bloque", 7), "tres_meses_acumulados": ("bloque", 90),
    "lectura_semanal_todo_el_anio": ("periodica", 7), "lectura_mensual_todo_el_anio": ("periodica", 30),
}


def ventanas(rng, tipo, par):
    # Ventanas [ini, fin] (índices 0-364) cuyo total se registra en el día fin
    if tipo == "bloque":
        ini = int(rng.integers(0, N_DIAS - par + 1))
        return [(ini, ini + par - 1)]
    off = int(rng.integers(0, par))
    fines = np.arange(off + par - 1, N_DIAS, par)
    return [(int(f - par + 1), int(f)) for f in fines]


def ajuste(B, y, obs, vent, tot):
    # c = (B_o' B_o + sum_w a_w a_w'/g_w)^-1 (B_o' y_o + sum_w a_w T_w/g_w), con a_w = suma de las filas de B en la ventana
    p = B.shape[1]
    G = (B[obs].T @ B[obs]) if obs.any() else np.zeros((p, p))
    b = B[obs].T @ y[obs] if obs.any() else np.zeros(p)
    for (i0, i1), T in zip(vent, tot):
        a = B[i0:i1 + 1].sum(axis=0)
        w = 1.0 / (i1 - i0 + 1)
        G = G + w * np.outer(a, a)
        b = b + w * a * T
    G = G + 1e-9 * np.trace(B.T @ B) / p * np.eye(p)
    return np.linalg.solve(G, b)


def validar_acumulados():
    # Acumulados con ventana como observaciones agregadas frente a descartarlos, al total ingenuo y a la desagregación
    rng = np.random.default_rng(SEMILLA)
    d = pd.read_csv(ruta("precip_diaria_depurada.csv"), usecols=["station_id", "year", "doy", "value_mm"])
    d = d[d.year.between(*VENTANA_EST) & ~d.station_id.isin(ids_ciegas())].copy()
    est = np.sort(d.station_id.unique())
    n_s = len(est)
    d["j"] = np.searchsorted(est, d.station_id.to_numpy())
    d["t"] = t_idx(d.year.to_numpy(), d.doy.to_numpy())
    A = construir_matriz(d.rename(columns={"value_mm": "v"}), n_s)
    n_anios = N_T // N_DIAS
    A3 = A.reshape(n_anios, N_DIAS, n_s)
    ok1 = ~np.isnan(A3)
    s_doy, n_doy = np.where(ok1, A3, 0.0).sum(0), ok1.sum(0)

    meta = pd.read_csv(ruta("metadata_depuracion.csv"), usecols=["station_id", "year", "cobertura", "nivel"])
    cand = meta[(meta.nivel == "A") & (meta.cobertura >= COB_REF) & meta.year.between(*VENTANA_EST)
                & meta.station_id.isin(estaciones_con_historia(meta, est))]
    cand = cand.sample(n=min(N_CASOS_ACUM, len(cand)), random_state=SEMILLA).reset_index(drop=True)
    jp = np.searchsorted(est, cand.station_id.to_numpy())
    yp = cand.year.to_numpy() - ANIO_BASE
    idx = yp[:, None] * N_DIAS + np.arange(N_DIAS)[None, :]
    Y0 = A[idx, jp[:, None]].astype(float)
    obs0 = ~np.isnan(Y0)
    Phi = base_bspline(K_VAL)
    Phi12 = base_bspline(12)
    X_true, _ = mco(Phi, Y0, obs0)
    tot_true = X_true.sum(1)

    ndo = n_doy[:, jp].T - obs0
    sdo = s_doy[:, jp].T - np.where(obs0, Y0, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(ndo >= 3, sdo / ndo, np.nan)
    clim = np.where(np.isnan(clim), np.nanmean(clim, axis=1, keepdims=True), clim)

    filas = []
    for nombre, (tipo, par) in ESCENARIOS.items():
        err = {}
        for i in range(len(cand)):
            y = np.nan_to_num(Y0[i])
            vent = ventanas(rng, tipo, par)
            tot = [y[a:b + 1].sum() for a, b in vent]
            oculto = np.zeros(N_DIAS, bool)
            for a, b in vent:
                oculto[a:b + 1] = True
            obs_d = obs0[i] & ~oculto if tipo == "bloque" else np.zeros(N_DIAS, bool)
            ult = np.array([b for _, b in vent])
            out = {}
            out["descartar"] = Phi @ ajuste(Phi, y, obs_d, [], []) if obs_d.sum() >= K_VAL else np.full(N_DIAS, np.nan)
            ynaive, onaive = y.copy(), obs_d.copy()
            ynaive[ult] = tot
            onaive[ult] = True
            out["naive_total_como_valor_diario"] = Phi @ ajuste(Phi, ynaive, onaive, [], [])
            Pb = Phi if tipo == "bloque" else Phi12
            out["agregado_mco"] = Pb @ ajuste(Pb, y, obs_d, vent, tot)
            yd = np.where(obs_d, y, 0.0)
            for (a, b), T in zip(vent, tot):
                w = clim[i][a:b + 1]
                yd[a:b + 1] = w * T / max(w.sum(), 1e-9)
            full = obs_d | oculto
            out["desagregacion_proporcional_climatologia"] = Phi @ ajuste(Phi, yd, full, [], []) if full.sum() >= K_VAL else np.full(N_DIAS, np.nan)
            out["solo_climatologia"] = Phi @ ajuste(Phi, clim[i], np.ones(N_DIAS, bool), [], [])
            for k, x in out.items():
                err.setdefault(k, []).append((np.linalg.norm(x - X_true[i]) / np.linalg.norm(X_true[i]),
                                              100 * (x.sum() - tot_true[i]) / tot_true[i]))
        for k, e in err.items():
            e = np.array(e)
            filas.append(dict(escenario=nombre, metodo=k, casos=int(np.isfinite(e[:, 0]).sum()),
                              error_l2_rel_mediana=float(np.nanmedian(e[:, 0])), error_l2_rel_p90=float(np.nanquantile(e[:, 0], 0.9)),
                              abs_error_total_pct_mediana=float(np.nanmedian(np.abs(e[:, 1]))),
                              sesgo_total_pct_mediana=float(np.nanmedian(e[:, 1]))))
    res = pd.DataFrame(filas)
    guardar("validacion_acumulados", res)
    guardar("validacion_acumulados_resumen", dict(semilla=SEMILLA, casos=len(cand), k=K_VAL))
    md = ["# Validación: acumulados con ventana como observaciones agregadas (Ítem 4.1)", "",
          f"{len(cand)} estación-años de nivel A con cobertura ≥ 99 % (entrenamiento {VENTANA_EST[0]}–{VENTANA_EST[1]}, sin estaciones ciegas, semilla {SEMILLA}). Se reproduce la estructura de los acumulados reales: "
          "el total de g días se registra en el último día de la ventana. Referencia: curva MCO del año completo.", ""]
    for col, tit in (("abs_error_total_pct_mediana", "|error| del total anual, mediana (%)"), ("error_l2_rel_mediana", "error L2 relativo de la curva, mediana"),
                     ("sesgo_total_pct_mediana", "sesgo del total anual, mediana (%)")):
        t = res.pivot(index="escenario", columns="metodo", values=col).loc[list(ESCENARIOS)].round(3).reset_index()
        md += [f"## {tit}", "", a_markdown(t).replace("| nan ", "| — "), ""]
    guardar("validacion_acumulados_md", "\n".join(md))


def totales_par(d):
    return d.groupby(["station_id", "year"]).value_mm.sum().rename("total").reset_index()


def comparar_corridas(base, var):
    # Cambios de una corrida con otros umbrales frente a la corrida de referencia
    t0, t1 = totales_par(base["maestro"]), totales_par(var["maestro"])
    m = t0.merge(t1, on=["station_id", "year"], how="outer", suffixes=("_base", "_var")).fillna(0.0)
    cambia = m.total_base != m.total_var
    rel = (m.total_var - m.total_base).abs() / m.total_base.where(m.total_base > 0)
    n0 = base["meta"].set_index(["station_id", "year"]).nivel
    n1 = var["meta"].set_index(["station_id", "year"]).nivel
    comunes = n0.index.intersection(n1.index)
    return dict(registros_diarios=len(var["maestro"]), ventanas=len(var["acum"]),
                pares_con_total_distinto_pct=100 * cambia.mean(),
                cambio_relativo_total_p99_pct=100 * float(rel[cambia].quantile(0.99)) if cambia.any() else 0.0,
                cambio_relativo_total_max_pct=100 * float(rel[cambia].max()) if cambia.any() else 0.0,
                pares_con_nivel_distinto=int((n0.loc[comunes] != n1.loc[comunes]).sum()))


def leer_corrida(base_dir):
    def leer(nombre, **kw):
        return pd.read_csv(Path(base_dir) / subcarpeta(nombre) / nombre, **kw)
    return dict(maestro=leer("precip_diaria_depurada.csv", usecols=["station_id", "year", "value_mm"]),
                meta=leer("metadata_depuracion.csv", usecols=["station_id", "year", "nivel"]),
                acum=leer("acumulados_ventana.csv", usecols=["station_id", "year"]))


def sensibilidad():
    # Una variación a la vez de los umbrales de la depuración; cada corrida completa se compara con la base
    variantes = {"base": {}, "PER_FRAC=0.3": {"PER_FRAC": 0.3}, "PER_FRAC=0.7": {"PER_FRAC": 0.7},
                 "PER_MIN_REG=3": {"PER_MIN_REG": 3}, "PER_MIN_REG=10": {"PER_MIN_REG": 10},
                 "ACUM_RAZON=(0.15,6)": {"ACUM_RAZON": (0.15, 6.0)}, "ACUM_RAZON=(0.4,2.5)": {"ACUM_RAZON": (0.4, 2.5)},
                 "Q_DIARIO=0.99": {"Q_DIARIO": 0.99}, "Q_DIARIO=0.999": {"Q_DIARIO": 0.999},
                 "CERO_ALFA=1e-3": {"CERO_ALFA": 1e-3}, "CERO_ALFA=1e-6": {"CERO_ALFA": 1e-6},
                 "A2_RETIRAR_PRIMERO=False": {"A2_RETIRAR_PRIMERO": False},
                 "GAP_MIN_ACUM=5": {"GAP_MIN_ACUM": 5}, "GAP_MIN_ACUM=15": {"GAP_MIN_ACUM": 15}}
    tmp = RAIZ / "data" / "interim" / "_sensibilidad"
    base = None
    filas = []
    for nombre, par in variantes.items():
        carpeta = tmp / nombre.replace("=", "_").replace("(", "").replace(")", "").replace(",", "_")
        u = depurar(dir_salida=carpeta, parametros=par)
        corrida = leer_corrida(carpeta)
        if base is None:
            base = corrida
        fila = dict(variante=nombre, **comparar_corridas(base, corrida))
        dest = u["acumulados"]["destinos"]
        fila.update(b2_recuperados=dest.get("B2_registro_unico_mes|recuperado_diario", 0),
                    ceros_falsos=u["regla_C"]["dias_cero_marcados"], periodicos=u["regla_D"]["registros"])
        filas.append(fila)
        shutil.rmtree(carpeta, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    t = pd.DataFrame(filas)
    guardar("sensibilidad_umbrales", t)
    meta = pd.read_csv(ruta("metadata_depuracion.csv"))
    cd = meta[(meta.nivel == "D") & meta.year.between(*VENTANA_EST) & ~meta.ciega]
    comp = pd.DataFrame([dict(cobertura_minima=c, pares=int((cd.cobertura >= c).sum()), registros=int(cd.n_validos[cd.cobertura >= c].sum()))
                         for c in (0.2, 0.3, 0.4, 0.5, 0.6)])
    guardar("sensibilidad_completar", comp)
    md = ["# Sensibilidad de los umbrales de la depuración (Ítem 4.1)", "",
          "Cada fila repite la depuración completa con un umbral modificado y compara el total anual de cada estación-año con la corrida base.", "",
          a_markdown(t.round(4)), "", f"## Cobertura mínima para el suavizamiento penalizado (nivel D, entrenamiento {VENTANA_EST[0]}-{VENTANA_EST[1]})", "", a_markdown(comp), ""]
    guardar("sensibilidad_umbrales_md", "\n".join(md))
    return t


def prueba_fuga():
    # Se sustituyen los años de prueba por datos falsos y se agrega una estación presente solo en el segundo año de prueba;
    # ninguna salida de años <= ANIO_EST_MAX debe cambiar
    def envenenado():
        d = leer_fuentes()
        rng = np.random.default_rng(0)
        base = d[d.year == ANIO_EST_MAX].copy()
        d = d[d.year <= ANIO_EST_MAX]
        partes = [d]
        for a in range(ANIO_EST_MAX + 1, ANIO_FIN + 1):
            z = base.copy()
            z["year"] = a
            z["v"] = np.where(rng.random(len(z)) < 0.5, rng.integers(0, 900, len(z)), 0).astype(float)
            partes.append(z)
        solo = base[base.station_id == base.station_id.iloc[0]].copy()
        solo["station_id"], solo["year"], solo["v"] = 9999999901, ANIO_EST_MAX + 2, 7.0
        partes.append(solo)
        return pd.concat(partes, ignore_index=True)

    tmp = RAIZ / "data" / "interim" / "_fuga"
    u1 = depurar(dir_salida=tmp, lector=envenenado)
    u0 = json.loads(ruta("umbrales_depuracion.json").read_text(encoding="utf-8"))
    claves = [("regla_A", "rachas_repetidas_observadas"), ("regla_A", "rachas_repetidas_esperadas_independencia"),
              ("regla_B", "pico_umbral_mm"), ("regla_A3", "umbral_mm"), ("regla_C", "p_lluvia_global_por_categoria"),
              ("regla_C", "n_dias_por_categoria"), ("cobertura", "cv_interanual_p90"), ("cobertura", "anios_min_estacion"),
              ("cobertura", "vif_nivel_a_por_k"), ("vecinos", "k")]
    res = {f"{a}.{k}": bool(u0[a][k] == u1[a][k]) for a, k in claves}

    def leer(base, nombre, **kw):
        return pd.read_csv(Path(base) / subcarpeta(nombre) / nombre, **kw)

    def hasta(df):
        return df[(df.year <= ANIO_EST_MAX) & (df.station_id != 9999999901)].reset_index(drop=True)

    for nombre in ("metadata_depuracion.csv", "precip_diaria_depurada.csv", "precip_diaria_imputada.csv", "acumulados_ventana.csv",
                   "correcciones_candidatas_x0.1.csv"):
        res[f"{nombre} (años <= {ANIO_EST_MAX})"] = bool(hasta(leer(comun.BASE_PROC, nombre)).equals(hasta(leer(tmp, nombre))))
    res["estaciones_ciegas_propias.csv"] = bool(leer(comun.BASE_PROC, "estaciones_ciegas_propias.csv").equals(leer(tmp, "estaciones_ciegas_propias.csv")))
    e0, e1 = leer(comun.BASE_PROC, "estaciones_resumen.csv"), leer(tmp, "estaciones_resumen.csv")
    e0, e1 = e0[e0.station_id.isin(e1.station_id)], e1[e1.station_id != 9999999901]
    res["estaciones_aptas_para_evaluacion (estaciones con datos <= %d)" % ANIO_EST_MAX] = bool(
        e0.set_index("station_id").apta_evaluacion.equals(e1.set_index("station_id").apta_evaluacion))
    m1 = leer(tmp, "metadata_depuracion.csv")
    res[f"estacion_solo_prueba_sin_pares_<={ANIO_EST_MAX}_con_datos"] = bool(m1[(m1.station_id == 9999999901) & (m1.year <= ANIO_EST_MAX)].n_validos.sum() == 0)
    shutil.rmtree(tmp, ignore_errors=True)
    resumen = dict(aprobada=all(res.values()), comprobaciones=res)
    guardar("prueba_fuga", resumen)
    md = ["# Prueba de fuga de información (Ítem 4.1)", "",
          f"Se sustituyeron los años de prueba ({ANIO_EST_MAX + 1}-{ANIO_FIN}) por datos falsos (ruido de hasta 900 mm) y se agregó una estación presente solo en {ANIO_EST_MAX + 2}. "
          "Cada fila compara una salida de la corrida real con la de la corrida envenenada.", "",
          "| Comprobación | Idéntica |", "|---|---|"] + [f"| {k} | {'sí' if v else 'NO'} |" for k, v in res.items()]
    guardar("prueba_fuga_md", "\n".join(md))
    print("prueba de fuga:", "APROBADA" if resumen["aprobada"] else "FALLA", resumen["comprobaciones"] if not resumen["aprobada"] else "")
    return resumen


def auditoria():
    # Precisión decimal de las estaciones y cobertura anual del satélite
    d = leer_fuentes()
    d = d[d.v.notna()]
    d["decimal"] = d.v != np.floor(d.v)
    d["tres_digitos"] = (d.v >= 100) & (d.v < 1000)
    est = d.groupby("station_id").agg(registros=("v", "size"), con_decimal=("decimal", "sum"), media_mm=("v", "mean"))
    est["solo_enteros"] = est.con_decimal == 0
    llu = d[d.v > 0].merge(est[["solo_enteros"]], left_on="station_id", right_index=True)
    est["media_dia_lluvioso_mm"] = llu.groupby("station_id").v.mean()
    guardar("diag_decimales_estaciones", est.reset_index())

    rangos = pd.cut(d.v, [0, 1e-9, 1, 10, 100, 1000, np.inf], labels=["0", "(0,1)", "[1,10)", "[10,100)", "[100,1000)", ">=1000"], right=False)
    por_rango = d.groupby(rangos, observed=True).agg(registros=("v", "size"), con_decimal=("decimal", "sum")).reset_index().rename(columns={"v": "rango_mm"})
    por_fuente = d.groupby("fuente").agg(registros=("v", "size"), con_decimal=("decimal", "sum"), tres_digitos=("tres_digitos", "sum")).reset_index()
    t3 = d[d.tres_digitos]
    resumen = dict(
        registros=int(len(d)), con_decimal=int(d.decimal.sum()), pct_con_decimal=round(100 * float(d.decimal.mean()), 3),
        con_un_decimal=int(((d.v * 10).round(6) % 1 == 0)[d.decimal].sum()), con_dos_decimales=int((~((d.v * 10).round(6) % 1 == 0))[d.decimal].sum()),
        estaciones=int(len(est)), estaciones_con_decimal=int((~est.solo_enteros).sum()), estaciones_solo_enteros=int(est.solo_enteros.sum()),
        registros_tres_digitos=int(len(t3)), tres_digitos_con_decimal=int(t3.decimal.sum()),
        tres_digitos_en_estaciones_con_decimal=int(t3.station_id.isin(est.index[~est.solo_enteros]).sum()),
        media_dia_lluvioso_solo_enteros_mm=round(float(llu.v[llu.solo_enteros].mean()), 3),
        media_dia_lluvioso_con_decimal_mm=round(float(llu.v[~llu.solo_enteros].mean()), 3),
        cociente_medias=round(float(llu.v[llu.solo_enteros].mean() / llu.v[~llu.solo_enteros].mean()), 3))
    resumen["por_fuente"] = por_fuente.to_dict("records")
    resumen["por_rango"] = por_rango.assign(**{"rango_mm": por_rango.iloc[:, 0].astype(str)}).drop(columns=por_rango.columns[0]).to_dict("records")
    guardar("diag_decimales", resumen)

    archivos = pd.Series([a.name for a in DIR_IMAGENES.glob("ppt*.tif")])
    sat = pd.DataFrame(dict(anio=archivos.str[3:7].astype(int), pentada=archivos.str[7:9].astype(int)))
    sat = sat[sat.anio.between(*PERIODO_PROYECTO)]
    pent = tabla_pentadas().set_index("pentada")
    s_anios = sat.groupby("anio").agg(pentadas=("pentada", "size"), ultima_pentada=("pentada", "max")).reset_index()
    s_anios["doy_fin_ultima"] = pent.doy_fin.reindex(s_anios.ultima_pentada).to_numpy()
    s_anios["completo"] = s_anios.pentadas == 72
    guardar("diag_satelite_anios", s_anios)


UMBRAL_TOTAL_DECIMAS = 12000.0


def candidatas_decimas():
    # Estaciones con algún año A-B sobre UMBRAL_TOTAL_DECIMAS mm: se conservan y quedan marcadas como posibles lecturas en
    # décimas de mm; no se modifica ningún valor
    meta = pd.read_csv(ruta("metadata_depuracion.csv"), usecols=["station_id", "year", "nivel", "cobertura", "total_mm"])
    ab = meta[meta.nivel.isin(["A", "B"]) & (meta.cobertura > 0)].copy()
    ab["total_anual_mm"] = ab.total_mm / ab.cobertura
    d = pd.read_csv(ruta("precip_diaria_depurada.csv"), usecols=["station_id", "value_mm"]).dropna()
    llu = d[d.value_mm > 0].assign(mult10=lambda x: x.value_mm % 10 == 0, decimal=lambda x: x.value_mm != np.floor(x.value_mm))
    por_est = llu.groupby("station_id").agg(dia_lluvioso_medio_mm=("value_mm", "mean"), pct_lluviosos_multiplo_10=("mult10", lambda x: 100 * x.mean()),
                                            pct_lluviosos_con_decimal=("decimal", lambda x: 100 * x.mean()), dia_lluvioso_max_mm=("value_mm", "max"))
    g = ab.groupby("station_id").total_anual_mm.agg(anios_ab="size", total_anual_mediana_mm="median", total_anual_max_mm="max",
                                                    anios_sobre_umbral=lambda x: int((x > UMBRAL_TOTAL_DECIMAS).sum()))
    cand = g[g.anios_sobre_umbral > 0].join(por_est).reset_index().sort_values("total_anual_max_mm", ascending=False)
    resto = por_est[~por_est.index.isin(cand.station_id)]
    cand["resto_dia_lluvioso_medio_mm"] = resto.dia_lluvioso_medio_mm.mean()
    cand["resto_pct_lluviosos_multiplo_10"] = resto.pct_lluviosos_multiplo_10.mean()
    cand["umbral_total_anual_mm"] = UMBRAL_TOTAL_DECIMAS
    cand["decision"] = "conservada en la base maestra; posible lectura en décimas; Ítem 3 con y sin estas estaciones"
    cand.to_csv(ruta("estaciones_candidatas_decimas.csv"), index=False, float_format="%.2f")
