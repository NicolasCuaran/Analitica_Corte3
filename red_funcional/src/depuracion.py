"""Depuración de la lluvia diaria: reglas A-E, estaciones ciegas, imputación, niveles de calidad, vistas y trazabilidad."""
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import BSpline

from . import comun
from .catalogos import catalogo_estaciones, haversine_km
from .comun import (ANIO_BASE, ANIO_FIN, CSV_COORDENADAS, CSV_CVC_ABIERTO, CSV_IDEAM, CSV_LIMITES, CSV_ORIGEN, CSV_SRTM,
                    DIR_BAD, DOMINIO, N_DIAS, PARTICION, PERIODO_PROYECTO, PLIEGUES_CV, SEMILLA, XLSX_ALTITUD, guardar, ruta)
from .satelite import tabla_pentadas

N_CASOS = 1500
ANIO_EST_MAX = PARTICION["entrenamiento"][1]
VENTANA_EST = (PERIODO_PROYECTO[0], ANIO_EST_MAX)
AMBITO = "proyecto_1981_2003"
AMBITOS = {AMBITO: PERIODO_PROYECTO}
ESTACIONES_RESERVADAS = ()
FRAC_CIEGAS, CIEGA_MIN_ANIOS, CIEGA_COB_DIAS = 0.15, 8, 329

V_MAX_MM = 1000.0
REP_DIAS, REP_MIN_MM = 4, 5.0
A2_RETIRAR_PRIMERO = True

PICO_VENTANA, PICO_CUANTIL, PICO_CORROB, CTX_RAZON = 3, 0.999, 0.25, 5.0

K_VECINOS, VEC_MIN_VALIDOS, CORR_MIN_DIAS = 5, 3, 500
BINS_Q = (0.25, 0.5, 0.75)
PSEUDO_OBS = 100
CERO_ALFA, CERO_MIN_RACHA, CERO_TOL_HUECO, CERO_PASADAS = 1e-4, 7, 3, 2
CERO_ALFAS_SENS = (1e-2, 1e-3, 1e-4, 1e-6)
# Regla C: solo los días con al menos la mitad de las vecinas húmedas (categoría >= 3) aportan evidencia contra el cero
CERO_CATEGORIA_EVIDENCIA = 3

EPS_REL = 0.10
VIF_MAX = (1 + EPS_REL) ** 2
VIF_B = 4.0
VIF_C = 10.0
K_VIF = (12, 20, 30)
K_REF_VIF, GRADO_VIF = 20, 3
COB_MIN, HUECO_MAX = 0.90, 30
# Año idóneo: al menos 80 % de los 365 días con dato tras las reglas (criterio de la OMM); sus faltantes se imputan
COB_IDONEA = 0.80
IMP_K, IMP_DIST_KM = 5, 50.0
# IDW con distancia efectiva d = sqrt(d_h^2 + (alfa dz)^2); alfa se elige por enmascaramiento con el entrenamiento
IMP_ALFAS, IMP_D_MIN_KM, IMP_FRAC_OCULTA = (0, 25, 50, 100), 0.5, 0.20
COB_GRILLA = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
HUECO_GRILLA = (15, 21, 30, 45, 365)
ANIOS_MIN_GRILLA = (5, 8, 10, 12, 15, 20)
MIN_ANIOS_CV, CV_CUANTIL = 5, 0.90
COB_COMPLETAR = 0.40
PER_MIN_REG, PER_FRAC, GAP_MAX = 5, 0.5, 400
ACUM_RAZON = (0.25, 4.0)
GAP_MIN_ACUM = 8
REGLA_E = "E_acumulado_tras_hueco"
# Esquema fijo de metadata_depuracion.csv: una columna por regla aunque no se dispare
REGLAS_DIARIAS = ("A1_desorbitado", "A1_negativo", "A2_repeticion", "A3_pico_desproporcionado", "B1_total_mensual_ultimo_dia",
                  "B2_registro_unico_mes", "B3_pico_aislado", "C_cero_falso", "D_gauge_periodico", REGLA_E)
Q_DIARIO = 0.995
REGLAS_ACUM = ("B1_total_mensual_ultimo_dia", "B2_registro_unico_mes", "B3_pico_aislado", "A3_pico_desproporcionado")


def t_idx(anio, doy):
    return (anio - ANIO_BASE) * N_DIAS + doy - 1


T_REF = (t_idx(VENTANA_EST[0], 1), t_idx(VENTANA_EST[1] + 1, 1))
N_T = t_idx(ANIO_FIN + 1, 1)


FUERA_PERIODO = {}


def leer_fuentes():
    # Registros de los dos CSV de estaciones dentro de PERIODO_PROYECTO; lo de afuera solo se cuenta (FUERA_PERIODO)
    partes = []
    for nombre, ruta in CSV_ORIGEN.items():
        d = pd.read_csv(ruta, usecols=["STATION_ID", "YEAR", "MONTH", "DAY", "VALUE"])
        d["fuente"] = nombre
        partes.append(d)
    d = pd.concat(partes, ignore_index=True)
    d.columns = ["station_id", "year", "month", "day", "v", "fuente"]
    dentro = d.year.between(*PERIODO_PROYECTO)
    FUERA_PERIODO.clear()
    FUERA_PERIODO.update(registros_csv=int((~dentro).sum()), anios_csv=sorted(int(a) for a in d.year[~dentro].unique()),
                         archivo_xlsx_1930_1979="no leído (fuera del periodo del proyecto)")
    return d[dentro].reset_index(drop=True)


def particion(anio):
    return np.where(np.asarray(anio) <= PARTICION["entrenamiento"][1], "entrenamiento", "prueba")


def elegir_ciegas(df, emp):
    # 15 % de las estaciones candidatas (en la malla, con altitud y con historia de entrenamiento). La unidad de muestreo es
    # el píxel, estratificado por altitud media con un píxel por bloque y semilla fija; solo usa años de entrenamiento.
    e = df[df.year.between(*VENTANA_EST) & df.v.between(0, V_MAX_MM)]
    n_cob = e.groupby(["station_id", "year"]).v.size().ge(CIEGA_COB_DIAS).groupby("station_id").sum()
    historia = n_cob[n_cob >= CIEGA_MIN_ANIOS].index
    cand = emp[emp.en_red & emp.station_id.isin(historia)]
    grupos = cand.groupby("pixel").altitud_m.mean().sort_values(kind="stable")
    n = int(round(FRAC_CIEGAS * len(cand)))
    bloques = np.array_split(np.arange(len(grupos)), n)
    rng = np.random.default_rng(SEMILLA)
    pixeles = [grupos.index[rng.choice(b)] for b in bloques]
    # Solo arrastra a las compañeras de píxel con datos de entrenamiento
    con_ent = e.station_id.unique()
    elegidas = emp.station_id[emp.dentro_pixel & emp.pixel.isin(pixeles) & emp.station_id.isin(con_ent)]
    emp["candidata_ciega"] = emp.station_id.isin(cand.station_id)
    return sorted(int(s) for s in elegidas), len(cand)


class Traza:
    # Conteo de estaciones, pares estación-año y registros válidos tras cada paso de la depuración
    def __init__(self):
        self.filas = []

    def registrar(self, paso, descripcion, df=None, pares=None):
        for amb, (a, b) in AMBITOS.items():
            if pares is None:
                s = df[df.year.between(a, b) & df.v.notna()]
                est, par, reg = s.station_id.nunique(), len(s[["station_id", "year"]].drop_duplicates()), len(s)
            else:
                s = pares[pares.year.between(a, b)]
                est, par, reg = s.station_id.nunique(), len(s), int(s.n_validos.sum())
            self.filas.append(dict(paso=paso, descripcion=descripcion, ambito=amb, n_estaciones=est,
                                   n_pares_estacion_anio=par, n_registros_validos=reg))

    def tabla(self):
        t = pd.DataFrame(self.filas)
        deltas = {"n_estaciones": "d_estaciones", "n_pares_estacion_anio": "d_pares",
                  "n_registros_validos": "d_registros"}
        es_p = t.paso.str.startswith("P")
        for c, d in deltas.items():
            t[d] = np.where(es_p, t.groupby("ambito")[c].diff().fillna(0), 0).astype(int)
        return t


def marcar(df, mascara, regla):
    nuevo = mascara & df.v.notna().to_numpy()
    df.loc[nuevo, "regla"] = regla
    df.loc[nuevo, "v"] = np.nan
    return int(nuevo.sum())


def construir_matriz(df, n_s):
    ok = df.v.notna().to_numpy()
    A = np.full((N_T, n_s), np.nan, dtype=np.float32)
    A[df.t.to_numpy()[ok], df.j.to_numpy()[ok]] = df.v.to_numpy()[ok]
    return A


def id_particion(t):
    # 0 entrenamiento, 1 prueba, a partir del índice temporal t
    anio = ANIO_BASE + np.asarray(t) // N_DIAS
    return (anio > PARTICION["entrenamiento"][1]).astype(int)


def desplazar(M, o):
    # Desplazamiento temporal que no cruza la frontera entre entrenamiento y prueba
    S = np.zeros_like(M)
    if o > 0:
        S[o:] = M[:-o]
    else:
        S[:o] = M[-o:]
    t = np.arange(M.shape[0])
    src = t - o
    cruza = (src >= 0) & (src < M.shape[0]) & (id_particion(t) != id_particion(np.clip(src, 0, M.shape[0] - 1)))
    S[cruza] = np.nan if np.isnan(M).any() else 0
    return S


def vecinos_por_ocurrencia(A, excluidas=()):
    # K_VECINOS estaciones con mayor correlación de ocurrencia; las ciegas reciben vecinas pero no son vecinas de nadie
    lo, hi = T_REF
    blk = A[lo:hi]
    occ = pd.DataFrame(np.where(np.isnan(blk), np.nan, (blk > 0).astype(np.float32)))
    C = occ.corr(min_periods=CORR_MIN_DIAS).to_numpy()
    np.fill_diagonal(C, np.nan)
    C[:, list(excluidas)] = np.nan
    vecinos = []
    for fila in C:
        ok = np.flatnonzero(~np.isnan(fila))
        vecinos.append(ok[np.argsort(-fila[ok])[:K_VECINOS]] if ok.size >= K_VECINOS else np.array([], dtype=int))
    return vecinos


def fraccion_vecinos_humedos(A, vecinos):
    humedo = np.where(np.isnan(A), np.nan, (A > 0).astype(np.float32))
    q = np.full(A.shape, np.nan, dtype=np.float32)
    for j, ix in enumerate(vecinos):
        if ix.size == 0:
            continue
        b = humedo[:, ix]
        nv = np.sum(~np.isnan(b), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.nansum(b, axis=1) / nv
        m[nv < VEC_MIN_VALIDOS] = np.nan
        q[:, j] = m
    return q


def categoria_vecinos(q):
    c = np.select([q <= 0, q < BINS_Q[0], q < BINS_Q[1], q < BINS_Q[2]], [0, 1, 2, 3], 4)
    return np.where(np.isnan(q), -1, c)


def tabla_ocurrencia(A, categoria, excluidas=()):
    # P(lluvia | categoría de vecinas, día previo seco): global (sin estaciones ciegas) y por estación (contracción PSEUDO_OBS)
    lo, hi = T_REF
    previo = np.zeros(A.shape, dtype=bool)
    previo[1:] = A[:-1] == 0
    valido = ~np.isnan(A) & previo
    valido[:lo] = False
    valido[hi:] = False
    humedo = np.nan_to_num(A) > 0
    n_s = np.zeros((A.shape[1], 5))
    w_s = np.zeros((A.shape[1], 5))
    for b in range(5):
        m = valido & (categoria == b)
        n_s[:, b] = m.sum(axis=0)
        w_s[:, b] = (m & humedo).sum(axis=0)
    usadas = np.ones(A.shape[1], dtype=bool)
    usadas[list(excluidas)] = False
    p_g = w_s[usadas].sum(axis=0) / n_s[usadas].sum(axis=0)
    p_s = (w_s + PSEUDO_OBS * p_g) / (n_s + PSEUDO_OBS)
    return p_g, p_s, n_s[usadas].sum(axis=0).astype(int)


def rachas_ceros(A, categoria, p_s):
    # Rachas de ceros por estación con su log-probabilidad bajo la ocurrencia de las vecinas
    resumen, miembros = [], []
    for j in range(A.shape[1]):
        idx = np.flatnonzero(~np.isnan(A[:, j]))
        if idx.size < 2:
            continue
        z = A[idx, j] == 0
        zi = np.flatnonzero(z)
        if zi.size == 0:
            continue
        corte = np.r_[True, ~z[1:] | ~z[:-1] | (np.diff(idx) > CERO_TOL_HUECO) | (np.diff(id_particion(idx)) != 0)]
        _, inv = np.unique(np.cumsum(corte)[zi], return_inverse=True)
        c = categoria[idx[zi], j]
        w = np.where(c >= CERO_CATEGORIA_EVIDENCIA, np.log1p(-p_s[j])[np.clip(c, 0, 4)], 0.0)
        L = np.bincount(inv)
        lp = np.bincount(inv, weights=w)
        resumen.append(np.column_stack([np.full(L.size, j), L, lp]))
        miembros.append((j, idx[zi], inv, L, lp))
    return np.vstack(resumen), miembros


def ceros_falsos(A, vecinos, alfa, excluidas=()):
    categoria = categoria_vecinos(fraccion_vecinos_humedos(A, vecinos))
    p_g, p_s, n_categoria = tabla_ocurrencia(A, categoria, excluidas)
    resumen, miembros = rachas_ceros(A, categoria, p_s)
    marca = np.zeros(A.shape, dtype=bool)
    for j, t, inv, L, lp in miembros:
        sel = (L[inv] >= CERO_MIN_RACHA) & (lp[inv] < np.log(alfa))
        marca[t[sel], j] = True
    return marca, resumen, p_g, n_categoria


def base_bspline(K, grado=GRADO_VIF):
    t = np.arange(1, N_DIAS + 1, dtype=float)
    interiores = np.linspace(1, N_DIAS, K - grado + 1)[1:-1]
    nodos = np.r_[[1.0] * (grado + 1), interiores, [float(N_DIAS)] * (grado + 1)]
    return BSpline.design_matrix(t, nodos, grado).toarray()


def inflacion_varianza(Phi, M, bloque=500):
    # VIF = tr[(Phi_o' Phi_o)^-1 Phi' Phi] / K: varianza media del ajuste MCO con los días observados relativa a la del
    # año completo; inf si Phi_o es singular
    K = Phi.shape[1]
    G = Phi.T @ Phi
    exterior = (Phi[:, :, None] * Phi[:, None, :]).reshape(N_DIAS, K * K)
    vif = np.full(M.shape[0], np.inf)
    for a in range(0, M.shape[0], bloque):
        Go = (M[a:a + bloque].astype(float) @ exterior).reshape(-1, K, K)
        lam = np.linalg.eigvalsh(Go)
        bien = lam[:, 0] > 1e-9 * lam[:, -1]
        if bien.any():
            sol = np.linalg.solve(Go[bien], np.broadcast_to(G, (int(bien.sum()), K, K)))
            tmp = np.full(Go.shape[0], np.inf)
            tmp[bien] = np.trace(sol, axis1=1, axis2=2) / K
            vif[a:a + bloque] = tmp
    return vif


def matriz_validez(df_val, pares):
    clave = pd.MultiIndex.from_frame(pares[["station_id", "year"]])
    pos = clave.get_indexer(pd.MultiIndex.from_frame(df_val[["station_id", "year"]]))
    M = np.zeros((len(pares), N_DIAS), dtype=bool)
    M[pos, df_val.doy.to_numpy() - 1] = True
    return M


def hueco_maximo(M):
    k = np.arange(N_DIAS)
    ultimo = np.maximum.accumulate(np.where(M, k, -1), axis=1)
    return np.where(~M, k - ultimo, 0).max(axis=1)


ENCABEZADOS_MD = {"descripcion": "descripción", "n_estaciones": "estaciones", "n_pares_estacion_anio": "pares estación-año",
                  "n_registros_validos": "registros válidos", "d_estaciones": "Δ estaciones", "d_pares": "Δ pares",
                  "d_registros": "Δ registros"}


def a_markdown(df):
    cols = list(df.columns)
    lineas = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for fila in df.itertuples(index=False):
        campos = []
        for x in fila:
            if isinstance(x, (int, np.integer)):
                campos.append(f"{x:,}")
            elif isinstance(x, (float, np.floating)):
                campos.append(f"{x:.3f}" if abs(x) < 1000 else f"{x:,.0f}")
            else:
                campos.append(str(x))
        lineas.append("| " + " | ".join(campos) + " |")
    return "\n".join(lineas)


def clasificar_acumulados(df, n_s):
    # Destino de los registros marcados por B1, B2, B3, A3 y E según su ventana (días desde el registro anterior):
    #   acumulado_ventana : suma de g días, coherente con g veces la lluvia diaria típica de la estación y el mes
    #   recuperado_diario : B2 con valor plausible como lluvia de un solo día (<= cuantil Q_DIARIO diario)
    #   candidato_x0.1    : pico con día previo presente y v/10 coherente con su contexto (posible dígito de más)
    #   retirado          : el resto
    ok = df.v.notna() & df.year.between(*VENTANA_EST)
    base = df[ok]
    mu_sm = base.groupby(["station_id", "month"]).v.mean().rename("mu")
    q_sm = base.groupby(["station_id", "month"]).v.quantile(Q_DIARIO).rename("q")
    base_g = base[~base.ciega]
    mu_m = base_g.groupby("month").v.mean().rename("mu_g")
    q_m = base_g.groupby("month").v.quantile(Q_DIARIO).rename("q_g")
    e = (df.loc[(df.regla == "") & df.v.notna() & df.gap.between(GAP_MIN_ACUM, GAP_MAX), ["station_id", "month", "v", "gap"]]
         .rename_axis("idx").reset_index().merge(mu_sm, on=["station_id", "month"], how="left")
         .merge(q_sm, on=["station_id", "month"], how="left").merge(mu_m, on="month", how="left").merge(q_m, on="month", how="left"))
    e["mu"], e["q"] = e.mu.fillna(e.mu_g), e.q.fillna(e.q_g)
    razon_e = e.v / (e.gap * e.mu.clip(lower=0.5))
    mascara_e = np.zeros(len(df), dtype=bool)
    mascara_e[e.idx[(e.v > e.q) & razon_e.between(*ACUM_RAZON)].to_numpy()] = True
    marcar(df, mascara_e, REGLA_E)
    c = df.loc[df.regla.isin(REGLAS_ACUM + (REGLA_E,)), ["station_id", "month", "v_crudo", "gap", "regla", "j", "t"]].copy()
    c = (c.rename_axis("idx").reset_index().merge(mu_sm, on=["station_id", "month"], how="left")
         .merge(q_sm, on=["station_id", "month"], how="left").merge(mu_m, on="month", how="left")
         .merge(q_m, on="month", how="left").set_index("idx"))
    c["mu"], c["q"] = c.mu.fillna(c.mu_g), c.q.fillna(c.q_g)
    dias_mes = pd.to_datetime(dict(year=df.loc[c.index, "year"], month=c.month, day=1)).dt.days_in_month
    sin_previo = (c.regla == REGLAS_ACUM[0]) & ~c.gap.between(1, GAP_MAX)
    c["gap"] = np.where(sin_previo, dias_mes, c.gap)
    df.loc[c.index, "gap"] = c.gap
    g = c.gap.clip(upper=GAP_MAX)
    c["razon"] = c.v_crudo / (g * c.mu.clip(lower=0.5))
    pico = c.regla.isin(["B3_pico_aislado", "A3_pico_desproporcionado"])
    con_ventana = c.gap.between(2, GAP_MAX)
    A = construir_matriz(df, n_s)
    ctx = np.maximum.reduce([np.nan_to_num(desplazar(A, o), nan=-1.0) for o in (-2, -1, 1, 2)])[c.t.to_numpy(), c.j.to_numpy()]
    reducido = c.v_crudo.to_numpy() / 10
    rel = reducido / np.where(ctx > 0, ctx, 1.0)
    c["ctx"] = ctx
    c["coherente_x0.1"] = np.where(ctx > 0, (rel >= 0.5) & (rel <= 3.0), reducido <= c.q.to_numpy())
    c["destino"] = np.select(
        [(c.regla == REGLAS_ACUM[0]) & con_ventana,
         c.regla == REGLA_E,
         (c.regla == REGLAS_ACUM[1]) & (c.v_crudo <= c.q),
         (c.regla == REGLAS_ACUM[1]) & con_ventana,
         pico & con_ventana & c.razon.between(*ACUM_RAZON),
         pico & (c.gap == 1) & c["coherente_x0.1"]],
        ["acumulado_ventana", "acumulado_ventana", "recuperado_diario", "acumulado_ventana", "acumulado_ventana", "candidato_x0.1"], "retirado")
    return c


def tabla_acumulados(df, c):
    # Registros con ventana: los de lectura periódica (regla D) y los reclasificados como acumulado_ventana
    idx_c = c.index[c.destino == "acumulado_ventana"]
    idx = df.index[(df.regla == "D_gauge_periodico") | df.index.isin(idx_c)]
    a = df.loc[idx, ["station_id", "year", "month", "day", "doy", "gap", "v_crudo", "regla"]].copy()
    a["razon"] = np.nan
    a.loc[idx_c, "razon"] = c.loc[idx_c, "razon"]
    a = a.rename(columns={"doy": "doy_fin", "gap": "n_dias", "v_crudo": "total_mm", "regla": "origen"})
    a["doy_ini"] = a.doy_fin - a.n_dias + 1
    a["cruza_anio"] = a.doy_ini < 1
    return a


def distancias_donantes(emp, est, ciegas, alfa=0.0):
    # Distancia efectiva d = sqrt(d_h^2 + (alfa dz)^2) en km entre estaciones; inf donde no se puede donar (a sí misma,
    # desde una ciega o a más de IMP_DIST_KM horizontales). Devuelve (d, d_h en km, |dz| en m).
    p = emp.set_index("station_id").loc[est]
    lat, lon, z = p.lat.to_numpy(), p.lon.to_numpy(), p.altitud_m.to_numpy(float)
    Dh = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])           # Dim: (n_s, n_s)
    Z = np.abs(z[:, None] - z[None, :])                                                   # Dim: (n_s, n_s), m
    Z = np.where(np.isnan(Z), np.inf if alfa > 0 else 0.0, Z)
    D = np.sqrt(Dh ** 2 + (alfa * Z / 1000) ** 2)
    np.fill_diagonal(D, np.inf)
    D[:, np.isin(est, list(ciegas))] = np.inf
    D[Dh > IMP_DIST_KM] = np.inf
    return D, Dh, Z


def vecinas_mismo_dia(A, D, j, tt, idw=True, Dh=None, Z=None):
    # Valor de cada día tt de la estación j con las IMP_K estaciones más cercanas que tienen dato ese día, con pesos 1/d^2
    # (IDW) o iguales (idw=False). Devuelve (valor, donantes, distancia horizontal media en km, desnivel medio en m).
    orden = np.argsort(D[j], kind="stable")
    orden = orden[np.isfinite(D[j, orden])]
    nada = np.full(len(tt), np.nan)
    if orden.size == 0:
        return nada, np.zeros(len(tt), dtype=int), nada, nada
    B = A[np.ix_(tt, orden)].astype(float)                                               # Dim: (días, donantes)
    ok = ~np.isnan(B)
    usa = ok & (np.cumsum(ok, axis=1) <= IMP_K)
    w = np.where(usa, 1.0 / np.maximum(D[j, orden], IMP_D_MIN_KM)[None, :] ** 2 if idw else 1.0, 0.0)
    n = usa.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = (w * np.where(usa, B, 0.0)).sum(axis=1) / w.sum(axis=1)
        dh = np.where(usa, (D if Dh is None else Dh)[j, orden][None, :], 0.0).sum(axis=1) / n
        dz = nada if Z is None else np.where(usa, Z[j, orden][None, :], 0.0).sum(axis=1) / n
    return np.where(n > 0, val, np.nan), n, np.where(n > 0, dh, np.nan), np.where(n > 0, dz, np.nan)


def elegir_alfa(A, emp, est, ciegas):
    # Prueba de enmascaramiento con años de entrenamiento y sin ciegas: en N_CASOS estación-años con >= 95 % de días se oculta
    # el IMP_FRAC_OCULTA de los días observados y se predicen por IDW para cada alfa; gana el de menor mediana del |error
    # relativo del total oculto|, con la misma muestra de días para todos
    rng = np.random.default_rng(SEMILLA)
    lo, hi = T_REF
    n_a = (hi - lo) // N_DIAS
    cob = (~np.isnan(A[lo:hi])).reshape(n_a, N_DIAS, -1).mean(axis=1)                   # Dim: (n_a, n_s)
    cob[:, np.isin(est, list(ciegas))] = 0.0
    ya, ja = np.nonzero(cob >= 0.95)
    s = rng.choice(len(ya), min(N_CASOS, len(ya)), replace=False)
    ya, ja = ya[s], ja[s]
    tt, jj, gg = [], [], []
    for g, (y, j) in enumerate(zip(ya, ja)):
        t0 = lo + y * N_DIAS
        dias = t0 + np.flatnonzero(~np.isnan(A[t0:t0 + N_DIAS, j]))
        h = np.sort(rng.choice(dias, int(IMP_FRAC_OCULTA * len(dias)), replace=False))
        tt.append(h); jj.append(np.full(len(h), j)); gg.append(np.full(len(h), g))
    tt, jj, gg = np.concatenate(tt), np.concatenate(jj), np.concatenate(gg)
    y_obs = A[tt, jj].astype(float)
    preds = {}
    for alfa in IMP_ALFAS:
        D = distancias_donantes(emp, est, ciegas, alfa)[0]
        p = np.full(len(tt), np.nan)
        for j in np.unique(jj):
            m = jj == j
            p[m] = vecinas_mismo_dia(A, D, j, tt[m])[0]
        preds[alfa] = p
    comun = np.logical_and.reduce([~np.isnan(p) for p in preds.values()])
    filas = []
    for alfa, p in preds.items():
        e = p[comun] - y_obs[comun]
        tot = pd.DataFrame({"g": gg[comun], "p": p[comun], "y": y_obs[comun]}).groupby("g").sum()
        rel = 100 * (tot.p - tot.y) / tot.y.where(tot.y > 0)
        filas.append(dict(alfa_m_por_m=alfa, desnivel_400m_equivale_a_km=0.4 * alfa, casos=len(ya), dias_ocultos=int(comun.sum()),
                          mae_mm=float(np.abs(e).mean()), rmse_mm=float(np.sqrt((e ** 2).mean())),
                          correlacion=float(np.corrcoef(p[comun], y_obs[comun])[0, 1]),
                          abs_error_total_oculto_pct_mediana=float(np.nanmedian(np.abs(rel))),
                          sesgo_total_oculto_pct_mediana=float(np.nanmedian(rel))))
    tabla = pd.DataFrame(filas)
    alfa = float(tabla.alfa_m_por_m[tabla.abs_error_total_oculto_pct_mediana.idxmin()])
    tabla["elegido"] = tabla.alfa_m_por_m == alfa
    return alfa, tabla


def imputar_vecinas(df, emp, est, ciegas, acum):
    # Días faltantes de los años idóneos completados por IDW del mismo día (alfa de elegir_alfa). No se imputan las
    # estaciones ciegas ni los días dentro de la ventana de un acumulado.
    n_s = len(est)
    A = construir_matriz(df, n_s)                                                        # Dim: (N_T, n_s)
    n_anios = N_T // N_DIAS
    cob = (~np.isnan(A)).reshape(n_anios, N_DIAS, n_s).sum(axis=1) / N_DIAS              # Dim: (n_anios, n_s)
    idoneo = (cob >= COB_IDONEA) & ~np.isin(est, list(ciegas))[None, :]
    ventana = np.zeros(A.shape, dtype=bool)
    if len(acum):
        jw = np.searchsorted(est, acum.station_id.to_numpy())
        t_fin = t_idx(acum.year.to_numpy(), acum.doy_fin.to_numpy())
        t_ini = np.maximum(t_fin - acum.n_dias.to_numpy() + 1, 0)
        for a, b, c in zip(t_ini, t_fin, jw):
            ventana[a:b + 1, c] = True
    alfa, tabla = elegir_alfa(A, emp, est, ciegas)
    guardar("validacion_imputacion_altura", tabla)
    D, Dh, Z = distancias_donantes(emp, est, ciegas, alfa)
    filas = []
    for j in np.flatnonzero(idoneo.any(axis=0)):
        tt = np.flatnonzero(np.isnan(A[:, j]) & ~ventana[:, j] & np.repeat(idoneo[:, j], N_DIAS))
        if tt.size == 0:
            continue
        val, n, dh, dz = vecinas_mismo_dia(A, D, j, tt, Dh=Dh, Z=Z)
        ok = n > 0
        filas.append(pd.DataFrame({"j": j, "t": tt[ok], "v": val[ok], "n_donantes": n[ok], "dist_media_km": dh[ok], "desnivel_medio_m": dz[ok]}))
    cols = ["j", "t", "v", "n_donantes", "dist_media_km", "desnivel_medio_m"]
    imp = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame({c: [] for c in cols})
    imp["station_id"] = est[imp.j.to_numpy().astype(int)]
    imp["year"] = ANIO_BASE + imp.t.to_numpy().astype(int) // N_DIAS
    imp["doy"] = imp.t.to_numpy().astype(int) % N_DIAS + 1
    fecha = pd.Timestamp("2001-01-01") + pd.to_timedelta(imp.doy.to_numpy() - 1, unit="D")   # año sin 29-feb: dominio de 365 días
    imp["month"], imp["day"] = fecha.month.to_numpy(), fecha.day.to_numpy()
    return imp, alfa


def asignar_nivel(pares):
    # Nivel de calidad A-D de cada estación-año; solo los años idóneos tienen curva
    idoneo = pares.cobertura >= COB_IDONEA
    estricto = (idoneo & (pares.cobertura_con_imputacion >= COB_MIN) & (pares.hueco_max_dias <= HUECO_MAX)
                & np.isfinite(pares.vif) & (pares.vif <= VIF_MAX))
    finito = idoneo & np.isfinite(pares.vif)
    pares["nivel"] = np.select([estricto, finito & (pares.vif <= VIF_B), finito & (pares.vif <= VIF_C)], ["A", "B", "C"], "D")
    pares["peso"] = np.where(finito, 1.0 / pares.vif.where(finito, 1.0), 0.0)
    return pares


def vistas_por_tarea(pares, estacion_apta):
    # Máscara de estación-años de cada uso posterior, con su criterio
    ab = pares.nivel.isin(["A", "B"])
    anio = pares.year
    apta = pares.station_id.isin(estacion_apta)
    propia = ~pares.ciega
    red = pares.en_red
    ent, pru = (anio.between(*PARTICION[k]) for k in ("entrenamiento", "prueba"))
    e0, e1 = PARTICION["entrenamiento"]
    p0, p1 = PARTICION["prueba"]
    return {
        "construccion": (pares.nivel.isin(["A", "B", "C"]), "Ítem 1: estimable por MCO (niveles A-C), 1981-2003, todas las estaciones"),
        "exploratorio_comparacion": (ab, "Ítems 2-3: niveles A-B, 1981-2003, todas las estaciones (con curva satelital)"),
        "base_fpca": (ab & ent & propia,
                      f"Decisión 1: media y autofunciones del FPCA solo con curvas A-B de entrenamiento ({e0}-{e1}), sin estaciones ciegas"),
        "entrenamiento": (ab & ent & propia & red, f"Ítems 4.2-4.3: curvas A-B, {e0}-{e1}, sin estaciones ciegas, en la malla del satélite y con altitud; "
                          f"la validación cruzada cronológica (bloques {PLIEGUES_CV[0][0]}-{PLIEGUES_CV[-1][1]}) elige hiperparámetros y épocas dentro de este bloque"),
        "suavizado_penalizado": ((pares.nivel == "D") & (pares.cobertura >= COB_COMPLETAR),
                                 f"Diagnóstico: años no idóneos con cobertura ≥ {COB_COMPLETAR:.0%}; suavizamiento penalizado (λ por VCG) solo para comprobar que excluirlos no sesga la muestra"),
        "acumulados": (((pares.n_validos > 0) | (pares.n_acumulados > 0)),
                       "Totales anuales y mensuales (PBIAS): todos los niveles, con las ventanas de acumulados como restricciones"),
        "prueba": ((pares.nivel == "A") & pru & apta & propia & red,
                   f"Evaluación final: nivel A, {p0}-{p1} (El Niño, La Niña y neutros), estación con historia suficiente, sin estaciones ciegas, en la malla y con altitud"),
        "ciegas_propias": (ab & pares.ciega & red,
                           "Proxy de las estaciones reservadas: niveles A-B de las estaciones ciegas (en la malla y con altitud), 1981-2003, nunca usadas para ajustar"),
    }


def tratamiento(pares):
    return np.select(
        [pares.nivel.isin(["A", "B", "C"]), (pares.nivel == "D") & (pares.cobertura >= COB_COMPLETAR)],
        ["mco_ponderado_1/vif", "suavizado_penalizado_vcg"], "solo_acumulados")


def _depurar(lector):
    # Depuración completa: integridad, dominio, ciegas, reglas D, A1, A2, B1-B3, A3, C y E, imputación, niveles, vistas,
    # tablas de trazabilidad y umbrales_depuracion.json
    traza = Traza()
    umbrales = {"entorno": dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                                scipy=scipy.__version__, plataforma=platform.platform())}

    df = lector()
    umbrales["fuera_del_periodo"] = dict(periodo=list(PERIODO_PROYECTO), **FUERA_PERIODO,
                                         decision="descartados de los datos procesados; solo se cuentan")
    traza.registrar("P0", f"Registros crudos de {PERIODO_PROYECTO[0]}-{PERIODO_PROYECTO[1]} (2 archivos CSV; lo anterior a {PERIODO_PROYECTO[0]} se descarta)", df)

    fecha = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day), errors="coerce")
    n_fecha_inval = int(fecha.isna().sum())
    df = df[fecha.notna()].copy()
    llave = ["station_id", "year", "month", "day"]
    dup = df.duplicated(llave, keep=False)
    n_dup = int(dup.sum())
    df = df[~dup | ~df.duplicated(llave, keep="first")].copy()
    df["fecha"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
    df = df[df.v.notna()].sort_values(["station_id", "fecha"]).copy()
    fecha = df.pop("fecha")
    df["gap"] = fecha.groupby(df.station_id).diff().dt.days.fillna(9999).astype(int)
    es_29f = ((df.month == 2) & (df.day == 29)).to_numpy()
    n_29f = int(es_29f.sum())
    df[es_29f & df.v.notna().to_numpy()][llave + ["v"]].rename(columns={"v": "value_mm"}).to_csv(
        ruta("registros_29_febrero.csv"), index=False)
    df, fecha = df[~es_29f].copy(), fecha[~es_29f]
    bisiesto = fecha.dt.is_leap_year.to_numpy() & (df.month > 2).to_numpy()
    df["doy"] = fecha.dt.dayofyear.to_numpy() - bisiesto.astype(int)
    df = df[df.v.notna()].copy()
    reservada = df.station_id.isin(ESTACIONES_RESERVADAS)
    n_reservadas = int(reservada.sum())
    df = df[~reservada].copy()
    traza.registrar("P1", "Integridad: fechas válidas, sin duplicados, 29-feb fuera del dominio de 365 días (guardados aparte), estaciones reservadas excluidas", df)
    # Dominio: solo el Valle del Cauca, antes de toda estadística de las reglas
    emp, fuera = catalogo_estaciones(np.sort(df.station_id.unique()))
    en_dominio = df.station_id.isin(emp.station_id).to_numpy()
    fuera = fuera.merge(df[~en_dominio].groupby("station_id").agg(registros=("v", "size"), anios=("year", "nunique")).reset_index(),
                        on="station_id", how="left")
    fuera.to_csv(ruta("estaciones_fuera_del_valle.csv"), index=False, float_format="%.6f")
    df = df[en_dominio].copy()
    n_fuera_lim, n_sin_coord = int(fuera.lat.notna().sum()), int(fuera.lat.isna().sum())
    traza.registrar("P1a", f"Dominio: solo estaciones con coordenadas dentro del límite GADM del {DOMINIO} "
                           f"({n_fuera_lim} fuera del límite y {n_sin_coord} sin coordenadas, excluidas de todo)", df)
    reemp = emp[emp.reemparejada_borde]
    umbrales["dominio"] = dict(fuente=CSV_LIMITES.name, criterio=f"coordenadas de la estación dentro del límite GADM del {DOMINIO}; "
                               "satélite: píxeles con centro dentro del límite", estaciones_dominio=int(len(emp)),
                               fuera_del_limite=n_fuera_lim, sin_coordenadas=n_sin_coord, registros_excluidos=int((~en_dominio).sum()),
                               por_departamento={str(k): int(v) for k, v in fuera.departamento.value_counts().items()},
                               reemparejadas_borde=[int(s) for s in reemp.station_id],
                               distancia_max_reemparejada_km=float(reemp.distancia_km.max()) if len(reemp) else 0.0)
    ciegas, n_elegibles = elegir_ciegas(df, emp)
    emp["ciega"] = emp.station_id.isin(ciegas)
    emp.to_csv(ruta("emparejamiento_estacion_pixel.csv"), index=False, float_format="%.6f")
    umbrales["emparejamiento"] = dict(
        regla="píxel de centro más cercano (Haversine) entre los píxeles del Valle; las estaciones del borde cuyo píxel tiene "
              "el centro fuera del límite toman el píxel del Valle más cercano",
        fuentes=[CSV_COORDENADAS.name, XLSX_ALTITUD.name, CSV_CVC_ABIERTO.name, CSV_IDEAM.name, CSV_SRTM.name], estaciones=int(len(emp)),
        coordenadas_por_fuente={str(k): int(v) for k, v in emp.fuente_coordenadas.value_counts().items()},
        altitud_por_fuente={str(k or "sin altitud"): int(v) for k, v in emp.fuente_altitud.value_counts().items()},
        con_coordenadas=int(emp.con_coordenadas.sum()), dentro_malla=int(emp.dentro_pixel.sum()),
        con_altitud=int(emp.altitud_m.notna().sum()), en_red=int(emp.en_red.sum()),
        pixeles_ocupados=int(emp.pixel[emp.dentro_pixel].nunique()),
        estaciones_que_comparten_pixel=int((emp.n_estaciones_en_pixel > 1).sum()))
    df["t"] = t_idx(df.year.to_numpy(), df.doy.to_numpy())
    est = np.sort(df.station_id.unique())
    n_s = len(est)
    df["j"] = np.searchsorted(est, df.station_id.to_numpy())
    ciega_j = np.flatnonzero(np.isin(est, ciegas))
    df["ciega"] = df.station_id.isin(ciegas)
    df["v_crudo"] = df.v
    df["regla"] = ""
    df["particion"] = particion(df.year)
    df["ambito"] = AMBITO
    df = df.sort_values(["j", "t"]).reset_index(drop=True)
    ref = (df.year.between(*VENTANA_EST) & ~df.ciega).to_numpy()
    anio_max_ajuste = int(df.year[ref].max())
    assert anio_max_ajuste <= ANIO_EST_MAX
    umbrales["estaciones_ciegas"] = dict(fraccion=FRAC_CIEGAS, semilla=SEMILLA, elegibles=n_elegibles, n=len(ciegas), ids=ciegas,
                                         criterio=f"candidatas: en la malla, con altitud y ≥ {CIEGA_MIN_ANIOS} años de entrenamiento con ≥ {CIEGA_COB_DIAS} días crudos; "
                                                  "un píxel por bloque de altitud creciente; las estaciones que comparten píxel van juntas")
    umbrales["integridad"] = dict(fechas_invalidas=n_fecha_inval, registros_duplicados=n_dup,
                                  registros_29_febrero_guardados_aparte=n_29f)
    umbrales["integridad"]["registros_estaciones_reservadas_excluidas"] = n_reservadas

    sy = df.groupby(["station_id", "year"]).gap.agg(n="size", f2=lambda g: g.between(2, GAP_MAX).mean())
    periodico = sy[(sy.n >= PER_MIN_REG) & (sy.f2 >= PER_FRAC)].index
    en_per = pd.MultiIndex.from_frame(df[["station_id", "year"]]).isin(periodico) & df.gap.between(2, GAP_MAX).to_numpy()
    n_per = marcar(df, en_per, "D_gauge_periodico")
    umbrales["regla_D"] = dict(min_registros=PER_MIN_REG, fraccion_tras_hueco=PER_FRAC, gap_max_dias=GAP_MAX,
                               estaciones_anio=int(len(periodico)), estaciones=int(periodico.get_level_values(0).nunique()),
                               registros=n_per)
    traza.registrar("P1b", f"Regla D: estación-año con ≥ {PER_FRAC:.0%} de registros tras un hueco (lectura periódica); pasan a acumulados con ventana", df)

    n_a1 = marcar(df, (df.v < 0).to_numpy(), "A1_negativo")
    n_a2 = marcar(df, (df.v > V_MAX_MM).to_numpy(), "A1_desorbitado")
    n_no_decimal = int(((df.v * 10 - np.round(df.v * 10)).abs() > 1e-6).sum())
    traza.registrar("P2a", f"Regla A1: lluvia < 0 o > {V_MAX_MM:.0f} mm/día", df)

    v = df.v.to_numpy()
    mismo = ((df.j.to_numpy()[1:] == df.j.to_numpy()[:-1]) & (np.diff(df.t.to_numpy()) == 1)
             & (np.diff(id_particion(df.t.to_numpy())) == 0) & (v[1:] == v[:-1]) & (v[1:] >= REP_MIN_MM))
    inicio = np.r_[True, ~mismo]
    long_rep = np.bincount(np.cumsum(inicio))[np.cumsum(inicio)]
    en_rep = (long_rep >= REP_DIAS) & (v >= REP_MIN_MM)
    freq = df.v[ref & (v >= REP_MIN_MM)].value_counts() / ref.sum()
    esperadas = float(ref.sum() * (freq ** REP_DIAS).sum())
    observadas = int((en_rep & inicio & ref).sum())
    n_rep_conservados = int((en_rep & inicio).sum())
    marcar(df, en_rep if A2_RETIRAR_PRIMERO else en_rep & ~inicio, "A2_repeticion")
    umbrales["regla_A"] = dict(v_max_mm=V_MAX_MM, negativos=n_a1, desorbitados=n_a2, rep_dias=REP_DIAS,
                               rep_min_mm=REP_MIN_MM, rachas_repetidas_observadas=observadas,
                               rachas_repetidas_esperadas_independencia=round(esperadas, 2),
                               rachas_con_primer_dia=n_rep_conservados, primer_dia_retirado=bool(A2_RETIRAR_PRIMERO),
                               valores_no_multiplo_0p1_documentados_no_removidos=n_no_decimal)
    traza.registrar("P2b", f"Regla A2: copias en ≥{REP_DIAS} días consecutivos con el mismo valor ≥{REP_MIN_MM:.0f} mm (se retira la racha completa, incluido el primer día)", df)

    A = construir_matriz(df, n_s)
    vecinos = vecinos_por_ocurrencia(A, ciega_j)
    umbrales["vecinos"] = dict(k=K_VECINOS, min_dias_correlacion=CORR_MIN_DIAS,
                               estaciones_sin_vecinos=int(sum(len(x) == 0 for x in vecinos)),
                               anio_max_usado=anio_max_ajuste)

    n_mes = df.groupby(["j", "year", "month"]).v.transform("count").to_numpy()
    dias_mes = pd.to_datetime(dict(year=df.year, month=df.month, day=1)).dt.days_in_month.to_numpy()
    ultimo = df.day.to_numpy() == dias_mes
    n_b1 = marcar(df, (n_mes == 1) & ultimo, "B1_total_mensual_ultimo_dia")
    n_b2 = marcar(df, (n_mes == 1) & ~ultimo, "B2_registro_unico_mes")
    traza.registrar("P3a", "Regla B1/B2: mes con un único registro (total mensual en un día; se recupera como total mensual)", df)

    A = construir_matriz(df, n_s)
    vecindad = sum(desplazar(np.nan_to_num(A), o) for o in range(-PICO_VENTANA, PICO_VENTANA + 1) if o != 0)
    cerca = np.maximum.reduce([np.nan_to_num(A if o == 0 else desplazar(A, o), nan=-1.0) for o in (-1, 0, 1)])
    lo, hi = T_REF
    humedo_ref = (A[lo:hi] > 0) & (vecindad[lo:hi] > 0)
    humedo_ref[:, ciega_j] = False
    u_pico = float(np.quantile(A[lo:hi][humedo_ref], PICO_CUANTIL))
    corrob = np.full(A.shape, -1.0, dtype=np.float32)
    for j, ix in enumerate(vecinos):
        if ix.size:
            corrob[:, j] = cerca[:, ix].max(axis=1)
    f_pico = (A >= u_pico) & (vecindad == 0) & (corrob < PICO_CORROB * np.nan_to_num(A))
    n_b3 = marcar(df, f_pico[df.t.to_numpy(), df.j.to_numpy()], "B3_pico_aislado")
    umbrales["regla_B"] = dict(meses_registro_unico_ultimo_dia=n_b1, meses_registro_unico_otro_dia=n_b2,
                               pico_ventana_dias=PICO_VENTANA, pico_cuantil_ref=PICO_CUANTIL,
                               pico_umbral_mm=round(u_pico, 1), pico_corroboracion_fraccion=PICO_CORROB,
                               picos_aislados=n_b3, anio_max_usado=anio_max_ajuste)
    traza.registrar("P3b", f"Regla B3: pico ≥ {u_pico:.0f} mm con ±{PICO_VENTANA} d secos y sin corroboración vecina", df)

    propio = np.maximum.reduce([np.nan_to_num(desplazar(A, o), nan=-1.0)
                                for o in range(-PICO_VENTANA, PICO_VENTANA + 1) if o != 0])
    contexto = np.maximum(np.maximum(propio, corrob), 0)
    f_desprop = (A >= u_pico) & (A > CTX_RAZON * contexto)
    n_a3 = marcar(df, f_desprop[df.t.to_numpy(), df.j.to_numpy()], "A3_pico_desproporcionado")
    umbrales["regla_A3"] = dict(umbral_mm=round(u_pico, 1), razon_contexto=CTX_RAZON, registros_marcados=n_a3,
                                anio_max_usado=anio_max_ajuste)
    traza.registrar("P3c", f"Regla A3: valor ≥ {u_pico:.0f} mm mayor que {CTX_RAZON:g}× el máximo de su contexto temporal y vecino", df)

    A = construir_matriz(df, n_s)
    marca, resumen, p_g, n_categoria = ceros_falsos(A, vecinos, CERO_ALFA, ciega_j)
    probadas = resumen[:, 1] >= CERO_MIN_RACHA
    sens = []
    for al in CERO_ALFAS_SENS:
        sel = probadas & (resumen[:, 2] < np.log(al))
        sens.append(dict(alfa=al, rachas=int(sel.sum()), dias_cero_marcados=int(resumen[sel, 1].sum()),
                         estaciones=int(np.unique(resumen[sel, 0]).size), rachas_probadas=int(probadas.sum()),
                         falsos_positivos_esperados=round(float(probadas.sum() * al), 1)))
    guardar("sensibilidad_ceros_falsos", pd.DataFrame(sens))
    n_c = 0
    for pasada in range(CERO_PASADAS):
        if pasada:
            A = construir_matriz(df, n_s)
            marca = ceros_falsos(A, vecinos, CERO_ALFA, ciega_j)[0]
        n_c += marcar(df, marca[df.t.to_numpy(), df.j.to_numpy()], "C_cero_falso")
    umbrales["regla_C"] = dict(alfa=CERO_ALFA, min_racha=CERO_MIN_RACHA, tolerancia_hueco_dias=CERO_TOL_HUECO,
                               pasadas=CERO_PASADAS, cortes_categoria_vecinos=list(BINS_Q), pseudo_obs=PSEUDO_OBS,
                               categoria_minima_evidencia=CERO_CATEGORIA_EVIDENCIA,
                               p_lluvia_global_por_categoria=[round(float(x), 4) for x in p_g],
                               n_dias_por_categoria=[int(x) for x in n_categoria], rachas_probadas=int(probadas.sum()),
                               falsos_positivos_esperados_bajo_nulo=round(float(probadas.sum() * CERO_ALFA), 1),
                               dias_cero_marcados=n_c, anio_max_usado=anio_max_ajuste)
    traza.registrar("P4", f"Regla C: racha de ceros improbable (α={CERO_ALFA:g}) dado el estado de 5 estaciones vecinas; solo cuentan los días con ≥ 50 % de vecinas húmedas", df)

    clas = clasificar_acumulados(df, n_s)
    rest = clas.index[clas.destino == "recuperado_diario"]
    df.loc[rest, "v"] = df.loc[rest, "v_crudo"]
    df["destino"] = np.where(df.regla != "", "retirado", "")
    df.loc[clas.index, "destino"] = clas.destino
    df.loc[df.regla == "D_gauge_periodico", "destino"] = "acumulado_ventana"
    df.loc[rest, "regla"] = ""
    df.loc[rest, "destino"] = ""
    acum = tabla_acumulados(df, clas)
    acum.to_csv(ruta("acumulados_ventana.csv"), index=False, float_format="%.3f")
    cand = clas[clas.regla.isin(["B3_pico_aislado", "A3_pico_desproporcionado"]) & (clas.gap == 1)]
    cand = df.loc[cand.index, ["station_id", "year", "month", "day", "v_crudo"]].assign(
        v_candidato=cand.v_crudo / 10, contexto_max_mm=cand.ctx, coherente=cand["coherente_x0.1"])
    cand.to_csv(ruta("correcciones_candidatas_x0.1.csv"), index=False)
    resumen_dest = clas.groupby(["regla", "destino"]).size().rename("n").reset_index()
    umbrales["acumulados"] = dict(
        destinos={f"{r.regla}|{r.destino}": int(r.n) for r in resumen_dest.itertuples()},
        ventanas=int(len(acum)), por_origen={k: int(v) for k, v in acum.origen.value_counts().items()},
        razon_mediana_por_origen={k: round(float(v), 3) for k, v in acum.groupby("origen").razon.median().dropna().items()},
        cruzan_anio=int(acum.cruza_anio.sum()), ventana_mediana_dias=float(acum.n_dias.median()),
        candidatos_x0_1=int(len(cand)), candidatos_coherentes=int(cand.coherente.sum()),
        razon_ventana_rango=list(ACUM_RAZON), cuantil_diario_plausible=Q_DIARIO)
    traza.registrar("P5", "Reclasificación de B1, B2, B3 y A3: acumulados con ventana, B2 diario recuperado y candidatos x0.1", df)
    imp, alfa_imp = imputar_vecinas(df, emp, est, ciegas, acum)
    traza.registrar("P6", f"Imputación de los años idóneos (≥ {COB_IDONEA:.0%} de días con dato): IDW (1/d²) de las {IMP_K} estaciones "
                          f"más cercanas con dato el mismo día, con distancia horizontal (≤ {IMP_DIST_KM:g} km) y de altura "
                          f"(1 m de desnivel = {alfa_imp:g} m horizontales); las ciegas ni donan ni se imputan",
                    pd.concat([df[["station_id", "year", "v"]], imp[["station_id", "year", "v"]]], ignore_index=True))

    todos = df[["station_id", "year", "j"]].drop_duplicates().sort_values(["j", "year"]).reset_index(drop=True)
    crudo = df.groupby(["station_id", "year"]).size().rename("n_crudo")
    val = df[df.v.notna()]
    agg = val.groupby(["station_id", "year"]).v.agg(n_validos="size", total_mm="sum")
    pares = todos.merge(crudo, on=["station_id", "year"], how="left").merge(agg, on=["station_id", "year"], how="left")
    pares[["n_validos", "total_mm"]] = pares[["n_validos", "total_mm"]].fillna(0)
    pares["n_validos"] = pares.n_validos.astype(int)
    assert set(df.regla[df.regla != ""]) <= set(REGLAS_DIARIAS), set(df.regla[df.regla != ""]) - set(REGLAS_DIARIAS)
    llave_par = pd.MultiIndex.from_frame(pares[["station_id", "year"]])
    for regla in REGLAS_DIARIAS:
        c = df[df.regla == regla].groupby(["station_id", "year"]).size()
        pares[regla] = c.reindex(llave_par, fill_value=0).to_numpy().astype(int)
    na = acum.groupby(["station_id", "year"]).total_mm.agg(n_acumulados="size", mm_acumulados="sum").reset_index()
    pares = pares.merge(na, on=["station_id", "year"], how="left")
    pares[["n_acumulados", "mm_acumulados"]] = pares[["n_acumulados", "mm_acumulados"]].fillna(0)
    pares["n_acumulados"] = pares.n_acumulados.astype(int)
    pares["dias_faltantes"] = N_DIAS - pares.n_validos
    pares["cobertura"] = pares.n_validos / N_DIAS
    n_imp = imp.groupby(["station_id", "year"]).size().rename("dias_imputados").reset_index()
    pares = pares.merge(n_imp, on=["station_id", "year"], how="left")
    pares["dias_imputados"] = pares.dias_imputados.fillna(0).astype(int)
    pares["cobertura_con_imputacion"] = (pares.n_validos + pares.dias_imputados) / N_DIAS
    pares["idoneo"] = pares.cobertura >= COB_IDONEA
    pares["ambito"] = AMBITO
    pares["particion"] = particion(pares.year)
    pares["ciega"] = pares.station_id.isin(ciegas)
    pares = pares.merge(emp[["station_id", "pixel", "altitud_m", "en_red"]], on="station_id", how="left")
    pares["en_red"] = pares.en_red.fillna(False).astype(bool)
    M = matriz_validez(pd.concat([val[["station_id", "year", "doy"]], imp[["station_id", "year", "doy"]]], ignore_index=True), pares)
    pares["hueco_max_dias"] = hueco_maximo(M)
    for K in K_VIF:
        pares[f"vif_k{K}"] = inflacion_varianza(base_bspline(K), M)
    pares["vif"] = pares[f"vif_k{K_REF_VIF}"]

    bandas = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001]
    etiq = ["<20%", "20-40%", "40-60%", "60-80%", "80-90%", "90-100%"]
    cob = lambda s: pd.cut(s, bandas, labels=etiq, right=False).value_counts().reindex(etiq).to_numpy()
    resumen_cob = pd.DataFrame({"banda_cobertura": etiq, "pares_tras_P1": cob(crudo / N_DIAS),
                                "pares_tras_P4": cob(pares.cobertura)})
    guardar("resumen_cobertura", resumen_cob)

    pr = pares[pares.year.between(*VENTANA_EST) & ~pares.ciega]
    filas = []
    for c in COB_GRILLA:
        for g in HUECO_GRILLA:
            ok = pr[(pr.cobertura >= c) & (pr.hueco_max_dias <= g)]
            fin = ok.vif[np.isfinite(ok.vif)]
            filas.append(dict(cobertura_min=c, hueco_max_dias=g, pares=len(ok), estaciones=ok.station_id.nunique(),
                              pct_singulares_k20=100 * (1 - len(fin) / max(len(ok), 1)),
                              vif_k20_mediana=fin.median(), vif_k20_p95=fin.quantile(0.95), vif_k20_max=fin.max()))
    guardar("sensibilidad_cobertura", pd.DataFrame(filas))

    pares = asignar_nivel(pares)
    niveles = pares.groupby(["ambito", "nivel"]).agg(pares=("year", "size"), estaciones=("station_id", "nunique"),
                                                    registros=("n_validos", "sum")).reset_index()
    guardar("niveles_calidad", niveles)
    for nv, desc in (("A", f"Nivel A: año idóneo; tras imputar, cobertura ≥ {COB_MIN:.0%}, hueco ≤ {HUECO_MAX} d y VIF ≤ {VIF_MAX:.2f}"),
                     ("B", f"Nivel B: año idóneo con días sin imputar, VIF ≤ {VIF_B:g} (desviación del ajuste ≤ {np.sqrt(VIF_B):.0f}× la del año completo)"),
                     ("C", f"Nivel C: año idóneo, {VIF_B:g} < VIF ≤ {VIF_C:g}"),
                     ("D", f"Nivel D: año no idóneo (< {COB_IDONEA:.0%} de días con dato), VIF > {VIF_C:g} o ajuste MCO singular: sin curva")):
        traza.registrar("Q" + nv, desc, pares=pares[pares.nivel == nv])

    base_todas = pares[(pares.nivel == "A") & pares.year.between(*VENTANA_EST)].copy()
    base = base_todas[~base_todas.ciega].copy()
    base["total_anual"] = base.total_mm / base.cobertura
    g = base.groupby("station_id").total_anual
    cv = (g.std(ddof=1) / g.mean())[g.size() >= MIN_ANIOS_CV]
    cv_ref = float(cv.quantile(CV_CUANTIL))
    anios_min = int(np.ceil((cv_ref / EPS_REL) ** 2))
    n_anios = base_todas.groupby("station_id").size()
    sens_n = pd.DataFrame([dict(anios_min=n, estaciones=int((n_anios >= n).sum()),
                                pares=int(n_anios[n_anios >= n].sum())) for n in ANIOS_MIN_GRILLA])
    guardar("sensibilidad_anios_min", sens_n)
    estacion_apta = set(n_anios[n_anios >= anios_min].index)
    pares["estacion_apta_evaluacion"] = pares.station_id.isin(estacion_apta)

    vistas = vistas_por_tarea(pares, estacion_apta)
    filas_v = []
    for nombre, (mascara, desc) in vistas.items():
        pares["uso_" + nombre] = mascara.to_numpy()
        for amb, (a, b) in AMBITOS.items():
            s = pares[mascara & pares.year.between(a, b)]
            filas_v.append(dict(vista=nombre, ambito=amb, estaciones=s.station_id.nunique(), pares=len(s),
                                registros=int(s.n_validos.sum()), criterio=desc))
    tabla_vistas = pd.DataFrame(filas_v)
    guardar("vistas_por_tarea", tabla_vistas)

    umbrales["cobertura"] = dict(
        cobertura_min=COB_MIN, hueco_max_dias=HUECO_MAX, eps_rel=EPS_REL, vif_max=round(VIF_MAX, 4), vif_b=VIF_B,
        k_ref_vif=K_REF_VIF, grado_vif=GRADO_VIF,
        vif_nivel_a_por_k={f"k{K}": dict(mediana=round(float(base[f"vif_k{K}"].median()), 4),
                                         p95=round(float(base[f"vif_k{K}"].quantile(0.95)), 4)) for K in K_VIF},
        cv_interanual_p50=round(float(cv.quantile(0.5)), 4), cv_interanual_p75=round(float(cv.quantile(0.75)), 4),
        cv_interanual_p90=round(cv_ref, 4), estaciones_con_cv=int(cv.size), anios_min_estacion=anios_min,
        anio_max_usado=int(base.year.max()))

    def motivo(r):
        if r.nivel == "D":
            if not r.n_validos:
                return "sin_dias_validos_tras_reglas"
            if not r.idoneo:
                return "anio_no_idoneo_menos_de_80pct_con_dato"
            return "ajuste_mco_singular" if not np.isfinite(r.vif) else "vif_sobre_umbral"
        return {"A": "nivel_A", "B": "nivel_B_peso_1/VIF", "C": "nivel_C_peso_1/VIF"}[r.nivel]
    pares["motivo_nivel"] = pares.apply(motivo, axis=1)

    cols = ["station_id", "year", "month", "day", "doy", "nivel", "value_mm"]
    maestro = (val.rename(columns={"v": "value_mm"})
               .merge(pares[["station_id", "year", "nivel", "peso"]], on=["station_id", "year"]))
    maestro = maestro[cols].sort_values(["station_id", "year", "doy"])
    maestro.to_csv(ruta("precip_diaria_depurada.csv"), index=False, float_format="%.1f")
    imputada = (imp.rename(columns={"v": "value_mm"}).merge(pares[["station_id", "year", "nivel"]], on=["station_id", "year"]))
    imputada[cols].sort_values(["station_id", "year", "doy"]).to_csv(ruta("precip_diaria_imputada.csv"), index=False, float_format="%.1f")
    # Inventario: porcentaje de días faltantes (tras las reglas, antes de imputar) por estación y año
    inv = pares.assign(pct=100 * (1 - pares.cobertura)).pivot(index="station_id", columns="year", values="pct")
    inv = inv.reindex(columns=range(ANIO_BASE, ANIO_FIN + 1)).round(1)
    inv.columns = [str(c) for c in inv.columns]
    inv["anios_idoneos"] = pares[pares.idoneo].groupby("station_id").size().reindex(inv.index, fill_value=0).astype(int)
    guardar("inventario_faltantes_estacion_anio", inv.reset_index())
    umbrales["imputacion"] = dict(
        criterio_anio_idoneo=f">= {COB_IDONEA:.0%} de los {N_DIAS} días con dato tras las reglas (OMM)",
        metodo=f"IDW del mismo día (pesos 1/d^2) de las {IMP_K} estaciones más cercanas con dato ese día; d = sqrt(d_h^2 + (alfa dz)^2), "
               f"d_h de Haversine <= {IMP_DIST_KM:g} km",
        alfa_m_por_m=alfa_imp, alfas_probados=list(IMP_ALFAS),
        criterio_alfa="menor mediana del |error del total oculto| en la prueba de enmascaramiento del entrenamiento (validacion_imputacion_altura.csv)",
        desnivel_medio_donantes_m_mediana=round(float(imp.desnivel_medio_m.median()), 1) if len(imp) else 0.0,
        anios_idoneos=int(pares.idoneo.sum()), anios_no_idoneos=int((~pares.idoneo).sum()),
        anios_idoneos_con_75pct=int((pares.cobertura >= 0.75).sum()),
        anios_imputados=int((pares.dias_imputados > 0).sum()), dias_imputados=int(len(imp)),
        anios_idoneos_incompletos_tras_imputar=int((pares.idoneo & ~pares.ciega & (pares.cobertura_con_imputacion < 1)).sum()),
        donantes_media=round(float(imp.n_donantes.mean()), 3) if len(imp) else 0.0,
        distancia_media_donantes_km_mediana=round(float(imp.dist_media_km.median()), 3) if len(imp) else 0.0,
        distancia_media_donantes_km_p90=round(float(imp.dist_media_km.quantile(0.9)), 3) if len(imp) else 0.0,
        ciegas="no donan ni se imputan", ventanas_de_acumulado="no se imputan")
    tabla_pentadas().to_csv(ruta("pentadas_dominio_365.csv"), index=False)

    reglas = list(REGLAS_DIARIAS)
    uso_cols = ["uso_" + k for k in vistas]
    pares["tratamiento"] = tratamiento(pares)
    meta_cols = (["station_id", "year", "tratamiento", "ambito", "particion", "ciega", "n_crudo", "n_validos",
                  "dias_faltantes", "cobertura", "hueco_max_dias", "total_mm", "n_acumulados", "mm_acumulados"] + [f"vif_k{K}" for K in K_VIF] + reglas +
                 ["nivel", "peso", "motivo_nivel", "estacion_apta_evaluacion", "en_red", "pixel", "altitud_m"] + uso_cols +
                 ["idoneo", "dias_imputados", "cobertura_con_imputacion"])
    pares[meta_cols].fillna({c: 0 for c in reglas}).to_csv(
        ruta("metadata_depuracion.csv"), index=False, float_format="%.4f")
    pd.DataFrame({"station_id": ciegas}).to_csv(ruta("estaciones_ciegas_propias.csv"), index=False)
    umbrales["niveles"] = dict(cobertura_idonea=COB_IDONEA, vif_max_A=round(VIF_MAX, 4), vif_B=VIF_B, vif_C=VIF_C,
                               justificacion="Error L2 relativo mediano del MCO por banda de VIF en validacion/validacion_faltantes_vif.csv")

    ev = df[df.regla != ""][["station_id", "year", "month", "day", "doy", "v_crudo", "regla", "destino", "particion"]]
    ev.rename(columns={"v_crudo": "value_mm_crudo"}).to_csv(ruta("eventos_depuracion.csv"), index=False)

    resumen_est = pares.groupby("station_id").agg(
        anios_totales=("year", "size"), primer_anio=("year", "min"), ultimo_anio=("year", "max"),
        anios_nivel_A=("nivel", lambda s: int((s == "A").sum())), anios_nivel_B=("nivel", lambda s: int((s == "B").sum())),
        anios_nivel_C=("nivel", lambda s: int((s == "C").sum())), anios_nivel_D=("nivel", lambda s: int((s == "D").sum())),
        apta_evaluacion=("estacion_apta_evaluacion", "first"), ciega_propia=("ciega", "first"),
        en_red=("en_red", "first"), pixel=("pixel", "first"), altitud_m=("altitud_m", "first")).reset_index()
    resumen_est.to_csv(ruta("estaciones_resumen.csv"), index=False)

    t = traza.tabla()
    guardar("tabla_trazabilidad", t)
    umbrales["decisiones"] = dict(
        D1=f"Media y autofunciones del FPCA solo con curvas de niveles A-B ({VENTANA_EST[0]}-{VENTANA_EST[1]}, sin estaciones ciegas); los años no idóneos (nivel D) no entran a ningún modelo: los de cobertura >= 40 % solo se suavizan con penalización (lambda por VCG) para comprobar que excluirlos no sesga la muestra",
        D6=f"El cero es un dato y año idóneo con >= {COB_IDONEA:.0%} de días con dato (OMM); días faltantes imputados por IDW del mismo día con distancia horizontal y de altura ({IMP_K} vecinas, <= {IMP_DIST_KM:g} km, alfa = {alfa_imp:g}); inventario en inventario_faltantes_estacion_anio.csv",
        D2=f"Todo registro fuera de 1981-2003 se descarta de los datos procesados (solo se cuenta en P0); partición {PARTICION['entrenamiento'][0]}-{PARTICION['entrenamiento'][1]} (entrenamiento, con validación cruzada cronológica) / {PARTICION['prueba'][0]}-{PARTICION['prueba'][1]} (prueba con El Niño, La Niña y neutros) y 15 % de estaciones ciegas propias (candidatas en la malla y con altitud, estratificadas por altitud, por píxel), excluidas de toda estadística de las reglas",
        D4="Coordenadas (CVC_Wheather_Stations_Long_Lat.csv) y altitud (CatalogoEstacionesElevacion.xlsx) de la consigna, completadas con los catálogos abiertos de la CVC y del IDEAM y, para la altitud que sigue faltando, con el SRTM (CGIAR), que también reemplaza la del IDEAM cuando discrepa de ella más que el error del SRTM en las estaciones de control (altitud_ideam_contraste_srtm.csv); píxel de centro más cercano; las vistas del Ítem 4 (entrenamiento, prueba, ciegas) exigen estación en la malla y con altitud",
        D3="Dominio t en [1, 365]; satélite: tasa diaria = valor / días de la péntada, ubicada en su baricentro (ver pentadas_dominio_365.csv)",
        D5=f"Dominio espacial: solo el {DOMINIO} (límite GADM). Estaciones fuera del límite o sin coordenadas y píxeles con centro fuera del límite se excluyen de todo (paso P1a, estaciones_fuera_del_valle.csv); las estaciones del borde toman el píxel del Valle más cercano")
    por_regla = ev.groupby(["regla", "particion"]).size().rename("n_registros").reset_index()
    md = ["# Tabla de trazabilidad de la depuración (Ítem 4.1)", "",
          "Par estación-año = (estación, año calendario) con al menos un día presente tras la depuración. "
          f"Ámbito único: periodo del proyecto {PERIODO_PROYECTO[0]}–{PERIODO_PROYECTO[1]}; lo anterior se descarta antes de P0. "
          "Los pasos P0–P4 retiran valores espurios. Los pasos Q asignan un nivel de calidad y no eliminan registros.", ""]
    for amb in AMBITOS:
        sub = t[t.ambito == amb].drop(columns=["ambito"]).rename(columns=ENCABEZADOS_MD)
        md += [f"## Ámbito {amb}", "", a_markdown(sub), ""]
    md += ["## Registros fuera del periodo (descartados)", "", a_markdown(pd.DataFrame([dict(
               registros_csv_fuera_de_1981_2003=FUERA_PERIODO["registros_csv"], anios=str(FUERA_PERIODO["anios_csv"]),
               xlsx_1930_1979="no leído")])), "",
           "## Vistas por tarea", "", a_markdown(tabla_vistas.drop(columns=["ambito"])), "",
           "## Distribución de cobertura anual (pares estación-año)", "", a_markdown(resumen_cob), "",
           "## Registros retirados por regla", "", a_markdown(por_regla), ""]
    guardar("tabla_trazabilidad_md", "\n".join(md))

    umbrales["auditoria_fuga"] = dict(
        ventana_estimacion=list(VENTANA_EST), anio_max_en_archivos=int(df.year.max()),
        anio_max_usado_para_estimar=anio_max_ajuste, registros_ventana_estimacion=int(ref.sum()),
        anios_de_prueba_presentes=sorted(int(a) for a in df.year.unique() if a > ANIO_EST_MAX))
    p0, p1a = t[t.paso == "P0"].iloc[0], t[t.paso == "P1a"].iloc[0]
    umbrales["totales"] = dict(
        registros_crudos=int(p0.n_registros_validos), registros_en_el_valle=int(p1a.n_registros_validos),
        registros_tras_reglas=int(len(val)),
        pct_retirado_por_reglas=round(100 * (1 - len(val) / p1a.n_registros_validos), 3),   # respecto al dominio (P1a)
        estaciones=int(val.station_id.nunique()), pares=int(len(pares)),
        pares_por_nivel={k: int(x) for k, x in pares.nivel.value_counts().sort_index().items()})
    (ruta("umbrales_depuracion.json")).write_text(json.dumps(umbrales, indent=2, ensure_ascii=False), encoding="utf-8")
    return umbrales


def depurar(dir_salida=None, lector=None, parametros=None):
    # Ejecuta la depuración; dir_salida, lector y parametros permiten las pruebas de fuga y de sensibilidad
    previo_base = comun.BASE_PROC
    previos = {k: globals()[k] for k in (parametros or {})}
    if dir_salida is not None:
        comun.BASE_PROC = Path(dir_salida)
    globals().update(parametros or {})
    try:
        return _depurar(lector or leer_fuentes)
    finally:
        comun.BASE_PROC = previo_base
        globals().update(previos)


def clave(s, a, m, d):
    return int(s), int(a), int(m), int(d)


def destinos_por_clave():
    # (regla, destino) de cada registro retirado, por (estación, año, mes, día)
    ev = pd.read_csv(ruta("eventos_depuracion.csv"), usecols=["station_id", "year", "month", "day", "regla", "destino"])
    ev = ev[ev.destino != "recuperado_diario"]
    f29 = pd.read_csv(ruta("registros_29_febrero.csv"), usecols=["station_id", "year", "month", "day"])
    f29["regla"], f29["destino"] = "29_febrero", "retirado"
    todo = pd.concat([ev, f29], ignore_index=True)
    assert not todo.duplicated(["station_id", "year", "month", "day"]).any()
    dest = {clave(r.station_id, r.year, r.month, r.day): [(r.regla, r.destino)] for r in todo.itertuples()}
    a2 = ev[(ev.regla == "A2_repeticion") & (not A2_RETIRAR_PRIMERO)].copy()
    a2["fecha"] = pd.to_datetime(dict(year=a2.year, month=a2.month, day=a2.day))
    claves_a2 = set(zip(a2.station_id, a2.fecha))
    for r in a2.itertuples():
        dest[clave(r.station_id, r.year, r.month, r.day)].append(("A2_repeticion", "con_primer_dia"))
        previo = r.fecha - pd.Timedelta(days=1)
        if (r.station_id, previo) not in claves_a2:
            dest[clave(r.station_id, previo.year, previo.month, previo.day)] = [("A2_repeticion", "con_primer_dia")]
    return dest, len(todo)


def exportar_baddata():
    # Registros retirados en su formato original, en data/baddata/<regla>/<destino>/<archivo de origen>.csv
    dest, n_esperado = destinos_por_clave()
    acum = {}

    def guardar(linea, k, nombre, cabecera):
        for regla, destino in dest[k]:
            acum.setdefault((regla, destino, nombre, cabecera), []).append(linea)

    for nombre, archivo in CSV_ORIGEN.items():
        with open(archivo, encoding="utf-8", newline="") as f:
            cabecera = f.readline().rstrip("\r\n")
            for linea in f:
                s, a, m, d, _ = linea.rstrip("\r\n").split(",")
                k = clave(s, a, m, d)
                if k in dest:
                    guardar(linea, k, nombre, cabecera)

    shutil.rmtree(DIR_BAD, ignore_errors=True)
    for (regla, destino, nombre, cabecera), lineas in sorted(acum.items()):
        carpeta = DIR_BAD / regla / destino
        carpeta.mkdir(parents=True, exist_ok=True)
        with open(carpeta / f"{nombre}.csv", "w", encoding="utf-8", newline="") as f:
            f.write(cabecera + "\r\n")
            f.writelines(lineas)
    total = sum(len(v) for (_, d, _, _), v in acum.items() if d != "con_primer_dia")
    assert total == n_esperado, (total, n_esperado)
