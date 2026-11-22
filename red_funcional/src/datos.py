"""Datos de la red funcional: vistas del preprocesamiento, satélite por píxel, ONI y registro de verificaciones."""
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .comun import FIG, N_DIAS, PARTICION, PROC, SAL, TABLAS

VISTAS = {"entrenamiento": "uso_entrenamiento", "prueba": "uso_prueba", "ciegas": "uso_ciegas_propias"}
TRAMOS = ("prueba", "ciegas")
ANIOS_ENT, ANIOS_PRU = PARTICION["entrenamiento"], PARTICION["prueba"]
TXT_ENT, TXT_PRU = "%d-%d" % ANIOS_ENT, "%d-%d" % ANIOS_PRU
FASES = ("nino", "nina", "neutro")
ETQ_FASE = {"nino": "El Niño", "nina": "La Niña", "neutro": "neutro"}


def anota(ctx, nombre, valor, tolerancia):
    # Registro de una verificación numérica: cumple si valor <= tolerancia
    ctx.verif.append(dict(verificacion=nombre, valor=float(valor), tolerancia=tolerancia, cumple=bool(valor <= tolerancia)))


def col_pixel(ctx, pixel):
    j = ctx.PX_VALLE.get_indexer(np.asarray(pixel, int))
    assert (j >= 0).all(), "píxel fuera del dominio del Valle"
    return j


def satelite_de(ctx, pixel, anio):
    return ctx.Xp[np.searchsorted(ctx.anios, anio), :, col_pixel(ctx, pixel)].T        # Dim: (72, n)


def cargar():
    # Contexto de la corrida: doble precisión en torch, carpetas de salida, contraste de los catálogos abiertos y vistas del
    # preprocesamiento (lluvia observada + imputada, satélite en mm/día, altitud normalizada con el entrenamiento y ONI)
    torch.set_default_dtype(torch.float64)
    SAL.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(t_ini=time.time(), verif=[], tablas={}, preproceso=TABLAS)          # tablas de resultados en memoria

    _cat = pd.read_csv(PROC / "validacion" / "diag_catalogos_abiertos.csv").set_index("indicador").valor
    anota(ctx, "Coordenadas CVC datos.gov.co vs consigna: distancia mediana (km)", _cat["coord_distancia_mediana_km"], 0.1)
    anota(ctx, "Altitud IDEAM datos.gov.co vs consigna: |diferencia| mediana (m)", _cat["altitud_dif_abs_mediana_m"], 10.0)
    anota(ctx, "Altitud SRTM (CGIAR, 90 m) vs consigna: |diferencia| mediana (m)", _cat["altitud_srtm_dif_abs_mediana_m"], 50.0)
    anota(ctx, "Estaciones del Valle sin altitud tras consigna, IDEAM y SRTM", _cat["altitud_sin_dato"], 0)

    meta = pd.read_csv(PROC / "metadatos/metadata_depuracion.csv")
    uso = {k: meta[v].astype(str).eq("True").to_numpy() for k, v in VISTAS.items()}
    meta = meta[np.logical_or.reduce(list(uso.values()))].reset_index(drop=True)
    uso = {k: meta[v].astype(str).eq("True").to_numpy() for k, v in VISTAS.items()}
    clave = meta.station_id.to_numpy() * 10000 + meta.year.to_numpy()
    d = pd.concat([pd.read_csv(PROC / f"datos_diarios/precip_diaria_{x}.csv", usecols=["station_id", "year", "doy", "value_mm"])
                   for x in ("depurada", "imputada")], ignore_index=True)                  # observado + días imputados
    d = d[d.doy <= N_DIAS]
    j = pd.Index(clave).get_indexer(d.station_id * 10000 + d.year)
    Yd = np.full((N_DIAS, len(meta)), np.nan)
    Yd[d.doy.to_numpy()[j >= 0] - 1, j[j >= 0]] = d.value_mm.to_numpy()[j >= 0]
    del d

    pen = pd.read_csv(PROC / "satelite/pentadas_dominio_365.csv")
    sat = pd.read_csv(PROC / "satelite/satelite_pentadal_mm.csv")
    assert all(c.startswith("px") for c in sat.columns[2:])
    PX_VALLE = pd.Index([int(c[2:]) for c in sat.columns[2:]])                           # píxeles con centro en el Valle (número de la malla completa)
    anios = np.sort(sat.anio.unique())
    nd = np.where((sat.anio % 4 == 0) & (sat.pentada == 12), pen.n_dias_anio_bisiesto.to_numpy()[sat.pentada - 1], pen.n_dias.to_numpy()[sat.pentada - 1])
    Xp = (sat.filter(like="px").to_numpy() / nd[:, None]).reshape(len(anios), 72, -1)  # Dim: (anio, péntada, píxel), mm/día
    t_sat = pen.t_baricentro.to_numpy()
    del sat

    oni_t = pd.read_csv(PROC / "oni_1981_2003.csv").set_index("anio")
    alt = meta.altitud_m.to_numpy()
    ALT_MU, ALT_SD = alt[uso["entrenamiento"]].mean(), alt[uso["entrenamiento"]].std()   # normalización solo con entrenamiento
    alt_n = (alt - ALT_MU) / ALT_SD
    oni_a = oni_t.oni_medio.reindex(meta.year).to_numpy()
    fase = oni_t.fase_codigo.reindex(meta.year).to_numpy()
    fase_i = pd.Index(FASES).get_indexer(fase)                                           # 0 Niño, 1 Niña, 2 neutro

    vars(ctx).update(meta=meta, uso=uso, Yd=Yd, PX_VALLE=PX_VALLE, anios=anios, Xp=Xp, t_sat=t_sat, oni_t=oni_t, alt=alt,
                     ALT_MU=ALT_MU, ALT_SD=ALT_SD, alt_n=alt_n, oni_a=oni_a, fase=fase, fase_i=fase_i)
    ctx.Xs = satelite_de(ctx, meta.pixel, meta.year)
    return ctx
