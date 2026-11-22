"""ONI de la NOAA y fase ENSO de cada año calendario."""
import hashlib
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

from .comun import RAIZ, guardar

URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
CRUDO = RAIZ / "data/raw/oni/oni_ascii_noaa_cpc.txt"
SALIDA = RAIZ / "data/processed/oni_1981_2003.csv"
ANIO_INI, ANIO_FIN = 1981, 2003
UMBRAL, MIN_TRIM = 0.5, 5
TRIMESTRES = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def descargar():
    CRUDO.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as r:
        CRUDO.write_bytes(r.read())


def leer_oni():
    # Tabla trimestral del ONI ordenada por año y trimestre; se descarga solo si falta en data/raw/
    if not CRUDO.exists():
        descargar()
    t = pd.read_csv(CRUDO, sep=r"\s+")
    t = t.rename(columns={"SEAS": "trimestre", "YR": "anio", "TOTAL": "sst", "ANOM": "anomalia"})
    t["pos"] = t.trimestre.map({s: i for i, s in enumerate(TRIMESTRES)})
    assert t.pos.notna().all(), "trimestre desconocido en la tabla del ONI"
    t = t.sort_values(["anio", "pos"]).reset_index(drop=True)
    assert (t.groupby("anio").size().loc[ANIO_INI:ANIO_FIN] == 12).all(), "años incompletos en 1981-2003"
    return t


def rachas(mascara):
    d = np.diff(np.concatenate(([0], mascara.astype(int), [0])))
    ini, fin = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ini, fin - ini


def mejor_racha(a, signo):
    # Racha más larga sobre el umbral (a igualdad, la de mayor suma de |anomalía|)
    ini, largo = rachas(a >= UMBRAL if signo > 0 else a <= -UMBRAL)
    if largo.size == 0:
        return 0, 0.0
    suma = np.array([np.abs(a[i:i + n]).sum() for i, n in zip(ini, largo)])
    k = np.lexsort((suma, largo))[-1]
    return int(largo[k]), float(suma[k])


def episodios(t):
    # Episodios de la NOAA: al menos MIN_TRIM trimestres consecutivos con |anomalía| >= UMBRAL
    a = t.anomalia.to_numpy()
    filas = []
    for signo, nombre in ((1, "El Niño"), (-1, "La Niña")):
        ini, largo = rachas(a >= UMBRAL if signo > 0 else a <= -UMBRAL)
        for i, n in zip(ini, largo):
            if n >= MIN_TRIM:
                tr = t.iloc[i:i + n]
                k = tr.anomalia.abs().to_numpy().argmax()
                filas.append(dict(episodio=nombre, inicio=f"{tr.trimestre.iloc[0]} {tr.anio.iloc[0]}", fin=f"{tr.trimestre.iloc[-1]} {tr.anio.iloc[-1]}",
                                  trimestres=int(n), anomalia_pico=float(tr.anomalia.iloc[k]), trimestre_pico=f"{tr.trimestre.iloc[k]} {tr.anio.iloc[k]}",
                                  i0=int(i), i1=int(i + n)))
    return pd.DataFrame(filas).sort_values("i0").reset_index(drop=True)


def clasificar(t, ep):
    # Fase ENSO de cada año calendario según sus 12 trimestres DJF..NDJ
    filas = []
    for anio in range(ANIO_INI, ANIO_FIN + 1):
        sub = t[t.anio == anio]
        a = sub.anomalia.to_numpy()
        n_nino, s_nino = mejor_racha(a, +1)
        n_nina, s_nina = mejor_racha(a, -1)
        es_nino, es_nina = n_nino >= MIN_TRIM, n_nina >= MIN_TRIM
        if es_nino and es_nina:
            nino_gana = (n_nino, s_nino) > (n_nina, s_nina)
            fase = "El Niño" if nino_gana else "La Niña"
        else:
            fase = "El Niño" if es_nino else "La Niña" if es_nina else "Neutro"
        pos = sub.index.to_numpy()
        solapa = ep[(ep.i0 <= pos.max()) & (ep.i1 > pos.min())]
        k = np.abs(a).argmax()
        filas.append(dict(anio=anio, fase=fase, fase_codigo={"El Niño": "nino", "La Niña": "nina", "Neutro": "neutro"}[fase], racha_nino=n_nino, racha_nina=n_nina,
                          trimestres_nino=int((a >= UMBRAL).sum()), trimestres_nina=int((a <= -UMBRAL).sum()),
                          oni_max=a.max(), oni_min=a.min(), oni_medio=a.mean(), oni_pico=a[k], trimestre_pico=sub.trimestre.iloc[k],
                          ambiguo=bool(es_nino and es_nina),
                          episodio_noaa="; ".join(f"{r.episodio} {r.inicio}-{r.fin}" for r in solapa.itertuples())))
    return pd.DataFrame(filas)


def oni_41():
    # Fase ENSO por año (se escribe: la leen la red y el R); el ONI trimestral y la ficha de la fuente quedan en memoria
    t = leer_oni()
    ep = episodios(t)
    out = clasificar(t, ep)
    tri = t[(t.anio >= ANIO_INI) & (t.anio <= ANIO_FIN)].copy()
    tri["fase_trimestre"] = np.where(tri.anomalia >= UMBRAL, "calido", np.where(tri.anomalia <= -UMBRAL, "frio", "neutro"))
    guardar("oni_trimestral_1981_2003", tri[["anio", "trimestre", "sst", "anomalia", "fase_trimestre"]])
    out.round(2).to_csv(SALIDA, index=False, encoding="utf-8")
    guardar("fuente_oni", {
        "fuente": "NOAA Climate Prediction Center, Oceanic Niño Index (ONI, ERSSTv5)", "url": URL, "archivo_crudo": CRUDO.relative_to(RAIZ).as_posix(),
        "md5_crudo": hashlib.md5(CRUDO.read_bytes()).hexdigest(), "fecha_consulta": date.today().isoformat(),
        "periodo": [ANIO_INI, ANIO_FIN], "criterio": f"racha de al menos {MIN_TRIM} trimestres consecutivos con |anomalía| >= {UMBRAL} °C dentro de los 12 trimestres DJF..NDJ del año",
        "desempate": "racha más larga y, a igualdad, mayor suma de |anomalía|; año marcado ambiguo"})
