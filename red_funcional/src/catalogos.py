"""Catálogos de estaciones: coordenadas, altitud, dominio del Valle y emparejamiento estación-píxel."""
import re
import unicodedata

import numpy as np
import pandas as pd

from .comun import (CSV_COORDENADAS, CSV_CVC_ABIERTO, CSV_IDEAM, CSV_LIMITES, CSV_SRTM, DOMINIO, RAIZ,
                    XLSX_ALTITUD, guardar, ruta)

R_TIERRA_KM = 6371.0088
MEDIO_PIXEL = 0.025
ALT_DIST_KM = 3.0
ALT_CONFIRMA_M = 100.0      # el IDEAM confirma la altitud de la consigna si difiere de ella en <= 100 m
PALABRAS_VACIAS = {"LA", "EL", "LOS", "LAS", "DE", "DEL", "AUT", "HDA", "HACIENDA", "PLTA", "PLANTA"}


def haversine_km(lat1, lon1, lat2, lon2):
    f1, f2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((f2 - f1) / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_TIERRA_KM * np.arcsin(np.sqrt(a))


def emparejar(cat, malla):
    # Píxel de centro más cercano (Haversine) para cada estación
    d = haversine_km(cat.lat.to_numpy()[:, None], cat.lon.to_numpy()[:, None], malla.lat.to_numpy()[None, :], malla.lon.to_numpy()[None, :])   # Dim: (E, P)
    k = d.argmin(axis=1)
    out = cat.copy()
    out["pixel"] = malla.pixel.to_numpy()[k]
    out["fila"], out["col"] = malla.fila.to_numpy()[k], malla.col.to_numpy()[k]
    out["distancia_km"] = d[np.arange(len(cat)), k]
    out["dentro_pixel"] = (np.abs(cat.lon.to_numpy() - malla.lon.to_numpy()[k]) <= MEDIO_PIXEL + 1e-9) & (np.abs(cat.lat.to_numpy() - malla.lat.to_numpy()[k]) <= MEDIO_PIXEL + 1e-9)
    n = out.groupby("pixel").station_id.transform("size")
    out["n_estaciones_en_pixel"] = n
    out["comparte_pixel_con"] = out.groupby("pixel").station_id.transform(lambda s: ";".join(map(str, s)))
    return out


def departamento_de(lon, lat, solo=None):
    # Departamento GADM de cada punto: punto en polígono por paridad de cruces de un rayo horizontal con las aristas de cada
    # anillo; "" si el punto no cae en ningún departamento
    lim = pd.read_csv(CSV_LIMITES)
    if solo is not None:
        lim = lim[lim.departamento == solo]
    x, y = np.asarray(lon, float)[:, None], np.asarray(lat, float)[:, None]
    out = np.full(len(x), "", dtype=object)
    for depto, p in lim.groupby("departamento", sort=False):
        a, lo, la = p.anillo.to_numpy(), p.lon.to_numpy(), p.lat.to_numpy()
        e = np.flatnonzero(a[:-1] == a[1:])
        x1, y1, x2, y2 = lo[e], la[e], lo[e + 1], la[e + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            cruce = ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))   # Dim: (n_puntos, n_aristas)
        out[(cruce.sum(axis=1) % 2 == 1) & (out == "")] = depto
    return out


def tm_a_geo(x, y, lat0, lon0, k0, fe, fn):
    # Inversa de la Transversa de Mercator sobre el elipsoide GRS80 (EPSG 9807); x, y en metros, devuelve (lon, lat) en grados
    a, f = 6378137.0, 1 / 298.257222101
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    c0 = 1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256
    p0 = np.radians(lat0)
    m0 = a * (c0 * p0 - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * p0)
              + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * p0) - (35 * e2 ** 3 / 3072) * np.sin(6 * p0))
    mu = (m0 + (np.asarray(y, float) - fn) / k0) / (a * c0)
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    c1, t1 = ep2 * np.cos(p1) ** 2, np.tan(p1) ** 2
    n1 = a / np.sqrt(1 - e2 * np.sin(p1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * np.sin(p1) ** 2) ** 1.5
    d = (np.asarray(x, float) - fe) / (n1 * k0)
    lat = p1 - (n1 * np.tan(p1) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
                                          + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = np.radians(lon0) + (d - (1 + 2 * t1 + c1) * d ** 3 / 6
                              + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / np.cos(p1)
    return np.degrees(lon), np.degrees(lat)


def geo_desde_magna(x, y):
    # El catálogo abierto de la CVC mezcla MAGNA-SIRGAS Colombia Oeste (EPSG:3115, este ≈ 1 000 000 m) y Origen Nacional
    # (EPSG:9377, este ≈ 4 600 000 m); se distinguen por la magnitud del este
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo_o, la_o = tm_a_geo(x, y, 4.596200416666667, -77.07750791666667, 1.0, 1e6, 1e6)
    lo_n, la_n = tm_a_geo(x, y, 4.0, -73.0, 0.9992, 5e6, 2e6)
    nacional = x > 3e6
    return np.where(nacional, lo_n, lo_o), np.where(nacional, la_n, la_o)


def nombre_clave(s):
    # Nombre comparable entre catálogos: sin tildes, sin el código entre corchetes, sin artículos ni prefijos y en orden alfabético
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    palabras = re.sub(r"[^A-Z0-9 ]", " ", re.sub(r"\[.*?\]", " ", s)).split()
    return " ".join(sorted(p for p in palabras if p not in PALABRAS_VACIAS))


def altitud_ideam(est):
    # Altitud de la estación del IDEAM con el mismo nombre más cercana a <= ALT_DIST_KM (NaN si no hay); est: nombre, lon, lat
    idm = pd.read_csv(CSV_IDEAM, dtype={"codigo": str}).drop_duplicates()
    idm["altitud"] = pd.to_numeric(idm.altitud, errors="coerce")
    idm = idm[(idm.altitud > 0) & idm.longitud.notna() & idm.latitud.notna()].reset_index(drop=True)
    if len(est) == 0:
        return np.array([]), np.array([])
    d = haversine_km(est.lat.to_numpy()[:, None], est.lon.to_numpy()[:, None], idm.latitud.to_numpy()[None, :],
                     idm.longitud.to_numpy()[None, :])                                                    # Dim: (E, I)
    igual = est.nombre.map(nombre_clave).to_numpy()[:, None] == idm.nombre.map(nombre_clave).to_numpy()[None, :]
    d = np.where(igual, d, np.inf)
    j = d.argmin(axis=1)
    dj = d[np.arange(len(est)), j]
    ok = dj <= ALT_DIST_KM
    return np.where(ok, idm.altitud.to_numpy()[j], np.nan), np.where(ok, dj, np.nan)


def catalogo_estaciones(estaciones):
    # Catálogo de las estaciones del Valle. Coordenadas: CSV de la consigna y, si faltan, catálogo abierto de la CVC.
    # Altitud: consigna, luego IDEAM y luego SRTM; la del IDEAM se reemplaza por la del SRTM cuando discrepa de ella más que el
    # mayor error del SRTM en las estaciones de control. Cada estación toma el píxel del Valle de centro más cercano.
    # Devuelve (catálogo del Valle, estaciones fuera del límite o sin coordenadas).
    cons = pd.read_csv(CSV_COORDENADAS, encoding="latin-1")
    cons = pd.DataFrame({"station_id": pd.to_numeric(cons.CODIGO, errors="coerce"), "nombre": cons.STATION_NAME,
                         "lon": cons.Lon, "lat": cons.lat}).dropna(subset=["station_id", "lon", "lat"])
    ab = pd.read_csv(CSV_CVC_ABIERTO, dtype={"codigo_estacion": str})
    lon_ab, lat_ab = geo_desde_magna(ab.este_x, ab.norte_y)
    ab = pd.DataFrame({"station_id": pd.to_numeric(ab.codigo_estacion, errors="coerce"), "nombre": ab.nombre_estacion,
                       "lon": lon_ab, "lat": lat_ab}).dropna(subset=["station_id", "lon", "lat"])
    cons["station_id"] = cons.station_id.astype("int64")
    ab["station_id"] = ab.station_id.astype("int64")
    cons, ab = cons.drop_duplicates("station_id"), ab.drop_duplicates("station_id")
    ambos = cons.merge(ab, on="station_id", suffixes=("", "_ab"))
    d_ambos = haversine_km(ambos.lat.to_numpy(), ambos.lon.to_numpy(), ambos.lat_ab.to_numpy(), ambos.lon_ab.to_numpy())
    co = pd.concat([cons.assign(fuente_coordenadas="consigna"),
                    ab[~ab.station_id.isin(cons.station_id)].assign(fuente_coordenadas="cvc_datosgov")], ignore_index=True)
    alt = pd.read_excel(XLSX_ALTITUD, engine="openpyxl")
    c_cod = next(c for c in alt.columns if "cod" in str(c).lower())
    c_alt = next(c for c in alt.columns if any(k in str(c).lower() for k in ("elev", "alt")))
    alt = pd.DataFrame({"station_id": pd.to_numeric(alt[c_cod], errors="coerce"),
                        "altitud_m": pd.to_numeric(alt[c_alt], errors="coerce")}).dropna().drop_duplicates("station_id")
    alt["station_id"] = alt.station_id.astype("int64")
    co = co[co.station_id.isin(estaciones)].reset_index(drop=True)
    co["departamento"] = departamento_de(co.lon, co.lat)
    todas = pd.DataFrame({"station_id": np.asarray(estaciones, dtype="int64")})
    fuera = todas.merge(co, on="station_id", how="left")
    fuera = fuera[fuera.departamento.ne(DOMINIO)].copy()
    fuera["motivo"] = np.where(fuera.lat.isna(), "sin coordenadas en la consigna ni en el catálogo abierto de la CVC",
                               f"coordenadas fuera del límite del {DOMINIO}")
    fuera["departamento"] = fuera.departamento.replace("", "ninguno").fillna("sin coordenadas")
    co = co[co.departamento.eq(DOMINIO)].drop(columns="departamento").reset_index(drop=True)
    emp = emparejar(co, pd.read_csv(RAIZ / "data" / "processed" / "satelite" / "pixeles_malla.csv"))
    emp["reemparejada_borde"] = ~emp.dentro_pixel
    emp["dentro_pixel"] = True
    n_px = emp.groupby("pixel").station_id.transform("size")
    emp["n_estaciones_en_pixel"] = n_px.astype(int)
    emp["comparte_pixel_con"] = emp.groupby("pixel").station_id.transform(lambda s: ";".join(map(str, s)))
    emp = emp.merge(alt, on="station_id", how="left")
    falta = emp.altitud_m.isna().to_numpy()
    a_idm, d_idm = altitud_ideam(emp[falta])
    emp["fuente_altitud"] = np.where(falta, "", "consigna")
    emp.loc[falta, "altitud_m"] = a_idm
    emp.loc[falta, "fuente_altitud"] = np.where(np.isnan(a_idm), "", "ideam_datosgov")
    # Control del cruce con el IDEAM en las estaciones que tienen la altitud de la consigna
    a_ctrl, _ = altitud_ideam(emp[~falta])
    dif = np.abs(a_ctrl - emp.altitud_m.to_numpy()[~falta])
    dif = dif[~np.isnan(dif)]
    srtm = pd.read_csv(CSV_SRTM).set_index("station_id").altitud_srtm_m.reindex(emp.station_id).to_numpy()
    # Error admisible del SRTM: mayor |IDEAM - SRTM| entre las estaciones de control en que el IDEAM confirma la consigna
    s_ctrl = srtm[~falta]
    d_ctrl = np.abs(a_ctrl - s_ctrl)
    confirmada = (np.abs(a_ctrl - emp.altitud_m.to_numpy()[~falta]) <= ALT_CONFIRMA_M) & ~np.isnan(s_ctrl)
    umbral_idm = float(d_ctrl[confirmada].max())
    es_idm = (emp.fuente_altitud == "ideam_datosgov").to_numpy()
    dist_idm = np.full(len(emp), np.nan)
    dist_idm[falta] = d_idm
    dif_is = emp.altitud_m.to_numpy() - srtm
    descarta = es_idm & (np.abs(dif_is) > umbral_idm)
    contraste = emp.loc[es_idm, ["station_id", "nombre", "lon", "lat"]].assign(
        altitud_ideam_m=emp.altitud_m.to_numpy()[es_idm], distancia_ideam_km=dist_idm[es_idm], altitud_srtm_m=srtm[es_idm],
        diferencia_ideam_srtm_m=dif_is[es_idm], umbral_m=umbral_idm, descartada=descarta[es_idm])
    emp.loc[descarta, "altitud_m"] = srtm[descarta]
    emp.loc[descarta, "fuente_altitud"] = "srtm_cgiar"
    contraste["altitud_asignada_m"] = emp.altitud_m.to_numpy()[es_idm]
    contraste["fuente_altitud"] = emp.fuente_altitud.to_numpy()[es_idm]
    contraste["motivo"] = np.where(
        contraste.descartada, f"|IDEAM - SRTM| mayor que {umbral_idm:.1f} m (el mayor error del SRTM en las estaciones de "
        "control con altitud confirmada por la consigna y el IDEAM): se descarta la del IDEAM y manda el SRTM",
        "dentro del error del SRTM en el control: se conserva la del IDEAM")
    guardar("altitud_ideam_contraste_srtm", contraste.sort_values("station_id"))
    falta = emp.altitud_m.isna().to_numpy() & ~np.isnan(srtm)
    emp.loc[falta, "altitud_m"] = srtm[falta]
    emp.loc[falta, "fuente_altitud"] = "srtm_cgiar"
    ctrl = (emp.fuente_altitud == "consigna").to_numpy() & ~np.isnan(srtm)
    dif_s = srtm[ctrl] - emp.altitud_m.to_numpy()[ctrl]
    pd.DataFrame([
        ("coord_estaciones_en_ambos_catalogos_cvc", len(ambos)),
        ("coord_distancia_mediana_km", float(np.median(d_ambos))),
        ("coord_distancia_p90_km", float(np.percentile(d_ambos, 90))),
        ("coord_mas_de_1km", int((d_ambos > 1).sum())),
        ("coord_estaciones_desde_cvc_datosgov", int((emp.fuente_coordenadas == "cvc_datosgov").sum())),
        ("altitud_control_estaciones", len(dif)),
        ("altitud_dif_abs_mediana_m", float(np.median(dif))),
        ("altitud_dif_mas_de_100m", int((dif > ALT_CONFIRMA_M).sum())),
        ("altitud_estaciones_desde_ideam", int((emp.fuente_altitud == "ideam_datosgov").sum())),
        ("altitud_distancia_max_km", float(np.nanmax(d_idm)) if np.isfinite(d_idm).any() else 0.0),
        ("altitud_ideam_srtm_control_confirmadas", int(confirmada.sum())),
        ("altitud_ideam_srtm_control_mas_de_300m", int((d_ctrl > 300).sum())),
        ("altitud_ideam_srtm_control_mas_de_300m_confirmadas", int((confirmada & (d_ctrl > 300)).sum())),
        ("altitud_ideam_srtm_umbral_m", umbral_idm),
        ("altitud_ideam_descartadas_por_srtm", int(descarta.sum())),
        ("altitud_srtm_control_estaciones", int(ctrl.sum())),
        ("altitud_srtm_dif_abs_mediana_m", float(np.median(np.abs(dif_s)))),
        ("altitud_srtm_sesgo_mediano_m", float(np.median(dif_s))),
        ("altitud_srtm_dif_abs_p90_m", float(np.percentile(np.abs(dif_s), 90))),
        ("altitud_srtm_dif_mas_de_300m", int((np.abs(dif_s) > 300).sum())),
        ("altitud_estaciones_desde_srtm", int((emp.fuente_altitud == "srtm_cgiar").sum())),
        ("altitud_sin_dato", int(emp.altitud_m.isna().sum()))], columns=["indicador", "valor"]
    ).to_csv(ruta("diag_catalogos_abiertos.csv"), index=False)
    emp["en_red"] = emp.dentro_pixel & emp.altitud_m.notna()
    emp["con_coordenadas"] = True
    emp = emp.sort_values("station_id").reset_index(drop=True)
    return emp, fuera[["station_id", "lon", "lat", "departamento", "motivo"]].reset_index(drop=True)
