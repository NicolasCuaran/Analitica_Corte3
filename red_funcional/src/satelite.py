"""Satélite pentadal: péntadas del dominio de 365 días, lector de GeoTIFF LZW y tabla de totales por píxel."""
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from .catalogos import departamento_de
from .comun import DIR_IMAGENES, DOMINIO, PERIODO_PROYECTO, ruta


def tabla_pentadas():
    # 72 péntadas por año (6 por mes; la sexta cubre del día 26 al fin de mes) en el dominio de 365 días sin 29-feb
    dias = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    acum = np.r_[0, np.cumsum(dias)[:-1]]
    filas = []
    for p in range(1, 73):
        m, k = (p - 1) // 6, (p - 1) % 6
        ini = 1 + 5 * k
        fin = 5 * (k + 1) if k < 5 else dias[m]
        n_bis = fin - ini + 1 + (1 if (m == 1 and k == 5) else 0)
        doy = np.arange(acum[m] + ini, acum[m] + fin + 1)
        filas.append(dict(pentada=p, mes=m + 1, doy_inicio=int(doy[0]), doy_fin=int(doy[-1]), n_dias=len(doy),
                          n_dias_anio_bisiesto=int(n_bis), t_baricentro=float(doy.mean())))
    return pd.DataFrame(filas)


def lzw_tiff(datos):
    # Decodificador LZW de TIFF: bits MSB primero, cambio anticipado de ancho, 256 = limpiar y 257 = fin
    bits = int.from_bytes(datos, "big")
    n_bits = len(datos) * 8
    pos = 0
    salida = bytearray()
    tabla = [bytes([i]) for i in range(256)] + [b"", b""]
    ancho = 9
    previo = None
    while pos + ancho <= n_bits:
        codigo = (bits >> (n_bits - pos - ancho)) & ((1 << ancho) - 1)
        pos += ancho
        if codigo == 257:
            break
        if codigo == 256:
            tabla = tabla[:258]
            ancho = 9
            previo = None
            continue
        if previo is None:
            entrada = tabla[codigo]
        elif codigo < len(tabla):
            entrada = tabla[codigo]
            tabla.append(previo + entrada[:1])
        else:
            entrada = previo + previo[:1]
            tabla.append(entrada)
        salida += entrada
        previo = entrada
        n = len(tabla) + 1
        ancho = 9 if n < 512 else 10 if n < 1024 else 11 if n < 2048 else 12
    return bytes(salida)


def leer_geotiff_int16(archivo):
    # GeoTIFF de una banda int16 con teselas LZW, sin librerías externas.
    # Devuelve (matriz, nodata, (escala_x, escala_y), (x0, y0)) con (x0, y0) la esquina superior izquierda.
    b = Path(archivo).read_bytes()
    orden = "<" if b[:2] == b"II" else ">"
    ifd = struct.unpack(orden + "I", b[4:8])[0]
    n = struct.unpack(orden + "H", b[ifd:ifd + 2])[0]
    tam = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 12: 8, 16: 8}
    fmt = {1: "B", 2: "c", 3: "H", 4: "I", 12: "d", 16: "Q"}
    tags = {}
    for i in range(n):
        e = b[ifd + 2 + 12 * i: ifd + 14 + 12 * i]
        cod, tipo, cnt = struct.unpack(orden + "HHI", e[:8])
        nbytes = tam[tipo] * cnt
        crudo = e[8:12] if nbytes <= 4 else b[struct.unpack(orden + "I", e[8:12])[0]:][:nbytes]
        if tipo == 2:
            tags[cod] = crudo[:cnt].decode("latin-1").rstrip("\x00")
        elif tipo in fmt:
            tags[cod] = struct.unpack(orden + str(cnt) + fmt[tipo], crudo[:nbytes])
    ancho, alto = tags[256][0], tags[257][0]
    tw, tl = tags[322][0], tags[323][0]
    assert tags[259][0] == 5 and tags[317][0] == 1 and tags[258][0] == 16 and tags[339][0] == 2
    nx = -(-ancho // tw)
    img = np.zeros((-(-alto // tl) * tl, nx * tw), dtype=orden + "i2")
    for k, (o, c) in enumerate(zip(tags[324], tags[325])):
        pix = np.frombuffer(lzw_tiff(b[o:o + c])[:tl * tw * 2], dtype=orden + "i2").reshape(tl, tw)
        r, q = divmod(k, nx)
        img[r * tl:(r + 1) * tl, q * tw:(q + 1) * tw] = pix
    nodata = float(tags[42113]) if 42113 in tags else None
    return img[:alto, :ancho].astype(np.int16), nodata, tags[33550][:2], tags[33922][3:5]


def satelite_a_csv():
    # Totales de péntada (mm) por píxel con centro en el Valle: filas = (año, péntada), columnas = píxeles
    filas, mats = [], []
    for a in range(PERIODO_PROYECTO[0], PERIODO_PROYECTO[1] + 1):
        for p in range(1, 73):
            f = DIR_IMAGENES / f"ppt{a}{p:02d}.tif"
            if not f.exists():
                continue
            img, nodata, (sx, sy), (x0, y0) = leer_geotiff_int16(f)
            m = img.astype(float)
            m[img == nodata] = np.nan
            filas.append((a, p))
            mats.append(m.ravel())
    M = np.vstack(mats)                                                               # Dim: (imágenes, 1443)
    nf, nc = img.shape
    px = pd.DataFrame({"pixel": np.arange(1, nf * nc + 1), "fila": np.repeat(np.arange(nf), nc), "col": np.tile(np.arange(nc), nf)})
    px["lon"] = x0 + (px.col + 0.5) * sx
    px["lat"] = y0 - (px.fila + 0.5) * sy
    # Solo los píxeles con centro dentro del Valle; conservan su número de la malla completa
    valle = departamento_de(px.lon, px.lat, solo=DOMINIO) == DOMINIO
    px = px[valle].reset_index(drop=True)
    df = pd.DataFrame(M[:, valle], columns=[f"px{k}" for k in px.pixel])
    df.insert(0, "pentada", [p for _, p in filas])
    df.insert(0, "anio", [a for a, _ in filas])
    df.to_csv(ruta("satelite_pentadal_mm.csv"), index=False, float_format="%.0f", na_rep="NA")
    px.to_csv(ruta("pixeles_malla.csv"), index=False, float_format="%.4f")
    tabla_pentadas().to_csv(ruta("pentadas_dominio_365.csv"), index=False)
