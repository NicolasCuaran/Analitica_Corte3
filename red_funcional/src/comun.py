"""Rutas y constantes comunes: periodo, partición, pliegues y carpetas de entrada y salida."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# Entradas de data/raw (solo lectura)
DIR_EST = RAIZ / "data" / "raw" / "estaciones"
DIR_IMAGENES = RAIZ / "data" / "raw" / "satelite" / "ImagenesSatelitePentadal" / "ImagenesSatelitePentadal"
CSV_ORIGEN = {"P_1980_1990": DIR_EST / "P_1980_1990.csv", "P_1991_2003": DIR_EST / "P_1991_2003.csv"}
DIR_CAT = RAIZ / "data" / "raw" / "catalogos"
CSV_COORDENADAS = DIR_CAT / "CVC_Wheather_Stations_Long_Lat.csv"
XLSX_ALTITUD = DIR_CAT / "CatalogoEstacionesElevacion.xlsx"
CSV_LIMITES = DIR_CAT / "limites_departamentos_gadm41.csv"
# Catálogos abiertos de datos.gov.co (API Socrata, descargados el 2026-11-19):
#   CVC (kfsz-ija2): https://www.datos.gov.co/resource/kfsz-ija2.csv?$limit=50000&$order=codigo_estacion
#   IDEAM (hp9r-jxuu), Valle y departamentos vecinos: https://www.datos.gov.co/resource/hp9r-jxuu.csv?$select=codigo,nombre,
#     categoria,estado,entidad,departamento,municipio,altitud,longitud,latitud&$where=departamento in('Valle Del Cauca','Cauca',
#     'Choco','Quindío','Risaralda','Tolima')&$order=codigo&$limit=50000
CSV_CVC_ABIERTO = DIR_CAT / "estaciones_cvc_datosgov.csv"
CSV_IDEAM = DIR_CAT / "estaciones_ideam_datosgov.csv"
# Altitud SRTM v4.1 (CGIAR-CSI, unos 90 m) interpolada (bilineal) en cada estación desde la tesela srtm_21_12
# (https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_21_12.zip, 2026-11-20, md5 7db42be86da0bf278f0964ac08d73c60).
# La licencia no permite redistribuir el ráster: solo se guarda el valor en las estaciones.
CSV_SRTM = DIR_CAT / "altitud_srtm_cgiar_estaciones.csv"

# Salidas
BASE_PROC = RAIZ / "data" / "processed"
DIR_BAD = RAIZ / "data" / "baddata"
PROC = RAIZ / "data/processed"
SAL = PROC / "item4"
FIG = RAIZ / "figures/item4_red"
CSV_PRED = RAIZ / "predicciones_estaciones_reservadas.csv"
CSV_BUSQUEDA = SAL / "busqueda_hiperparametros.csv"

# Periodo, partición cronológica y validación cruzada de ventana creciente
PERIODO_PROYECTO = (1981, 2003)
ANIO_BASE, ANIO_FIN = PERIODO_PROYECTO
PARTICION = {"entrenamiento": (1981, 1996), "prueba": (1997, 2003)}
PLIEGUES_CV = ((1989, 1990), (1991, 1992), (1993, 1994), (1995, 1996))
N_DIAS = 365
SEMILLA = 42
DOMINIO = "Valle del Cauca"


# Resultados del preprocesamiento que se quedan en memoria en lugar de escribirse; las corridas de sensibilidad y de fuga,
# que redirigen BASE_PROC a una carpeta temporal, no los pisan
TABLAS = {}


def guardar(nombre, objeto):
    if BASE_PROC == PROC:
        TABLAS[nombre] = objeto


def subcarpeta(nombre):
    # Carpeta de data/processed/ que le corresponde a cada archivo del preprocesamiento
    if nombre.startswith(("validacion_", "sensibilidad_", "diag_", "prueba_fuga")):
        return "validacion"
    if nombre in ("precip_diaria_depurada.csv", "precip_diaria_imputada.csv", "registros_29_febrero.csv"):
        return "datos_diarios"
    if nombre.startswith(("acumulados", "correcciones_candidatas")):
        return "acumulados"
    if nombre.startswith("tabla_trazabilidad") or nombre == "resumen_cobertura.csv":
        return "trazabilidad"
    if nombre in ("pentadas_dominio_365.csv", "satelite_pentadal_mm.csv", "pixeles_malla.csv"):
        return "satelite"
    if nombre.startswith("reglas"):
        return "documentacion"
    return "metadatos"


def ruta(nombre):
    # Ruta de un archivo del preprocesamiento bajo BASE_PROC (que la depuración puede redirigir a una carpeta temporal)
    carpeta = BASE_PROC / subcarpeta(nombre)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / nombre
