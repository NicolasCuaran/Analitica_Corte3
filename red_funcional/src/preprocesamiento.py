"""Preprocesamiento desde data/raw/ con caché en data/processed/: satélite, depuración, validaciones y ONI.

Solo se escriben los insumos que leen la red y el R; los diagnósticos, las validaciones y la trazabilidad quedan en
comun.TABLAS (ctx.preproceso en la red) y solo existen en la corrida que rehace el preprocesamiento."""
import shutil
import time

from . import comun
from .comun import DIR_CAT, ruta
from .depuracion import depurar, exportar_baddata
from .oni import oni_41
from .satelite import satelite_a_csv
from .validacion import (auditoria, candidatas_decimas, diagnosticos, prueba_fuga, sensibilidad, validar_acumulados,
                         validar_faltantes)

PASOS = {"satelite": satelite_a_csv, "depurar": depurar, "exportar": exportar_baddata, "faltantes": validar_faltantes,
         "acumulados": validar_acumulados, "diagnosticos": diagnosticos, "auditoria": auditoria, "decimas": candidatas_decimas,
         "fuga": prueba_fuga, "sensibilidad": sensibilidad}

# Archivos de data/processed/ que, si existen todos, permiten saltar el preprocesamiento
INSUMOS_41 = [comun.BASE_PROC / p for p in ("datos_diarios/precip_diaria_depurada.csv", "metadatos/metadata_depuracion.csv",
                                           "metadatos/emparejamiento_estacion_pixel.csv", "metadatos/estaciones_fuera_del_valle.csv",
                                           "validacion/diag_catalogos_abiertos.csv", "datos_diarios/precip_diaria_imputada.csv",
                                           "metadatos/estaciones_candidatas_decimas.csv",
                                           "metadatos/eventos_depuracion.csv", "acumulados/acumulados_ventana.csv",
                                           "datos_diarios/registros_29_febrero.csv", "satelite/satelite_pentadal_mm.csv",
                                           "satelite/pentadas_dominio_365.csv", "satelite/pixeles_malla.csv", "oni_1981_2003.csv")]


def ejecutar(forzar=False):
    # Rehace el preprocesamiento si falta algún insumo o si se fuerza; siempre copia el límite GADM que lee el R
    faltan = [p for p in INSUMOS_41 if not p.exists()]
    if not forzar and not faltan:
        print("Preprocesamiento: data/processed/ completo, se reutiliza")
    else:
        print(f"Preprocesamiento desde data/raw/ ({'forzado' if forzar else f'faltan {len(faltan)} insumos'})")
        for paso, fn in PASOS.items():
            t0 = time.time()
            fn()
            print(f"  {paso}: {time.time() - t0:.0f} s")
        oni_41()
    shutil.copyfile(DIR_CAT / "limites_departamentos_gadm41.csv", ruta("limites_departamentos_gadm41.csv"))
