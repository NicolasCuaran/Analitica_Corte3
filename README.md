# Proyecto final de Analítica de Datos 2026-2S

Precipitación del Valle del Cauca, 1981–2003, con datos funcionales y una red neuronal funcional que corrige el sesgo de la lluvia satelital, con tres cabezas comparadas en igualdad de condiciones: lineal, MLP y KAN.

**Integrantes:** Nicolás Santiago Cuarán Sotelo, Juan José Orozco, Yaxul Santiago Cárdenas y David Felipe Martínez Viracacha.
**Curso:** Analítica de Datos, Universidad Autónoma de Occidente.

## Contenido

```
red_funcional/notebooks/red_funcional.ipynb   Preprocesamiento y red funcional, lo primero
red_funcional/src/                            Módulos del notebook 
script_analitica_fda.R                        Construcción de curvas, análisis exploratorio, comparación por fuente e inferencia funcional
predicciones_estaciones_reservadas.csv        Predicciones de la red funcional (station_id, year, day, value_mm) de la cabeza elegida por validación cruzada
documentos/                                   Informe (Word y PDF)
documentos/figuras/                           Diagrama de la arquitectura de la red funcional
data/raw/                                     Datos entregados con la consigna y catálogos abiertos 
data/processed/                               Lo que escribe el notebook: insumos del script de R y caché de la búsqueda de hiperparámetros

figures/                                      Figuras, una carpeta por tema
```

| Módulo de `red_funcional/src/` | Contenido |
|---|---|
|  |  |
| `oni.py` | ONI de la NOAA y fase ENSO de cada año |
| `catalogos.py` | Coordenadas, altitud, dominio del Valle y emparejamiento estación-píxel |
| `satelite.py` | Lector de los GeoTIFF y totales pentadales por píxel |
| `depuracion.py` | Reglas de depuración, estaciones ciegas, imputación, niveles de calidad y vistas por tarea |
| `validacion.py` | Diagnósticos de las reglas, enmascaramiento, acumulados, sensibilidad y prueba de fuga |
| `preprocesamiento.py` | Orden del preprocesamiento y caché en `data/processed/` |
| `datos.py` | Vistas del preprocesamiento para la red y registro de verificaciones |
| `bases.py` | B-splines, matrices de Gram, MCO enmascarado y K por VCG |
| `curvas.py` | Curvas funcionales, FPCA, pliegues y referencia de las fechas de los picos |
| `modelo.py` | Capa KAN, cabezas lineal, MLP y KAN y red funcional |
| `entrenamiento.py` | Prueba sintética de la KAN, búsqueda de hiperparámetros, ajuste final y diagnósticos |
| `evaluacion.py` | Métricas, bootstrap por estación, aporte de las covariables, predicciones y verificaciones |
| `interpretacion.py` | Funciones de peso, aristas de la KAN, ajuste simbólico, dependencia parcial y fases ENSO |
| `figuras.py` | Figuras de `figures/item4_red/` |

| Tema | Figuras | Tablas |
|---|---|---|
| Preprocesamiento | `figures/item3_comparacion/mapa_cobertura_valle.png` | Objetos del notebook (trazabilidad, umbrales y validaciones) |
| Construcción de curvas | `figures/item1_construccion/` | Objetos del script de R |
| Análisis exploratorio | `figures/item2_exploratorio/` | Objetos del script de R |
| Comparación por fuente | `figures/item3_comparacion/` | Objetos del script de R |
| Red funcional | `figures/item4_red/` | Objetos del notebook |



## Instalación

Python 3.12 y R 4.6, con las versiones verificadas:

```
pip install numpy==1.26.4 pandas==2.3.3 scipy==1.15.3 matplotlib==3.11.2 openpyxl==3.1.5 nbformat==5.11.1 nbclient==0.11.0 ipykernel==7.4.0
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

En R:

```r
install.packages(c("fda", "fda.usc", "fdANOVA"))
```

## Datos de entrada

La entrega no incluye la data cruda ni `data/processed/`: el notebook recrea todo desde `data/raw/`. Antes de ejecutar, se copia la data cruda con estas rutas y nombres:

```
data/raw/estaciones/P_1980_1990.csv
data/raw/estaciones/P_1991_2003.csv
data/raw/catalogos/CVC_Wheather_Stations_Long_Lat.csv
data/raw/catalogos/CatalogoEstacionesElevacion.xlsx
data/raw/satelite/ImagenesSatelitePentadal/ImagenesSatelitePentadal/pptAAAAPP.tif   (imágenes descomprimidas)
```


## Ejecución

1. Abrir `red_funcional/notebooks/red_funcional.ipynb` con un kernel de Python que tenga torch y usar «Ejecutar todo». Los parámetros van al inicio del notebook:
   - `PREPROCESAR = True` rehace el preprocesamiento desde `data/raw/` (≈ 20 min); con `False`, reutiliza `data/processed/` si está completo.
   - `BUSCAR = True` repite la búsqueda de hiperparámetros; con `False`, la lee de `data/processed/item4/busqueda_hiperparametros.csv`.
2. Después, `Rscript script_analitica_fda.R`, que lee solo `data/processed/` y escribe solo figuras.

Sin interfaz, desde la raíz del repositorio y con el Python que tiene torch:

```
python -c "import nbformat, nbclient; nb=nbformat.read('red_funcional/notebooks/red_funcional.ipynb', 4); nbclient.NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': 'red_funcional/notebooks'}}).execute(); nbformat.write(nb, 'red_funcional/notebooks/red_funcional.ipynb')"
Rscript script_analitica_fda.R
```

- Las rutas se calculan desde la ubicación de los módulos y del script de R, así que las salidas quedan en el mismo lugar desde cualquier carpeta.
- El 22-nov-2026, el notebook completo, con el preprocesamiento desde `data/raw/` y la búsqueda de hiperparámetros guardada, tardó unos 29–31 min, y el script de R unos 5 min.
- El notebook y el script de R no se llaman entre sí: se comunican solo por los archivos de `data/processed/`.
- Semillas fijas: 42 en R y Python; 42, 123 y 999 en el ajuste final y la ablación de la red funcional.

**Versiones verificadas.** Python 3.12.10 con numpy 1.26.4, pandas 2.3.3, scipy 1.15.3, matplotlib 3.11.2, openpyxl 3.1.5, torch 2.15.0 (CPU), nbformat 5.11.1, nbclient 0.11.0 e ipykernel 7.4.0. R 4.6.1 con fda 6.3.0, fda.usc 2.2.0 y fdANOVA 0.1.3; la construcción de curvas usa solo R base.

## Datos

- **Estaciones:** precipitación diaria de la red de la CVC (`data/raw/estaciones/`). Las coordenadas salen de la consigna (167 estaciones) y del catálogo abierto de la CVC (45); la altitud, de la consigna (153), del catálogo abierto del IDEAM (32) y del SRTM de 90 m del CGIAR (27). Todo está en `data/raw/catalogos/`; del SRTM solo se guarda la altitud interpolada en cada estación.
- **Altitud del IDEAM contrastada con el SRTM:** se reemplaza por la del SRTM cuando las dos difieren más que el mayor error del SRTM en las estaciones de control, donde el IDEAM confirma la altitud de la consigna (678,3 m). Solo cambia EL ALCAZAR (2614500102): de 17 m a 977,2 m (objeto `ctx.preproceso["altitud_ideam_contraste_srtm"]` del notebook).
- **Satélite:** producto pentadal de 0,05° (`data/raw/satelite/`).
- **ONI:** tabla oficial de NOAA CPC (`data/raw/oni/`).
- **Dominio:** límite GADM 4.1 del Valle del Cauca (`data/raw/catalogos/limites_departamentos_gadm41.csv`). Se usan 212 estaciones (44 quedan fuera del límite y 67 no tienen coordenadas en ningún catálogo) y los 686 píxeles del satélite con centro en el departamento.
- **Partición:** entrenamiento 1981–1996 con validación cruzada cronológica de 4 pliegues (valida 1989–90, 1991–92, 1993–94 y 1995–96, cada pliegue entrenando con los años anteriores) y prueba 1997–2003, con años El Niño, La Niña y neutros. Además, 31 estaciones ciegas que el modelo nunca ve, elegidas por bloques de altitud con semilla 42.

## Red funcional

- **Tres cabezas** (lineal, MLP y KAN) con el mismo codificador funcional, el mismo decodificador FPCA y la misma búsqueda de hiperparámetros; solo cambia la cabeza.
- **CSV de predicciones:** lo produce la cabeza que gana la validación cruzada con la regla de empate de la búsqueda, nunca la prueba. En la corrida del 22-nov-2026 es el MLP (media de 3 semillas); la comparación queda en el objeto `seleccion` del notebook. Hoy contiene las 31 estaciones ciegas.
