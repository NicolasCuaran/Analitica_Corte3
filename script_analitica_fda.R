#Precipitación del Valle del Cauca, 1981-2003 construcción de curvas, análisis exploratorio y comparación entre fuentes.
#. Rutas, insumos y funciones comunes

ARCHIVO_MAESTRO <- local({
  arg  <- commandArgs(trailingOnly = FALSE)
  arch <- sub("^--file=", "", arg[grep("^--file=", arg)])
  if (!length(arch)) arch <- head(unlist(lapply(rev(sys.frames()), function(f) if (is.character(f$ofile)) f$ofile)), 1L)   # «Source» de RStudio: archivo de source()
  if (length(arch) == 1L) normalizePath(arch) else normalizePath("script_analitica_fda.R")
})
setwd(dirname(ARCHIVO_MAESTRO))
ENTRADAS_MAESTRO <- c("data/processed/metadatos/metadata_depuracion.csv", "data/processed/datos_diarios/precip_diaria_depurada.csv",
                      "data/processed/datos_diarios/precip_diaria_imputada.csv", "data/processed/datos_diarios/registros_29_febrero.csv",
                      "data/processed/metadatos/eventos_depuracion.csv", "data/processed/acumulados/acumulados_ventana.csv",
                      "data/processed/satelite/satelite_pentadal_mm.csv", "data/processed/satelite/pentadas_dominio_365.csv",
                      "data/processed/satelite/pixeles_malla.csv",
                      "data/processed/metadatos/emparejamiento_estacion_pixel.csv", "data/processed/oni_1981_2003.csv",
                      "data/processed/metadatos/estaciones_candidatas_decimas.csv", "data/processed/metadatos/limites_departamentos_gadm41.csv",
                      "data/processed/metadatos/estaciones_fuera_del_valle.csv")
if (!all(file.exists(ENTRADAS_MAESTRO))) stop("Faltan entradas en data/processed/ (ejecute antes red_funcional/notebooks/red_funcional.ipynb, Ítem 4.1): ",
                                            paste(ENTRADAS_MAESTRO[!file.exists(ENTRADAS_MAESTRO)], collapse = ", "))
invisible(lapply(c("figures/item1_construccion", "figures/item2_exploratorio", "figures/item3_comparacion"), dir.create, showWarnings = FALSE, recursive = TRUE))

#Lluvia diaria: lo observado más los días imputados por IDW en los años con >= 80 % de días con dato
precip_diaria <- do.call(rbind, lapply(c("depurada", "imputada"), function(x)
  read.csv(sprintf("data/processed/datos_diarios/precip_diaria_%s.csv", x),
           colClasses = c("numeric", "integer", "NULL", "NULL", "integer", "NULL", "numeric"))))
metadata_depuracion <- read.csv("data/processed/metadatos/metadata_depuracion.csv", stringsAsFactors = FALSE)
satelite_pentadal   <- read.csv("data/processed/satelite/satelite_pentadal_mm.csv", na.strings = "NA")
pentadas_365        <- read.csv("data/processed/satelite/pentadas_dominio_365.csv")
pixeles_malla       <- read.csv("data/processed/satelite/pixeles_malla.csv")
emparejamiento      <- read.csv("data/processed/metadatos/emparejamiento_estacion_pixel.csv", stringsAsFactors = FALSE)
oni_anual           <- read.csv("data/processed/oni_1981_2003.csv", stringsAsFactors = FALSE)

#Funciones de nodos y base B-spline por la recursión de Cox-de Boor (nodos repetidos en los bordes)

nodos_bspline <- function(K, rango, grado) {
  c(rep(rango[1], grado + 1L), seq(rango[1], rango[2], length.out = K - grado + 1L)[-c(1L, K - grado + 1L)], rep(rango[2], grado + 1L))
}

base_nodos <- function(t, nodos, grado) {
  m <- length(nodos) - 1L; K <- m - grado
  B <- (outer(t, nodos[seq_len(m)], ">=") & outer(t, nodos[seq_len(m) + 1L], "<")) * 1
  B[t == nodos[m + 1L], ] <- 0
  B[t == nodos[m + 1L], K] <- 1
  elevar <- function(B, d) {
    j  <- seq_len(ncol(B) - 1L)
    d1 <- nodos[j + d] - nodos[j]
    d2 <- nodos[j + d + 1L] - nodos[j + 1L]
    sweep(outer(t, nodos[j], "-"), 2, ifelse(d1 > 0, 1 / d1, 0), "*") * B[, j, drop = FALSE] +
      sweep(outer(-t, -nodos[j + d + 1L], "-"), 2, ifelse(d2 > 0, 1 / d2, 0), "*") * B[, j + 1L, drop = FALSE]
  }
  Reduce(elevar, seq_len(grado), B)
}

#Función que arma la matriz D (K-1, K) con c' = D c, coeficientes de la derivada del spline
coef_derivada <- function(nodos, grado) {
  K <- length(nodos) - grado - 1L; i <- seq_len(K - 1L)
  w <- grado / (nodos[i + grado + 1L] - nodos[i + 1L])
  D <- matrix(0, K - 1L, K)
  D[cbind(i, i)] <- -ifelse(is.finite(w), w, 0)
  D[cbind(i, i + 1L)] <- ifelse(is.finite(w), w, 0)
  D
}

#Función que evalúa la base o su derivada de orden deriv: Phi^(r)(t) = B_{p-r}(t) D_r ... D_1
bspline_base <- function(t, K, rango = c(1, 365), grado = 3L, deriv = 0L) {
  nodos <- nodos_bspline(K, rango, grado); n <- length(nodos)
  if (deriv == 0L) return(base_nodos(t, nodos, grado))   # Dim: (length(t), K)
  Dtot <- Reduce(function(acc, k) coef_derivada(nodos[k:(n - k + 1L)], grado - k + 1L) %*% acc, seq_len(deriv), diag(K))
  base_nodos(t, nodos[(deriv + 1L):(n - deriv)], grado - deriv) %*% Dtot
}

#Matriz de rugosidad R = int Phi''(t) Phi''(t)' dt por la regla del trapecio
rugosidad <- function(K, rango = c(1, 365), grado = 3L, paso = 0.1) {
  malla <- seq(rango[1], rango[2], by = paso)
  w <- rep(paso, length(malla)); w[c(1L, length(malla))] <- paso / 2
  D2 <- bspline_base(malla, K, rango, grado, deriv = 2L)
  crossprod(D2 * sqrt(w))
}

#Producto fila a fila de Phi: permite G_p = Phi' diag(m_p) Phi para todas las curvas
kron_filas <- function(Phi) {
  K <- ncol(Phi)
  Phi[, rep(seq_len(K), times = K), drop = FALSE] * Phi[, rep(seq_len(K), each = K), drop = FALSE]
}

#Mínimos cuadrados: C = (Phi'Phi)^-1 Phi'Y, S = Phi (Phi'Phi)^-1 Phi'

#Ajuste con datos completos: una sola factorización para todas las curvas
ajuste_completo <- function(Phi, Y) {
  Ginv <- chol2inv(chol(crossprod(Phi)))
  C    <- Ginv %*% crossprod(Phi, Y)                 # Dim: (K, P)
  list(C = C, Ginv = Ginv, h = rowSums((Phi %*% Ginv) * Phi))
}

#Ajuste con días faltantes (NA): c_p = G_p^-1 Phi' y_p con G_p = Phi' diag(m_p) Phi, todas las curvas a la vez
#extra: observaciones fuera de la rejilla (29 de febrero, t = 59.5) que se suman a G_p y b_p
ajuste_enmascarado <- function(Phi, Y, extra = NULL) {
  K <- ncol(Phi); M <- !is.na(Y); Y0 <- ifelse(M, Y, 0); P <- ncol(Y)
  G <- crossprod(M * 1, kron_filas(Phi))             # Dim: (P, K^2)
  b <- crossprod(Y0, Phi)                            # Dim: (P, K)
  if (!is.null(extra)) {
    gx <- rowsum(kron_filas(extra$A), extra$p)
    bx <- rowsum(extra$A * extra$y, extra$p)
    ip <- as.integer(rownames(gx))
    G[ip, ] <- G[ip, ] + gx
    b[ip, ] <- b[ip, ] + bx
  }
  Ginv <- t(vapply(seq_len(nrow(G)), function(p) {
    R <- tryCatch(chol(matrix(G[p, ], K)), error = function(e) NULL)
    if (is.null(R) || min(diag(R)) < 1e-7 * max(diag(R))) rep(NA_real_, K * K) else as.vector(chol2inv(R))
  }, numeric(K * K)))                                # Dim: (P, K^2)
  C <- matrix(vapply(seq_len(nrow(G)), function(p) as.vector(matrix(Ginv[p, ], K) %*% b[p, ]), numeric(K)), nrow = K)
  sce_x <- numeric(P); n_x <- numeric(P)
  if (!is.null(extra)) {
    r <- extra$y - rowSums(extra$A * t(C)[extra$p, , drop = FALSE])
    sx <- rowsum(r^2, extra$p)
    sce_x[as.integer(rownames(sx))] <- sx[, 1]
    n_x <- tabulate(extra$p, P)
  }
  list(C = C, Ginv = Ginv, M = M, n = colSums(M), sce_x = sce_x, n_x = n_x,
       h = kron_filas(Phi) %*% t(Ginv))              # h: (N, P), apalancamiento s_ii de cada curva
}

#Suavizamiento penalizado: c = (Phi'Phi + lambda R)^-1 Phi'y, nu = tr(S_lambda)
ajuste_penalizado <- function(Phi, Y, R, lambda) {
  K <- ncol(Phi); M <- !is.na(Y); Y0 <- ifelse(M, Y, 0)
  G <- crossprod(M * 1, kron_filas(Phi)); b <- crossprod(Y0, Phi)
  Ainv <- t(vapply(seq_len(nrow(G)), function(p) tryCatch(as.vector(solve(matrix(G[p, ], K) + lambda * R)), error = function(e) rep(NA_real_, K * K)), numeric(K * K)))
  C <- t(vapply(seq_len(nrow(G)), function(p) as.vector(matrix(Ainv[p, ], K) %*% b[p, ]), numeric(K)))   # Dim: (P, K)
  E <- (Y0 - Phi %*% t(C)) * M
  list(C = t(C), nu = rowSums(Ainv * G), sce = colSums(E^2), n = colSums(M))
}

#Criterios por curva: ECM, VC con diag(S) y VCG
criterios_completo <- function(Phi, Y) {
  aj <- ajuste_completo(Phi, Y); K <- ncol(Phi); n <- nrow(Phi)
  E  <- Y - Phi %*% aj$C
  cbind(ECM = colMeans(E^2), VC = colMeans((E / (1 - aj$h))^2), VCG = colMeans(E^2) / (1 - K / n)^2)
}

criterios_enmascarado <- function(Phi, Y) {
  aj <- ajuste_enmascarado(Phi, Y); K <- ncol(Phi)
  E  <- (ifelse(aj$M, Y, 0) - Phi %*% aj$C) * aj$M
  d  <- ifelse(aj$M, 1 - aj$h, 1)
  cbind(ECM = colSums(E^2) / aj$n, VC = colSums((E / d)^2 * aj$M) / aj$n, VCG = colSums(E^2) / aj$n / (1 - K / aj$n)^2)
}

#Matriz S de un patrón de días observados y verificación de sus propiedades
matriz_S <- function(Phi) Phi %*% solve(crossprod(Phi), t(Phi))

verificar_S <- function(caso, Phi) {
  S <- matriz_S(Phi); K <- ncol(Phi)
  ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
  data.frame(caso = caso, n = nrow(Phi), K = K,
             error_idempotencia_F = norm(S %*% S - S, "F"),
             error_traza = abs(sum(diag(S)) - K),
             error_simetria = max(abs(S - t(S))),
             autovalores_cercanos_a_1 = sum(ev > 0.5),
             distancia_autovalores_a_01 = max(pmin(abs(ev), abs(ev - 1))),
             condicion_PhiTPhi = kappa(crossprod(Phi), exact = TRUE))
}

#Partición temporal: entrenamiento 1981-1996 (con los bloques de validación cruzada 1989-1996) y prueba 1997-2003
ANIO_ENTRENA <- 1996L
ANIOS_VALIDA <- c(1989L, 1996L)

NUCLEO <- c(ls(), "NUCLEO")

# Construcción de curvas 
rm(list = setdiff(ls(), NUCLEO))
set.seed(42)

dir.create("figures", showWarnings = FALSE)
res_construccion <- list()                             # tablas de resultados del bloque

N_DIAS    <- 365L
GRILLA_EST <- c(4:20, 22, 24, 26, 28, 30, 34, 38, 42, 48)
GRILLA_SAT <- c(4:20, 22, 24, 26, 28, 30, 34)   # K <= n/2 con n = 72 para que la VC esté definida
TOL_IDEMP <- 1e-12
B_BOOT    <- 200L
K_PEN     <- 20L
LAMBDAS_REL <- 10^seq(-4, 2, length.out = 10)

#Lectura de datos: estaciones (niveles A-C), nivel D y satélite pentadal, 1981-2003

meta_todo <- metadata_depuracion
meta_todo$clave <- meta_todo$station_id * 10000 + meta_todo$year
meta   <- meta_todo[as.logical(meta_todo$uso_construccion), ]
meta_D <- meta_todo[as.logical(meta_todo$uso_suavizado_penalizado), ]
meta$sigma2 <- NA_real_

d <- precip_diaria
armar_Y <- function(m) {
  idx <- match(d$station_id * 10000 + d$year, m$clave)
  Y <- matrix(NA_real_, N_DIAS, nrow(m))
  Y[cbind(d$doy[!is.na(idx)], idx[!is.na(idx)])] <- d$value_mm[!is.na(idx)]
  Y
}
Y_est <- armar_Y(meta)
Y_D   <- armar_Y(meta_D)
rm(d)
es_A <- meta$nivel == "A"

pent <- pentadas_365
sat  <- satelite_pentadal
stopifnot(all(sat$anio >= 1981L & sat$anio <= 2003L), all(table(sat$anio) == 72L))
n_px <- ncol(sat) - 2L; n_an <- length(unique(sat$anio))
nd   <- ifelse(sat$anio %% 4 == 0 & sat$pentada == 12, pent$n_dias_anio_bisiesto[sat$pentada], pent$n_dias[sat$pentada])
Y_sat <- matrix(as.matrix(sat[, -(1:2)]) / nd, nrow = 72L)         # Dim: (72, n_an * n_px), tasa mm/día
anio_sat  <- rep(sort(unique(sat$anio)), times = n_px)
pixel_sat <- rep(as.integer(sub("^px", "", names(sat)[-(1:2)])), each = n_an)   # número del píxel en la malla completa (solo celdas del Valle)
t_sat <- pent$t_baricentro
rm(sat)

#Rejilla diaria, base de referencia y derivada por diferencias centrales

t_dia <- as.numeric(seq_len(N_DIAS))
Phi_ref <- bspline_base(t_dia, 20L)
t_ver <- seq(3, 363, by = 0.5)
dif_central <- (bspline_base(t_ver + 1e-4, 20L) - bspline_base(t_ver - 1e-4, 20L)) / 2e-4

#Función de selección de K por ECM, VC y VCG y su gráfica

seleccion <- function(f_criterios, Y, t_obs, etiqueta, grilla, grado = 3L, rango = c(1, 365)) {
  crit <- lapply(grilla, function(K) f_criterios(bspline_base(t_obs, K, rango, grado), Y))
  por_curva <- vapply(c("ECM", "VC", "VCG"), function(m) vapply(crit, function(x) x[, m], numeric(ncol(Y))),
                      matrix(0, ncol(Y), length(grilla)))   # Dim: (P, nK, 3)
  validas <- apply(is.finite(por_curva), 1, all)
  tab <- data.frame(K = grilla,
                    ECM = colMeans(por_curva[validas, , "ECM"]), VC = colMeans(por_curva[validas, , "VC"]),
                    VCG = colMeans(por_curva[validas, , "VCG"]))
  list(tabla = tab, por_curva = por_curva[validas, , , drop = FALSE], validas = validas, grilla = grilla,
       K_VC = grilla[which.min(tab$VC)], K_VCG = grilla[which.min(tab$VCG)], etiqueta = etiqueta)
}

grafica_seleccion <- function(sel, archivo) {
  png(archivo, width = 1500, height = 480, res = 130)
  par(mfrow = c(1, 3), mar = c(4.2, 4.5, 3, 1))
  invisible(lapply(c("ECM", "VC", "VCG"), function(m) {
    y_m <- sel$tabla[[m]]
    plot(sel$tabla$K, y_m, type = "b", pch = 19, xlab = "Número de bases (K)", ylab = paste(m, "(mm²/día²)"),
         ylim = if (m == "ECM") range(y_m) else c(min(y_m), 1.5 * min(y_m)),   # la VC diverge cuando K se acerca a n/2
         main = sprintf("%s: %s", sel$etiqueta, c(ECM = "ECM de ajuste", VC = "Validación cruzada", VCG = "VCG")[m]))
    if (m != "ECM") {
      k_m <- sel$tabla$K[which.min(sel$tabla[[m]])]
      abline(v = k_m, lty = 3, col = "red"); abline(h = min(sel$tabla[[m]]), lty = 2, col = "red")
      mtext(sprintf("K = %d", k_m), side = 3, line = -1.5, col = "red", cex = 0.8)
    }
    grid()
  }))
  invisible(dev.off())
}

sel_est <- seleccion(criterios_enmascarado, Y_est[, es_A], t_dia, "Estaciones", GRILLA_EST)
sel_sat <- seleccion(criterios_completo, Y_sat, t_sat, "Satélite", GRILLA_SAT)
grafica_seleccion(sel_est, "figures/item1_construccion/seleccion_K_estaciones.png")
grafica_seleccion(sel_sat, "figures/item1_construccion/seleccion_K_satelite.png")
res_construccion$criterios_K_estaciones <- sel_est$tabla
res_construccion$criterios_K_satelite <- sel_sat$tabla

K_EST <- sel_est$K_VCG
K_SAT <- sel_sat$K_VCG
K_EST
K_SAT

#Sensibilidad a la base: lineal por tramos (orden 2) frente a cúbica (orden 4)

sel_est_lin <- seleccion(criterios_enmascarado, Y_est[, es_A], t_dia, "Estaciones, orden 2", GRILLA_EST, grado = 1L)
sel_sat_lin <- seleccion(criterios_completo, Y_sat, t_sat, "Satélite, orden 2", GRILLA_SAT, grado = 1L)

fila_base <- function(fuente, sel, orden, K_comun, tr_S) {
  i <- match(sel$K_VCG, sel$tabla$K); j <- match(K_comun, sel$tabla$K)
  data.frame(fuente = fuente, base = ifelse(orden == 2L, "lineal por tramos (orden 2)", "cúbica (orden 4)"), orden = orden, K_VCG = sel$K_VCG,
             tr_S = tr_S, ECM = sel$tabla$ECM[i], VC = sel$tabla$VC[i], VCG = sel$tabla$VCG[i], K_comun = K_comun, VCG_en_K_comun = sel$tabla$VCG[j],
             derivada = ifelse(orden == 2L, "discontinua (constante por tramos)", "continua (C2)"), seleccionada = orden == 4L)
}
tr_est <- function(K, grado) {
  aj <- ajuste_enmascarado(bspline_base(t_dia, K, grado = grado), Y_est[, es_A])
  mean(colSums(aj$h * aj$M)[is.finite(colSums(aj$h * aj$M))])
}
tr_sat <- function(K, grado) sum(ajuste_completo(bspline_base(t_sat, K, grado = grado), Y_sat[, 1:50])$h)
sens_bases <- rbind(fila_base("estaciones", sel_est_lin, 2L, K_EST, tr_est(sel_est_lin$K_VCG, 1L)),
                    fila_base("estaciones", sel_est, 4L, K_EST, tr_est(K_EST, 3L)),
                    fila_base("satelite", sel_sat_lin, 2L, K_SAT, tr_sat(sel_sat_lin$K_VCG, 1L)),
                    fila_base("satelite", sel_sat, 4L, K_SAT, tr_sat(K_SAT, 3L)))
res_construccion$sensibilidad_bases <- sens_bases

png("figures/item1_construccion/sensibilidad_bases.png", width = 1300, height = 520, res = 130)
par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
invisible(lapply(list(list(sel_est_lin, sel_est, "Estaciones"), list(sel_sat_lin, sel_sat, "Satélite")), function(z) {
  rr <- range(c(z[[1]]$tabla$VCG, z[[2]]$tabla$VCG)); lim <- c(rr[1], min(rr[2], 1.6 * rr[1]))
  plot(z[[1]]$tabla$K, z[[1]]$tabla$VCG, type = "b", pch = 17, col = "firebrick", ylim = lim, xlab = "Número de bases (K)",
       ylab = "VCG (mm²/día²)", main = paste("VCG por base:", z[[3]]))
  lines(z[[2]]$tabla$K, z[[2]]$tabla$VCG, type = "b", pch = 19, col = "navy")
  abline(v = c(z[[1]]$K_VCG, z[[2]]$K_VCG), lty = 3, col = c("firebrick", "navy"))
  legend("topright", c("orden 2 (lineal)", "orden 4 (cúbica)"), col = c("firebrick", "navy"), pch = c(17, 19), lty = 1, bty = "n", cex = 0.8)
  grid()
}))
invisible(dev.off())

#Sesgo y varianza según K, con la media funcional entre curvas como referencia

sesgo_varianza <- function(clima, t_obs, sel, n, K_ref = 30L) {
  ajustar <- function(K) { B <- bspline_base(t_obs, K); bspline_base(t_dia, K) %*% solve(crossprod(B), crossprod(B, clima)) }
  ref <- ajustar(K_ref)
  sesgo2 <- vapply(sel$grilla, function(K) mean((ajustar(K) - ref)^2), numeric(1))
  sigma2 <- sel$tabla$ECM * n / (n - sel$tabla$K)
  data.frame(K = sel$grilla, sesgo2 = sesgo2, varianza = sigma2 * sel$grilla / n, ECM_estimado = sesgo2 + sigma2 * sel$grilla / n)
}
sv_est <- sesgo_varianza(rowMeans(Y_est[, es_A], na.rm = TRUE), t_dia, sel_est, mean(colSums(!is.na(Y_est[, es_A]))))
sv_sat <- sesgo_varianza(rowMeans(Y_sat), t_sat, sel_sat, 72)
res_construccion$sesgo_varianza_estaciones <- sv_est
res_construccion$sesgo_varianza_satelite <- sv_sat
png("figures/item1_construccion/sesgo_varianza.png", width = 1300, height = 520, res = 130)
par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
invisible(lapply(list(list(sv_est, "Estaciones"), list(sv_sat, "Satélite")), function(z) {
  matplot(z[[1]]$K, z[[1]][, c("sesgo2", "varianza", "ECM_estimado")], type = "b", pch = 19, lty = 1, xlab = "Número de bases (K)",
          ylab = "mm²/día²", main = paste("Sesgo² + varianza:", z[[2]]))
  legend("top", c("sesgo²", "varianza", "suma"), col = 1:3, pch = 19, lty = 1, bty = "n", cex = 0.8)
}))
invisible(dev.off())

#Máximos y mínimos locales de la media funcional ajustada con el K elegido
extremos <- function(clima, t_obs, K) {
  B <- bspline_base(t_obs, K); x <- bspline_base(t_dia, K) %*% solve(crossprod(B), crossprod(B, clima))
  d <- diff(sign(diff(drop(x))))
  list(picos = which(d == -2) + 1L, valles = which(d == 2) + 1L)
}
ext_est <- extremos(rowMeans(Y_est[, es_A], na.rm = TRUE), t_dia, sel_est$K_VCG)
ext_sat <- extremos(rowMeans(Y_sat), t_sat, sel_sat$K_VCG)

#Estabilidad de la selección de K por remuestreo de curvas

boot_K <- function(sel) {
  VCG <- sel$por_curva[, , "VCG"]
  tabla <- table(vapply(seq_len(B_BOOT), function(b) sel$grilla[which.min(colMeans(VCG[sample.int(nrow(VCG), replace = TRUE), , drop = FALSE]))], numeric(1)))
  moda <- as.numeric(names(tabla)[which.max(tabla)])
  c(frecuencia_K_elegido = sum(tabla[names(tabla) == as.character(sel$K_VCG)]) / B_BOOT, moda = moda,
    K_optimo_por_curva_mediana = median(sel$grilla[apply(VCG, 1, which.min)]),
    VCG_relativo_K_vecino_inferior = VCG_rel(sel, -1), VCG_relativo_K_vecino_superior = VCG_rel(sel, 1))
}
VCG_rel <- function(sel, paso) {
  i <- which(sel$grilla == sel$K_VCG) + paso
  if (i < 1 || i > length(sel$grilla)) NA_real_ else sel$tabla$VCG[i] / min(sel$tabla$VCG) - 1
}
estab <- rbind(estaciones = boot_K(sel_est), satelite = boot_K(sel_sat))
res_construccion$estabilidad_K <- data.frame(fuente = rownames(estab), estab, K_VCG = c(K_EST, K_SAT))

#Verificación de S: idempotencia, traza, simetría y autovalores

incompletas <- which(es_A & colSums(!is.na(Y_est)) < N_DIAS)
incompletas <- incompletas[order(colSums(!is.na(Y_est[, incompletas])))]
obs_ej <- incompletas[c(1L, length(incompletas) %/% 2L, length(incompletas))]
verif <- do.call(rbind, c(list(
  verificar_S(sprintf("Estaciones, año completo (K = %d)", K_EST), bspline_base(t_dia, K_EST)),
  verificar_S(sprintf("Satélite, 72 péntadas (K = %d)", K_SAT), bspline_base(t_sat, K_SAT)),
  verificar_S("Estaciones, año completo (K = 48)", bspline_base(t_dia, 48L))),
  lapply(obs_ej, function(p) verificar_S(sprintf("Estaciones, patrón enmascarado (%d días, K = %d)", sum(!is.na(Y_est[, p])), K_EST),
                                         bspline_base(t_dia[!is.na(Y_est[, p])], K_EST)))))
res_construccion$verificaciones_S <- verif
verif

#Ajuste vectorizado frente a qr.solve curva a curva
p_ver <- sample.int(ncol(Y_est), 50L)
aj_ver <- ajuste_enmascarado(bspline_base(t_dia, K_EST), Y_est[, p_ver])
C_qr <- vapply(seq_along(p_ver), function(i) { o <- !is.na(Y_est[, p_ver[i]]); qr.solve(bspline_base(t_dia[o], K_EST), Y_est[o, p_ver[i]]) }, numeric(K_EST))
aj_ver <- ajuste_completo(bspline_base(t_sat, K_SAT), Y_sat[, 1:50])
C_qr <- qr.solve(bspline_base(t_sat, K_SAT), Y_sat[, 1:50])

#Ajuste final de todas las curvas, covarianza residual y bandas punto a punto

Phi_est <- bspline_base(t_dia, K_EST)

#El 29 de febrero entra como observación fuera de la rejilla (t = 59.5) salvo que un vecino esté retirado;
#los acumulados de varios días quedan como faltantes (NA) en su ventana
ev <- read.csv("data/processed/metadatos/eventos_depuracion.csv")
ac <- read.csv("data/processed/acumulados/acumulados_ventana.csv")
ac$clave <- ac$station_id * 10000 + ac$year
ac <- ac[ac$clave %in% meta$clave & !as.logical(ac$cruza_anio), ]

f29 <- read.csv("data/processed/datos_diarios/registros_29_febrero.csv")
f29$clave <- f29$station_id * 10000 + f29$year
f29$p <- match(f29$clave, meta$clave)
retirados <- paste(ev$station_id * 10000 + ev$year, ev$doy)
en_ventana <- c(paste(ac$clave, 59L)[ac$doy_ini <= 59L & ac$doy_fin >= 59L], paste(ac$clave, 60L)[ac$doy_ini <= 60L & ac$doy_fin >= 60L])
vecino_malo <- paste(f29$clave, 59L) %in% c(retirados, en_ventana) | paste(f29$clave, 60L) %in% c(retirados, en_ventana)
f29_apto <- !is.na(f29$p) & !vecino_malo & f29$value_mm <= 1000
n_f29_curvas <- sum(!is.na(f29$p))
f29 <- f29[f29_apto, ]

extra_est <- list(A = bspline_base(rep(59.5, nrow(f29)), K_EST), y = f29$value_mm, p = f29$p)
aj_base <- ajuste_enmascarado(Phi_est, Y_est)
aj_est  <- ajuste_enmascarado(Phi_est, Y_est, extra_est)
Phi_sat <- bspline_base(t_sat, K_SAT)
aj_sat  <- ajuste_completo(Phi_sat, Y_sat)

E_est <- (ifelse(aj_est$M, Y_est, 0) - Phi_est %*% aj_est$C) * aj_est$M     # residuos con 0 en los días faltantes
E_sat <- Y_sat - Phi_sat %*% aj_sat$C
sigma2_est <- (colSums(E_est^2) + aj_est$sce_x) / (aj_est$n + aj_est$n_x - K_EST)   # SCE / (n - K), con las observaciones del 29-feb
sigma2_sat <- colSums(E_sat^2) / (nrow(Phi_sat) - K_SAT)
estable_est <- is.finite(sigma2_est)

Phi_grilla <- bspline_base(t_dia, K_SAT)
s_sat <- rowSums((Phi_grilla %*% aj_sat$Ginv) * Phi_grilla)                    # Var[x(t)] / sigma^2 = diag(S) en t = 1..365
s_est <- aj_est$h                                                              # misma grilla: días 1..365

#Covarianza residual (1/N) sum_j e_j e_j'; en estaciones, por pares de días observados conjuntamente
Sigma_sat <- tcrossprod(E_sat) / ncol(E_sat)
Mn <- aj_est$M[, estable_est] * 1
Sigma_est <- tcrossprod(E_est[, estable_est]) / tcrossprod(Mn)                # Dim: (365, 365)
sd_dia_est <- sqrt(diag(Sigma_est)); sd_dia_sat <- sqrt(diag(Sigma_sat))
rho1 <- function(S) mean(S[cbind(1:(nrow(S) - 1), 2:nrow(S))] / sqrt(diag(S)[-nrow(S)] * diag(S)[-1]))

afect <- sort(unique(extra_est$p))
x0 <- Phi_est %*% aj_base$C[, afect, drop = FALSE]; x1 <- Phi_est %*% aj_est$C[, afect, drop = FALSE]
cambio <- sqrt(colSums((x1 - x0)^2) / colSums(x0^2))

#Valor esperado de la curva MCO y su error estándar en los días retirados de curvas A-C
rt <- ev[ev$destino == "retirado", ]
rt$p <- match(rt$station_id * 10000 + rt$year, meta$clave)
rt <- rt[!is.na(rt$p), ]
P_rt   <- bspline_base(as.numeric(rt$doy), K_EST)
est_rt <- rowSums(P_rt * t(aj_est$C)[rt$p, , drop = FALSE])
ee_rt  <- sqrt(sigma2_est[rt$p] * rowSums(kron_filas(P_rt) * aj_est$Ginv[rt$p, , drop = FALSE]))
restaurados <- data.frame(station_id = rt$station_id, year = rt$year, month = rt$month, day = rt$day, doy = rt$doy, regla = rt$regla,
                          valor_crudo_mm = rt$value_mm_crudo, estimado_mm = pmax(est_rt, 0), ee_curva_mm = ee_rt,
                          banda_inf_mm = pmax(est_rt - 2 * ee_rt, 0), banda_sup_mm = est_rt + 2 * ee_rt, truncado_en_cero = est_rt < 0)
res_construccion$restauracion_dias_retirados <- restaurados

s_med_est <- rowMeans(s_est[, estable_est])

#Truncamiento en cero solo al evaluar x(t) S, C y las derivadas se conservan sin truncar

x_est <- Phi_est %*% aj_est$C                                                  # x(t) = Phi(t) C, sin truncar, (365, P)
x_sat <- Phi_grilla %*% aj_sat$C
evaluar_fisico <- function(x) pmax(x, 0)

indicadores_trunc <- function(x) {
  neg <- x < 0
  m_neg <- colSums(pmax(-x, 0)); m_pos <- colSums(pmax(x, 0))
  c(curva_dias = length(x), pct_valores_truncados = 100 * mean(neg), pct_menores_que_menos_1 = 100 * mean(x < -1), minimo_mm_dia = min(x),
    curvas = ncol(x), pct_curvas_con_truncamiento = 100 * mean(colSums(neg) > 0),
    masa_truncada_sobre_masa_positiva_global_pct = 100 * sum(m_neg) / sum(m_pos),
    masa_truncada_mediana_curvas_afectadas_pct = 100 * median((m_neg / m_pos)[colSums(neg) > 0]),
    pct_truncados_en_bordes_t_le_15_o_t_ge_351 = 100 * sum(neg[t_dia <= 15 | t_dia >= 351, ]) / sum(neg))
}
tab_trunc <- data.frame(indicador = names(indicadores_trunc(x_est[, estable_est])),
                        estaciones = unname(indicadores_trunc(x_est[, estable_est])), satelite = unname(indicadores_trunc(x_sat)))
res_construccion$tabla_truncamiento <- tab_trunc
borde <- t_dia <= 15 | t_dia >= 351
tot_obs <- colSums(Y_sat * pent$n_dias)

#Nivel D suavizamiento penalizado con lambda por VCG

Phi_pen <- bspline_base(t_dia, K_PEN)
R_pen   <- rugosidad(K_PEN)
escala  <- sum(diag(crossprod(Phi_pen))) / sum(diag(R_pen))
ajustes_pen <- lapply(LAMBDAS_REL * escala, function(l) ajuste_penalizado(Phi_pen, Y_D, R_pen, l))
vcg_pen <- vapply(ajustes_pen, function(a) (a$sce / a$n) / (1 - a$nu / a$n)^2, numeric(ncol(Y_D)))   # Dim: (P_D, nLambda)
lam_ok  <- apply(is.finite(vcg_pen), 1, all)
tab_pen <- data.frame(log10_lambda_rel = log10(LAMBDAS_REL), VCG = colMeans(vcg_pen[lam_ok, ]),
                      nu_medio = vapply(ajustes_pen, function(a) mean(a$nu[lam_ok]), numeric(1)))
i_pen <- which.min(tab_pen$VCG)
aj_pen <- ajustes_pen[[i_pen]]
sigma2_pen <- aj_pen$sce / (aj_pen$n - aj_pen$nu)
x_pen <- Phi_pen %*% aj_pen$C
res_construccion$criterios_lambda_nivel_D <- tab_pen
coef_D <- cbind(meta_D[, c("station_id", "year", "nivel", "cobertura")], sigma2 = sigma2_pen, nu = aj_pen$nu, log10_lambda_rel = tab_pen$log10_lambda_rel[i_pen],
                t(aj_pen$C))   # coeficientes del nivel D en las columnas 8 a 7 + K_PEN
png("figures/item1_construccion/nivel_D_lambda.png", width = 1300, height = 520, res = 130)
par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
plot(tab_pen$log10_lambda_rel, tab_pen$VCG, type = "b", pch = 19, xlab = "log10(lambda relativo)", ylab = "VCG (mm²/día²)", main = "Nivel D: VCG frente a lambda")
abline(v = tab_pen$log10_lambda_rel[i_pen], lty = 3, col = "red"); grid()
plot(tab_pen$log10_lambda_rel, tab_pen$nu_medio, type = "b", pch = 19, xlab = "log10(lambda relativo)", ylab = "nu = tr(S_lambda)", main = "Grados de libertad efectivos")
abline(v = tab_pen$log10_lambda_rel[i_pen], lty = 3, col = "red"); grid()
invisible(dev.off())

#K  selección con el entrenamiento 1981-1996 sin estaciones ciegas los bloques de validación solo contrastan

en_train <- as.logical(meta$uso_entrenamiento)
en_valid <- en_train & meta$year >= ANIOS_VALIDA[1]
sat_train <- anio_sat <= ANIO_ENTRENA
sat_valid <- anio_sat >= ANIOS_VALIDA[1] & anio_sat <= ANIOS_VALIDA[2]
sel_est4 <- seleccion(criterios_enmascarado, Y_est[, en_train], t_dia, "Estaciones, entrenamiento 1981-1996", GRILLA_EST)
sel_sat4 <- seleccion(criterios_completo, Y_sat[, sat_train], t_sat, "Satélite, entrenamiento 1981-1996", GRILLA_SAT)
val_est4 <- seleccion(criterios_enmascarado, Y_est[, en_valid], t_dia, "Estaciones, bloques de validación cruzada 1989-1996", GRILLA_EST)
val_sat4 <- seleccion(criterios_completo, Y_sat[, sat_valid], t_sat, "Satélite, bloques de validación cruzada 1989-1996", GRILLA_SAT)
grafica_seleccion(sel_est4, "figures/item1_construccion/seleccion_K_item4_estaciones.png")
grafica_seleccion(sel_sat4, "figures/item1_construccion/seleccion_K_item4_satelite.png")
estab4 <- rbind(estaciones = boot_K(sel_est4), satelite = boot_K(sel_sat4))
meseta <- function(sel) { r <- range(sel$tabla$K[sel$tabla$VCG <= 1.01 * min(sel$tabla$VCG)]); paste(r, collapse = "-") }
costo_K_completo <- function(sel, K) sel$tabla$VCG[match(K, sel$tabla$K)] / min(sel$tabla$VCG) - 1
k_item4 <- data.frame(
  fuente = c("estaciones", "satelite"),
  K_item4_VCG_entrenamiento = c(sel_est4$K_VCG, sel_sat4$K_VCG),
  K_VC_entrenamiento = c(sel_est4$K_VC, sel_sat4$K_VC),
  K_VCG_validacion = c(val_est4$K_VCG, val_sat4$K_VCG),
  K_items_1_a_3_periodo_completo = c(K_EST, K_SAT),
  meseta_VCG_1pct = c(meseta(sel_est4), meseta(sel_sat4)),
  costo_VCG_con_K_completo_en_entrenamiento = c(costo_K_completo(sel_est4, K_EST), costo_K_completo(sel_sat4, K_SAT)),
  curvas_entrenamiento = c(sum(sel_est4$validas), sum(sel_sat4$validas)),
  curvas_validacion = c(sum(val_est4$validas), sum(val_sat4$validas)),
  estaciones_ciegas_excluidas = c(length(unique(meta$station_id[as.logical(meta$ciega)])), NA),
  frecuencia_K_elegido_bootstrap = estab4[, "frecuencia_K_elegido"])
res_construccion$k_item4 <- k_item4
res_construccion$criterios_K_item4_estaciones <- sel_est4$tabla
res_construccion$criterios_K_item4_satelite <- sel_sat4$tabla

#Derivada analítica Phi'(t) C frente a diferencias centrales

C_ej   <- aj_est$C[, estable_est][, 1:200]
d_anal <- bspline_base(t_ver, K_EST, deriv = 1L) %*% C_ej
d_num  <- (bspline_base(t_ver + 1e-4, K_EST) - bspline_base(t_ver - 1e-4, K_EST)) %*% C_ej / 2e-4

#Sensibilidad de las curvas de estaciones al K
resumen_K <- function(K) {
  Phi <- bspline_base(t_dia, K); aj <- ajuste_enmascarado(Phi, Y_est)
  E <- (ifelse(aj$M, Y_est, 0) - Phi %*% aj$C) * aj$M
  sig2 <- colSums(E^2) / (aj$n - K); ok <- is.finite(sig2); x <- Phi %*% aj$C
  ext <- extremos(rowMeans(Y_est[, es_A], na.rm = TRUE), t_dia, K)
  data.frame(K = K, curvas_estables = sum(ok), sigma_mediana = sqrt(median(sig2[ok])),
             ancho_banda = mean(2 * sqrt(sig2[ok]) * colMeans(sqrt(pmax(aj$h[, ok], 0)))),
             pct_negativos = 100 * mean(x[, ok] < 0), pct_negativos_bordes = 100 * mean(x[t_dia <= 15 | t_dia >= 351, ok] < 0),
             maximos_media_funcional = paste(ext$picos, collapse = "-"))
}
sens_K <- do.call(rbind, lapply(c(6, 7, 8, 11), resumen_K))
res_construccion$sensibilidad_K_estaciones <- sens_K

png("figures/item1_construccion/base_bspline.png", width = 1400, height = 520, res = 130)
par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
matplot(t_dia, bspline_base(t_dia, K_EST), type = "l", lty = 1, xlab = "Día del año", ylab = "B-spline", main = sprintf("Base cúbica, K = %d", K_EST))
matplot(t_dia, bspline_base(t_dia, K_SAT), type = "l", lty = 1, xlab = "Día del año", ylab = "B-spline", main = sprintf("Base cúbica, K = %d (satélite)", K_SAT))
invisible(dev.off())

ej_est <- sample(which(estable_est & colSums(aj_est$M) >= 330L), 4L)
ej_sat <- sample(ncol(Y_sat), 4L)
png("figures/item1_construccion/curvas_bandas.png", width = 1600, height = 1000, res = 130)
par(mfrow = c(2, 4), mar = c(4.2, 4.5, 3, 1))
dibujar <- function(t_obs, y, x, sigma2, s, titulo) {
  b <- 2 * sqrt(sigma2 * pmax(s, 0))
  plot(t_obs, y, pch = 16, cex = 0.35, col = "grey55", xlab = "Día del año", ylab = "mm/día", main = titulo, ylim = range(c(y, x + b, x - b), na.rm = TRUE))
  polygon(c(t_dia, rev(t_dia)), c(x - b, rev(x + b)), col = adjustcolor("steelblue", 0.35), border = NA)
  lines(t_dia, x, col = "navy", lwd = 2)
}
invisible(lapply(ej_est, function(p) dibujar(t_dia, Y_est[, p], x_est[, p], sigma2_est[p], s_est[, p],
                                             sprintf("Est. %.0f - %d", meta$station_id[p], meta$year[p]))))
invisible(lapply(ej_sat, function(p) dibujar(t_sat, Y_sat[, p], x_sat[, p], sigma2_sat[p], s_sat,
                                             sprintf("Píxel %d - %d", pixel_sat[p], anio_sat[p]))))
invisible(dev.off())

png("figures/item1_construccion/covarianza_residual.png", width = 1700, height = 1000, res = 130)
par(mfrow = c(2, 3), mar = c(4.2, 4.5, 3, 1))
invisible(lapply(list(list(Sigma_est, "Estaciones", t_dia, sd_dia_est), list(Sigma_sat, "Satélite", t_sat, sd_dia_sat)), function(z) {
  C_cor <- z[[1]] / outer(sqrt(diag(z[[1]])), sqrt(diag(z[[1]])))
  image(z[[3]], z[[3]], C_cor, xlab = "t", ylab = "s", main = paste("Correlación residual:", z[[2]]), col = hcl.colors(30, "Blue-Red 3"), zlim = c(-1, 1))
  plot(z[[3]], z[[4]], type = "l", lwd = 2, xlab = "Día del año", ylab = "sd residual (mm/día)", main = paste("sd residual por día:", z[[2]]))
  hist(sqrt(if (z[[2]] == "Estaciones") sigma2_est else sigma2_sat), breaks = 40, main = paste("sigma por curva:", z[[2]]), xlab = "mm/día", col = "grey80")
}))
invisible(dev.off())

#Coeficientes de las curvas que usan los bloques siguientes
coef_est <- list(C = aj_est$C, K = K_EST, curvas = meta[, c("station_id", "year", "nivel")], sigma2 = sigma2_est)
coef_sat <- list(C = aj_sat$C, K = K_SAT, pixel = pixel_sat, anio = anio_sat, sigma2 = sigma2_sat)
res_construccion$resumen_item1 <- data.frame(
  fuente = c("estaciones", "estaciones_nivel_D_penalizado", "satelite"),
  K = c(K_EST, K_PEN, K_SAT), curvas = c(sum(estable_est), sum(lam_ok), ncol(Y_sat)),
  sigma_mediana = c(sqrt(median(sigma2_est, na.rm = TRUE)), sqrt(median(sigma2_pen[lam_ok])), sqrt(median(sigma2_sat))),
  rho1 = c(rho1(Sigma_est), NA, rho1(Sigma_sat)),
  rho1_nulo = c(rho1(diag(N_DIAS) - matriz_S(Phi_est)), NA, rho1(diag(nrow(Phi_sat)) - matriz_S(Phi_sat))))   # residuo (I - S) e con e iid

#Ruta penalizada: base cúbica amplia (K = 55) y lambda por VCG, con VCG = (SCE/n)/(1 - nu/n)^2;
#es la curva de referencia de las fechas de las temporadas

K_RUTA <- 55L
LAMBDAS_RUTA_REL <- 10^seq(-3, 3, by = 0.25)
Phi_ruta <- bspline_base(t_dia, K_RUTA)                                   # Dim: (365, 55)
R_ruta   <- rugosidad(K_RUTA)                                             # Dim: (55, 55)
escala_ruta <- sum(diag(crossprod(Phi_ruta))) / sum(diag(R_ruta))

#Función de ajuste penalizado para varios lambda: G_p y b_p se calculan una vez y nu_p = tr(A_p^-1 G_p)
penalizado_lote <- function(Phi, Y, R, lambdas) {
  K <- ncol(Phi); M <- !is.na(Y); Y0 <- ifelse(M, Y, 0); n <- colSums(M)
  G <- crossprod(M * 1, kron_filas(Phi))                                  # Dim: (P, K^2)
  b <- crossprod(Y0, Phi)                                                 # Dim: (P, K)
  lapply(lambdas, function(l) {
    sol <- vapply(seq_len(nrow(G)), function(p) {
      Gp <- matrix(G[p, ], K)
      r <- tryCatch(solve(Gp + l * R, cbind(b[p, ], Gp)), error = function(e) NULL)
      if (is.null(r)) rep(NA_real_, K + 1L) else c(r[, 1L], sum(diag(r[, -1L, drop = FALSE])))
    }, numeric(K + 1L))                                                   # Dim: (K + 1, P)
    C <- sol[seq_len(K), , drop = FALSE]
    E <- (Y0 - Phi %*% C) * M
    list(C = C, nu = sol[K + 1L, ], sce = colSums(E^2), n = n, lambda = l)
  })
}

aj_ruta_A <- penalizado_lote(Phi_ruta, Y_est[, es_A], R_ruta, LAMBDAS_RUTA_REL * escala_ruta)
vcg_ruta  <- vapply(aj_ruta_A, function(a) (a$sce / a$n) / (1 - a$nu / a$n)^2, numeric(sum(es_A)))   # Dim: (P_A, nLambda)
nu_ruta   <- vapply(aj_ruta_A, function(a) a$nu, numeric(sum(es_A)))
#Comparación con la ruta K sobre las mismas curvas de nivel A estables en ambas grillas
comunes <- which(sel_est$validas)[apply(is.finite(vcg_ruta[sel_est$validas, , drop = FALSE]), 1, all)]
fila_K  <- match(comunes, which(sel_est$validas))
tab_ruta <- data.frame(log10_lambda_rel = log10(LAMBDAS_RUTA_REL), VCG = colMeans(vcg_ruta[comunes, , drop = FALSE]),
                       nu_medio = colMeans(nu_ruta[comunes, , drop = FALSE]))
i_ruta <- which.min(tab_ruta$VCG)
vcg_K_comunes <- colMeans(sel_est$por_curva[fila_K, , "VCG", drop = FALSE])
vcg_K_min <- min(vcg_K_comunes); K_vcg_comunes <- sel_est$grilla[which.min(vcg_K_comunes)]
lambda_ruta <- LAMBDAS_RUTA_REL[i_ruta] * escala_ruta
res_construccion$criterios_lambda_ruta_penalizada <- tab_ruta

#Ajuste de todas las curvas A-C con el lambda elegido (sin el 29 de febrero)
aj_ruta <- penalizado_lote(Phi_ruta, Y_est, R_ruta, lambda_ruta)[[1]]
coef_pen <- list(C = aj_ruta$C, K = K_RUTA, lambda = lambda_ruta, log10_lambda_rel = tab_ruta$log10_lambda_rel[i_ruta], nu = aj_ruta$nu,
                 curvas = meta[, c("station_id", "year", "nivel")])

#Verificación de S_lambda en un año completo: simétrica, tr = nu, autovalores en [0, 1]
G_ruta <- crossprod(Phi_ruta)
S_ruta <- Phi_ruta %*% solve(G_ruta + lambda_ruta * R_ruta, t(Phi_ruta))
ev_ruta <- eigen((S_ruta + t(S_ruta)) / 2, symmetric = TRUE, only.values = TRUE)$values
verif_ruta <- data.frame(
  caso = "Estaciones, ruta penalizada, año completo (K = 55)", lambda_rel = LAMBDAS_RUTA_REL[i_ruta],
  nu = sum(diag(S_ruta)), error_traza_vs_formula = abs(sum(diag(S_ruta)) - sum(diag(solve(G_ruta + lambda_ruta * R_ruta, G_ruta)))),
  error_simetria = max(abs(S_ruta - t(S_ruta))), autovalor_min = min(ev_ruta), autovalor_max = max(ev_ruta),
  norma_SS_menos_S = norm(S_ruta %*% S_ruta - S_ruta, "F"),
  VCG_penalizada = tab_ruta$VCG[i_ruta], VCG_mejor_K = vcg_K_min, K_mejor_mismas_curvas = K_vcg_comunes,
  mejora_VCG_pct = 100 * (1 - tab_ruta$VCG[i_ruta] / vcg_K_min), curvas_comparadas = length(comunes),
  curvas_ajustadas = sum(colSums(!is.finite(aj_ruta$C)) == 0), curvas_total = ncol(Y_est))
res_construccion$verificacion_ruta_penalizada <- verif_ruta
stopifnot(verif_ruta$error_simetria <= 1e-12, verif_ruta$error_traza_vs_formula <= 1e-10, verif_ruta$autovalor_min > -1e-10, verif_ruta$autovalor_max < 1 + 1e-10)

png("figures/item1_construccion/ruta_penalizada.png", width = 1400, height = 520, res = 130)
par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
plot(tab_ruta$log10_lambda_rel, tab_ruta$VCG, type = "b", pch = 19, ylim = range(c(tab_ruta$VCG, vcg_K_min)),
     xlab = "log10(lambda relativo)", ylab = "VCG (mm²/día²)", main = "Ruta penalizada (K = 55): VCG")
abline(h = vcg_K_min, lty = 2, col = "firebrick"); abline(v = tab_ruta$log10_lambda_rel[i_ruta], lty = 3, col = "navy")
legend("topleft", c("VCG penalizada", sprintf("mejor ruta K (K = %d), mismas curvas", K_vcg_comunes)), col = c("black", "firebrick"),
       lty = c(1, 2), pch = c(19, NA), bty = "n", cex = 0.8); grid()
plot(tab_ruta$log10_lambda_rel, tab_ruta$nu_medio, type = "b", pch = 19, xlab = "log10(lambda relativo)", ylab = "nu = tr(S_lambda)",
     main = "Grados de libertad efectivos"); abline(v = tab_ruta$log10_lambda_rel[i_ruta], lty = 3, col = "navy"); grid()
invisible(dev.off())

#Resultados que pasan a los bloques siguientes
NUCLEO <- c(NUCLEO, "res_construccion", "coef_est", "coef_sat", "coef_D", "coef_pen")

# Análisis exploratorio (a mano y contrastado con fda y fda.usc)

rm(list = setdiff(ls(), NUCLEO))
set.seed(42)
suppressMessages({library(fda); library(fda.usc)})

dir.create("figures/item2_exploratorio", showWarnings = FALSE, recursive = TRUE)
res_exploratorio <- list()                             # tablas de resultados del bloque

N_DIAS        <- 365L
BORDE_INF     <- 14L                                   # bordes de apalancamiento [1, 14] y [352, 365]; soporte interior [15, 351]
BORDE_SUP     <- 352L
B_BOOT        <- 1000L
FACTOR_BIGOTE <- 1.5
PROP_CENTRAL  <- 0.5
GRILLA_K_EST  <- c(6L, 7L, 8L, 10L, 11L, 14L, 20L)
GRILLA_K_SAT  <- c(8L, 11L, 14L, 20L)
MES_INI <- c(1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
MES_ETQ <- c("E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")
MESES   <- c("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
COLORES <- c(estaciones = "#1f77b4", satelite = "#d95f02")
t_dia   <- as.numeric(seq_len(N_DIAS))
en_borde <- t_dia <= BORDE_INF | t_dia >= BORDE_SUP

fecha_doy <- function(t) {
  d <- as.Date(round(t) - 1, origin = "2001-01-01")
  ifelse(is.na(t), NA_character_, paste(format(d, "%d"), MESES[as.integer(format(d, "%m"))]))
}
guardar <- function(archivo, ancho, alto, expr) {
  png(archivo, width = ancho, height = alto, res = 130)
  on.exit(dev.off())
  expr
}
eje_meses <- function() axis(1, at = MES_INI, labels = MES_ETQ)
dif_rel   <- function(a, b) sqrt(sum((a - b)^2) / sum(b^2))

#Curvas del Ítem 1 (estaciones A-B y satélite) evaluadas en la rejilla diaria t = 1..365

est <- coef_est
sat <- coef_sat
usa <- est$curvas$nivel %in% c("A", "B")
id_e <- est$curvas[usa, ]
ids <- list(estaciones = data.frame(id = id_e$station_id, anio = id_e$year, nivel = id_e$nivel, stringsAsFactors = FALSE),
            satelite   = data.frame(id = sat$pixel, anio = sat$anio, nivel = NA_character_, stringsAsFactors = FALSE))
fuentes <- lapply(list(estaciones = list(C = est$C[, usa, drop = FALSE], K = est$K, grupo = id_e$station_id),
                       satelite   = list(C = sat$C, K = sat$K, grupo = sat$pixel)),
                  function(f) {
                    f$Phi <- bspline_base(t_dia, f$K)                    # Dim: (365, K)
                    f$Y   <- f$Phi %*% f$C                               # Dim: (365, N)
                    f$mu  <- drop(f$Phi %*% colMeans(t(f$C)))
                    f
                  })
d_diario <- precip_diaria
clave_e <- id_e$station_id * 10000 + id_e$year
idx <- match(d_diario$station_id * 10000 + d_diario$year, clave_e)
ok  <- !is.na(idx)
Y_diario <- matrix(NA_real_, N_DIAS, length(clave_e))                      # Dim: (365, N), NA donde se retiró el registro
Y_diario[cbind(d_diario$doy[ok], idx[ok])] <- d_diario$value_mm[ok]
rm(d_diario)

#Media funcional a mano frente a mean.fd() y func.mean()

verif_media <- t(vapply(fuentes, function(f) {
  base   <- create.bspline.basis(rangeval = c(1, N_DIAS), nbasis = f$K, norder = 4L)
  fd_obj <- fd(f$C, base)
  c(dif_base_fda = max(abs(eval.basis(t_dia, base) - f$Phi)),
    dif_mean_fd = max(abs(f$mu - drop(eval.fd(t_dia, mean.fd(fd_obj))))),
    dif_func_mean = max(abs(f$mu - drop(func.mean(fdata(t(f$Y), argvals = t_dia))$data))),
    dif_media_curvas = max(abs(f$mu - rowMeans(f$Y))))
}, numeric(4)))
stopifnot(all(verif_media[, c("dif_mean_fd", "dif_media_curvas")] <= 1e-12))

media_balanceada <- function(f) {
  S <- rowsum(t(f$C), f$grupo)
  n <- drop(rowsum(rep(1, ncol(f$C)), f$grupo))
  drop(f$Phi %*% colMeans(S / n))
}
fuentes <- lapply(fuentes, function(f) { f$mu_bal <- media_balanceada(f); f })
dif_balanceada <- vapply(fuentes, function(f) dif_rel(f$mu_bal, f$mu), numeric(1))

#Varianza, desviación estándar, coeficiente de variación, covarianza y correlación: v(s,t) = Phi(s)' Sigma_c Phi(t)

fuentes <- lapply(fuentes, function(f) {
  f$Sigma    <- cov(t(f$C))                                              # Dim: (K, K), divisor N - 1
  PhiS       <- f$Phi %*% f$Sigma
  f$G        <- tcrossprod(PhiS, f$Phi)                                  # Dim: (365, 365)
  f$varianza <- rowSums(PhiS * f$Phi)
  f$sd       <- sqrt(f$varianza)
  f$cv       <- f$sd / f$mu
  f$Rc       <- f$G / tcrossprod(f$sd)
  f
})
verif_cov <- t(vapply(fuentes, function(f) {
  fd_obj <- fd(f$C, create.bspline.basis(rangeval = c(1, N_DIAS), nbasis = f$K, norder = 4L))
  c(dif_cov_var_fd = max(abs(f$G - eval.bifd(t_dia, t_dia, var.fd(fd_obj)))) / max(abs(f$G)),
    dif_var_curvas = max(abs(f$varianza - apply(f$Y, 1L, var))) / max(f$varianza),
    dif_cor_fd = max(abs(f$Rc - cor.fd(t_dia, fd_obj))),
    asimetria = max(abs(f$G - t(f$G))) / max(abs(f$G)),
    autovalor_minimo_Sigma = min(eigen(f$Sigma, symmetric = TRUE, only.values = TRUE)$values))
}, numeric(5)))
stopifnot(all(verif_cov[, c("dif_cov_var_fd", "dif_var_curvas")] <= 1e-10))

resumen_var <- t(vapply(fuentes, function(f) {
  borde <- en_borde
  c(sd_min = min(f$sd), dia_sd_min = t_dia[which.min(f$sd)], sd_max = max(f$sd), dia_sd_max = t_dia[which.max(f$sd)],
    razon_var_borde_mediana = max(f$varianza[borde]) / median(f$varianza),
    cv_mediano = median(f$cv), cv_min = min(f$cv), cv_max = max(f$cv))
}, numeric(8)))

#Efecto de borde: inflación del apalancamiento en t = 1 y t = 365 frente a datos completos (umbral 4)
UMBRAL_BORDE <- 4
Phi_e  <- fuentes$estaciones$Phi
aj_b   <- ajuste_enmascarado(Phi_e, Y_diario)
h0     <- rowSums((Phi_e %*% chol2inv(chol(crossprod(Phi_e)))) * Phi_e)
infl   <- aj_b$h / h0                                                      # Dim: (365, N), inflación de la varianza del ajuste en cada día
infl_borde <- apply(infl[c(1L, N_DIAS), , drop = FALSE], 2L, max)
inestable  <- is.na(infl_borde) | infl_borde > UMBRAL_BORDE

#Meseta de valores altos: racha máxima de días consecutivos con >= 100 mm
UMBRAL_ALTO <- 100
racha_max <- function(x) { r <- rle(x); max(c(0L, r$lengths[r$values])) }
racha_alta <- apply(!is.na(Y_diario) & Y_diario >= UMBRAL_ALTO, 2L, racha_max)
meseta <- racha_alta >= 5L
id_meseta <- id_e[meseta, ]
res_exploratorio$mesetas_valores_altos <- cbind(id_e, racha_dias_100mm = racha_alta)[racha_alta >= 3L, ]

excluir    <- inestable | meseta
C_estable  <- fuentes$estaciones$C[, !excluir, drop = FALSE]
Sigma_est  <- cov(t(C_estable))
fuentes$estaciones$var_estable <- rowSums((Phi_e %*% Sigma_est) * Phi_e)
fuentes$estaciones$mu_estable  <- drop(Phi_e %*% rowMeans(C_estable))
fuentes$estaciones$var_ruido <- rowMeans(sweep(infl * h0, 2L, est$sigma2[usa], "*"), na.rm = TRUE)   # E[sigma_i^2 h_i(t)]: Var[x_hat(t)] = Var[x(t)] + E[sigma^2 h(t)]
ids$estaciones$infl_borde <- infl_borde; ids$estaciones$borde_inestable <- inestable; ids$estaciones$racha_100mm <- racha_alta
ids$satelite$infl_borde <- NA_real_;     ids$satelite$borde_inestable <- FALSE;   ids$satelite$racha_100mm <- NA_integer_

h_lag <- 0:182
corr_lag <- vapply(fuentes, function(f) vapply(h_lag, function(h) mean(f$Rc[cbind(seq_len(N_DIAS - h), seq_len(N_DIAS - h) + h)]), numeric(1)),
                   numeric(length(h_lag)))

#Derivadas: extremos y puntos de inflexión exactos de la media funcional

estructura_extremos <- function(K) {
  nodos <- unique(nodos_bspline(K, c(1, N_DIAS), 3L))
  a <- nodos[-length(nodos)]; b <- nodos[-1L]
  list(K = K, nodos = nodos, a = a, b = b, nI = length(a),
       D1 = bspline_base(c(a, (a + b) / 2, b), K, deriv = 1L),          # Dim: (3 nI, K)
       D2 = bspline_base(nodos, K, deriv = 2L))                         # Dim: (nodos, K)
}

#Función de extremos: en cada tramo x'(t) es cuadrática y sus raíces se resuelven exactamente
extremos_media <- function(cm, ex) {
  d <- drop(ex$D1 %*% cm); nI <- ex$nI
  fa <- d[seq_len(nI)]; fm <- d[nI + seq_len(nI)]; fb <- d[2L * nI + seq_len(nI)]
  p2 <- 2 * fb + 2 * fa - 4 * fm; p1 <- fb - fa - p2; p0 <- fa
  disc <- p1^2 - 4 * p2 * p0
  cuad <- abs(p2) > 1e-12 * pmax(1, abs(p1))
  raiz <- sqrt(pmax(disc, 0))
  u <- cbind(ifelse(cuad & disc >= 0, (-p1 + raiz) / (2 * p2), NA_real_),
             ifelse(cuad & disc >= 0, (-p1 - raiz) / (2 * p2), NA_real_),
             ifelse(!cuad & abs(p1) > 0, -p0 / p1, NA_real_))
  t_raiz <- ex$a + u * (ex$b - ex$a)
  t_r <- sort(unique(round(t_raiz[!is.na(u) & u >= -1e-12 & u <= 1 + 1e-12], 6)))
  d2 <- drop(bspline_base(t_r, ex$K, deriv = 2L) %*% cm)
  data.frame(t = t_r, tipo = ifelse(d2 < 0, "maximo", "minimo"), d2 = d2, stringsAsFactors = FALSE)
}

#Función de inflexiones: x''(t) es lineal a trozos y sus raíces se interpolan exactamente
inflexiones_media <- function(cm, ex) {
  v <- drop(ex$D2 %*% cm); n <- length(v)
  fa <- v[-n]; fb <- v[-1L]
  cruce <- fa * fb < 0
  t_i <- ex$a[cruce] + (ex$b[cruce] - ex$a[cruce]) * fa[cruce] / (fa[cruce] - fb[cruce])
  d1 <- drop(bspline_base(t_i, ex$K, deriv = 1L) %*% cm)
  data.frame(t = t_i, tipo = ifelse(d1 > 0, "ascenso_maximo", "descenso_maximo"), d1 = d1, stringsAsFactors = FALSE)
}

eventos_media <- function(cm, ex) {
  e <- extremos_media(cm, ex); i <- inflexiones_media(cm, ex)
  ev <- rbind(data.frame(evento = e$tipo, t = e$t), data.frame(evento = i$tipo, t = i$t))
  ev <- ev[order(ev$t), ]
  ev$fecha <- fecha_doy(ev$t)
  ev$mu_mm_dia <- drop(bspline_base(ev$t, ex$K) %*% cm)
  ev$dmu_mm_dia2 <- drop(bspline_base(ev$t, ex$K, deriv = 1L) %*% cm)
  rownames(ev) <- NULL
  ev
}

#Función de fechas clave: los dos picos más altos de la media y el valle más bajo entre ellos
fechas_clave <- function(cm, ex) {
  e <- extremos_media(cm, ex)
  mx <- e$t[e$tipo == "maximo"]; mn <- e$t[e$tipo == "minimo"]
  if (length(mx) < 2L) return(c(pico1 = NA_real_, valle = NA_real_, pico2 = NA_real_))
  top <- sort(mx[order(drop(bspline_base(mx, ex$K) %*% cm), decreasing = TRUE)[1:2]])
  v <- mn[mn > top[1L] & mn < top[2L]]
  c(pico1 = top[1L], valle = if (length(v)) v[which.min(drop(bspline_base(v, ex$K) %*% cm))] else NA_real_, pico2 = top[2L])
}

estruct <- lapply(fuentes, function(f) estructura_extremos(f$K))
eventos <- Map(function(f, ex) eventos_media(colMeans(t(f$C)), ex), fuentes, estruct)
dif_deriv <- vapply(fuentes, function(f) {
  D1 <- bspline_base(t_dia, f$K, deriv = 1L)
  max(abs(D1 %*% colMeans(t(f$C)) - rowMeans(D1 %*% f$C)))
}, numeric(1))
stopifnot(all(dif_deriv < 1e-10))

fechas_base <- t(vapply(seq_along(fuentes), function(i) fechas_clave(colMeans(t(fuentes[[i]]$C)), estruct[[i]]), numeric(3)))
rownames(fechas_base) <- names(fuentes)

#Fechas de picos y valle en cada curva
fechas_curvas <- Map(function(f, ex) t(apply(f$C, 2L, function(cm) fechas_clave(cm, ex))), fuentes, estruct)
resumen_fechas <- do.call(rbind, Map(function(m, n) {
  q <- t(apply(m, 2L, quantile, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE))
  data.frame(fuente = n, evento = colnames(m), p05 = q[, 1], p25 = q[, 2], mediana = q[, 3], p75 = q[, 4], p95 = q[, 5],
             pct_curvas_con_evento = 100 * colMeans(!is.na(m)), row.names = NULL)
}, fechas_curvas, names(fechas_curvas)))
res_exploratorio$fechas_por_curva_resumen <- cbind(resumen_fechas[, 1:2], round(resumen_fechas[, 3:8], 2))

#Incertidumbre de las fechas: bootstrap por estación o píxel y sensibilidad a K

bootstrap_fechas <- function(f, ex) {
  S  <- rowsum(t(f$C), f$grupo)
  nc <- drop(rowsum(rep(1, ncol(f$C)), f$grupo))
  W  <- rmultinom(B_BOOT, nrow(S), rep(1, nrow(S)))                     # Dim: (grupos, B)
  cm <- crossprod(W, S) / drop(crossprod(W, nc))                        # Dim: (B, K)
  t(vapply(seq_len(B_BOOT), function(b) fechas_clave(cm[b, ], ex), numeric(3)))
}
set.seed(42)
boot <- Map(bootstrap_fechas, fuentes, estruct)
ic_boot <- lapply(boot, function(b) apply(b, 2L, quantile, probs = c(0.025, 0.5, 0.975), na.rm = TRUE))

pent <- pentadas_365
sat_csv <- satelite_pentadal
nd <- ifelse(sat_csv$anio %% 4 == 0 & sat_csv$pentada == 12, pent$n_dias_anio_bisiesto[sat_csv$pentada], pent$n_dias[sat_csv$pentada])
y_pent_medio <- rowMeans(matrix(as.matrix(sat_csv[, -(1:2)]) / nd, nrow = 72L))   # Dim: (72), tasa mm/día media de todas las curvas
rm(sat_csv)
Phi_s <- fuentes$satelite$Phi
h_sat <- rowSums((Phi_s %*% solve(crossprod(bspline_base(pent$t_baricentro, fuentes$satelite$K)))) * Phi_s)   # leverage en la rejilla diaria con 72 péntadas
fuentes$satelite$var_ruido <- mean(sat$sigma2) * h_sat

#Apalancamiento por zona: estaciones con datos completos y satélite con 72 péntadas
apalancamiento_zona <- function(h, nombre, var_ruido, varianza) {
  base <- mean(h[!en_borde]); ok <- which(h <= 1.5 * base)
  data.frame(fuente = nombre, K = if (nombre == "estaciones") fuentes$estaciones$K else fuentes$satelite$K,
             leverage_medio_interior = base, leverage_medio_borde = mean(h[en_borde]), leverage_max_borde = max(h[en_borde]),
             razon_borde_interior = mean(h[en_borde]) / base, t_primer_dia_dentro_de_1p5 = t_dia[ok[1L]], t_ultimo_dia_dentro_de_1p5 = t_dia[ok[length(ok)]],
             pct_ruido_interior = 100 * median(var_ruido[!en_borde] / varianza[!en_borde]), pct_ruido_borde = 100 * median(var_ruido[en_borde] / varianza[en_borde]))
}
apalancamiento <- rbind(apalancamiento_zona(h0, "estaciones", fuentes$estaciones$var_ruido, fuentes$estaciones$varianza),
                        apalancamiento_zona(h_sat, "satelite", fuentes$satelite$var_ruido, fuentes$satelite$varianza))
res_exploratorio$apalancamiento_borde <- cbind(apalancamiento[, 1:2], round(apalancamiento[, 3:ncol(apalancamiento)], 5))

cm_K_est <- lapply(GRILLA_K_EST, function(K) rowMeans(ajuste_enmascarado(bspline_base(t_dia, K), Y_diario)$C, na.rm = TRUE))
cm_K_sat <- lapply(GRILLA_K_SAT, function(K) { Phi <- bspline_base(pent$t_baricentro, K); drop(solve(crossprod(Phi), crossprod(Phi, y_pent_medio))) })
dif_lineal_sat <- max(abs(cm_K_sat[[match(fuentes$satelite$K, GRILLA_K_SAT)]] - colMeans(t(fuentes$satelite$C))))
dif_29feb_est  <- max(abs(drop(bspline_base(t_dia, est$K) %*% (cm_K_est[[match(est$K, GRILLA_K_EST)]] - colMeans(t(fuentes$estaciones$C))))))

sens_K <- rbind(
  data.frame(fuente = "estaciones", K = GRILLA_K_EST, t(vapply(seq_along(GRILLA_K_EST), function(i) fechas_clave(cm_K_est[[i]], estructura_extremos(GRILLA_K_EST[i])), numeric(3)))),
  data.frame(fuente = "satelite", K = GRILLA_K_SAT, t(vapply(seq_along(GRILLA_K_SAT), function(i) fechas_clave(cm_K_sat[[i]], estructura_extremos(GRILLA_K_SAT[i])), numeric(3)))))

corr_picos <- t(vapply(names(fuentes), function(n) {
  p <- round(fechas_base[n, ]); R <- fuentes[[n]]$Rc
  c(r_pico1_pico2 = R[p["pico1"], p["pico2"]], r_pico1_valle = R[p["pico1"], p["valle"]], r_valle_pico2 = R[p["valle"], p["pico2"]])
}, numeric(3)))
res_exploratorio$sensibilidad_fechas_K <- cbind(sens_K[, 1:2], round(sens_K[, 3:5], 3))

#Función de profundidad por rangos: Fraiman-Muniz, D_FM = mean_t [1 - |1/2 - F_t(x_i(t))|], y MBD

profundidades <- function(Y) {
  n <- ncol(Y)
  r <- apply(Y, 1L, rank, ties.method = "average")                       # Dim: (N, T)
  cbind(FM = rowMeans(1 - abs(0.5 - r / n)), MBD = rowMeans((r - 1) * (n - r) + (n - 1)) / choose(n, 2))
}

#Función de boxplot funcional: región central con el 50 % más profundo y bigotes a 1.5 veces su ancho
boxplot_funcional <- function(Y, prof, factor = FACTOR_BIGOTE) {
  n <- ncol(Y)
  orden <- order(prof, decreasing = TRUE)
  central <- orden[seq_len(ceiling(PROP_CENTRAL * n))]
  inf <- apply(Y[, central, drop = FALSE], 1L, min); sup <- apply(Y[, central, drop = FALSE], 1L, max)
  cerca_inf <- inf - factor * (sup - inf); cerca_sup <- sup + factor * (sup - inf)
  E <- pmax(Y - cerca_sup, cerca_inf - Y, 0)                             # Dim: (T, N), exceso fuera de los bigotes en mm/día
  exceso <- apply(E, 2L, max)
  dia    <- apply(E, 2L, which.max)
  list(central = central, mediana = orden[1L], inf = inf, sup = sup, cerca_inf = cerca_inf, cerca_sup = cerca_sup,
       atipica = exceso > 0, exceso = exceso, dia = dia,
       direccion = ifelse(Y[cbind(dia, seq_len(n))] > cerca_sup[dia], "alto", "bajo"))
}

prof <- lapply(fuentes, function(f) profundidades(f$Y))
caja_mbd <- Map(function(f, p) boxplot_funcional(f$Y, p[, "MBD"]), fuentes, prof)
caja_fm  <- Map(function(f, p) boxplot_funcional(f$Y, p[, "FM"]), fuentes, prof)

#Atipicidad de forma cada curva dividida por su media anual
Y_forma    <- lapply(fuentes, function(f) sweep(f$Y, 2L, colMeans(f$Y), "/"))
prof_forma <- lapply(Y_forma, profundidades)
caja_forma <- Map(function(Yf, p) boxplot_funcional(Yf, p[, "MBD"]), Y_forma, prof_forma)
factores   <- c(1, 1.5, 2, 3)
sens_factor <- data.frame(factor = factores,
                          estaciones_nivel = vapply(factores, function(fc) sum(boxplot_funcional(fuentes$estaciones$Y, prof$estaciones[, "MBD"], fc)$atipica), numeric(1)),
                          satelite_nivel = vapply(factores, function(fc) sum(boxplot_funcional(fuentes$satelite$Y, prof$satelite[, "MBD"], fc)$atipica), numeric(1)),
                          estaciones_forma = vapply(factores, function(fc) sum(boxplot_funcional(Y_forma$estaciones, prof_forma$estaciones[, "MBD"], fc)$atipica), numeric(1)),
                          satelite_forma = vapply(factores, function(fc) sum(boxplot_funcional(Y_forma$satelite, prof_forma$satelite[, "MBD"], fc)$atipica), numeric(1)))
res_exploratorio$sensibilidad_factor_bigote <- sens_factor

set.seed(42)
verif_prof <- t(vapply(names(fuentes), function(nombre) {
  Y <- fuentes[[nombre]]$Y
  sub <- sort(sample(ncol(Y), min(ncol(Y), 2000L)))
  sub_fm <- sort(sample(ncol(Y), 500L))
  p <- profundidades(Y[, sub]); cj <- boxplot_funcional(Y[, sub], p[, "MBD"])
  fb <- fbplot(Y[, sub], x = t_dia, method = "MBD", plot = FALSE)
  fm_usc <- depth.FM(fdata(t(Y[, sub_fm]), argvals = t_dia), dfunc = "FM1", par.dfunc = list(scale = FALSE))$dep
  c(n_curvas = length(sub),
    dif_MBD_fbplot = max(abs(p[, "MBD"] - fb$depth)),
    mediana_igual = as.numeric(sub[cj$mediana] == sub[fb$medcurve]),
    atipicas_mano = sum(cj$atipica), atipicas_fbplot = length(fb$outpoint),
    mismas_atipicas = as.numeric(setequal(which(cj$atipica), fb$outpoint)),
    dif_FM_depthFM = max(abs(profundidades(Y[, sub_fm])[, "FM"] - fm_usc)))
}, numeric(7)))
fb_est <- fbplot(fuentes$estaciones$Y, x = t_dia, method = "MBD", plot = FALSE)

resumen_atipicas <- t(vapply(names(fuentes), function(n) {
  c(N = ncol(fuentes[[n]]$Y), atipicas_MBD = sum(caja_mbd[[n]]$atipica), atipicas_FM = sum(caja_fm[[n]]$atipica),
    atipicas_ambas = sum(caja_mbd[[n]]$atipica & caja_fm[[n]]$atipica),
    pct_MBD = 100 * mean(caja_mbd[[n]]$atipica), pct_FM = 100 * mean(caja_fm[[n]]$atipica),
    correlacion_FM_MBD = cor(prof[[n]][, "FM"], prof[[n]][, "MBD"]),
    atipicas_forma = sum(caja_forma[[n]]$atipica), atipicas_forma_y_nivel = sum(caja_forma[[n]]$atipica & caja_mbd[[n]]$atipica))
}, numeric(9)))
resumen_atipicas

reporte_profundidad <- function(nombre) {
  f <- fuentes[[nombre]]; cm <- caja_mbd[[nombre]]; cf <- caja_fm[[nombre]]; p <- prof[[nombre]]; cs <- caja_forma[[nombre]]
  data.frame(fuente = nombre, ids[[nombre]], profundidad_FM = p[, "FM"], profundidad_MBD = p[, "MBD"],
             rango_MBD = rank(-p[, "MBD"], ties.method = "first"), total_anual_mm = colSums(f$Y),
             outlier_forma = cs$atipica, dia_exceso_forma = ifelse(cs$atipica, cs$dia, NA_integer_), direccion_forma = ifelse(cs$atipica, cs$direccion, NA_character_),
             outlier_mbd = cm$atipica, dia_exceso_mbd = ifelse(cm$atipica, cm$dia, NA_integer_),
             exceso_mbd_mm_dia = ifelse(cm$atipica, cm$exceso, NA_real_), direccion_mbd = ifelse(cm$atipica, cm$direccion, NA_character_),
             outlier_fm = cf$atipica, dia_exceso_fm = ifelse(cf$atipica, cf$dia, NA_integer_),
             exceso_fm_mm_dia = ifelse(cf$atipica, cf$exceso, NA_real_), direccion_fm = ifelse(cf$atipica, cf$direccion, NA_character_),
             stringsAsFactors = FALSE)
}
rep_prof <- lapply(names(fuentes), reporte_profundidad)
names(rep_prof) <- names(fuentes)
res_exploratorio$profundidades_estaciones <- rep_prof$estaciones
res_exploratorio$profundidades_satelite   <- rep_prof$satelite
outliers <- do.call(rbind, lapply(rep_prof, function(r) r[r$outlier_mbd | r$outlier_fm | r$outlier_forma, ]))
outliers <- outliers[order(outliers$fuente, -outliers$total_anual_mm), ]
res_exploratorio$outliers_funcionales <- outliers
est_ok <- which(!excluir)
caja_estable <- boxplot_funcional(fuentes$estaciones$Y[, est_ok], profundidades(fuentes$estaciones$Y[, est_ok])[, "MBD"])

indice_anual <- sapply(rep_prof, function(r) tapply(r$total_anual_mm / ave(r$total_anual_mm, r$id), r$anio, median))
indice_anual <- data.frame(anio = 1981:2003, indice_estaciones = indice_anual[, "estaciones"], indice_satelite = indice_anual[, "satelite"])
res_exploratorio$indice_anual <- round(indice_anual, 4)

oni <- oni_anual
ind <- merge(indice_anual, oni[, c("anio", "fase_codigo", "oni_medio")], by = "anio")
fases <- c("nino", "nina", "neutro")
enso <- do.call(rbind, lapply(c("indice_estaciones", "indice_satelite"), function(v) {
  g <- split(ind[[v]], factor(ind$fase_codigo, levels = fases))
  data.frame(indice = v, fase = fases, n_anios = lengths(g), mediana = vapply(g, median, numeric(1)), media = vapply(g, mean, numeric(1)),
             minimo = vapply(g, min, numeric(1)), maximo = vapply(g, max, numeric(1)), row.names = NULL)
}))
prueba_enso <- do.call(rbind, lapply(c("indice_estaciones", "indice_satelite"), function(v) {
  nino <- ind[[v]][ind$fase_codigo == "nino"]; nina <- ind[[v]][ind$fase_codigo == "nina"]
  data.frame(indice = v, mediana_nino_menos_nina = median(nino) - median(nina), correlacion_con_oni_medio = cor(ind[[v]], ind$oni_medio))
}))
res_exploratorio$indice_anual_por_fase_enso <- cbind(enso[, 1:3], round(enso[, 4:7], 4))
res_exploratorio$indice_anual_enso_pruebas <- cbind(prueba_enso[1], signif(prueba_enso[, -1L], 4))

anios_forma <- lapply(names(fuentes), function(n) {
  r <- rep_prof[[n]]; data.frame(anio = 1981:2003, curvas = as.vector(table(factor(r$anio, levels = 1981:2003))),
                                 atipicas_forma = as.vector(table(factor(r$anio[r$outlier_forma], levels = 1981:2003))))
})
names(anios_forma) <- names(fuentes)
mes_exceso <- function(dia) tabulate(as.integer(format(as.Date(dia - 1, origin = "2001-01-01"), "%m")), 12L)

umbrales_total <- c(6000, 8000, 10000, 12000, 15000)
tot_extremos <- data.frame(umbral_mm = umbrales_total,
                           estacion_anios = vapply(umbrales_total, function(u) sum(rep_prof$estaciones$total_anual_mm > u), numeric(1)),
                           estaciones = vapply(umbrales_total, function(u) length(unique(rep_prof$estaciones$id[rep_prof$estaciones$total_anual_mm > u])), numeric(1)))
res_exploratorio$totales_anuales_extremos <- tot_extremos

por_anio <- lapply(rep_prof, function(r) {
  tot <- table(factor(r$anio, levels = 1981:2003)); at <- table(factor(r$anio[r$outlier_mbd], levels = 1981:2003))
  data.frame(anio = 1981:2003, curvas = as.vector(tot), atipicas_mbd = as.vector(at), pct = 100 * as.vector(at) / as.vector(tot))
})
res_exploratorio$atipicas_por_anio_estaciones <- por_anio$estaciones
res_exploratorio$atipicas_por_anio_satelite <- por_anio$satelite

por_estacion <- aggregate(cbind(curvas = 1, atipicas = outlier_mbd) ~ id, data = rep_prof$estaciones, FUN = sum)
por_estacion <- por_estacion[order(-por_estacion$atipicas, -por_estacion$curvas), ]
res_exploratorio$atipicas_por_estacion <- por_estacion

malla <- pixeles_malla
por_pixel <- aggregate(cbind(curvas = 1, atipicas = outlier_mbd, total = total_anual_mm) ~ id, data = rep_prof$satelite, FUN = sum)
por_pixel$total_anual_medio <- por_pixel$total / por_pixel$curvas
res_exploratorio$atipicas_por_pixel <- merge(por_pixel[, c("id", "curvas", "atipicas", "total_anual_medio")], malla, by.x = "id", by.y = "pixel")

#Representatividad de las estaciones A-B en 1981-2003

meta_todo <- metadata_depuracion
cD <- coef_D
Phi_D <- bspline_base(t_dia, K_PEN <- 20L)
Y_C <- fuentes$estaciones$Phi %*% est$C[, est$curvas$nivel == "C", drop = FALSE]
Y_D <- Phi_D %*% t(as.matrix(cD[, 8:(7 + K_PEN)]))
Y_AB <- fuentes$estaciones$Y
tot <- list(AB = colSums(Y_AB), C = colSums(Y_C), D = colSums(Y_D))
#Comparación de los totales anuales por cuantiles
PROBS_Q <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
dif_cuantiles <- function(a, b) max(abs(quantile(a, PROBS_Q) - quantile(b, PROBS_Q))) / median(a)
q_AB_C <- dif_cuantiles(tot$AB, tot$C); q_AB_D <- dif_cuantiles(tot$AB, tot$D)
mu_AB <- rowMeans(Y_AB); mu_ABC <- rowMeans(cbind(Y_AB, Y_C)); mu_todo <- rowMeans(cbind(Y_AB, Y_C, Y_D))
anios <- 1981:2003
cobertura <- data.frame(anio = anios, pares = as.vector(table(factor(meta_todo$year, levels = anios))),
                        pares_AB = as.vector(table(factor(meta_todo$year[meta_todo$nivel %in% c("A", "B")], levels = anios))),
                        estaciones_AB = vapply(anios, function(a) length(unique(meta_todo$station_id[meta_todo$year == a & meta_todo$nivel %in% c("A", "B")])), numeric(1)))
cobertura$pct_AB <- 100 * cobertura$pares_AB / cobertura$pares
res_exploratorio$representatividad_por_anio <- cobertura
est_AB <- unique(id_e$station_id); est_todas <- unique(meta_todo$station_id)
est_fuera <- setdiff(est_todas, est_AB)
tot_est_AB <- tapply(tot$AB, id_e$station_id, mean)
tot_fuera <- tapply(c(tot$C, colSums(Y_D)), c(est$curvas$station_id[est$curvas$nivel == "C"], cD$station_id), mean)
tot_fuera <- tot_fuera[names(tot_fuera) %in% est_fuera]
part_AB <- table(meta_todo$particion[meta_todo$nivel %in% c("A", "B")]); part_todo <- table(meta_todo$particion)

picos <- lapply(fuentes, function(f) NULL)
picos$estaciones <- na.omit(fechas_base["estaciones", c("pico1", "pico2")]); picos$satelite <- na.omit(fechas_base["satelite", c("pico1", "pico2")])

guardar("figures/item2_exploratorio/media_funcional.png", 1500, 560, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
  invisible(Map(function(f, nombre, col) {
    muestra <- sample(ncol(f$Y), 120)
    matplot(t_dia, f$Y[, muestra], type = "l", lty = 1, col = adjustcolor("grey55", 0.35), xaxt = "n", xlab = "Día del año",
            ylab = "Precipitación (mm/día)", main = sprintf("%s: N = %d curvas, K = %d", nombre, ncol(f$Y), f$K),
            ylim = c(min(f$mu - 2 * f$sd, quantile(f$Y, 0.005)), max(f$mu + 2 * f$sd, quantile(f$Y, 0.995))))
    lines(t_dia, f$mu + 2 * f$sd, lty = 2, col = col); lines(t_dia, f$mu - 2 * f$sd, lty = 2, col = col)
    lines(t_dia, f$mu, lwd = 3, col = col)
    lines(t_dia, drop(eval.fd(t_dia, mean.fd(fd(f$C, create.bspline.basis(c(1, N_DIAS), f$K, 4L))))), lty = 3, lwd = 2)
    eje_meses(); abline(h = 0, col = "grey60")
    legend("topright", c("120 curvas", "media (paso a paso)", "mean.fd()", "media ± 2 sd"), lty = c(1, 1, 3, 2), lwd = c(1, 3, 2, 1),
           col = c("grey55", col, "black", col), bty = "n", cex = 0.8)
  }, fuentes, names(fuentes), COLORES))
})

guardar("figures/item2_exploratorio/varianza_cv.png", 1500, 900, {
  par(mfrow = c(2, 2), mar = c(4.2, 4.5, 3, 1))
  invisible(Map(function(f, nombre, col) {
    plot(t_dia, f$sd, type = "l", lwd = 2.5, col = col, xaxt = "n", xlab = "Día del año", ylab = "Desviación estándar (mm/día)",
         main = sprintf("%s: desviación estándar funcional", nombre))
    if (!is.null(f$var_estable)) {
      lines(t_dia, sqrt(f$var_estable), lwd = 2, lty = 2)
      legend("top", c("todas las curvas", "sin borde inestable ni meseta"), lty = c(1, 2), lwd = c(2.5, 2), col = c(col, "black"), bty = "n", cex = 0.8)
    }
    eje_meses(); abline(v = c(BORDE_INF + 0.5, BORDE_SUP - 0.5), lty = 3, col = "grey50"); grid()
  }, fuentes, names(fuentes), COLORES))
  invisible(Map(function(f, nombre, col) {
    plot(t_dia, f$cv, type = "l", lwd = 2.5, col = col, xaxt = "n", xlab = "Día del año", ylab = "sd / media (adimensional)",
         main = sprintf("%s: coeficiente de variación funcional", nombre), ylim = c(0, max(f$cv)))
    if (!is.null(f$var_estable)) lines(t_dia, sqrt(f$var_estable) / f$mu_estable, lwd = 2, lty = 2)
    eje_meses(); abline(v = c(BORDE_INF + 0.5, BORDE_SUP - 0.5), lty = 3, col = "grey50"); grid()
  }, fuentes, names(fuentes), COLORES))
})

guardar("figures/item2_exploratorio/efecto_borde.png", 1500, 560, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
  hist(pmin(infl_borde, 6), breaks = seq(0, 6, by = 0.25), col = COLORES["estaciones"], border = "white", xlab = "Inflación de la varianza en t = 1 o t = 365 (máximo, truncada en 6)",
       ylab = "Curvas", main = sprintf("Inflación en el borde: %d de %d curvas > %g", sum(inestable), length(inestable), UMBRAL_BORDE))
  abline(v = UMBRAL_BORDE, lty = 2, col = "red", lwd = 2)
  neta <- lapply(fuentes, function(f) sqrt(pmax(f$varianza - f$var_ruido, 0)))
  matplot(t_dia, cbind(fuentes$estaciones$sd, sqrt(fuentes$estaciones$var_estable), neta$estaciones), type = "l", lty = c(1, 2, 3), lwd = c(3, 2, 2),
          col = c(COLORES["estaciones"], "black", "red"), xaxt = "n", xlab = "Día del año", ylab = "Desviación estándar (mm/día)",
          main = "Estaciones: sd funcional y componentes")
  eje_meses(); abline(v = c(BORDE_INF + 0.5, BORDE_SUP - 0.5), lty = 3, col = "grey50")
  legend("top", c("todas las curvas", "sin borde inestable ni meseta", "sin ruido de estimación"), lty = c(1, 2, 3), lwd = c(3, 2, 2),
         col = c(COLORES["estaciones"], "black", "red"), bty = "n", cex = 0.8)
})

invisible(Map(function(f, nombre, pk) {
  guardar(sprintf("figures/item2_exploratorio/covarianza_correlacion_%s.png", nombre), 1500, 640, {
    par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 4))
    image(t_dia, t_dia, f$G, col = hcl.colors(60, "YlOrRd", rev = TRUE), xlab = "s (día del año)", ylab = "t (día del año)",
          main = sprintf("%s: covarianza v(s,t) (mm²/día²)", nombre)); contour(t_dia, t_dia, f$G, add = TRUE, nlevels = 8, drawlabels = FALSE)
    abline(v = pk, h = pk, lty = 3)
    image(t_dia, t_dia, f$Rc, zlim = c(-1, 1), col = hcl.colors(60, "Blue-Red 3"), xlab = "s (día del año)", ylab = "t (día del año)",
          main = sprintf("%s: correlación r(s,t)", nombre)); contour(t_dia, t_dia, f$Rc, add = TRUE, levels = c(0, 0.5, 0.8), drawlabels = FALSE)
    abline(v = pk, h = pk, lty = 3)
  })
}, fuentes, names(fuentes), picos))

guardar("figures/item2_exploratorio/fechas_por_curva.png", 1500, 780, {
  par(mfrow = c(2, 3), mar = c(4.2, 4.5, 3, 1))
  invisible(Map(function(m, nombre, col) invisible(lapply(colnames(m), function(e) {
    hist(m[, e], breaks = seq(0, 370, by = 10), col = col, border = "white", xaxt = "n", xlab = "Día del año", ylab = "Curvas",
         main = sprintf("%s: %s (%.0f%% de las curvas)", nombre, e, 100 * mean(!is.na(m[, e]))))
    eje_meses()
  })), fechas_curvas, names(fechas_curvas), COLORES))
})

guardar("figures/item2_exploratorio/sensibilidad_K_fechas.png", 1500, 560, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
  invisible(Map(function(nombre, col) {
    s <- sens_K[sens_K$fuente == nombre, ]; ic <- ic_boot[[nombre]]
    matplot(s$K, s[, c("pico1", "valle", "pico2")], type = "b", pch = c(19, 17, 15), lty = 1, col = c("#1b9e77", "#7570b3", "#d95f02"),
            xlab = "Número de bases (K)", ylab = "Día del año", main = sprintf("%s: fechas de la media según K", nombre),
            ylim = range(c(unlist(s[, c("pico1", "valle", "pico2")]), ic), na.rm = TRUE) + c(0, 55))
    kb <- fuentes[[nombre]]$K
    arrows(kb + c(-0.15, 0, 0.15), ic["2.5%", ], kb + c(-0.15, 0, 0.15), ic["97.5%", ], angle = 90, code = 3, length = 0.04, lwd = 2)
    legend("top", c("pico 1", "valle", "pico 2", "IC bootstrap 95% en K elegido"), pch = c(19, 17, 15, NA), lty = c(1, 1, 1, 1), lwd = c(1, 1, 1, 2),
           col = c("#1b9e77", "#7570b3", "#d95f02", "black"), bty = "n", cex = 0.75, ncol = 2)
  }, names(fuentes), COLORES))
})

guardar("figures/item2_exploratorio/profundidad.png", 1500, 640, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
  invisible(Map(function(p, cm, cf, nombre, col) {
    ver <- if (nrow(p) > 4000L) sort(sample(nrow(p), 4000L)) else seq_len(nrow(p))
    plot(p[ver, "FM"], p[ver, "MBD"], pch = 19, cex = 0.5, col = adjustcolor(col, 0.4), xlab = "Profundidad Fraiman-Muniz", ylab = "Profundidad MBD",
         main = sprintf("%s (correlación = %.2f)", nombre, cor(p[, "FM"], p[, "MBD"])))
    puntos <- which(cm$atipica)
    puntos <- if (length(puntos) > 1500L) sample(puntos, 1500L) else puntos
    points(p[puntos, "FM"], p[puntos, "MBD"], pch = 19, cex = 0.6, col = "red")
    legend("bottomright", c("curvas", "atípicas (boxplot MBD)"), pch = 19, col = c(col, "red"), bty = "n", cex = 0.8)
  }, prof, caja_mbd, caja_fm, names(fuentes), COLORES))
})

figura_caja <- function(Y, cm, nombre, col, archivo, etiqueta, ylab) {
  guardar(archivo, 1500, 640, {
    par(mar = c(4.2, 4.5, 3, 1))
    normales <- setdiff(seq_len(ncol(Y)), which(cm$atipica))
    gris <- sample(normales, min(200L, length(normales)))
    atip <- which(cm$atipica); atip <- if (length(atip) > 60L) sample(atip, 60L) else atip
    matplot(t_dia, Y[, gris], type = "l", lty = 1, col = adjustcolor("grey60", 0.25), xaxt = "n", xlab = "Día del año", ylab = ylab,
            main = sprintf("%s, %s: boxplot funcional (MBD), %d atípicas de %d curvas (%.1f%%)", nombre, etiqueta, sum(cm$atipica), ncol(Y), 100 * mean(cm$atipica)),
            ylim = range(c(cm$cerca_sup, cm$cerca_inf, Y[, atip])))
    polygon(c(t_dia, rev(t_dia)), c(cm$inf, rev(cm$sup)), col = adjustcolor(col, 0.3), border = NA)
    lines(t_dia, cm$cerca_sup, lty = 2, lwd = 1.5); lines(t_dia, cm$cerca_inf, lty = 2, lwd = 1.5)
    if (length(atip)) matlines(t_dia, Y[, atip], lty = 1, col = adjustcolor("red", 0.35))
    lines(t_dia, Y[, cm$mediana], lwd = 3); eje_meses(); abline(h = 0, col = "grey60")
    legend("topright", c("curvas (muestra)", "región central 50%", "bigotes (1.5 veces)", "mediana funcional", "atípicas (muestra)"),
           lty = 1, lwd = c(1, 6, 1.5, 3, 1), col = c("grey60", adjustcolor(col, 0.5), "black", "black", "red"), bty = "n", cex = 0.75)
  })
}
invisible(Map(function(f, cm, nombre, col) figura_caja(f$Y, cm, nombre, col, sprintf("figures/item2_exploratorio/boxplot_funcional_%s.png", nombre), "nivel", "Precipitación (mm/día)"),
              fuentes, caja_mbd, names(fuentes), COLORES))
invisible(Map(function(Yf, cm, nombre, col) figura_caja(Yf, cm, nombre, col, sprintf("figures/item2_exploratorio/boxplot_forma_%s.png", nombre), "forma", "Precipitación / media anual de la curva"),
              Y_forma, caja_forma, names(fuentes), COLORES))

guardar("figures/item2_exploratorio/atipicas_anio_estacion.png", 1500, 900, {
  par(mfrow = c(2, 2), mar = c(4.2, 4.5, 3, 1))
  barplot(por_anio$estaciones$pct, names.arg = por_anio$estaciones$anio, col = COLORES["estaciones"], las = 2, cex.names = 0.7,
          ylab = "% de curvas atípicas (MBD)", main = sprintf("Estaciones: atípicas por año (satélite: %d en %d años)", sum(por_anio$satelite$atipicas_mbd), sum(por_anio$satelite$atipicas_mbd > 0)))
  matplot(indice_anual$anio, indice_anual[, c("indice_estaciones", "indice_satelite")], type = "b", pch = 19, lty = 1, col = COLORES,
          xlab = "Año", ylab = "Total anual / total medio de la curva", main = "Índice anual (mediana entre estaciones o píxeles)")
  abline(h = 1, lty = 2, col = "grey40"); legend("topleft", c("estaciones", "satélite"), pch = 19, lty = 1, col = COLORES, bty = "n", cex = 0.8)
  hist(por_estacion$atipicas, breaks = seq(-0.5, max(por_estacion$atipicas) + 0.5, 1), col = COLORES["estaciones"], border = "white",
       xlab = "Años atípicos por estación", ylab = "Estaciones", main = "Estaciones: años atípicos por estación")
  top <- head(por_estacion[por_estacion$atipicas > 0, ], 12)
  barplot(rev(top$atipicas / top$curvas), names.arg = rev(top$id), horiz = TRUE, las = 1, cex.names = 0.7, col = COLORES["estaciones"],
          xlab = "Fracción de años atípicos", main = "Estaciones con más años atípicos")
})

COL_FASE <- c(nino = "#d62728", nina = "#1f77b4", neutro = "grey60")
guardar("figures/item2_exploratorio/indice_anual_enso.png", 1500, 900, {
  par(mfrow = c(2, 2), mar = c(4.2, 4.5, 3, 1))
  f_fase <- factor(ind$fase_codigo, levels = fases)
  invisible(Map(function(v, nombre, p) {
    boxplot(ind[[v]] ~ f_fase, names = c("El Niño", "La Niña", "Neutro"), col = adjustcolor(COL_FASE, 0.35), outline = FALSE, ylim = range(ind[[v]]),
            ylab = "Total anual / total medio del sitio", xlab = "Fase ENSO del año (ONI de la NOAA)", main = sprintf("%s (mediana Niño - Niña = %.2f)", nombre, p))
    stripchart(ind[[v]] ~ f_fase, vertical = TRUE, add = TRUE, pch = 19, col = COL_FASE, method = "jitter")
    abline(h = 1, lty = 2, col = "grey40")
  }, c("indice_estaciones", "indice_satelite"), c("Estaciones", "Satélite"), prueba_enso$mediana_nino_menos_nina))
  invisible(Map(function(v, nombre) {
    barplot(ind[[v]] - 1, names.arg = ind$anio, col = COL_FASE[ind$fase_codigo], las = 2, cex.names = 0.7, ylab = "Índice anual − 1", main = sprintf("%s: índice anual por año", nombre))
    legend("bottomleft", c("El Niño", "La Niña", "Neutro"), fill = COL_FASE, bty = "n", cex = 0.8, ncol = 3)
  }, c("indice_estaciones", "indice_satelite"), c("Estaciones", "Satélite")))
})

lon_u <- sort(unique(malla$lon)); lat_u <- sort(unique(malla$lat))
matriz_malla <- function(v) { M <- matrix(NA_real_, length(lat_u), length(lon_u)); M[cbind(match(malla$lat[match(por_pixel$id, malla$pixel)], lat_u), match(malla$lon[match(por_pixel$id, malla$pixel)], lon_u))] <- v; M }
guardar("figures/item2_exploratorio/mapa_satelite.png", 1500, 640, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 5))
  image(lon_u, lat_u, t(matriz_malla(por_pixel$total_anual_medio)), col = hcl.colors(60, "YlGnBu", rev = TRUE), xlab = "Longitud (°)", ylab = "Latitud (°)",
        main = "Satélite: total anual medio (mm)"); box()
  image(lon_u, lat_u, t(matriz_malla(tapply(rep_prof$satelite$profundidad_MBD, rep_prof$satelite$id, mean))), col = hcl.colors(60, "Purples 3", rev = TRUE),
        xlab = "Longitud (°)", ylab = "Latitud (°)", main = "Satélite: profundidad MBD media (menor = más periférico)"); box()
})

guardar("figures/item2_exploratorio/representatividad.png", 1500, 520, {
  par(mfrow = c(1, 3), mar = c(4.2, 4.5, 3, 1))
  barplot(rbind(cobertura$pares_AB, cobertura$pares - cobertura$pares_AB), names.arg = cobertura$anio, las = 2, cex.names = 0.65,
          col = c("#1f77b4", "grey75"), ylab = "Pares estación-año", main = "Pares por año: niveles A-B frente al resto")
  legend("bottomright", c("A-B (usados)", "C y D"), fill = c("#1f77b4", "grey75"), bty = "n", cex = 0.8)
  boxplot(tot, names = c(sprintf("A-B\nn=%d", length(tot$AB)), sprintf("C\nn=%d", length(tot$C)), sprintf("D pen.\nn=%d", length(tot$D))),
          ylab = "Total anual de la curva ajustada (mm)", main = "Total anual por nivel de calidad", col = c("#1f77b4", "grey80", "grey80"))
  matplot(t_dia, cbind(mu_AB, mu_ABC, mu_todo), type = "l", lty = c(1, 2, 3), lwd = c(3, 2, 2), col = c("#1f77b4", "red", "black"), xaxt = "n",
          xlab = "Día del año", ylab = "Media funcional (mm/día)", main = "Media funcional según el subconjunto"); eje_meses()
  legend("topright", c("A-B", "A-C", "A-C + D penalizado"), lty = c(1, 2, 3), lwd = c(3, 2, 2), col = c("#1f77b4", "red", "black"), bty = "n", cex = 0.8)
})

indic <- c(N_estaciones = ncol(fuentes$estaciones$Y), N_satelite = ncol(fuentes$satelite$Y), K_estaciones = fuentes$estaciones$K, K_satelite = fuentes$satelite$K,
           dif_media_mean_fd_estaciones = verif_media["estaciones", "dif_mean_fd"], dif_media_mean_fd_satelite = verif_media["satelite", "dif_mean_fd"],
           dif_cov_var_fd_estaciones = verif_cov["estaciones", "dif_cov_var_fd"], dif_cov_var_fd_satelite = verif_cov["satelite", "dif_cov_var_fd"],
           atipicas_mbd_estaciones = sum(caja_mbd$estaciones$atipica), atipicas_mbd_satelite = sum(caja_mbd$satelite$atipica),
           atipicas_fm_estaciones = sum(caja_fm$estaciones$atipica), atipicas_fm_satelite = sum(caja_fm$satelite$atipica),
           dif_cuantiles_rel_AB_vs_C = q_AB_C, dif_cuantiles_rel_AB_vs_D = q_AB_D, dif_L2_AB_vs_ABC = dif_rel(mu_AB, mu_ABC), dif_L2_AB_vs_todo = dif_rel(mu_AB, mu_todo))
res_exploratorio$resumen_item2 <- data.frame(indicador = names(indic), valor = signif(indic, 6))
res_exploratorio$eventos_media_funcional <- do.call(rbind, Map(function(e, n) data.frame(fuente = n, e), eventos, names(eventos)))
res_exploratorio$fechas_bootstrap <- do.call(rbind, Map(function(ic, n) data.frame(fuente = n, estadistico = rownames(ic), ic), ic_boot, names(ic_boot)))

#Fechas de las temporadas: ruta penalizada frente al K de la VCG, K = 8 y medias móviles circulares de 15 y 31 días

pen <- coef_pen
stopifnot(identical(pen$curvas$station_id * 10000 + pen$curvas$year, est$curvas$station_id * 10000 + est$curvas$year))
C_pen <- pen$C[, usa, drop = FALSE]
ok_pen <- colSums(!is.finite(C_pen)) == 0
cm_pen <- rowMeans(C_pen[, ok_pen, drop = FALSE])
mu_pen <- drop(bspline_base(t_dia, pen$K) %*% cm_pen)
fd_pen <- fd(C_pen[, ok_pen, drop = FALSE], create.bspline.basis(rangeval = c(1, N_DIAS), nbasis = pen$K, norder = 4L))
dif_mean_fd_pen <- max(abs(mu_pen - drop(eval.fd(t_dia, mean.fd(fd_pen)))))

aj8 <- ajuste_enmascarado(bspline_base(t_dia, 8L), Y_diario)
ok8 <- colSums(!is.finite(aj8$C)) == 0
cm8 <- rowMeans(aj8$C[, ok8, drop = FALSE])
cm6 <- rowMeans(fuentes$estaciones$C)

y_obs <- rowMeans(Y_diario, na.rm = TRUE)                                 # media diaria observada (sin modelo)
media_movil <- function(y, w) {
  h <- (w - 1L) %/% 2L
  as.numeric(stats::filter(c(tail(y, h), y, head(y, h)), rep(1 / w, w), sides = 2))[h + seq_along(y)]
}
eventos_ventana <- function(x) c(pico1 = 59 + which.max(x[60:181]), valle = 151 + which.min(x[152:273]), pico2 = 243 + which.max(x[244:349]))
curvas_fecha <- list(penalizada = mu_pen, K8 = drop(bspline_base(t_dia, 8L) %*% cm8), Kvcg = fuentes$estaciones$mu,
                     movil15 = media_movil(y_obs, 15L), movil31 = media_movil(y_obs, 31L))
dias_fecha <- rbind(penalizada = fechas_clave(cm_pen, estructura_extremos(pen$K)),
                    K8 = fechas_clave(cm8, estructura_extremos(8L)),
                    Kvcg = fechas_clave(cm6, estructura_extremos(fuentes$estaciones$K)),
                    movil15 = eventos_ventana(curvas_fecha$movil15),
                    movil31 = eventos_ventana(curvas_fecha$movil31))
fechas_oficiales <- data.frame(curva = c("ruta penalizada (oficial)", "MCO K = 8 (Ítem 4)", sprintf("MCO K = %d (VCG)", fuentes$estaciones$K), "media móvil 15 d", "media móvil 31 d"),
                               nu_o_K = c(round(verif_media_nu <- mean(pen$nu[usa][ok_pen]), 2), 8, fuentes$estaciones$K, NA, NA),
                               dia_pico1 = dias_fecha[, "pico1"], dia_valle = dias_fecha[, "valle"], dia_pico2 = dias_fecha[, "pico2"],
                               pico1 = fecha_doy(dias_fecha[, "pico1"]), valle = fecha_doy(dias_fecha[, "valle"]), pico2 = fecha_doy(dias_fecha[, "pico2"]),
                               row.names = NULL)
fechas_oficiales$dif_valle_vs_movil_dias <- fechas_oficiales$dia_valle - mean(dias_fecha[c("movil15", "movil31"), "valle"])
res_exploratorio$fechas_oficiales_temporadas <- fechas_oficiales
fechas_oficiales
res_exploratorio$eventos_media_funcional_penalizada <- data.frame(fuente = "estaciones_penalizada", eventos_media(cm_pen, estructura_extremos(pen$K)))
stopifnot(dif_mean_fd_pen <= 1e-12)
res_exploratorio$verificacion_media_penalizada <- data.frame(indicador = "dif_media_mean_fd_penalizada", valor = dif_mean_fd_pen)

guardar("figures/item2_exploratorio/fechas_oficiales.png", 1500, 620, {
  par(mar = c(4.2, 4.5, 3, 1))
  plot(t_dia, y_obs, col = "grey75", type = "l", xaxt = "n", xlab = "Día del año", ylab = "Lluvia media de las estaciones (mm/día)",
       main = sprintf("Fechas de las temporadas: ruta penalizada (oficial) frente a K = %d (VCG), K = 8 (Ítem 4) y medias móviles", fuentes$estaciones$K))
  eje_meses()
  col_f <- c(penalizada = "black", K8 = "darkorange", Kvcg = "firebrick", movil15 = "royalblue", movil31 = "navy")
  lty_f <- c(penalizada = 1, K8 = 1, Kvcg = 1, movil15 = 2, movil31 = 3)
  invisible(Map(function(x, n) lines(t_dia, x, col = col_f[n], lty = lty_f[n], lwd = ifelse(n == "penalizada", 3, 1.6)), curvas_fecha, names(curvas_fecha)))
  points(dias_fecha[, "valle"], vapply(seq_len(nrow(dias_fecha)), function(i) curvas_fecha[[i]][round(dias_fecha[i, "valle"])], numeric(1)),
         pch = 25, bg = col_f, col = col_f)
  legend("topleft", c("media diaria observada", "ruta penalizada (oficial)", "MCO K = 8 (Ítem 4)", sprintf("MCO K = %d (VCG)", fuentes$estaciones$K), "media móvil 15 d", "media móvil 31 d"),
         col = c("grey75", col_f), lty = c(1, lty_f), lwd = c(1, 3, 1.6, 1.6, 1.6, 1.6), bty = "n", cex = 0.8, ncol = 2)
})

#Derivada de la media y banda 10-90 % de las derivadas: estaciones con la ruta penalizada y satélite con el K de su VCG
C_pen_ok  <- C_pen[, ok_pen, drop = FALSE]
fuentes_d <- list(estaciones = list(C = C_pen_ok, K = pen$K, mu = mu_pen, sd = apply(bspline_base(t_dia, pen$K) %*% C_pen_ok, 1L, sd)),
                  satelite = fuentes$satelite)
eventos_d <- list(estaciones = eventos_media(cm_pen, estructura_extremos(pen$K)), satelite = eventos$satelite)
banda_d1  <- lapply(fuentes_d, function(f) apply(bspline_base(t_dia, f$K, deriv = 1L) %*% f$C, 1L, quantile, probs = c(0.1, 0.9)))   # Dim: (2, 365)
TIT_D <- c("estaciones (ruta penalizada)", sprintf("satélite (K = %d)", fuentes$satelite$K))

guardar("figures/item2_exploratorio/derivada_temporadas.png", 1500, 940, {
  par(mfrow = c(2, 2), mar = c(4.2, 4.5, 3, 1))
  invisible(Map(function(f, nombre, col, ev) {
    plot(t_dia, f$mu, type = "l", lwd = 3, col = col, xaxt = "n", xlab = "Día del año", ylab = "Precipitación (mm/día)",
         main = sprintf("%s: media funcional y extremos", nombre), ylim = c(0, max(f$mu + f$sd)))
    lines(t_dia, f$mu + f$sd, lty = 2, col = col); lines(t_dia, pmax(f$mu - f$sd, 0), lty = 2, col = col)
    eje_meses(); abline(v = ev$t[ev$evento == "maximo"], lty = 1, lwd = 2, col = "red"); abline(v = ev$t[ev$evento == "minimo"], lty = 1, lwd = 2, col = "darkgreen")
    abline(v = ev$t[ev$evento %in% c("ascenso_maximo", "descenso_maximo")], lty = 3, col = "grey40")
    legend("topleft", c("máximo (x' = 0, x'' < 0)", "mínimo (x' = 0, x'' > 0)", "inflexión", "media ± 1 sd"), lty = c(1, 1, 3, 2),
           col = c("red", "darkgreen", "grey40", col), bty = "n", cex = 0.75)
  }, fuentes_d, TIT_D, COLORES, eventos_d))
  invisible(Map(function(f, nombre, col, banda) {
    plot(t_dia, banda[1, ], type = "n", ylim = range(banda), xaxt = "n", xlab = "Día del año", ylab = "Derivada (mm/día por día)",
         main = sprintf("%s: derivada y banda 10-90%%", nombre))
    polygon(c(t_dia, rev(t_dia)), c(banda[1, ], rev(banda[2, ])), col = adjustcolor(col, 0.2), border = NA)
    lines(t_dia, drop(bspline_base(t_dia, f$K, deriv = 1L) %*% colMeans(t(f$C))), lwd = 3, col = col)
    abline(h = 0, col = "black", lty = 2); eje_meses(); grid()
  }, fuentes_d, TIT_D, COLORES, banda_d1))
})

#Tramos del soporte interior en que la banda 10-90 % de las derivadas excluye el cero
tramos_banda <- function(b, fuente) {
  s <- ifelse(en_borde, 0L, ifelse(b[2, ] < 0, -1L, ifelse(b[1, ] > 0, 1L, 0L)))
  r <- rle(s); fin <- cumsum(r$lengths); ini <- fin - r$lengths + 1L; k <- which(r$values != 0L)
  data.frame(fuente = rep(fuente, length(k)), dia_desde = ini[k], dia_hasta = fin[k], desde = fecha_doy(ini[k]), hasta = fecha_doy(fin[k]),
             dias = r$lengths[k], signo = ifelse(r$values[k] > 0, "sobre cero (ascenso)", "bajo cero (descenso)"), stringsAsFactors = FALSE)
}
res_exploratorio$banda_derivada <- do.call(rbind, Map(tramos_banda, banda_d1, c("estaciones_penalizada", "satelite")))

#Resultados que pasan a los bloques siguientes
NUCLEO <- c(NUCLEO, "res_exploratorio")

#Comparación entre fuentes

dir.create("figures/item3_comparacion", showWarnings = FALSE, recursive = TRUE)
res_comparacion <- list()                              # tablas de resultados del bloque

guardar <- function(archivo, ancho, alto, expr) {
  png(archivo, width = ancho, height = alto, res = 130)
  on.exit(dev.off())
  expr
}

emp   <- emparejamiento
meta  <- metadata_depuracion
malla <- pixeles_malla
sat   <- satelite_pentadal
logico <- function(x) toupper(as.character(x)) %in% c("TRUE", "1")
emp[c("dentro_pixel", "en_red", "ciega", "candidata_ciega", "con_coordenadas", "reemparejada_borde")] <-
  lapply(emp[c("dentro_pixel", "en_red", "ciega", "candidata_ciega", "con_coordenadas", "reemparejada_borde")], logico)

usa <- function(col) unique(meta$station_id[logico(meta[[col]])])
emp$rol <- ifelse(!emp$con_coordenadas, "sin_coordenadas",
           ifelse(!emp$dentro_pixel, "fuera_malla",
           ifelse(!emp$en_red, "malla_sin_altitud",
           ifelse(emp$ciega, "ciega",
           ifelse(emp$station_id %in% usa("uso_entrenamiento"), "entrenamiento", "red_sin_entrenamiento")))))

#Fondo de los mapas: total anual medio del satélite por píxel
px_cols  <- grep("^px", names(sat), value = TRUE)
n_anios  <- length(unique(sat$anio))
tot_px   <- colSums(as.matrix(sat[, px_cols])) / n_anios                       # Dim: (686,), celdas del Valle
lon_u <- sort(unique(malla$lon)); lat_u <- sort(unique(malla$lat))
M_tot <- matrix(NA_real_, length(lat_u), length(lon_u))
M_tot[cbind(match(malla$lat, lat_u), match(malla$lon, lon_u))] <- tot_px[paste0("px", malla$pixel)]
paso <- diff(lon_u[1:2]) / 2
lim   <- read.csv("data/processed/metadatos/limites_departamentos_gadm41.csv", stringsAsFactors = FALSE)
fuera <- read.csv("data/processed/metadatos/estaciones_fuera_del_valle.csv", stringsAsFactors = FALSE)
con_cortes <- function(p) do.call(rbind, lapply(split(p[c("lon", "lat")], p$anillo), rbind, c(NA, NA)))
borde_valle <- con_cortes(lim[lim$departamento == "Valle del Cauca", ])

geo <- emp[emp$con_coordenadas, ]
xl <- range(c(lon_u, geo$lon)) + c(-0.05, 0.05); yl <- range(c(lat_u, geo$lat)) + c(-0.05, 0.05)
fondo <- function(main) {
  image(lon_u, lat_u, t(M_tot), col = hcl.colors(40, "Blues 3", rev = TRUE, alpha = 0.55), xlim = xl, ylim = yl, asp = 1,
        xlab = "Longitud (°)", ylab = "Latitud (°)", main = main)
  lines(borde_valle, col = "grey20", lwd = 0.9)
}

#Cobertura espacial y temporal de las estaciones en el Valle del Cauca (límite GADM)
px_con_est <- malla$pixel %in% emp$pixel
M_cob <- matrix(NA_real_, length(lat_u), length(lon_u))
M_cob[cbind(match(malla$lat, lat_u), match(malla$lon, lon_u))] <- 1 + px_con_est

anios_ab <- tapply(meta$nivel %in% c("A", "B"), meta$station_id, sum)
geo$anios_ab <- as.integer(anios_ab[as.character(geo$station_id)])
geo$anios_ab[is.na(geo$anios_ab)] <- 0L
RANGOS_AB <- c("0", "1-7", "8-15", "16-22", "23 (completa)")
pal_ab <- hcl.colors(length(RANGOS_AB), "YlGnBu", rev = TRUE)
geo$rango_ab <- cut(geo$anios_ab, c(-0.5, 0.5, 7.5, 15.5, 22.5, 23.5), RANGOS_AB)
o_ab <- order(geo$anios_ab)
excl <- fuera[!is.na(fuera$lon), ]
ab_m <- meta$nivel %in% c("A", "B")
T_ab <- table(factor(meta$nivel[ab_m], c("A", "B")), factor(meta$year[ab_m], 1981:2003))   # Dim: (2, 23)
cob <- c(celdas_valle = nrow(malla), celdas_valle_con_estacion = sum(px_con_est), pct_celdas_valle_con_estacion = 100 * mean(px_con_est),
         estaciones_valle = nrow(emp), estaciones_valle_en_red = sum(emp$en_red), reemparejadas_borde = sum(emp$reemparejada_borde),
         excluidas_fuera_del_limite = nrow(excl), excluidas_sin_coordenadas = sum(is.na(fuera$lon)),
         anios_ab_mediana = median(geo$anios_ab), estaciones_23_anios_ab = sum(geo$anios_ab == 23),
         estaciones_ab_por_anio_min = min(colSums(T_ab)), estaciones_ab_por_anio_max = max(colSums(T_ab)))
res_comparacion$cobertura_valle <- data.frame(indicador = names(cob), valor = signif(cob, 6))

guardar("figures/item3_comparacion/mapa_cobertura_valle.png", 1700, 950, {
  layout(matrix(1:2, 1), widths = c(1.2, 1))
  par(mar = c(4.2, 4.5, 4.2, 1))
  image(lon_u, lat_u, t(M_cob), col = c("grey88", "grey68"), breaks = c(0.5, 1.5, 2.5), asp = 1,
        xlim = c(-77.62, -75.62), ylim = c(2.1, 5.06), xlab = "Longitud (°)", ylab = "Latitud (°)",
        main = "Cobertura de las estaciones en el Valle del Cauca, 1981-2003")
  u <- par("usr")
  clip(max(u[1], -78.19), min(u[2], -75.21), max(u[3], 1.91), min(u[4], 5.29))       # oculta los bordes del recorte del GADM
  lines(con_cortes(lim[lim$departamento != "Valle del Cauca", ]), col = "grey65", lwd = 0.7)
  lines(borde_valle, col = "grey10", lwd = 1.2)
  do.call(clip, as.list(par("usr")))
  text(c(-77.2, -76.95, -75.55, -75.62), c(4.85, 2.3, 3.75, 4.5), c("Chocó", "Cauca", "Tolima", "Quindío"), col = "grey45", font = 3, cex = 0.85)
  text(-77.7, 3.75, "Océano\nPacífico", col = "#4a7fb0", font = 3, cex = 0.85)
  points(excl$lon, excl$lat, pch = 4, col = "grey50", cex = 0.8)
  points(geo$lon[o_ab], geo$lat[o_ab], pch = 21, bg = pal_ab[geo$rango_ab[o_ab]], col = "grey15", lwd = 0.6, cex = 1.1)
  legend("bottomleft", c(sprintf("Sin estación (%d)", sum(!px_con_est)), sprintf("Con al menos una estación (%d)", sum(px_con_est))),
         fill = c("grey88", "grey68"), border = "grey60", bg = "white", cex = 0.75, title = "Celdas del Valle (0,05°)")
  legend("bottomright", c(sprintf("%s (%d)", RANGOS_AB, as.integer(table(geo$rango_ab))), sprintf("Excluida, fuera del Valle (%d)", nrow(excl))),
         pch = c(rep(21, length(RANGOS_AB)), 4), pt.bg = c(pal_ab, NA), col = c(rep("grey15", length(RANGOS_AB)), "grey50"),
         pt.cex = c(rep(1.2, length(RANGOS_AB)), 1), bg = "white", cex = 0.75, title = "Años con curva A-B")
  mtext(sprintf("%d estaciones del Valle con datos 1981-2003 (las únicas del proyecto); excluidas: %d fuera del límite y %d sin coordenadas",
                nrow(emp), nrow(excl), sum(is.na(fuera$lon))), 3, 0.3, cex = 0.72)
  par(mar = c(4.2, 4.5, 4.2, 1))
  y_max <- max(colSums(T_ab))
  bp <- barplot(T_ab, col = c("#2c7fb8", "#9ecae1"), border = NA, las = 2, cex.names = 0.75, ylim = c(0, y_max * 1.32),
                ylab = "Estaciones del Valle", main = "Estaciones con curva A-B por año")
  n_ent <- ANIO_ENTRENA - 1980L                                                    # años de entrenamiento (1981-1996)
  n_cv <- ANIOS_VALIDA[1] - 1981L                                                   # primer bloque de validación cruzada
  segments(mean(bp[n_ent + 0:1]), 0, y1 = y_max * 1.12, lty = 2, col = "grey30")
  segments(mean(bp[n_cv + 0:1]), 0, y1 = y_max * 1.0, lty = 3, col = "grey55")
  text(c(mean(bp[1:n_ent]), mean(bp[(n_ent + 1):23])), y_max * 1.07,
       c(sprintf("Entrenamiento (validación cruzada %d-%d)", ANIOS_VALIDA[1], ANIOS_VALIDA[2]), "Prueba"), cex = 0.75)
  legend(mean(range(bp)), y_max * 1.3, c("Nivel A", "Nivel B"), fill = c("#2c7fb8", "#9ecae1"), border = NA, horiz = TRUE, xjust = 0.5,
         bty = "n", cex = 0.75)
})

compartidos <- unique(emp[emp$dentro_pixel & emp$n_estaciones_en_pixel > 1, c("pixel", "n_estaciones_en_pixel", "comparte_pixel_con")])
px_comp <- malla[match(compartidos$pixel, malla$pixel), ]
ESTILO <- data.frame(rol = c("red", "malla_sin_altitud"), col = c("#08306b", "#e6550d"), pch = c(19, 17),
                     etiqueta = c("En la red (píxel y altitud)", "Con píxel, sin altitud"))
geo$grupo <- ifelse(geo$en_red, "red", "malla_sin_altitud")
borde <- geo[geo$reemparejada_borde, ]
est <- ESTILO[match(geo$grupo, ESTILO$rol), ]
n_por_px <- table(factor(emp$n_estaciones_en_pixel[emp$dentro_pixel], levels = sort(unique(emp$n_estaciones_en_pixel[emp$dentro_pixel]))))
guardar("figures/item3_comparacion/mapa_emparejamiento.png", 1600, 800, {
  layout(matrix(1:2, 1), widths = c(2.2, 1))
  par(mar = c(4.2, 4.5, 3, 1))
  fondo("Emparejamiento estación-píxel (fondo: total anual medio del satélite)")
  rect(px_comp$lon - paso, px_comp$lat - paso, px_comp$lon + paso, px_comp$lat + paso, border = "#cb181d", lwd = 1.6)
  points(geo$lon, geo$lat, col = est$col, pch = est$pch, cex = 0.85)
  points(borde$lon, borde$lat, pch = 1, col = "#31a354", cex = 1.9, lwd = 1.5)
  legend("topleft", c(sprintf("%s (%d)", ESTILO$etiqueta, as.integer(table(factor(geo$grupo, ESTILO$rol)))), "Píxel con ≥ 2 estaciones",
                      sprintf("Borde: píxel del Valle más cercano (%d)", nrow(borde)), "Límite del Valle (GADM)"),
         col = c(ESTILO$col, "#cb181d", "#31a354", "grey20"), pch = c(ESTILO$pch, 0, 1, NA), lty = c(NA, NA, NA, NA, 1), pt.cex = c(1, 1, 1.6, 1.6, 1),
         bg = "white", cex = 0.75)
  mtext(sprintf("%d estaciones del Valle con datos 1981-2003; excluidas: %d fuera del límite y %d sin coordenadas", nrow(emp),
                sum(!is.na(fuera$lon)), sum(is.na(fuera$lon))), 3, 0.2, cex = 0.75)
  par(mar = c(4.2, 4.5, 3, 1))
  n_px <- as.integer(n_por_px) / as.integer(names(n_por_px))
  bp <- barplot(n_px, names.arg = names(n_por_px), col = "#6baed6", border = "white", ylim = c(0, max(n_px) * 1.12),
                xlab = "Estaciones en el mismo píxel", ylab = "Número de píxeles", main = "Estaciones por píxel")
  text(bp, n_px, n_px, pos = 3, font = 2, xpd = TRUE)
})

red <- geo[geo$en_red, ]
cortes <- c(0, 500, 1000, 1500, 2000, 2500, 4000)
pal <- hcl.colors(length(cortes) - 1, "Terrain")
red$col_alt <- pal[findInterval(red$altitud_m, cortes, rightmost.closed = TRUE)]
FORMA <- c(entrenamiento = 21, red_sin_entrenamiento = 22, ciega = 24)
guardar("figures/item3_comparacion/mapa_particion.png", 1400, 900, {
  par(mar = c(4.2, 4.5, 3, 1))
  fondo("Partición espacial del Ítem 4: entrenamiento frente a estaciones ciegas")
  points(geo$lon[!geo$en_red], geo$lat[!geo$en_red], pch = 20, col = "grey65", cex = 0.6)
  points(red$lon, red$lat, pch = FORMA[red$rol], bg = red$col_alt, col = ifelse(red$rol == "ciega", "#cb181d", "grey20"),
         lwd = ifelse(red$rol == "ciega", 2.2, 0.8), cex = ifelse(red$rol == "ciega", 1.5, 1.1))
  legend("topleft", c(sprintf("Entrenamiento (%d)", sum(red$rol == "entrenamiento")), sprintf("En la red, sin años de entrenamiento A-B (%d)", sum(red$rol == "red_sin_entrenamiento")),
                      sprintf("Ciega (%d)", sum(red$rol == "ciega")), "Fuera de la red"),
         pch = c(21, 22, 24, 20), pt.bg = "grey85", col = c("grey20", "grey20", "#cb181d", "grey65"), pt.lwd = c(0.8, 0.8, 2.2, 1), bg = "white", cex = 0.8)
  legend("bottomright", sprintf("%d–%d m", head(cortes, -1), cortes[-1]), fill = pal, title = "Altitud", bg = "white", cex = 0.75)
})

#Distribución de altitud de candidatas y ciegas: cuantiles y densidades
cand <- emp$altitud_m[emp$candidata_ciega]
cieg <- emp$altitud_m[emp$ciega & emp$en_red]
PROBS <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
cuantiles <- data.frame(probabilidad = PROBS, candidatas_m = unname(quantile(cand, PROBS)), ciegas_m = unname(quantile(cieg, PROBS)))
cuantiles$diferencia_m <- cuantiles$ciegas_m - cuantiles$candidatas_m
cuantiles$diferencia_rel_rango <- cuantiles$diferencia_m / diff(range(cand))
guardar("figures/item3_comparacion/altitud_ciegas.png", 1500, 650, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
  br <- seq(0, ceiling(max(cand) / 250) * 250, 250)
  d_cand <- density(cand, from = 0, to = max(br)); d_cieg <- density(cieg, from = 0, to = max(br))
  hist(cand, breaks = br, freq = FALSE, col = "grey85", border = "white", xlab = "Altitud (m s.n.m.)", ylab = "Densidad",
       ylim = c(0, max(d_cand$y, d_cieg$y, hist(cand, br, plot = FALSE)$density, hist(cieg, br, plot = FALSE)$density)),
       main = sprintf("Altitud: %d candidatas frente a %d ciegas", length(cand), length(cieg)))
  hist(cieg, breaks = br, freq = FALSE, col = adjustcolor("#cb181d", 0.35), border = "white", add = TRUE)
  lines(d_cand, col = "grey30", lwd = 2.5); lines(d_cieg, col = "#cb181d", lwd = 2.5)
  rug(cieg, col = "#cb181d")
  legend("topright", c("Candidatas", "Ciegas"), fill = c("grey85", adjustcolor("#cb181d", 0.35)), col = c("grey30", "#cb181d"),
         lwd = 2.5, border = NA, bty = "n")
  lim <- range(cuantiles[, c("candidatas_m", "ciegas_m")])
  plot(cuantiles$candidatas_m, cuantiles$ciegas_m, pch = 19, col = "#cb181d", xlim = lim, ylim = lim, asp = 1,
       xlab = "Cuantil de las candidatas (m)", ylab = "Cuantil de las ciegas (m)",
       main = sprintf("Cuantiles (máx. |diferencia| = %.0f m)", max(abs(cuantiles$diferencia_m))))
  abline(0, 1, lty = 2, col = "grey40")
  text(cuantiles$candidatas_m, cuantiles$ciegas_m, sprintf("q%02.0f", 100 * PROBS), pos = 4, cex = 0.75)
})

res_comparacion$pixeles_compartidos <- compartidos
res <- c(estaciones = nrow(emp), con_coordenadas = sum(emp$con_coordenadas), dentro_malla = sum(emp$dentro_pixel), en_red = sum(emp$en_red),
         pixeles_ocupados = length(unique(emp$pixel[emp$dentro_pixel])), pixeles_compartidos = nrow(compartidos),
         candidatas_ciegas = length(cand), ciegas_en_red = length(cieg), ciegas_total = sum(emp$ciega),
         entrenamiento = sum(emp$rol == "entrenamiento"), altitud_mediana_candidatas = median(cand), altitud_mediana_ciegas = median(cieg),
         max_abs_dif_cuantil_altitud_m = max(abs(cuantiles$diferencia_m)),
         max_abs_dif_cuantil_rel_rango = max(abs(cuantiles$diferencia_rel_rango)))
res_comparacion$resumen_emparejamiento <- data.frame(indicador = names(res), valor = signif(res, 6))
res_comparacion$cuantiles_altitud_ciegas <- cuantiles

#Resultados que pasan a los bloques siguientes
NUCLEO <- c(NUCLEO, "res_comparacion")

#Comparación funcional estación - satélite: diferencia, covarianza cruzada, inferencia y modelo concurrente

rm(list = setdiff(ls(), NUCLEO))
set.seed(42)
suppressMessages(library(fda))

dir.create("figures/item3_comparacion", showWarnings = FALSE, recursive = TRUE)

N_DIAS     <- 365L
INTERIOR   <- c(15L, 351L)                             # soporte interior; bordes de apalancamiento [1, 14] y [352, 365]
INTERIOR_2 <- c(28L, 338L)                             # apalancamiento <= 1.5x en ambas fuentes (Ítem 2): sensibilidad
COTA_VALLE <- 1000                                     # fondo del valle geográfico del río Cauca (~900-1 000 m s.n.m.)
COTA_LADERA <- 1500                                    # laderas de las cordilleras Occidental y Central
MES_INI <- c(1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
MES_ETQ <- c("E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")
MESES   <- c("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
#Temporadas del régimen bimodal: secas dic-feb y jun-ago, lluviosas mar-may y sep-nov
TEMPORADA <- c("DEF seca", "DEF seca", "MAM lluviosa", "MAM lluviosa", "MAM lluviosa", "JJA seca", "JJA seca", "JJA seca",
               "SON lluviosa", "SON lluviosa", "SON lluviosa", "DEF seca")
t_dia   <- as.numeric(seq_len(N_DIAS))
mes_dia <- as.integer(format(as.Date(t_dia - 1, origin = "2001-01-01"), "%m"))
interior <- t_dia >= INTERIOR[1] & t_dia <= INTERIOR[2]

fecha_doy <- function(t) {
  d <- as.Date(round(t) - 1, origin = "2001-01-01")
  paste(format(d, "%d"), MESES[as.integer(format(d, "%m"))])
}
guardar <- function(archivo, ancho, alto, expr) {
  png(archivo, width = ancho, height = alto, res = 130)
  on.exit(dev.off())
  expr
}
eje_meses <- function() axis(1, at = MES_INI, labels = MES_ETQ)
sombrear_bordes <- function() {
  u <- par("usr")
  rect(c(u[1], INTERIOR[2] + 0.5), u[3], c(INTERIOR[1] - 0.5, u[2]), u[4], col = adjustcolor("grey60", 0.25), border = NA)
}
logico <- function(x) toupper(as.character(x)) %in% c("TRUE", "1")

#Pares estación-año (niveles A-B) con la curva del píxel emparejado del mismo año

est <- coef_est
sat <- coef_sat
#Cálculo de la proyección del satélite a la base de las estaciones en la rejilla diaria: c = (Phi_e' Phi_e)^-1 Phi_e' Phi_s c_s.
#Las dos fuentes quedan con la misma base; la constante está en ambas bases, así que el total anual no cambia.
#El satélite en su base propia queda como sensibilidad.
K_SAT_PROPIO <- sat$K
C_SAT_PROPIO <- sat$C                                                    # Dim: (11, n_sat)
P_e0  <- bspline_base(t_dia, est$K)
sat$C <- solve(crossprod(P_e0), crossprod(P_e0, bspline_base(t_dia, K_SAT_PROPIO) %*% sat$C))   # Dim: (6, n_sat)
sat$K <- est$K
rm(P_e0)
emp <- emparejamiento
emp[c("dentro_pixel", "ciega")] <- lapply(emp[c("dentro_pixel", "ciega")], logico)
oni <- oni_anual
decimas <- read.csv("data/processed/metadatos/estaciones_candidatas_decimas.csv")$station_id

ab     <- est$curvas$nivel %in% c("A", "B")
cur    <- est$curvas[ab, ]
f_emp  <- match(cur$station_id, emp$station_id)
dentro <- emp$dentro_pixel[f_emp] %in% TRUE
j_sat  <- match(emp$pixel[f_emp] * 1e4 + cur$year, sat$pixel * 1e4 + sat$anio)
ok     <- dentro & !is.na(j_sat)
pares  <- data.frame(station_id = cur$station_id[ok], year = cur$year[ok], nivel = cur$nivel[ok], pixel = emp$pixel[f_emp[ok]],
                     altitud_m = emp$altitud_m[f_emp[ok]], ciega = emp$ciega[f_emp[ok]],
                     fase = oni$fase[match(cur$year[ok], oni$anio)], decimas = cur$station_id[ok] %in% decimas)
pares$piso <- cut(pares$altitud_m, c(-Inf, COTA_VALLE, COTA_LADERA, Inf), right = FALSE,
                  labels = c(sprintf("Valle (< %d m)", COTA_VALLE), sprintf("Intermedio (%d-%d m)", COTA_VALLE, COTA_LADERA),
                             sprintf("Ladera (>= %d m)", COTA_LADERA)))
#Subregión el primer dígito del código IDEAM da la vertiente (5 = Pacífico); en la cuenca del Cauca se usa el piso altitudinal
AREA_PACIFICO <- "5"
pares$vertiente <- ifelse(substr(sprintf("%.0f", pares$station_id), 1, 1) == AREA_PACIFICO, "Pacífico", "Cauca")
pares$subregion <- factor(ifelse(pares$vertiente == "Pacífico", "Pacífico", paste("Cauca", tolower(as.character(pares$piso)))),
                          levels = c("Pacífico", paste("Cauca", tolower(levels(pares$piso)))))

Phi_e <- bspline_base(t_dia, est$K)                                      # Dim: (365, K_est = 6)
Phi_s <- bspline_base(t_dia, sat$K)                                      # Dim: (365, 6) base común de las dos fuentes
C_e   <- est$C[, ab, drop = FALSE][, ok, drop = FALSE]                   # Dim: (6, N)
C_s   <- sat$C[, j_sat[ok], drop = FALSE]                                # Dim: (6, N)
Y     <- Phi_e %*% C_e                                                   # Dim: (365, N) estación, mm/día (MCO lineal sin truncar)
X     <- Phi_s %*% C_s                                                   # Dim: (365, N) satélite en el píxel emparejado, mm/día
D     <- Y - X                                                           # Dim: (365, N) d_i(t) = estación - satélite
N     <- ncol(D)
stopifnot(N > 2L, !anyNA(D), all(!is.na(pares$fase)))

#Banda de la diferencia entre fuentes: d_bar(t) ± 2 EE(t), con EE(t) = DE_d(t) / sqrt(N)

banda <- function(cols, grupo = NULL, M = D) {
  Dg <- M[, cols, drop = FALSE]
  if (!is.null(grupo)) {
    g  <- grupo[cols]
    Dg <- t(rowsum(t(Dg), g) / drop(rowsum(rep(1, length(g)), g)))      # Dim: (365, n_estaciones): media por estación
  }
  n  <- ncol(Dg)
  m  <- rowMeans(Dg)
  ee <- sqrt(rowSums((Dg - m)^2) / (n - 1)) / sqrt(n)
  data.frame(t = t_dia, d = m, ee = ee, li = m - 2 * ee, ls = m + 2 * ee, n = n)
}

intervalos <- function(b, rango = INTERIOR) {
  zona  <- b$t >= rango[1] & b$t <= rango[2]
  signo <- ifelse(!zona, 0L, ifelse(b$li > 0, 1L, ifelse(b$ls < 0, -1L, 0L)))
  r   <- rle(signo)
  fin <- cumsum(r$lengths); ini <- fin - r$lengths + 1L
  k   <- which(r$values != 0L)
  data.frame(inicio = ini[k], fin = fin[k], dias = r$lengths[k],
             fecha_inicio = fecha_doy(ini[k]), fecha_fin = fecha_doy(fin[k]),
             sentido = ifelse(r$values[k] > 0, "estación > satélite (subestima)", "satélite > estación (sobrestima)"),
             d_medio_mm_dia = vapply(k, function(i) mean(b$d[ini[i]:fin[i]]), numeric(1)),
             stringsAsFactors = FALSE)
}

resumen_banda <- function(b, cols) {
  y_med <- mean(Y[, cols])
  c(n = b$n[1], d_medio_interior = mean(b$d[interior]), sesgo_anual_mm = sum(b$d),
    pbias_pct = 100 * sum(b$d) / (N_DIAS * y_med),
    dias_estacion_mayor = sum(interior & b$li > 0), dias_satelite_mayor = sum(interior & b$ls < 0),
    dias_sin_diferencia = sum(interior & b$li <= 0 & b$ls >= 0),
    ee_borde_sobre_interior = mean(b$ee[!interior]) / mean(b$ee[interior]))
}

todos <- seq_len(N)
b_par <- banda(todos)
b_est <- banda(todos, pares$station_id)
int_par   <- intervalos(b_par)
int_est   <- intervalos(b_est)
int_par_2 <- intervalos(b_par, INTERIOR_2)

#Sensibilidad a la resolución satélite en su base propia (K de su VCG), con bases distintas en cada fuente;
#S_e = Phi_e (Phi_e' Phi_e)^-1 Phi_e' lo devuelve a la base común
S_e   <- matriz_S(Phi_e)                                                 # Dim: (365, 365), idempotente, tr = K de las estaciones
X_ke  <- bspline_base(t_dia, K_SAT_PROPIO) %*% C_SAT_PROPIO[, j_sat[ok], drop = FALSE]   # Dim: (365, N) satélite en K = 11
D_ke  <- Y - X_ke
b_ke  <- banda(todos, M = D_ke)
int_ke <- intervalos(b_ke)
r_diag_ke <- rowSums((X_ke - rowMeans(X_ke)) * (Y - rowMeans(Y))) / sqrt(rowSums((X_ke - rowMeans(X_ke))^2) * rowSums((Y - rowMeans(Y))^2))
stopifnot(max(abs(S_e %*% S_e - S_e)) <= 1e-12, abs(sum(diag(S_e)) - est$K) <= 1e-10, max(abs(S_e %*% X_ke - X)) <= 1e-10)

base_fd <- function(K) create.bspline.basis(rangeval = c(1, N_DIAS), nbasis = K, norder = 4L)
fdY <- fd(C_e, base_fd(est$K))
fdX <- fd(C_s, base_fd(sat$K))
verif_dif <- c(base_est = max(abs(eval.basis(t_dia, base_fd(est$K)) - Phi_e)),
               base_sat = max(abs(eval.basis(t_dia, base_fd(sat$K)) - Phi_s)),
               media_dif_mean_fd = max(abs(b_par$d - drop(eval.fd(t_dia, mean.fd(fdY)) - eval.fd(t_dia, mean.fd(fdX))))),
               media_dif_coeficientes = max(abs(b_par$d - drop(Phi_e %*% rowMeans(C_e) - Phi_s %*% rowMeans(C_s)))))
stopifnot(all(verif_dif <= 1e-12))

#Covarianza y correlación cruzada satélite (s) - estación (t): v_XY(s,t) = phi(s)' Sigma_cd psi(t)

Xc   <- X - rowMeans(X)
Yc   <- Y - rowMeans(Y)
V_xy <- tcrossprod(Xc, Yc) / (N - 1)                                    # Dim: (365 s satélite, 365 t estación)
S_cd <- cov(t(C_s), t(C_e))                                             # Dim: (6, 6) covarianza cruzada de coeficientes
V_xy_coef <- Phi_s %*% S_cd %*% t(Phi_e)
v_x  <- rowSums(Xc^2) / (N - 1)
v_y  <- rowSums(Yc^2) / (N - 1)
R_xy <- V_xy / sqrt(outer(v_x, v_y))
r_diag <- diag(R_xy)

verif_cruz <- c(cov_cruzada_var_fd = max(abs(V_xy - eval.bifd(t_dia, t_dia, var.fd(fdX, fdY)))) / max(abs(V_xy)),
                cov_cruzada_coeficientes = max(abs(V_xy - V_xy_coef)) / max(abs(V_xy)),
                cor_cruzada_cor_fd = max(abs(R_xy - cor.fd(t_dia, fdX, t_dia, fdY))))
stopifnot(all(verif_cruz[c("cov_cruzada_var_fd", "cov_cruzada_coeficientes")] <= 1e-10), verif_cruz["cor_cruzada_cor_fd"] <= 1e-8)

cor_rezago <- function(R, h) { i <- seq(INTERIOR[1], INTERIOR[2] - abs(h)); mean(R[cbind(i + max(-h, 0), i + max(h, 0))]) }
rezagos    <- -60:60
r_rezago   <- vapply(rezagos, function(h) cor_rezago(R_xy, h), numeric(1))   # h > 0: el satélite (s) antecede a la estación (t = s + h)

#Correlación cruzada de anomalías interanuales: cada curva centrada en la media de su estación
centrar_estacion <- function(M) M - t(rowsum(t(M), pares$station_id) / drop(rowsum(rep(1, N), pares$station_id)))[, as.character(pares$station_id)]
Xa <- centrar_estacion(X); Ya <- centrar_estacion(Y)                  # Dim: (365, N) anomalías respecto de la climatología de la estación
r_diag_anom <- rowSums(Xa * Ya) / sqrt(rowSums(Xa^2) * rowSums(Ya^2))

#Covarianza de cada fuente sobre los mismos pares: v(s,t) = Phi(s)' Sigma_c Phi(t)

cov_fuente <- function(Phi, C) {
  Sigma <- cov(t(C))
  G <- Phi %*% Sigma %*% t(Phi)                                         # Dim: (365, 365)
  list(G = G, sd = sqrt(diag(G)), R = G / tcrossprod(sqrt(diag(G))))
}
cov_e <- cov_fuente(Phi_e, C_e)
cov_s <- cov_fuente(Phi_s, C_s)
stopifnot(max(abs(cov_e$G - tcrossprod(Yc) / (N - 1))) / max(cov_e$G) <= 1e-10,
          max(abs(cov_s$G - tcrossprod(Xc) / (N - 1))) / max(cov_s$G) <= 1e-10)
memoria <- function(R) vapply(1:120, function(h) cor_rezago(R, h), numeric(1))
mem_e <- memoria(cov_e$R); mem_s <- memoria(cov_s$R)
mem_tabla <- rbind(estaciones = mem_e[c(15, 30, 60, 90)], satelite = mem_s[c(15, 30, 60, 90)])
colnames(mem_tabla) <- paste0("h", c(15, 30, 60, 90))
var_total <- c(estaciones = sum(diag(cov_e$G)), satelite = sum(diag(cov_s$G)))

por_temporada <- function(v) tapply(v[interior], TEMPORADA[mes_dia][interior], mean)
tabla_temporada <- data.frame(
  temporada = names(por_temporada(b_par$d)),
  media_estacion_mm_dia = por_temporada(rowMeans(Y)), media_satelite_mm_dia = por_temporada(rowMeans(X)),
  d_medio_mm_dia = por_temporada(b_par$d), sd_estacion = por_temporada(cov_e$sd), sd_satelite = por_temporada(cov_s$sd),
  razon_sd_est_sat = por_temporada(cov_e$sd / cov_s$sd), r_diagonal = por_temporada(r_diag), row.names = NULL)

#Sesgo por subregión, fase ENSO y sensibilidad a las estaciones candidatas a décimas

grupos <- c(list(Todos = todos),
            split(todos, pares$subregion), list(`Cauca (todos los pisos)` = which(pares$vertiente == "Cauca")),
            split(todos, factor(pares$fase, c("El Niño", "Neutro", "La Niña"))),
            list(`Sin estaciones candidatas a décimas` = which(!pares$decimas),
                 `Solo estaciones candidatas a décimas` = which(pares$decimas)))
grupos <- grupos[vapply(grupos, length, integer(1)) > 2L]
bandas <- lapply(grupos, banda)
tabla_grupos <- data.frame(grupo = names(bandas), t(mapply(resumen_banda, bandas, grupos)),
                           estaciones = vapply(grupos, function(g) length(unique(pares$station_id[g])), integer(1)),
                           check.names = FALSE, row.names = NULL)
intervalos_grupos <- do.call(rbind, Map(function(b, nombre) {
  i <- intervalos(b); if (nrow(i)) cbind(grupo = nombre, i) else NULL }, bandas, names(bandas)))

bandas_ke <- lapply(grupos, banda, M = D_ke)
tabla_grupos_ke <- data.frame(grupo = names(bandas_ke), t(mapply(resumen_banda, bandas_ke, grupos)), check.names = FALSE, row.names = NULL)
contraste <- function(a, b, bandas = get("bandas", envir = globalenv())) {
  d  <- bandas[[a]]$d - bandas[[b]]$d
  ee <- sqrt(bandas[[a]]$ee^2 + bandas[[b]]$ee^2)                       # grupos disjuntos: varianzas de medias independientes se suman
  data.frame(t = t_dia, d = d, ee = ee, li = d - 2 * ee, ls = d + 2 * ee, n = NA)
}
NM_SUB <- levels(pares$subregion)
CONTRASTES <- setNames(list(c(NM_SUB[1], "Cauca (todos los pisos)"), c(NM_SUB[2], NM_SUB[4]), c("El Niño", "La Niña")),
                       c("Pacífico - Cauca", "Cauca: valle - ladera", "El Niño - La Niña"))
contrastes <- lapply(CONTRASTES, function(g) contraste(g[1], g[2]))
int_contrastes <- do.call(rbind, Map(function(b, nombre) { i <- intervalos(b); if (nrow(i)) cbind(contraste = nombre, i) else NULL },
                                     contrastes, names(contrastes)))
contrastes_ke <- lapply(CONTRASTES, function(g) contraste(g[1], g[2], bandas_ke))
int_contrastes_ke <- do.call(rbind, Map(function(b, nombre) { i <- intervalos(b); if (nrow(i)) cbind(contraste = nombre, i) else NULL },
                                        contrastes_ke, names(contrastes_ke)))
NM_SIN <- "Sin estaciones candidatas a décimas"
COLS_S <- c("n", "d_medio_interior", "sesgo_anual_mm", "pbias_pct", "dias_estacion_mayor", "dias_satelite_mayor", "dias_sin_diferencia")
fila_grupo <- function(g) as.numeric(unlist(tabla_grupos[tabla_grupos$grupo == g, COLS_S]))
cambio <- abs(bandas$Todos$d - bandas[[NM_SIN]]$d) * interior
sens_decimas <- data.frame(indicador = c(COLS_S, "max_abs_cambio_d_bar_mm_dia", "dia_max_cambio"),
                           con_decimas = c(fila_grupo("Todos"), NA, NA),
                           sin_decimas = c(fila_grupo(NM_SIN), max(cambio), which.max(cambio)))
tabla_grupos
n_dec <- length(unique(pares$station_id[pares$decimas]))

#Funciones de gráfica de bandas; el eje y se fija con el soporte interior
rango_int <- function(...) range(unlist(lapply(list(...), function(b) c(b$li[interior], b$ls[interior]))), 0)
panel_banda <- function(b, col, main, ylim = NULL, nuevo = TRUE, ylab = "d(t) = estación - satélite (mm/día)", marcas = TRUE, alto_marca = 0.03) {
  if (nuevo) {
    plot(t_dia, b$d, type = "n", ylim = if (is.null(ylim)) rango_int(b) else ylim, xaxt = "n",
         xlab = "Día del año", ylab = ylab, main = main)
    eje_meses(); sombrear_bordes(); abline(h = 0, lty = 3)
  }
  sig <- interior & (b$li > 0 | b$ls < 0)
  polygon(c(t_dia, rev(t_dia)), c(b$li, rev(b$ls)), col = adjustcolor(col, 0.2), border = NA)
  lines(t_dia, b$d, col = col, lwd = 2.5)
  if (marcas) points(t_dia[sig], rep(par("usr")[3] + alto_marca * diff(par("usr")[3:4]), sum(sig)), pch = 15, cex = 0.35, col = col)
}

guardar("figures/item3_comparacion/diferencia_bandas.png", 1700, 700, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
  muestra <- sort(sample(todos, min(300L, N)))
  matplot(t_dia, D[, muestra], type = "l", lty = 1, col = adjustcolor("grey40", 0.15), xaxt = "n",
          xlab = "Día del año", ylab = "d_i(t) = estación - satélite (mm/día)",
          main = sprintf("Diferencias por par estación-año (%d de %d)", length(muestra), N))
  eje_meses(); sombrear_bordes(); abline(h = 0, lty = 3); lines(t_dia, b_par$d, lwd = 3, col = "#08306b")
  panel_banda(b_par, "#08306b", "Diferencia media ± 2 EE(t)", ylim = rango_int(b_par, b_est, b_ke))
  lines(t_dia, b_est$li, lty = 2, col = "#d95f02"); lines(t_dia, b_est$ls, lty = 2, col = "#d95f02")
  polygon(c(t_dia, rev(t_dia)), c(b_ke$li, rev(b_ke$ls)), col = adjustcolor("#1b9e77", 0.2), border = NA)
  lines(t_dia, b_ke$d, col = "#1b9e77", lwd = 2.5)
  legend("topright", c(sprintf("d(t) ± 2 EE, unidad par (N = %d)", N), sprintf("± 2 EE, unidad estación (n = %d)", b_est$n[1]),
                       sprintf("Satélite en su base K = %d (bases distintas)", K_SAT_PROPIO), "Días con banda sin cero (par)", "Borde: extrapolación"),
         col = c("#08306b", "#d95f02", "#1b9e77", "#08306b", NA), lty = c(1, 2, 1, NA, NA), lwd = c(2.5, 1, 2.5, NA, NA),
         pch = c(NA, NA, NA, 15, 15), pt.cex = c(1, 1, 1, 0.8, 2), bg = "white", cex = 0.7)
})

guardar("figures/item3_comparacion/correlacion_cruzada.png", 1700, 750, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3, 1))
  image(t_dia, t_dia, R_xy, zlim = c(-1, 1), col = hcl.colors(40, "Blue-Red 3"), xaxt = "n", yaxt = "n",
        xlab = "s: satélite (día del año)", ylab = "t: estación (día del año)", main = "Correlación cruzada r_XY(s, t)")
  contour(t_dia, t_dia, R_xy, levels = seq(-1, 1, 0.1), labcex = 0.6, add = TRUE)
  eje_meses(); axis(2, at = MES_INI, labels = MES_ETQ); abline(0, 1, lty = 3, col = "black")
  plot(t_dia, r_diag, type = "l", lwd = 3, col = "#08306b", ylim = c(min(0, r_diag, r_diag_anom), 1), xaxt = "n",
       xlab = "Día del año", ylab = expression(r[XY](t, t)), main = "Diagonal de la correlación cruzada")
  eje_meses(); sombrear_bordes(); abline(h = 0, lty = 3)
  lines(t_dia, r_diag_anom, lwd = 2.5, col = "#d95f02")
  lines(t_dia, r_diag_ke, lwd = 2, lty = 2, col = "#1b9e77")
  legend("bottomright", c(sprintf("Pares (mediana interior %.2f)", median(r_diag[interior])),
                          sprintf("Satélite en su base K = %d, bases distintas (mediana %.2f)", K_SAT_PROPIO, median(r_diag_ke[interior])),
                          sprintf("Anomalías interanuales (mediana %.2f)", median(r_diag_anom[interior])),
                          sprintf("Rezago de máxima correlación: %d días", rezagos[which.max(r_rezago)])),
         lty = c(1, 2, 1, NA), lwd = c(3, 2, 2.5, NA), col = c("#08306b", "#1b9e77", "#d95f02", NA), bg = "white", cex = 0.72)
})

guardar("figures/item3_comparacion/covarianzas.png", 1800, 650, {
  par(mfrow = c(1, 3), mar = c(4.2, 4.5, 3, 1))
  niveles_e <- pretty(range(cov_e$G), 10)
  contour(t_dia, t_dia, cov_e$G, levels = niveles_e, labcex = 0.6, xaxt = "n", yaxt = "n",
          xlab = "s (día)", ylab = "t (día)", main = expression("Estaciones: v(s,t) (mm/día)"^2))
  eje_meses(); axis(2, at = MES_INI, labels = MES_ETQ)
  contour(t_dia, t_dia, cov_s$G, levels = pretty(range(cov_s$G), 10), labcex = 0.6, xaxt = "n", yaxt = "n",
          xlab = "s (día)", ylab = "t (día)", main = expression("Satélite: v(s,t) (mm/día)"^2))
  eje_meses(); axis(2, at = MES_INI, labels = MES_ETQ)
  plot(1:120, mem_e, type = "l", lwd = 2.5, col = "#1f77b4", ylim = c(min(mem_e, mem_s, 0), 1),
       xlab = "Separación h (días)", ylab = "Correlación media r(t, t + h)", main = "Memoria temporal de cada fuente")
  lines(1:120, mem_s, lwd = 2.5, col = "#d95f02"); abline(h = 0.5, lty = 3)
  legend("topright", c(sprintf("Estaciones (sd media %.1f mm/día)", mean(cov_e$sd[interior])),
                       sprintf("Satélite (sd media %.1f mm/día)", mean(cov_s$sd[interior]))),
         col = c("#1f77b4", "#d95f02"), lwd = 2.5, bg = "white", cex = 0.8)
})

COL_SUB  <- setNames(c("#0868ac", "#1b9e77", "#7570b3", "#d95f02"), levels(pares$subregion))
COL_FASE <- c(`El Niño` = "#cb181d", Neutro = "grey35", `La Niña` = "#2171b5")
guardar("figures/item3_comparacion/sesgo_altitud_enso.png", 1800, 1250, {
  par(mfrow = c(2, 2), mar = c(4.2, 4.5, 3, 1))
  yl <- do.call(rango_int, bandas[names(COL_SUB)])
  panel_banda(bandas[[names(COL_SUB)[1]]], COL_SUB[1], "Sesgo por subregión y piso", ylim = yl, marcas = FALSE)
  invisible(Map(function(nm, cl) panel_banda(bandas[[nm]], cl, "", nuevo = FALSE, marcas = FALSE), names(COL_SUB)[-1], COL_SUB[-1]))
  legend("topright", sprintf("%s: %d pares, %d est.", names(COL_SUB), tabla_grupos$n[match(names(COL_SUB), tabla_grupos$grupo)],
                             tabla_grupos$estaciones[match(names(COL_SUB), tabla_grupos$grupo)]),
         col = COL_SUB, lwd = 2.5, bg = "white", cex = 0.7)
  panel_banda(contrastes[[1]], COL_SUB[1], "Contrastes ± 2 EE: Pacífico - Cauca y valle - ladera del Cauca",
              ylim = rango_int(contrastes[[1]], contrastes[[2]]), ylab = "Diferencia de d(t) entre grupos (mm/día)")
  panel_banda(contrastes[[2]], COL_SUB[4], "", nuevo = FALSE, alto_marca = 0.07)
  legend("topright", c(names(contrastes)[1:2], "Días con banda sin cero"), col = c(COL_SUB[c(1, 4)], "grey30"), lwd = c(2.5, 2.5, NA),
         pch = c(NA, NA, 15), bg = "white", cex = 0.75)
  yl <- do.call(rango_int, bandas[names(COL_FASE)])
  panel_banda(bandas$`El Niño`, COL_FASE[1], "Sesgo por fase ENSO (ONI, NOAA)", ylim = yl, marcas = FALSE)
  panel_banda(bandas$Neutro, COL_FASE[2], "", nuevo = FALSE, marcas = FALSE)
  panel_banda(bandas$`La Niña`, COL_FASE[3], "", nuevo = FALSE, marcas = FALSE)
  legend("topright", sprintf("%s: %d pares", names(COL_FASE), tabla_grupos$n[match(names(COL_FASE), tabla_grupos$grupo)]),
         col = COL_FASE, lwd = 2.5, bg = "white", cex = 0.75)
  panel_banda(contrastes$`El Niño - La Niña`, "black", "Contraste El Niño - La Niña ± 2 EE", ylab = "d_Niño(t) - d_Niña(t) (mm/día)")
})

guardar("figures/item3_comparacion/sensibilidad_decimas.png", 1300, 700, {
  par(mar = c(4.2, 4.5, 3, 1))
  b_sin <- bandas[[NM_SIN]]
  panel_banda(b_par, "#08306b", sprintf("Sensibilidad a las %d estaciones con totales > 12 000 mm/año", n_dec), ylim = rango_int(b_par, b_sin), marcas = FALSE)
  panel_banda(b_sin, "#e6550d", "", nuevo = FALSE, marcas = FALSE)
  legend("topright", c(sprintf("Con las %d estaciones (N = %d)", n_dec, N), sprintf("Sin las %d estaciones (N = %d)", n_dec, b_sin$n[1])),
         col = c("#08306b", "#e6550d"), lwd = 2.5, bg = "white", cex = 0.8)
})

res_comparacion$curva_diferencia <- data.frame(
  t = t_dia, fecha = fecha_doy(t_dia), interior = interior, media_estacion = rowMeans(Y), media_satelite = rowMeans(X),
  d_bar = b_par$d, ee_par = b_par$ee, li_par = b_par$li, ls_par = b_par$ls,
  ee_estacion = b_est$ee, li_estacion = b_est$li, ls_estacion = b_est$ls, d_bar_satelite_K11 = b_ke$d, ee_satelite_K11 = b_ke$ee,
  r_diagonal = r_diag, r_diagonal_anomalias = r_diag_anom, r_diagonal_satelite_K11 = r_diag_ke,
  sd_estacion = cov_e$sd, sd_satelite = cov_s$sd)
res_comparacion$intervalos_sesgo <- rbind(cbind(unidad = "par", rango = "15-351", int_par), cbind(unidad = "estacion", rango = "15-351", int_est),
                                          cbind(unidad = "par", rango = "28-338", int_par_2), cbind(unidad = "par_satelite_K11", rango = "15-351", int_ke))
res_comparacion$sesgo_por_grupo <- tabla_grupos
res_comparacion$intervalos_por_grupo <- intervalos_grupos
res_comparacion$intervalos_contrastes <- rbind(cbind(resolucion = "misma_base", int_contrastes), cbind(resolucion = "satelite_K11", int_contrastes_ke))
res_comparacion$sesgo_por_grupo_satelite_K11 <- tabla_grupos_ke
res_comparacion$covarianza_por_temporada <- tabla_temporada
res_comparacion$sensibilidad_decimas <- sens_decimas
res_comparacion$pares_item3 <- pares
res <- c(pares = N, estaciones = length(unique(pares$station_id)), pixeles = length(unique(pares$pixel)),
         d_medio_interior = mean(b_par$d[interior]), sesgo_anual_mm = sum(b_par$d),
         dias_sin_cero_par = sum(interior & (b_par$li > 0 | b_par$ls < 0)), dias_sin_cero_estacion = sum(interior & (b_est$li > 0 | b_est$ls < 0)),
         r_diag_mediana = median(r_diag[interior]), r_diag_min = min(r_diag[interior]), r_diag_max = max(r_diag[interior]),
         rezago_max_cor = rezagos[which.max(r_rezago)], var_total_razon_est_sat = unname(var_total[1] / var_total[2]),
         memoria_r30_estaciones = mem_tabla["estaciones", "h30"], memoria_r30_satelite = mem_tabla["satelite", "h30"],
         memoria_r90_estaciones = mem_tabla["estaciones", "h90"], memoria_r90_satelite = mem_tabla["satelite", "h90"],
         r_diag_anomalias_mediana = median(r_diag_anom[interior]), r_rezago_0 = r_rezago[rezagos == 0],
         dias_sin_cero_satelite_K11 = sum(interior & (b_ke$li > 0 | b_ke$ls < 0)), r_diag_mediana_satelite_K11 = median(r_diag_ke[interior]),
         amplitud_d_K11 = diff(range(b_ke$d[interior])), amplitud_d_Kest = diff(range(b_par$d[interior])),
         ee_borde_sobre_interior = mean(b_par$ee[!interior]) / mean(b_par$ee[interior]),
         verif_media_mean_fd = unname(verif_dif["media_dif_mean_fd"]), verif_cov_cruzada_var_fd = unname(verif_cruz["cov_cruzada_var_fd"]),
         verif_cor_cruzada_cor_fd = unname(verif_cruz["cor_cruzada_cor_fd"]))
res_comparacion$resumen_sesgo <- data.frame(indicador = names(res), valor = signif(res, 6))

#Inferencia funcional sobre d(t) en [15, 351]: prueba pareada y ANOVA funcional de una vía por subregión y por fase ENSO
#Unidades: la estación (media de sus años) por subregión y el año sin efecto de sitio por fase ENSO

B_PERM <- 5000L                                        # permutaciones
B_CFF  <- 2000L                                        # trayectorias gaussianas de Cuevas et al. y réplicas de fanova.tests()
ALFA   <- 0.05
t_int  <- t_dia[interior]
M_int  <- length(t_int)
w_int  <- rep(1, M_int); w_int[c(1L, M_int)] <- 0.5   # regla del trapecio en el soporte interior
set.seed(42)
suppressWarnings(suppressMessages(library(fdANOVA)))   # aviso de compilación con R 4.6.2, sin efecto

#Función de trazas de la covarianza combinada con la corrección de sesgo de Zhang (2013)
trazas_zhang <- function(G, w, N_gl) {
  DG  <- G * w                                         # diag(w) G
  tr1 <- sum(diag(DG)); tr2 <- sum(DG * t(DG))
  tr2_c <- N_gl^2 / ((N_gl - 1) * (N_gl + 2)) * (tr2 - tr1^2 / N_gl)
  tr1_c <- N_gl * (N_gl + 1) / ((N_gl - 1) * (N_gl + 2)) * (tr1^2 - 2 * tr2 / (N_gl + 1))
  c(tr_gamma = tr1, tr_gamma2 = tr2, beta = tr2_c / tr1, d = tr1_c / tr2_c)
}

#Función que agrupa en intervalos los días consecutivos que cumplen una condición
intervalos_dias <- function(ind) {
  r <- rle(ind); fin <- cumsum(r$lengths); ini <- fin - r$lengths + 1L; k <- which(r$values)
  data.frame(inicio = t_int[ini[k]], fin = t_int[fin[k]], dias = r$lengths[k], fecha_inicio = fecha_doy(t_int[ini[k]]),
             fecha_fin = fecha_doy(t_int[fin[k]]), stringsAsFactors = FALSE)
}

#Función de prueba pareada: T(t) = |d_bar(t)| / (DE(t) / sqrt(n)) por cambio de signo y F pareada con el d de Zhang
prueba_pareada <- function(Dg, w = w_int) {
  n <- ncol(Dg); m <- rowMeans(Dg); S2 <- rowSums(Dg^2)
  t_abs  <- function(mb) abs(mb) / sqrt(pmax(S2 - n * mb^2, 0) / ((n - 1) * n))
  exacto <- 2^n <= 20000
  Sg <- if (exacto) t(as.matrix(expand.grid(rep(list(c(-1, 1)), n)))) else matrix(sample(c(-1, 1), n * B_PERM, replace = TRUE), n)   # Dim: (n, B)
  T_obs <- t_abs(m)
  T_nul <- t_abs(Dg %*% Sg / n)                                                    # Dim: (M, B) estadísticos bajo H0
  Tmax_nul <- apply(T_nul, 2, max)
  #Valor p: en el caso exacto B incluye la asignación observada; si no, se suma 1
  p_de <- function(cuenta, B) if (exacto) cuenta / B else (1 + cuenta) / (B + 1)
  tz <- trazas_zhang(tcrossprod(Dg - m) / (n - 1), w, n - 1)
  F_par <- n * sum(w * m^2) / tz[["tr_gamma"]]
  list(n = n, exacto = exacto, B = ncol(Sg), d_bar = m, ee = sqrt((S2 - n * m^2) / (n - 1)) / sqrt(n), T = T_obs,
       c_puntual = apply(T_nul, 1, quantile, probs = 1 - ALFA), c_max = unname(quantile(Tmax_nul, 1 - ALFA)),
       p_puntual = p_de(rowSums(T_nul >= T_obs - 1e-12), ncol(Sg)), p_global = p_de(sum(Tmax_nul >= max(T_obs) - 1e-12), ncol(Sg)),
       F = F_par, d = tz[["d"]], p_F = pf(F_par, tz[["d"]], (n - 1) * tz[["d"]], lower.tail = FALSE))
}

#Función de sumas del ANOVA funcional de una vía: F puntual, L2, tipo F, GPF y Cuevas et al.
sumas_anova <- function(Xg, grupo, w) {
  k <- nlevels(grupo); n_i <- as.numeric(table(grupo)); n <- sum(n_i); g <- as.integer(grupo)
  Z      <- model.matrix(~ grupo - 1)                                              # Dim: (n, k) indicadora de grupo
  Medias <- Xg %*% Z %*% diag(1 / n_i)                                             # Dim: (M, k)
  m_glob <- rowMeans(Xg)
  SSR_t  <- as.numeric((Medias - m_glob)^2 %*% n_i)
  SSE_t  <- rowSums((Xg - Medias[, g])^2)
  SST_t  <- rowSums((Xg - m_glob)^2)
  F_t    <- (SSR_t / (k - 1)) / (SSE_t / (n - k))                                  # F puntual
  G      <- tcrossprod(Xg - Medias[, g]) / (n - k)                                 # Dim: (M, M) covarianza combinada, N = n - k
  tz     <- trazas_zhang(G, w, n - k)
  SSR_n  <- sum(w * SSR_t); SSE_n <- sum(w * SSE_t); F_n <- (SSR_n / (k - 1)) / (SSE_n / (n - k))
  s_t    <- sqrt(diag(G)); DR <- G / (s_t %o% s_t) * w                              # correlación rho(s, t) para la GPF
  tr_rho <- sum(diag(DR)); tr_rho2 <- sum(DR * t(DR))
  pares_g <- combn(k, 2)
  V_n <- sum(apply(pares_g, 2, function(p) n_i[p[1]] * sum(w * (Medias[, p[1]] - Medias[, p[2]])^2)))   # estadístico de Cuevas et al.
  colnames(Medias) <- levels(grupo)
  list(niveles = levels(grupo), k = k, n_i = n_i, n = n, g = g, Z = Z, Medias = Medias, m_glob = m_glob, SSR_t = SSR_t, SSE_t = SSE_t, SST_t = SST_t, F_t = F_t,
       tz = tz, SSR_n = SSR_n, SSE_n = SSE_n, F_n = F_n, T_GPF = sum(w * F_t), V_n = V_n, pares_g = pares_g,
       b_rho = tr_rho2 / ((k - 1) * tr_rho), d_rho = (k - 1) * tr_rho^2 / tr_rho2,
       p_L2 = pchisq(SSR_n / tz[["beta"]], (k - 1) * tz[["d"]], lower.tail = FALSE),
       p_F = pf(F_n, (k - 1) * tz[["d"]], (n - k) * tz[["d"]], lower.tail = FALSE),
       err_SST = max(abs(SST_t - (SSR_t + SSE_t))) / max(SST_t),                  # SST(t) = SSR(t) + SSE(t)
       err_SSE_traza = abs(SSE_n - (n - k) * tz[["tr_gamma"]]) / SSE_n)            # SSE_n = (n - k) tr(gamma)
}

anova_funcional <- function(Xg, grupo, w = w_int) {
  a <- sumas_anova(Xg, grupo, w); k <- a$k; n <- a$n; n_i <- a$n_i; M <- nrow(Xg)
  a$p_GPF <- pchisq(a$T_GPF / a$b_rho, a$d_rho, lower.tail = FALSE)
  #Permutación de etiquetas: SST(t) no cambia, basta recalcular SSR(t)
  SSR_nul  <- vapply(seq_len(B_PERM), function(b) as.numeric((Xg %*% a$Z[sample(n), , drop = FALSE] %*% diag(1 / n_i) - a$m_glob)^2 %*% n_i),
                     numeric(M))                                                   # Dim: (M, B)
  F_nul    <- (SSR_nul / (k - 1)) / ((a$SST_t - SSR_nul) / (n - k))
  SSRn_nul <- colSums(w * SSR_nul)
  Fn_nul   <- (SSRn_nul / (k - 1)) / ((sum(w * a$SST_t) - SSRn_nul) / (n - k))
  Fmax_nul <- apply(F_nul, 2, max)
  p_perm   <- function(nul, obs) (1 + sum(nul >= obs - 1e-12)) / (length(nul) + 1)
  a$F_max <- max(a$F_t)
  a$p_F_perm <- p_perm(Fn_nul, a$F_n); a$p_GPF_perm <- p_perm(colSums(w * F_nul), a$T_GPF); a$p_Fmax_perm <- p_perm(Fmax_nul, a$F_max)
  a$c_puntual <- apply(F_nul, 1, quantile, probs = 1 - ALFA)                       # curva crítica puntual
  a$c_max     <- unname(quantile(Fmax_nul, 1 - ALFA))                              # valor crítico del máximo
  a$p_puntual <- (1 + rowSums(F_nul >= a$F_t - 1e-12)) / (B_PERM + 1)
  #Prueba de Cuevas, Febrero y Fraiman con procesos gaussianos de la covarianza de cada grupo
  Raiz <- lapply(seq_len(k), function(i) {
    R_i <- Xg[, a$g == i, drop = FALSE] - a$Medias[, i]
    E <- eigen(tcrossprod(R_i) / (n_i[i] - 1), symmetric = TRUE); r <- min(n_i[i] - 1, M)
    E$vectors[, seq_len(r), drop = FALSE] %*% diag(sqrt(pmax(E$values[seq_len(r)], 0)), r)
  })
  Zs <- lapply(Raiz, function(A) A %*% matrix(rnorm(ncol(A) * B_CFF), ncol(A)))   # Dim: (M, B_CFF) por grupo
  V_nul <- Reduce(`+`, lapply(seq_len(ncol(a$pares_g)), function(j) {
    p <- a$pares_g[, j]; colSums(w * (Zs[[p[1]]] - sqrt(n_i[p[1]] / n_i[p[2]]) * Zs[[p[2]]])^2) }))
  a$p_CFF <- (1 + sum(V_nul >= a$V_n)) / (B_CFF + 1)
  #Comparaciones por pares con la covarianza combinada y ajuste de Holm
  pg <- a$pares_g
  a$pares <- data.frame(grupo_a = levels(grupo)[pg[1, ]], grupo_b = levels(grupo)[pg[2, ]],
                        F = apply(pg, 2, function(p) n_i[p[1]] * n_i[p[2]] / (n_i[p[1]] + n_i[p[2]]) *
                                    sum(w * (a$Medias[, p[1]] - a$Medias[, p[2]])^2) / a$tz[["tr_gamma"]]), stringsAsFactors = FALSE)
  a$pares$p <- pf(a$pares$F, a$tz[["d"]], (n - k) * a$tz[["d"]], lower.tail = FALSE)
  a$pares$p_holm <- p.adjust(a$pares$p, method = "holm")
  #Pares con Cuevas et al. (covarianza de cada grupo) y ajuste de Holm, con los mismos procesos Zs de la prueba global
  a$pares$V_CFF <- apply(pg, 2, function(p) n_i[p[1]] * sum(w * (a$Medias[, p[1]] - a$Medias[, p[2]])^2))
  a$pares$p_CFF <- vapply(seq_len(ncol(pg)), function(j) { p <- pg[, j]
    (1 + sum(colSums(w * (Zs[[p[1]]] - sqrt(n_i[p[1]] / n_i[p[2]]) * Zs[[p[2]]])^2) >= a$pares$V_CFF[j])) / (B_CFF + 1) }, numeric(1))
  a$pares$p_CFF_holm <- p.adjust(a$pares$p_CFF, method = "holm")
  a
}

#Función de contraste con fdANOVA::fanova.tests() (pesos 1; columnas ordenadas por grupo)
contraste_fdanova <- function(Xg, grupo, a) {
  o  <- order(grupo)
  fa <- fanova.tests(x = Xg[, o], group.label = as.character(grupo[o]), test = c("L2B", "FB", "GPF", "Fmaxb", "CS"),
                     params = list(paramFmaxb = B_CFF, paramCS = B_CFF))
  u  <- sumas_anova(Xg, grupo, rep(1, nrow(Xg)))
  est_mano <- c(L2 = u$SSR_n, F = u$F_n, GPF = u$T_GPF / nrow(Xg), Fmax = max(u$F_t), CFF = u$V_n)
  est_fd   <- c(L2 = fa$L2B$statL2, F = fa$FB$statF, GPF = fa$GPF$statGPF, Fmax = fa$Fmaxb$statFmax, CFF = fa$CS$statCS)
  list(tabla = data.frame(prueba = c("L2 (sesgo corregido)", "Tipo F (sesgo corregido)", "GPF", "F_max", "Cuevas-Febrero-Fraiman"),
                          p_manual = c(a$p_L2, a$p_F, a$p_GPF, a$p_Fmax_perm, a$p_CFF),
                          p_fdANOVA = c(fa$L2B$pvalueL2B, fa$FB$pvalueFB, fa$GPF$pvalueGPF, fa$Fmaxb$pvalueFmaxb, fa$CS$pvalueCS),
                          stringsAsFactors = FALSE),
       err_est = max(abs(est_mano - est_fd) / abs(est_fd)),
       err_p = max(abs(c(u$p_L2, u$p_F) - c(fa$L2B$pvalueL2B, fa$FB$pvalueFB))))
}

#Unidades independientes: curvas por estación y por año
D_int <- D[interior, , drop = FALSE]                                               # Dim: (337, N)
media_por <- function(M, g) t(rowsum(t(M), g) / drop(rowsum(rep(1, length(g)), g)))   # media de columnas por grupo
D_est <- media_por(D_int, pares$station_id)                                        # Dim: (337, 207) d_bar de cada estación
est_u <- unique(pares[c("station_id", "subregion")])
subregion_est <- est_u$subregion[match(as.numeric(colnames(D_est)), est_u$station_id)]
A_par <- D_int - D_est[, as.character(pares$station_id)]                           # anomalía de cada par respecto de su estación
A_anio <- media_por(A_par, pares$year)                                             # Dim: (337, 23) curva anual sin efecto de sitio
D_anio <- media_por(D_int, pares$year)                                             # Dim: (337, 23) sin quitar el sitio (sensibilidad)
anios <- as.integer(colnames(A_anio))
fase_anio <- factor(oni$fase[match(anios, oni$anio)], c("El Niño", "Neutro", "La Niña"))

#Función que fija la semilla 42 antes de cada análisis
con_semilla <- function(f, ...) { set.seed(42); f(...) }
anova_sub <- con_semilla(anova_funcional, D_est, subregion_est)
anova_fase <- con_semilla(anova_funcional, A_anio, fase_anio)
anova_fase_sc <- con_semilla(anova_funcional, D_anio, fase_anio)
fd_sub <- con_semilla(contraste_fdanova, D_est, subregion_est, anova_sub)
fd_fase <- con_semilla(contraste_fdanova, A_anio, fase_anio, anova_fase)
pareadas <- c(list(`Todas (unidad estación)` = con_semilla(prueba_pareada, D_est)),
              lapply(split(seq_len(ncol(D_est)), subregion_est), function(j) con_semilla(prueba_pareada, D_est[, j, drop = FALSE])),
              lapply(split(seq_len(ncol(D_anio)), fase_anio), function(j) con_semilla(prueba_pareada, D_anio[, j, drop = FALSE])))

#Tablas: pruebas globales, comparaciones por pares, intervalos significativos y curvas puntuales
NM_SUBR <- "ANOVA por subregión y piso"; NM_FASE <- "ANOVA por fase ENSO"; NM_FASE_SC <- "ANOVA por fase ENSO (sin quitar el sitio)"
U_EST <- "estación (media de sus años)"; U_ANIO <- "año (pares menos la media de su estación)"; U_ANIO_SC <- "año (media de los pares)"
#Prueba que decide (su p es p_remuestreo): Cuevas et al. por subregión (varianzas muy distintas entre grupos),
#F por permutación por fase ENSO (varianzas parecidas) y T_max en las pareadas
filas_anova <- function(nombre, unidad, a, p_fd = rep(NA_real_, 5), decide = "Tipo F (sesgo corregido)") {
  k <- a$k; n <- a$n; d <- a$tz[["d"]]
  f <- data.frame(analisis = nombre, unidad = unidad, n = n, tamanos = paste(sprintf("%s: %d", a$niveles, a$n_i), collapse = "; "),
                  prueba = c("L2 (sesgo corregido)", "Tipo F (sesgo corregido)", "GPF", "F_max", "Cuevas-Febrero-Fraiman"),
                  remuestreo = c(NA, sprintf("%d permutaciones", B_PERM), sprintf("%d permutaciones", B_PERM), sprintf("%d permutaciones", B_PERM),
                                 sprintf("%d trayectorias gaussianas", B_CFF)),
                  estadistico = c(a$SSR_n, a$F_n, a$T_GPF, a$F_max, a$V_n), beta = c(a$tz[["beta"]], NA, a$b_rho, NA, NA),
                  d = c(d, d, a$d_rho, NA, NA), gl1 = c((k - 1) * d, (k - 1) * d, a$d_rho, NA, NA), gl2 = c(NA, (n - k) * d, NA, NA, NA),
                  p_aproximacion = c(a$p_L2, a$p_F, a$p_GPF, NA, NA), p_remuestreo = c(NA, a$p_F_perm, a$p_GPF_perm, a$p_Fmax_perm, a$p_CFF),
                  p_fdANOVA = p_fd, valor_critico_5pct = c(NA, NA, NA, a$c_max, NA),
                  dias_sig_puntual = c(NA, NA, NA, sum(a$p_puntual < ALFA), NA), dias_sig_maximo = c(NA, NA, NA, sum(a$F_t > a$c_max), NA),
                  stringsAsFactors = FALSE)
  f$decide <- f$prueba == decide
  f
}
filas_pareada <- function(nombre, unidad, p) {
  data.frame(analisis = paste("Pareada:", nombre), unidad = unidad, n = p$n, tamanos = as.character(p$n),
             prueba = c("F pareada (sesgo corregido)", "T_max por cambio de signo"),
             remuestreo = c(NA, if (p$exacto) sprintf("exacta (%d signos)", p$B) else sprintf("%d vectores de signos", p$B)),
             estadistico = c(p$F, max(p$T)), beta = NA, d = c(p$d, NA), gl1 = c(p$d, NA), gl2 = c((p$n - 1) * p$d, NA),
             p_aproximacion = c(p$p_F, NA), p_remuestreo = c(NA, p$p_global), p_fdANOVA = NA, valor_critico_5pct = c(NA, p$c_max),
             dias_sig_puntual = c(NA, sum(p$p_puntual < ALFA)), dias_sig_maximo = c(NA, sum(p$T > p$c_max)), decide = c(FALSE, TRUE),
             stringsAsFactors = FALSE)
}
U_PAR <- c(rep(U_EST, 1 + nlevels(subregion_est)), rep(U_ANIO_SC, nlevels(fase_anio)))
tabla_pruebas <- rbind(filas_anova(NM_SUBR, U_EST, anova_sub, fd_sub$tabla$p_fdANOVA, decide = "Cuevas-Febrero-Fraiman"),
                       filas_anova(NM_FASE, U_ANIO, anova_fase, fd_fase$tabla$p_fdANOVA),
                       filas_anova(NM_FASE_SC, U_ANIO_SC, anova_fase_sc),
                       do.call(rbind, Map(filas_pareada, names(pareadas), U_PAR, pareadas)))
tabla_pruebas$version_fdANOVA <- as.character(packageVersion("fdANOVA"))
tabla_pares <- rbind(cbind(analisis = NM_SUBR, anova_sub$pares), cbind(analisis = NM_FASE, anova_fase$pares),
                     cbind(analisis = NM_FASE_SC, anova_fase_sc$pares))

detalle_anova <- function(a) function(i) vapply(seq_len(nrow(i)), function(r) {
  j <- match(i$inicio[r]:i$fin[r], t_int)
  paste(sprintf("%s %+.2f", a$niveles, colMeans(a$Medias[j, , drop = FALSE])), collapse = "; ") }, character(1))
detalle_pareada <- function(p) function(i) vapply(seq_len(nrow(i)), function(r) {
  dm <- mean(p$d_bar[match(i$inicio[r]:i$fin[r], t_int)])
  sprintf("d medio %+.2f mm/día: %s", dm, ifelse(dm > 0, "estación > satélite (subestima)", "satélite > estación (sobrestima)")) }, character(1))
tabla_int <- function(nombre, ind_punt, ind_max, detalle) do.call(rbind, Map(function(crit, ind) {
  i <- intervalos_dias(ind); if (nrow(i)) cbind(analisis = nombre, criterio = crit, i, detalle = detalle(i), stringsAsFactors = FALSE) else NULL },
  c("puntual, sin corrección (p(t) < 0,05)", "máximo, error controlado en [15, 351]"), list(ind_punt, ind_max)))
tabla_intervalos <- rbind(
  do.call(rbind, Map(function(a, nm) tabla_int(nm, a$p_puntual < ALFA, a$F_t > a$c_max, detalle_anova(a)),
                     list(anova_sub, anova_fase, anova_fase_sc), c(NM_SUBR, NM_FASE, NM_FASE_SC))),
  do.call(rbind, Map(function(p, nm) tabla_int(paste("Pareada:", nm), p$p_puntual < ALFA, p$T > p$c_max, detalle_pareada(p)),
                     pareadas, names(pareadas))))

p_tod <- pareadas[[1]]
curvas_inf <- data.frame(t = t_int, fecha = fecha_doy(t_int),
                         F_subregion = anova_sub$F_t, c05_subregion = anova_sub$c_puntual, p_subregion = anova_sub$p_puntual,
                         setNames(as.data.frame(anova_sub$Medias), c("media_pacifico", "media_cauca_valle", "media_cauca_intermedio", "media_cauca_ladera")),
                         setNames(as.data.frame(vapply(levels(subregion_est), function(l) apply(D_est[, subregion_est == l, drop = FALSE], 1, sd), numeric(M_int))),
                                  c("sd_pacifico", "sd_cauca_valle", "sd_cauca_intermedio", "sd_cauca_ladera")),
                         F_fase = anova_fase$F_t, c05_fase = anova_fase$c_puntual, p_fase = anova_fase$p_puntual,
                         setNames(as.data.frame(anova_fase$Medias), c("anomalia_nino", "anomalia_neutro", "anomalia_nina")),
                         d_bar_estaciones = p_tod$d_bar, ee_estaciones = p_tod$ee, T_estaciones = p_tod$T,
                         c05_T_estaciones = p_tod$c_puntual, p_T_estaciones = p_tod$p_puntual)
verif_inf <- c(sst_ssr_sse = max(anova_sub$err_SST, anova_fase$err_SST, anova_fase_sc$err_SST),
               sse_traza = max(anova_sub$err_SSE_traza, anova_fase$err_SSE_traza, anova_fase_sc$err_SSE_traza),
               fdanova_estadisticos = max(fd_sub$err_est, fd_fase$err_est), fdanova_p_L2_F = max(fd_sub$err_p, fd_fase$err_p))
res_inf <- c(b_permutaciones = B_PERM, b_cuevas = B_CFF, dias_interior = M_int,
             p_F_perm_subregion = anova_sub$p_F_perm, p_Fmax_perm_subregion = anova_sub$p_Fmax_perm, c_max_subregion = anova_sub$c_max,
             dias_max_subregion = sum(anova_sub$F_t > anova_sub$c_max), p_CFF_subregion = anova_sub$p_CFF, p_F_perm_fase = anova_fase$p_F_perm,
             p_Fmax_perm_fase = anova_fase$p_Fmax_perm, c_max_fase = anova_fase$c_max, dias_max_fase = sum(anova_fase$F_t > anova_fase$c_max),
             p_CFF_fase = anova_fase$p_CFF,
             p_global_pareada_estaciones = p_tod$p_global, dias_max_pareada_estaciones = sum(p_tod$T > p_tod$c_max),
             setNames(verif_inf, paste0("verif_", names(verif_inf))))
res_inf
res_comparacion$inferencia_d_pruebas <- tabla_pruebas
res_comparacion$inferencia_d_pares <- tabla_pares
res_comparacion$inferencia_d_intervalos <- tabla_intervalos
res_comparacion$inferencia_d_curvas <- curvas_inf
res_comparacion$resumen_inferencia_d <- data.frame(indicador = names(res_inf), valor = signif(res_inf, 6))
stopifnot(all(verif_inf[c("sst_ssr_sse", "sse_traza")] <= 1e-12), all(verif_inf[c("fdanova_estadisticos", "fdanova_p_L2_F")] <= 1e-10))

#Gráfica de F(t) con la curva crítica puntual y el valor crítico del máximo
sombrear_sig <- function(ind) {
  i <- intervalos_dias(ind); u <- par("usr")
  if (nrow(i)) rect(i$inicio - 0.5, u[3], i$fin + 0.5, u[4], col = adjustcolor("#fdae61", 0.35), border = NA)
}
eje_int <- function() { eje_meses(); abline(h = 0, lty = 3) }
panel_F <- function(a, main) {
  plot(t_int, a$F_t, type = "n", xlim = range(t_dia), ylim = c(0, max(a$F_t, a$c_max, a$c_puntual) * 1.25), xaxt = "n",
       xlab = "Día del año", ylab = "F(t)", main = main)
  sombrear_sig(a$F_t > a$c_max); sombrear_bordes(); eje_meses()
  lines(t_int, a$c_puntual, col = "#2171b5", lty = 3, lwd = 2)
  abline(h = a$c_max, lty = 2)
  lines(t_int, a$F_t, col = "#cb181d", lwd = 2.2)
  legend("topleft", c("F(t) observado", "Crítico puntual 5 % (permutación)", sprintf("Crítico del máximo 5 %% = %.2f", a$c_max),
                      sprintf("F(t) > crítico del máximo: %d días", sum(a$F_t > a$c_max))),
         col = c("#cb181d", "#2171b5", "black", adjustcolor("#fdae61", 0.6)), lty = c(1, 3, 2, NA), lwd = c(2.2, 2, 1, NA), pch = c(NA, NA, NA, 15),
         pt.cex = 2, bg = "white", cex = 0.68)
  mtext(sprintf("p por permutación: F %.2g · GPF %.2g · F_max %.2g · Cuevas %.2g", a$p_F_perm, a$p_GPF_perm, a$p_Fmax_perm, a$p_CFF),
        3, 0.2, cex = 0.68)
}
panel_medias <- function(a, cols, main, ylab, fondo = NULL, fondo_col = NULL) {
  yl <- range(a$Medias, fondo, 0); yl[2] <- yl[2] + 0.3 * diff(yl)                  # espacio para la leyenda
  plot(t_int, a$Medias[, 1], type = "n", xlim = range(t_dia), ylim = yl, xaxt = "n", xlab = "Día del año", ylab = ylab, main = main)
  sombrear_sig(a$F_t > a$c_max); sombrear_bordes(); eje_int()
  if (!is.null(fondo)) matlines(t_int, fondo, lty = 1, col = adjustcolor(fondo_col, 0.35))
  matlines(t_int, a$Medias, lty = 1, lwd = 2.6, col = cols)
  legend("topright", sprintf("%s (n = %d)", a$niveles, a$n_i), col = cols, lwd = 2.6, bg = "white", cex = 0.7)
}
guardar("figures/item3_comparacion/anova_funcional_d.png", 1800, 1250, {
  par(mfrow = c(2, 2), mar = c(4.2, 4.5, 3.4, 1))
  panel_medias(anova_sub, COL_SUB, "d(t) medio por subregión y piso (unidad: estación)", "d(t) = estación - satélite (mm/día)")
  panel_F(anova_sub, sprintf("ANOVA funcional por subregión y piso (%d estaciones)", anova_sub$n))
  panel_medias(anova_fase, COL_FASE, "Anomalía de d(t) por fase ENSO (unidad: año, sin efecto de sitio)", "Anomalía de d(t) (mm/día)",
               fondo = A_anio, fondo_col = COL_FASE[as.character(fase_anio)])
  panel_F(anova_fase, sprintf("ANOVA funcional por fase ENSO (%d años)", anova_fase$n))
})

guardar("figures/item3_comparacion/prueba_pareada_d.png", 1800, 700, {
  par(mfrow = c(1, 2), mar = c(4.2, 4.5, 3.4, 1))
  sig <- p_tod$T > p_tod$c_max
  plot(t_int, p_tod$d_bar, type = "n", xlim = range(t_dia), ylim = range(p_tod$d_bar - 2 * p_tod$ee, p_tod$d_bar + 2 * p_tod$ee, 0),
       xaxt = "n", xlab = "Día del año", ylab = "d(t) = estación - satélite (mm/día)",
       main = sprintf("d(t) medio ± 2 EE(t) por estación (n = %d)", p_tod$n))
  sombrear_sig(sig); sombrear_bordes(); eje_int()
  polygon(c(t_int, rev(t_int)), c(p_tod$d_bar - 2 * p_tod$ee, rev(p_tod$d_bar + 2 * p_tod$ee)), col = adjustcolor("#08306b", 0.2), border = NA)
  lines(t_int, p_tod$d_bar, lwd = 2.5, col = "#08306b")
  legend("topright", c("d(t) medio ± 2 EE(t)", sprintf("Significativo con control global: %d días", sum(sig)),
                       sprintf("Banda sin cero (sin corrección): %d días", sum(p_tod$d_bar - 2 * p_tod$ee > 0 | p_tod$d_bar + 2 * p_tod$ee < 0))),
         col = c("#08306b", adjustcolor("#fdae61", 0.6), NA), lwd = c(2.5, NA, NA), pch = c(NA, 15, NA), pt.cex = 2, bg = "white", cex = 0.7)
  plot(t_int, p_tod$T, type = "n", xlim = range(t_dia), ylim = c(0, max(p_tod$T, p_tod$c_max) * 1.15), xaxt = "n",
       xlab = "Día del año", ylab = "|t(t)|", main = "Prueba pareada por cambio de signo (criterio del máximo)")
  sombrear_sig(sig); sombrear_bordes(); eje_meses()
  lines(t_int, p_tod$c_puntual, col = "#2171b5", lty = 3, lwd = 2); abline(h = p_tod$c_max, lty = 2)
  lines(t_int, p_tod$T, col = "#cb181d", lwd = 2.2)
  legend("topleft", c("|t(t)| observado", "Crítico puntual 5 %", sprintf("Crítico del máximo 5 %% = %.2f", p_tod$c_max)),
         col = c("#cb181d", "#2171b5", "black"), lty = c(1, 3, 2), lwd = c(2.2, 2, 1), bg = "white", cex = 0.7)
  mtext(sprintf("p global (T_max) %.4f con %d vectores de signos; F pareada = %.2f, d = %.2f, p = %.1e",
                p_tod$p_global, p_tod$B, p_tod$F, p_tod$d, p_tod$p_F), 3, 0.2, cex = 0.68)
})

#Modelo concurrente y_i(t) = beta0(t) + beta1(t) x_i(t) + e_i(t) con fRegress() y R2(t) por día, beta en la base común de las dos fuentes;
#contraste a mano por mínimos cuadrados en cada día y por ecuaciones normales en la base

K_B    <- sat$K
base_b <- base_fd(K_B)
fit_c  <- fRegress(fdY, list(const = rep(1, N), sat = fdX), list(const = fdPar(base_b), sat = fdPar(base_b)))
Theta  <- bspline_base(t_dia, K_B)                                                # Dim: (365, 6), igual a eval.basis (bloque 3)
b_f    <- cbind(b0 = drop(eval.fd(t_dia, fit_c$betaestlist$const$fd)), b1 = drop(eval.fd(t_dia, fit_c$betaestlist$sat$fd)))
#Ajuste b0(t) + b1(t) x_i(t) en la rejilla diaria (yhatfdobj no se usa: está re-suavizado en la base de la respuesta)
Yhat_f <- b_f[, "b0"] + b_f[, "b1"] * X                                            # Dim: (365, N)
E_f    <- Y - Yhat_f
SCT_t  <- rowSums((Y - rowMeans(Y))^2)
R2_f   <- 1 - rowSums(E_f^2) / SCT_t
R2_yhatfd <- 1 - rowSums((Y - eval.fd(t_dia, fit_c$yhatfdobj))^2) / SCT_t          # solo para documentar la diferencia

#Paso 1: mínimos cuadrados en cada día con sumas por fila; Var(beta(t)) = s2(t) (Z'Z)^-1
Sx <- rowSums(X); Sy <- rowSums(Y); Sxx <- rowSums(X^2); Sxy <- rowSums(X * Y); det_t <- N * Sxx - Sx^2
b1_p <- (N * Sxy - Sx * Sy) / det_t
b0_p <- (Sy - b1_p * Sx) / N
E_p  <- Y - (b0_p + b1_p * X)
s2_p <- rowSums(E_p^2) / (N - 2)
ee_p <- cbind(b0 = sqrt(s2_p * Sxx / det_t), b1 = sqrt(s2_p * N / det_t))
R2_p <- 1 - rowSums(E_p^2) / SCT_t                                                # en regresión simple R2(t) = r_XY(t, t)^2 (diagonal de arriba)

#Paso 2: ecuaciones normales en la base con la regla del trapecio en una malla de 0,05 días
h_g  <- 0.05
t_g  <- seq(1, N_DIAS, by = h_g)
w_g  <- rep(h_g, length(t_g)); w_g[c(1L, length(t_g))] <- h_g / 2
Th_g <- bspline_base(t_g, K_B); Pe_g <- bspline_base(t_g, est$K); Ps_g <- bspline_base(t_g, sat$K)
sx_g  <- drop(Ps_g %*% rowSums(C_s)); sy_g <- drop(Pe_g %*% rowSums(C_e))
sxx_g <- rowSums((Ps_g %*% tcrossprod(C_s)) * Ps_g); syy_g <- rowSums((Pe_g %*% tcrossprod(C_e)) * Pe_g)
sxy_g <- rowSums((Ps_g %*% tcrossprod(C_s, C_e)) * Pe_g)
A_ne  <- rbind(cbind(crossprod(Th_g, Th_g * (w_g * N)), crossprod(Th_g, Th_g * (w_g * sx_g))),
               cbind(crossprod(Th_g, Th_g * (w_g * sx_g)), crossprod(Th_g, Th_g * (w_g * sxx_g))))      # Dim: (12, 12)
c_mano <- solve(A_ne, c(crossprod(Th_g, w_g * sy_g), crossprod(Th_g, w_g * sxy_g)))
c_fr   <- c(fit_c$betaestlist$const$fd$coefs, fit_c$betaestlist$sat$fd$coefs)
sise   <- function(b0, b1) sum(w_g * (syy_g - 2 * b0 * sy_g - 2 * b1 * sxy_g + N * b0^2 + 2 * b0 * b1 * sx_g + b1^2 * sxx_g))
b1p_g  <- (N * sxy_g - sx_g * sy_g) / (N * sxx_g - sx_g^2)
SISE   <- c(puntual = sise((sy_g - b1p_g * sx_g) / N, b1p_g),
            mano = sise(drop(Th_g %*% c_mano[seq_len(K_B)]), drop(Th_g %*% c_mano[K_B + seq_len(K_B)])),
            fRegress = sise(drop(Th_g %*% c_fr[seq_len(K_B)]), drop(Th_g %*% c_fr[K_B + seq_len(K_B)])))
b1_mano <- drop(Theta %*% c_mano[K_B + seq_len(K_B)])

#Errores estándar de beta(t) como en fRegress(y2cMap, SigmaE), sumando por bloques en vez del producto de Kronecker
y2c_e  <- solve(crossprod(Phi_e), t(Phi_e))                                       # Dim: (6, 365) mapa de datos a coeficientes (MCO, Ítem 1)
SigmaE <- var(t(E_f))                                                              # Dim: (365, 365) covarianza de los residuos
bvar_bloques <- function(Xfd, Cmat) {
  tfine <- seq(1, N_DIAS, length.out = max(501, 10 * est$K + 1)); dt <- tfine[2] - tfine[1]
  Bf <- eval.basis(tfine, base_b); Yb <- eval.basis(tfine, base_fd(est$K)); Xf <- eval.fd(tfine, Xfd)
  P  <- lapply(seq_len(est$K), function(k) rbind(dt * crossprod(Bf * Yb[, k], matrix(1, length(tfine), ncol(Xf))),
                                                 dt * crossprod(Bf * Yb[, k], Xf)))   # Dim: (12, n) bloque k de basisprodmat
  V  <- y2c_e %*% SigmaE %*% t(y2c_e)
  Mv <- Reduce(`+`, lapply(seq_len(est$K), function(l) Reduce(`+`, lapply(seq_len(est$K), function(k) V[k, l] * P[[k]])) %*% t(P[[l]])))
  solve(Cmat, t(solve(Cmat, Mv)))
}
bvar_c <- bvar_bloques(fdX, fit_c$Cmat)
ee_f   <- cbind(b0 = sqrt(rowSums((Theta %*% bvar_c[seq_len(K_B), seq_len(K_B)]) * Theta)),
                b1 = sqrt(rowSums((Theta %*% bvar_c[K_B + seq_len(K_B), K_B + seq_len(K_B)]) * Theta)))
set.seed(42)
sub_c  <- sort(sample(N, 150L))
fit_s  <- fRegress(fdY[sub_c], list(const = rep(1, length(sub_c)), sat = fdX[sub_c]), list(const = fdPar(base_b), sat = fdPar(base_b)),
                   y2cMap = y2c_e, SigmaE = SigmaE)
verif_conc <- c(r2_puntual_r_diagonal = max(abs(R2_p - r_diag^2)), r2_fRegress_sobre_puntual = max(R2_f - R2_p),
                bvar_bloques_fRegress = max(abs(bvar_bloques(fdX[sub_c], fit_s$Cmat) - fit_s$bvar)) / max(abs(fit_s$bvar)),
                sise_exceso_fRegress = unname(SISE["fRegress"] / SISE["mano"] - 1),
                sise_exceso_mano_puntual = unname(SISE["mano"] / SISE["puntual"] - 1))
stopifnot(verif_conc["r2_puntual_r_diagonal"] <= 1e-12, verif_conc["r2_fRegress_sobre_puntual"] <= 1e-12, verif_conc["bvar_bloques_fRegress"] <= 1e-10,
          SISE["puntual"] <= SISE["mano"], SISE["mano"] <= SISE["fRegress"], verif_conc["sise_exceso_fRegress"] <= 0.01)

r2_temp <- data.frame(temporada = names(por_temporada(R2_f)), R2_fRegress = por_temporada(R2_f), R2_puntual = por_temporada(R2_p),
                      beta1_fRegress = por_temporada(b_f[, "b1"]), row.names = NULL)
r2_temp
res_comparacion$modelo_concurrente_curvas <- data.frame(
  t = t_dia, fecha = fecha_doy(t_dia), interior = interior, temporada = TEMPORADA[mes_dia],
  beta0_fRegress = b_f[, "b0"], ee_beta0_fRegress = ee_f[, "b0"], beta1_fRegress = b_f[, "b1"], ee_beta1_fRegress = ee_f[, "b1"],
  beta0_puntual = b0_p, ee_beta0_puntual = ee_p[, "b0"], beta1_puntual = b1_p, ee_beta1_puntual = ee_p[, "b1"],
  beta1_ecuaciones_normales = b1_mano, R2_fRegress = R2_f, R2_puntual = R2_p)
res_conc <- c(pares = N, K_beta = K_B, R2_medio_interior_fRegress = mean(R2_f[interior]), R2_medio_interior_puntual = mean(R2_p[interior]),
              R2_min_fRegress = min(R2_f[interior]), dia_R2_min = t_dia[interior][which.min(R2_f[interior])],
              R2_max_fRegress = max(R2_f[interior]), dia_R2_max = t_dia[interior][which.max(R2_f[interior])],
              beta1_min = min(b_f[interior, "b1"]), beta1_max = max(b_f[interior, "b1"]),
              dias_beta1_sobre_1 = sum(interior & b_f[, "b1"] - 2 * ee_f[, "b1"] > 1), dias_beta1_bajo_1 = sum(interior & b_f[, "b1"] + 2 * ee_f[, "b1"] < 1),
              max_abs_beta1_fRegress_puntual = max(abs(b_f[, "b1"] - b1_p)[interior]),
              R2_medio_interior_yhatfdobj = mean(R2_yhatfd[interior]), dias_R2_yhatfdobj_sobre_puntual = sum(interior & R2_yhatfd > R2_p),
              setNames(SISE, paste0("SISE_", names(SISE))), setNames(verif_conc, paste0("verif_", names(verif_conc))))
res_comparacion$modelo_concurrente_resumen <- rbind(data.frame(indicador = names(res_conc), valor = signif(res_conc, 6)),
                                                    data.frame(indicador = paste0("R2_", gsub(" ", "_", r2_temp$temporada)), valor = signif(r2_temp$R2_fRegress, 6)))

guardar("figures/item3_comparacion/modelo_concurrente.png", 1900, 650, {
  par(mfrow = c(1, 3), mar = c(4.2, 4.5, 3.4, 1))
  panel_beta <- function(j, main, ylab, ref) {
    li <- b_f[, j] - 2 * ee_f[, j]; ls <- b_f[, j] + 2 * ee_f[, j]
    plot(t_dia, b_f[, j], type = "n", ylim = range(li[interior], ls[interior], if (j == "b0") b0_p[interior] else b1_p[interior], ref),
         xaxt = "n", xlab = "Día del año", ylab = ylab, main = main)
    sombrear_bordes(); eje_meses(); abline(h = ref, lty = 3)
    polygon(c(t_dia, rev(t_dia)), c(li, rev(ls)), col = adjustcolor("#08306b", 0.2), border = NA)
    lines(t_dia, b_f[, j], lwd = 2.5, col = "#08306b")
    lines(t_dia, if (j == "b0") b0_p else b1_p, lwd = 1.8, lty = 2, col = "#d95f02")
    legend("topright", c(sprintf("fRegress (K = %d) ± 2 EE", K_B), "Mínimos cuadrados en cada día"), col = c("#08306b", "#d95f02"),
           lty = c(1, 2), lwd = c(2.5, 1.8), bg = "white", cex = 0.72)
  }
  panel_beta("b0", "Intercepto beta0(t)", "beta0(t) (mm/día)", 0)
  panel_beta("b1", "Efecto del satélite beta1(t)", "beta1(t)", 1)
  plot(t_dia, R2_f, type = "n", ylim = c(0, 1), xaxt = "n", xlab = "Día del año", ylab = expression(R^2*(t)),
       main = sprintf("Varianza de la estación explicada por el satélite (N = %d pares)", N))
  sombrear_bordes(); eje_meses()
  lines(t_dia, R2_f, lwd = 2.5, col = "#08306b"); lines(t_dia, R2_p, lwd = 1.8, lty = 2, col = "#d95f02")
  text(c(30, 105, 197, 289), 0.95, sprintf("%s\n%.2f", r2_temp$temporada[match(c("DEF seca", "MAM lluviosa", "JJA seca", "SON lluviosa"), r2_temp$temporada)],
                                          r2_temp$R2_fRegress[match(c("DEF seca", "MAM lluviosa", "JJA seca", "SON lluviosa"), r2_temp$temporada)]), cex = 0.75)
  legend("bottomright", c("fRegress", "Mínimos cuadrados en cada día (= r_XY(t,t)^2)"), col = c("#08306b", "#d95f02"), lty = c(1, 2),
         lwd = c(2.5, 1.8), bg = "white", cex = 0.72)
})

# Verificaciones finales

rm(list = setdiff(ls(), NUCLEO))
indicadores <- function(x) setNames(as.numeric(x$valor), x$indicador)   # tabla indicador-valor a vector con nombres
vS   <- res_construccion$verificaciones_S
vpen <- res_construccion$verificacion_ruta_penalizada
h03  <- indicadores(res_exploratorio$resumen_item2)
vfp  <- indicadores(res_exploratorio$verificacion_media_penalizada)
h04  <- indicadores(res_comparacion$resumen_sesgo)
h05  <- indicadores(res_comparacion$resumen_inferencia_d)
h06  <- indicadores(res_comparacion$modelo_concurrente_resumen)
pico_sat_emparejado <- 59 + which.max(res_comparacion$curva_diferencia$media_satelite[60:181])

lineas <- readLines(ARCHIVO_MAESTRO, encoding = "UTF-8")                 # el propio script
codigo <- sub("#.*$", "", lineas)
patron <- paste0("\\b(", paste0("fo", "r"), "|", paste0("whi", "le"), "|", paste0("repe", "at"), ")\\b\\s*[({]")
ini_b1 <- grep("^# BLOQUE 1\\.", lineas); ini_b2 <- grep("^# BLOQUE 2\\.", lineas)
lib_b1 <- sum(grepl(paste0("libr", "ary\\(|requ", "ire\\("), codigo[ini_b1:ini_b2]))

verificaciones <- data.frame(
  verificacion = c("||S S - S||_F, máximo en los casos de K (Ítem 1)", "|tr(S) - K|, máximo (Ítem 1)",
                   "Media a mano frente a mean.fd(), estaciones con el K de la VCG (Ítem 2)", "Media a mano frente a mean.fd(), satélite (Ítem 2)",
                   "Media de la ruta penalizada frente a mean.fd()", "S_lambda simétrica (ruta penalizada)",
                   "Media de d(t) frente a mean.fd() (Ítem 3)", "Covarianza cruzada frente a var.fd() (Ítem 3)",
                   "SST(t) = SSR(t) + SSE(t) y SSE_n = (n - k) tr(gamma), ANOVA funcional de d(t) (Ítem 3)",
                   "Estadísticos del ANOVA funcional a mano frente a fanova.tests() (Ítem 3)",
                   "Valores p de L2 y tipo F a mano frente a fanova.tests() (Ítem 3)",
                   "Modelo concurrente: R2(t) de mínimos cuadrados en cada día frente a r_XY(t,t)^2 (Ítem 3)",
                   "Modelo concurrente: EE de beta(t) por bloques frente a fRegress() (Ítem 3)",
                   "Modelo concurrente: exceso relativo de SISE de fRegress() sobre las ecuaciones normales (Ítem 3)",
                   "Bucles explícitos en el archivo maestro", "Librerías cargadas en el bloque del Ítem 1"),
  valor = c(max(vS$error_idempotencia_F[!grepl("K = 48", vS$caso)]), max(vS$error_traza), h03["dif_media_mean_fd_estaciones"],
            h03["dif_media_mean_fd_satelite"], vfp["dif_media_mean_fd_penalizada"], vpen$error_simetria, h04["verif_media_mean_fd"], h04["verif_cov_cruzada_var_fd"],
            max(h05["verif_sst_ssr_sse"], h05["verif_sse_traza"]), h05["verif_fdanova_estadisticos"], h05["verif_fdanova_p_L2_F"],
            h06["verif_r2_puntual_r_diagonal"], h06["verif_bvar_bloques_fRegress"], h06["verif_sise_exceso_fRegress"],
            sum(grepl(patron, codigo)), lib_b1),
  tolerancia = c(1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-10, 1e-10, 1e-12, 1e-10, 0.01, 0, 0))
verificaciones$cumple <- verificaciones$valor <= verificaciones$tolerancia
verificaciones

if (!all(verificaciones$cumple)) stop("Verificaciones que no se cumplen: ", paste(verificaciones$verificacion[!verificaciones$cumple], collapse = "; "))
