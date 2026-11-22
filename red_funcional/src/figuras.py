"""Figuras de la red funcional en figures/item4_red/."""
import matplotlib
matplotlib.use("Agg")   # backend sin pantalla: las figuras se guardan igual dentro y fuera de Jupyter
import matplotlib.pyplot as plt
import numpy as np
import torch

from .comun import FIG
from .curvas import M_FPCA, t_dia
from .datos import ANIOS_ENT, ETQ_FASE, FASES, TRAMOS, TXT_PRU
from .modelo import CABEZAS, COMBOS, COVARIABLES, K_NEU, NOMBRES

COLOR = {"lineal": "tab:gray", "mlp": "tab:blue", "kan": "tab:red"}
MES_INI = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]; MES_ETQ = list("EFMAMJJASOND")


def kan_sintetica(x_s, f_sint, kan_1d, rmse_sint):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    with torch.no_grad():
        ax.plot(x_s, f_sint(x_s), "k-", lw=3, label="f(x) = sin(πx) + exp(x/2)"); ax.plot(x_s, kan_1d(x_s), "r--", lw=1.5, label=f"KAN (G = 20), RMSE = {rmse_sint:.4f}")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.legend(frameon=False); ax.set_title("Prueba sintética previa de la capa KAN")
    fig.tight_layout(); fig.savefig(FIG / "kan_sintetica.png", dpi=130); plt.close(fig)


def busqueda_hiperparametros(busqueda, n_pliegues):
    fig, ejes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
    for ax, cab in zip(ejes, CABEZAS):
        b = busqueda[busqueda.cabeza == cab].reset_index(drop=True)
        x = np.arange(len(b)) + (b.pasada.astype(str) + b.etapa).factorize()[0] * 0.8
        for n, r in b.iterrows():
            ax.scatter([x[n]] * len(r.val_pliegues), r.val_pliegues, s=14, color=COLOR[cab], alpha=0.35, linewidths=0)
            ax.errorbar(x[n], r.media_val, yerr=r.sd_val, fmt="o", ms=7 if r.elegida else 5, color=COLOR[cab] if r.elegida else "0.35",
                        mfc=COLOR[cab] if r.elegida else "white", capsize=3, lw=1.2)
        ax.axhline(b.media_val[b.partida & (b.etapa == "A") & (b.pasada == 1)].iloc[0], color="0.5", ls=":", lw=1, label="configuración inicial")
        ax.set_xticks(x, [f"{r.pasada}{r.etapa}: {r.valor}" for r in b.itertuples()], rotation=60, ha="right", fontsize=6.5)
        ax.set_title(f"Cabeza {cab}: {b.parametros.iloc[-1]} parámetros", fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    ejes[0].set_ylabel("pérdida L2 de validación / la de la climatología")
    fig.suptitle(f"Búsqueda por coordenadas, 2 pasadas (A estructura, B rugosidad λ_w, C tasa, D regularización): media ± desv. de {n_pliegues} pliegues de la validación cruzada cronológica; relleno = elegida", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "busqueda_hiperparametros.png", dpi=130); plt.close(fig)


def capacidad_esparsidad(capac, esp, busqueda):
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for cab in ("mlp", "kan"):
        c_ = capac[capac.cabeza == cab]
        ejes[0].errorbar(c_.parametros, c_.media_val, yerr=c_.sd_val, fmt="o-", color=COLOR[cab], capsize=3, lw=1.5, label=f"cabeza {cab}")
    ejes[0].axhline(busqueda[(busqueda.cabeza == "lineal") & busqueda.elegida].media_val.iloc[-1], color=COLOR["lineal"], ls="--", lw=1, label="cabeza lineal elegida")
    ejes[0].set_xscale("log"); ejes[0].set_xlabel("parámetros entrenables (modelo completo, escala log)")
    ejes[0].set_ylabel("pérdida L2 relativa en validación cruzada"); ejes[0].legend(frameon=False, fontsize=8)
    ejes[0].set_title("Capacidad con parámetros igualados (±10 %)", fontsize=10)
    ejes[1].errorbar(esp.aristas_activas_media, esp.media_val, yerr=esp.sd_val, fmt="o-", color=COLOR["kan"], capsize=3, lw=1.5)
    for r in esp.itertuples():
        ejes[1].annotate(f"λ_L1 = {r.lambda_L1:g}", (r.aristas_activas_media, r.media_val), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ejes[1].set_xlabel(f"aristas activas de la primera capa KAN tras la poda (amplitud > 10 % de la mayor, de {int(esp.aristas_totales[0])})")
    ejes[1].set_ylabel("pérdida L2 relativa en validación cruzada"); ejes[1].set_title("Costo de simplificar la KAN con L1 + entropía y poda", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "capacidad_esparsidad.png", dpi=130); plt.close(fig)


def curvas_validacion(hist_cv, perdida_clima, cv_final, n_pliegues):
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4.2))
    for kind in CABEZAS:
        for f in range(n_pliegues):
            h = hist_cv[(kind, f)]
            ejes[0].plot(np.array(h) / perdida_clima[f], color=COLOR[kind], alpha=0.35 + 0.15 * f, lw=1, label=kind if f == 0 else None)
        c_ = cv_final[cv_final.cabeza == kind]
        ejes[1].plot(c_.pliegue, c_.perdida_rel, "o-", color=COLOR[kind], lw=1.6, label=f"cabeza {kind}")
    ejes[0].set_yscale("log"); ejes[0].set_xlabel("época"); ejes[0].set_ylabel("pérdida L2 de validación / climatología")
    ejes[0].legend(frameon=False); ejes[0].set_title("Curvas de validación de los 4 pliegues (más oscuro = pliegue más reciente)", fontsize=10)
    ejes[1].axhline(1, color="0.6", ls=":", lw=1)
    ejes[1].set_xticks(cv_final.pliegue.unique(), [f"{r.validacion}\n{r.fases_validacion}" for r in cv_final[cv_final.cabeza == "kan"].itertuples()], fontsize=8)
    ejes[1].set_ylabel("pérdida relativa a la climatología"); ejes[1].legend(frameon=False, fontsize=8)
    ejes[1].set_title("Puntaje de cada pliegue (bloque de validación y fase ENSO de sus años)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "curvas_validacion.png", dpi=130); plt.close(fig)


def prueba_por_anio_enso(por_anio):
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4.2))
    x = por_anio.anio.to_numpy()
    ejes[0].plot(x, por_anio.total_estacion_mm, "k-o", lw=2.4, label="estaciones (curva MCO)")
    ejes[0].plot(x, por_anio.total_satelite_mm, "-s", color="0.55", lw=1.4, label="satélite sin corregir")
    for kind in CABEZAS:
        ejes[0].plot(x, por_anio[f"total_{kind}_mm"], "-o", color=COLOR[kind], lw=1.4, label=f"cabeza {kind}")
    for ax in ejes:
        for a, f_ in zip(x, por_anio.fase):
            ax.axvspan(a - 0.5, a + 0.5, color={"nino": "#fdd0a2", "nina": "#c6dbef", "neutro": "white"}[f_], alpha=0.6, lw=0)
        ax.set_xticks(x, [f"{a}\n{ETQ_FASE[f_]}" for a, f_ in zip(x, por_anio.fase)], fontsize=8)
    ejes[0].set_ylabel("total anual medio de las estaciones de prueba (mm)"); ejes[0].legend(frameon=False, fontsize=8)
    ejes[0].set_title(f"Prueba {TXT_PRU} año a año (naranja El Niño, azul La Niña)", fontsize=10)
    for modelo, col in (("satelite", "0.55"), ("lineal", COLOR["lineal"]), ("mlp", COLOR["mlp"]), ("kan", COLOR["kan"])):
        ejes[1].plot(x, por_anio[f"PBIAS_{modelo}"], "-o", color=col, lw=1.4, label=NOMBRES[modelo])
    ejes[1].axhline(0, color="k", lw=0.8); ejes[1].set_ylabel("PBIAS (%)"); ejes[1].legend(frameon=False, fontsize=8)
    ejes[1].set_title("Sesgo de volumen por año de prueba", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "prueba_por_anio_enso.png", dpi=130); plt.close(fig)


def ablacion(maestra):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, k in zip(ejes, ("L2_rel", "KGE")):
        for d_, (tramo, col) in enumerate((("prueba", "tab:blue"), ("ciegas", "tab:red"))):
            t_ = maestra[maestra.tramo == tramo]
            ic = np.array([[float(x) for x in s.strip("[]").split(",")] for s in t_[k.split("_")[0] + ("_IC" if k == "KGE" else "_IC")]]) if k == "KGE" else \
                 np.array([[float(x) for x in s.strip("[]").split(",")] for s in t_["L2_IC"]])
            x = np.arange(4) + (d_ - 0.5) * 0.38
            ax.bar(x, t_[k], width=0.38, color=col, alpha=0.75, label=f"prueba {TXT_PRU}" if tramo == "prueba" else "estaciones ciegas")
            ax.errorbar(x, t_[k], yerr=[t_[k] - ic[:, 0], ic[:, 1] - t_[k]], fmt="none", ecolor="k", capsize=3)
        ax.set_xticks(np.arange(4), ["Satélite", "Lineal", "MLP", "KAN"]); ax.set_ylabel(k.replace("_", " "))
        ax.set_title(("Error L2 relativo (menor es mejor)" if k == "L2_rel" else "KGE (mayor es mejor)") + ", IC 95 % bootstrap por estación")
        ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "ablacion.png", dpi=130); plt.close(fig)


def ventanas_wk(Phi_x, ml, mk):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    for ax, m, tit in zip(ejes, (ml, mk), ("cabeza lineal", "cabeza KAN")):
        Wk = Phi_x @ m.A.detach().numpy().T                                             # Dim: (365, K_NEU), w_k(t)
        Wk = Wk / np.abs(Wk).max(0)
        ax.plot(t_dia, Wk, lw=1.6, label=[f"w{k + 1} (z{k + 1})" for k in range(K_NEU)]); ax.axhline(0, color="k", lw=0.6)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
        ax.set_xticks(MES_INI, MES_ETQ); ax.set_xlabel("día del año"); ax.set_ylabel("w_k(t) normalizada")
        ax.set_title(f"w_k(t) de las {K_NEU} neuronas funcionales ({tit}, semilla 42)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "ventanas_wk.png", dpi=130); plt.close(fig)


def activaciones_kan(curvas, activa, nom_in, nom_out, B, contraste, forma, alt_mu, alt_sd):
    n_o, n_i = activa.shape
    fig, ejes = plt.subplots(n_o, n_i, figsize=(2.1 * n_i, 1.7 * n_o), sharex="col")
    for i_ in range(n_i):
        for o in range(n_o):
            _, xf, y_ = curvas[(o, i_)]
            ax = ejes[o, i_]
            if not activa[o, i_]:
                ax.set_facecolor("0.93"); ax.text(0.5, 0.5, "podada", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="0.5")
            ax.plot(xf, y_ - y_.mean(), color="tab:red", lw=1.2)
            if contraste and i_ >= K_NEU:
                lin = B[o, i_] * ((xf - alt_mu) / alt_sd if i_ == K_NEU else xf)
                ax.plot(xf, lin - lin.mean(), color="tab:gray", ls="--", lw=1)
            ax.axhline(0, color="0.8", lw=0.5); ax.tick_params(labelsize=6)
            if o == 0: ax.set_title(f"entrada {nom_in[i_]}", fontsize=8)
            if i_ == 0: ax.set_ylabel(nom_out[o], fontsize=8)
    fig.suptitle(f"Aristas φ de la primera capa KAN (forma {forma}, semilla 42, {int(activa.sum())} de {activa.size} activas), centradas"
                 + ("; gris discontinuo: recta de la cabeza lineal" if contraste else ""), fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "activaciones_kan.png", dpi=130); plt.close(fig)


def ajuste_simbolico(aristas, curvas, simb, candidatas, nom_in, fisica):
    fig, ejes = plt.subplots(1, len(aristas), figsize=(5.4 * len(aristas), 3.8), squeeze=False)
    for ax, (tipo, (o_s, i_s)) in zip(ejes[0], aristas.items()):
        xs, xf, ys = curvas[(o_s, i_s)]
        s_ = simb[(simb.criterio == tipo) & (simb.n_parametros <= 4)].iloc[0]
        ax.plot(xf, ys, "k-", lw=3, label=f"arista {s_.arista}")
        ax.plot(xf, candidatas[s_.expresion](ys)[0](xs, *s_.parametros), "r--", lw=1.6, label=f"{s_.expresion}, R² = {s_.R2:.3f}")
        r_lin = simb[(simb.criterio == tipo) & (simb.expresion == "a x + b")].R2
        ax.set_xlabel(nom_in[i_s] if fisica else "entrada normalizada"); ax.set_ylabel("φ(x)")
        ax.set_title(f"{tipo}: recta R² = {r_lin.iloc[0]:.3f}" if len(r_lin) else tipo, fontsize=9); ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Ajuste simbólico de aristas KAN (x normalizada en el ajuste; eje en unidades de la variable)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "ajuste_simbolico.png", dpi=130); plt.close(fig)


def dependencia_parcial(dep_curvas, alt_g, oni_g, oni_lo, oni_hi, alt_tr, oni_tr, n_tr):
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
    for kind in CABEZAS:
        r_alt, r_oni, d_oni = dep_curvas[kind]
        ejes[0].plot(alt_g, r_alt, color=COLOR[kind], lw=2, label=f"cabeza {kind}")
        ejes[1].plot(oni_g, r_oni, color=COLOR[kind], lw=2, label=f"cabeza {kind}")
        ejes[2].plot(alt_g, d_oni, color=COLOR[kind], lw=2, label=f"cabeza {kind}")
    for ax, v_ in ((ejes[0], np.unique(alt_tr)), (ejes[1], np.unique(oni_tr)), (ejes[2], np.unique(alt_tr))):
        ax.plot(v_, np.full(len(v_), 0.02), "|", color="k", ms=9, transform=ax.get_xaxis_transform())   # valores observados en el entrenamiento
    ejes[0].set_xlabel("altitud de la estación (m); marcas: estaciones de entrenamiento"); ejes[0].set_title("Respuesta a la altitud", fontsize=10)
    ejes[1].set_xlabel(f"ONI medio del año (°C); marcas: los {ANIOS_ENT[1] - ANIOS_ENT[0] + 1} años de entrenamiento"); ejes[1].set_title("Respuesta al ENSO", fontsize=10)
    for v_, t_ in ((-0.5, "Niña"), (0.5, "Niño")):
        ejes[1].axvline(v_, color="0.7", ls=":", lw=1); ejes[1].text(v_, ejes[1].get_ylim()[1], t_, fontsize=7, ha="center", va="top")
    ejes[2].axhline(0, color="0.7", lw=0.8); ejes[2].set_xlabel("altitud de la estación (m)")
    ejes[2].set_title(f"Interacción: efecto de pasar de ONI {oni_lo:.1f} a {oni_hi:.1f} según la altitud", fontsize=10)
    ejes[0].set_ylabel("total anual medio predicho (mm)"); ejes[2].set_ylabel("cambio del total anual (mm)")
    ejes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Dependencia parcial sobre las {n_tr:,} curvas de entrenamiento (satélite observado; media de 3 semillas)".replace(",", " "), fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "dependencia_parcial.png", dpi=130); plt.close(fig)


def aporte_atenuacion_enso(fisica, curvas, activa, nom_out, oni_tip, ap_curvas):
    n_o = activa.shape[0]
    fig, ejes = plt.subplots(1, 2 if fisica else 1, figsize=(14 if fisica else 8.5, 4.2), squeeze=False)
    ejes = ejes[0]
    if fisica:
        i_oni = K_NEU + COVARIABLES.index("oni")
        for o in range(n_o):
            xs_, xf_, y_ = curvas[(o, i_oni)]
            if activa[o, i_oni]:
                ejes[0].plot(xf_, y_ - np.interp(oni_tip["neutro"], xf_, y_), lw=1.6, label=nom_out[o])
        if not activa[:, i_oni].any():
            ejes[0].text(0.5, 0.5, "todas las aristas del ONI quedaron podadas", transform=ejes[0].transAxes, ha="center", fontsize=9)
        ejes[0].axvspan(ejes[0].get_xlim()[0], -0.5, color="#c6dbef", alpha=0.5, lw=0); ejes[0].axvspan(0.5, ejes[0].get_xlim()[1], color="#fdd0a2", alpha=0.5, lw=0)
        ejes[0].axhline(0, color="k", lw=0.6); ejes[0].legend(frameon=False, fontsize=7)
        ejes[0].set_xlabel("ONI medio del año (°C); azul La Niña, naranja El Niño"); ejes[0].set_ylabel("φ(ONI) - φ(neutro)")
        ejes[0].set_title("Aristas del ONI en la KAN (semilla 42): signo de su aporte a cada score", fontsize=10)
    for kind in CABEZAS:
        for f_, ls in (("nina", "-"), ("nino", "--")):
            ejes[-1].plot(t_dia, ap_curvas[(kind, f_)], ls, color=COLOR[kind], lw=1.6 if kind == "kan" else 1.0,
                          label=f"{kind}, {ETQ_FASE[f_]}")
    ejes[-1].axhline(0, color="k", lw=0.6); ejes[-1].set_xticks(MES_INI, MES_ETQ); ejes[-1].legend(frameon=False, fontsize=7, ncol=3)
    ejes[-1].set_ylabel("cambio de la lluvia media predicha (mm/día)")
    ejes[-1].set_title("Aporte (+) y atenuación (-) de un año Niña y uno Niño frente a uno neutro", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "aporte_atenuacion_enso.png", dpi=130); plt.close(fig)


def ablacion_covariables(abl, res):
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, tramo in zip(ejes, TRAMOS):
        for d_, kind in enumerate(CABEZAS):
            t_ = abl[abl.cabeza == kind]
            ic = np.array([[float(x) for x in s.strip("[]").split(",")] for s in t_[f"{tramo}_L2_IC"]])
            xb = np.arange(len(COMBOS)) + (d_ - 1) * 0.27
            ax.bar(xb, t_[f"{tramo}_L2"], width=0.27, color=COLOR[kind], alpha=0.75, label=f"cabeza {kind}")
            ax.errorbar(xb, t_[f"{tramo}_L2"], yerr=[t_[f"{tramo}_L2"] - ic[:, 0], ic[:, 1] - t_[f"{tramo}_L2"]], fmt="none", ecolor="k", capsize=2)
        ax.axhline(res[(tramo, "satelite")]["L2_rel"].mean(), color="0.4", ls="--", lw=1, label="satélite sin corregir")
        ax.set_xticks(np.arange(len(COMBOS)), list(COMBOS), fontsize=8)
        ax.set_title(f"{'Prueba ' + TXT_PRU if tramo == 'prueba' else 'Estaciones ciegas'}: error L2 relativo (IC 95 % bootstrap)", fontsize=10)
    ejes[0].set_ylabel("error L2 relativo (menor es mejor)"); ejes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ablacion_covariables.png", dpi=130); plt.close(fig)


def cabeza_lineal(B):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(B, cmap="RdBu_r", vmin=-np.abs(B).max(), vmax=np.abs(B).max(), aspect="auto")
    ax.set_xticks(range(K_NEU + 2), [f"z{k + 1}" for k in range(K_NEU)] + ["altitud", "ONI"]); ax.set_yticks(range(M_FPCA), [f"s{m + 1}" for m in range(M_FPCA)])
    for (a_, b_), v in np.ndenumerate(B):
        ax.text(b_, a_, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax); ax.set_title("Cabeza lineal: coeficientes B (semilla 42)")
    fig.tight_layout(); fig.savefig(FIG / "cabeza_lineal.png", dpi=130); plt.close(fig)


def fpca_autofunciones(Psi_t, FP_FASE, Phi_y, lam_fp):
    fig, ejes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for ax, m_ in zip(ejes, range(2)):
        ax.plot(t_dia, Psi_t[m_], "k-", lw=2.4, label="base común (todas las fases)")
        for f, col in zip(range(len(FASES)), ("tab:orange", "tab:blue", "0.5")):
            pf = (FP_FASE[f][1] @ Phi_y.T)[m_]
            ax.plot(t_dia, pf * np.sign(pf @ Psi_t[m_]), color=col, lw=1, ls="--", label=f"solo años {ETQ_FASE[FASES[f]]}")
        ax.set_ylabel(f"ψ{m_ + 1}(t)"); ax.axhline(0, color="grey", lw=0.6)
        if m_ == 0: ax.legend(frameon=False, fontsize=7, ncol=4)
        ax.set_title(f"Autofunción {m_ + 1} del decodificador (sqrt mm/día), varianza {100 * (lam_fp[m_] / lam_fp.sum()):.1f} %", fontsize=9)
    ejes[1].set_xticks(MES_INI, MES_ETQ)
    fig.tight_layout(); fig.savefig(FIG / "fpca_autofunciones.png", dpi=130); plt.close(fig)


def peores_casos(ctx, i, peor):
    meta, res, fase, Y_true, X_mm, preds_mm = ctx.meta, ctx.res, ctx.fase, ctx.Y_true, ctx.X_mm, ctx.preds_mm
    fig, ejes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for ax, p_ in zip(ejes.ravel(), peor):
        c = i[p_]
        ax.plot(t_dia, Y_true[:, c], "k-", lw=2.2, label="estación (curva MCO)")
        ax.plot(t_dia, X_mm[:, c], color="tab:gray", lw=1.2, label="satélite")
        ax.plot(t_dia, preds_mm[("ciegas", "lineal", 42)][:, p_], color="tab:blue", lw=1.2, label="lineal")
        ax.plot(t_dia, preds_mm[("ciegas", "kan", 42)][:, p_], color="tab:red", lw=1.4, label="KAN")
        ax.set_title(f"Estación {meta.station_id[c]} ({meta.altitud_m[c]:.0f} m), {meta.year[c]} ({fase[c]}); "
                     f"L2 rel. KAN = {res[('ciegas', 'kan')]['L2_rel'][p_]:.2f}", fontsize=8)
        ax.set_xticks(MES_INI, MES_ETQ)
    ejes[0, 0].legend(frameon=False, fontsize=7); ejes[0, 0].set_ylabel("mm/día"); ejes[1, 0].set_ylabel("mm/día")
    fig.tight_layout(); fig.savefig(FIG / "peores_casos.png", dpi=130); plt.close(fig)
