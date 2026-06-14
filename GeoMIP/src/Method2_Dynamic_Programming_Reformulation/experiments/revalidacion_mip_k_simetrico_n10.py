"""
experiments/revalidacion_mip_k_simetrico_n10.py

Re-validacion del fenomeno "MIP en k>2" del modelo SIMETRICO (KGeometricSIAHeuristica)
para las pruebas 10, 24, 31, 34, 38, 39 de la hoja 10A-Elementos.

Motivacion
----------
Resultados historicos reportaron el MIP en k>2 para esos 6 casos. Se generaron
ANTES del fix de no-determinismo (random.sample sin seed en el truncado de
candidatos). Hay que confirmar si el fenomeno es GENUINO o ARTEFACTO.

Probe previo (probe_determinismo_simetrico.py) confirmo que el numero de
candidatos unicos por k para estos 6 casos es <= 1537 < 2000 = m_max_candidatos,
por lo que el branch random.sample NUNCA se dispara aqui: la heuristica es
deterministica para estos casos.

Que hace este script
--------------------
Por cada caso y k in {2,3,4,5}:
  1. HEURISTICA run1 y run2 (n_max_exhaustivo=6 -> modo heuristico para n=10).
     Reporta phi y delta entre corridas (debe ser 0 -> determinista).
  2. EXACTO ground truth (n_max_exhaustivo=999 -> enumeracion S(n,k) completa).
     Es el arbitro definitivo de en que k cae el MIP real.

Verdicto por caso:
  GENUINO   : el MIP exacto cae en k>2 (y la heuristica lo reproduce).
  ARTEFACTO : el MIP exacto cae en k=2 (el k>2 historico era ruido/heuristica pobre).
  CAMBIO    : el MIP cae en k>2 pero en un k distinto al historico.

Como ejecutar (desde Method2_Dynamic_Programming_Reformulation/):
    .venv/Scripts/python.exe experiments/revalidacion_mip_k_simetrico_n10.py

Salida Excel: GeoMIP/results/revalidacion_mip_k_simetrico_n10.xlsx
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

METHOD2_ROOT = Path(__file__).resolve().parents[1]
GEOMIP_ROOT  = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHOD2_ROOT))

logging.disable(logging.CRITICAL)

import src.middlewares.slogger as _slogger_mod

class _NullSafeLogger:
    def __init__(self, name): pass
    def debug(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def error(self, *a, **k): pass
    def critic(self, *a, **k): pass
    def fatal(self, *a, **k): pass
    def set_log(self, *a, **k): pass

_slogger_mod.SafeLogger = _NullSafeLogger

from src.models.base.application import aplicacion
from src.middlewares.profile import profiler_manager
from src.controllers.manager import Manager
from src.controllers.strategies.kgeometric_heuristica import KGeometricSIAHeuristica
from src.funcs.decay import decay_exponencial

from src.ejecutar_kgeomip import (
    convertir_a_binario, parsear_hoja, resolver_tpm, leer_excel, INPUT_XLSX,
)

profiler_manager.enabled = False

HOJA = "10A-Elementos"
PRUEBAS_OBJETIVO = [10, 24, 31, 34, 38, 39]
K_VALUES = [2, 3, 4, 5]

# MIP historico reportado (antes del fix de no-determinismo)
HISTORICO = {
    10: {2: 1.398, 3: 1.303, 4: 1.277, 5: 1.373, "k_mip": 4},
    24: {2: 1.398, 3: 1.303, 4: 1.277, 5: 1.373, "k_mip": 4},
    31: {2: 0.988, 3: 0.961, 4: 0.979, 5: 1.014, "k_mip": 3},
    34: {2: 0.219, 3: 0.209, 4: None,  5: None,  "k_mip": 3},
    38: {2: 0.992, 3: 0.969, 4: 0.963, 5: None,  "k_mip": 4},
    39: {2: 0.797, 3: 0.758, 4: 0.783, 5: None,  "k_mip": 3},
}


def _correr(estado_inicial, condiciones, alcance, mecanismo, tpm, n_max_exh):
    """Devuelve {k: phi} ejecutando la estrategia simetrica una vez."""
    gestor = Manager(estado_inicial=estado_inicial)
    sia = KGeometricSIAHeuristica(
        gestor,
        k_max=5,
        k_min=2,
        n_max_exhaustivo=n_max_exh,
        m_max_candidatos=2000,   # default real de produccion
        decay_fn=decay_exponencial,
    )
    sia.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
    return {k: d["phi"] for k, d in sia._resultados_por_k.items()}, sia._modo_usado


def _k_mip(phis: dict) -> int | None:
    validos = {k: v for k, v in phis.items() if v is not None}
    if not validos:
        return None
    return min(validos, key=lambda k: validos[k])


def main() -> None:
    n, variante = parsear_hoja(HOJA)
    estado_inicial, todos_casos = leer_excel(INPUT_XLSX, HOJA)
    tpm_path, tpm = resolver_tpm(n, variante)
    condiciones = "1" * len(estado_inicial)
    aplicacion.pagina_sample_network = variante

    por_num = {c[0]: c for c in todos_casos}

    print("=" * 100)
    print(f"RE-VALIDACION MIP k>2 — Modelo SIMETRICO — Hoja {HOJA}  n={len(estado_inicial)}")
    print(f"Estado inicial: {estado_inicial}   Pruebas: {PRUEBAS_OBJETIVO}")
    print("=" * 100)

    filas = []

    for num in PRUEBAS_OBJETIVO:
        if num not in por_num:
            print(f"[SKIP] Prueba {num} no encontrada en la hoja.")
            continue
        _, alcance_str, mecanismo_str = por_num[num]
        alcance   = convertir_a_binario(alcance_str,   len(estado_inicial))
        mecanismo = convertir_a_binario(mecanismo_str, len(estado_inicial))

        print(f"\n[Prueba {num}]  Alcance={alcance_str!r}  Mecanismo={mecanismo_str!r}")

        # Heuristica x2 (determinismo)
        t0 = time.perf_counter()
        phis_h1, modo_h = _correr(estado_inicial, condiciones, alcance, mecanismo, tpm, 6)
        phis_h2, _      = _correr(estado_inicial, condiciones, alcance, mecanismo, tpm, 6)
        t_heur = time.perf_counter() - t0

        # Exacto ground truth
        t0 = time.perf_counter()
        phis_ex, modo_ex = _correr(estado_inicial, condiciones, alcance, mecanismo, tpm, 999)
        t_exac = time.perf_counter() - t0

        # Delta determinismo (max sobre k)
        deltas = []
        for k in K_VALUES:
            a, b = phis_h1.get(k), phis_h2.get(k)
            if a is not None and b is not None:
                deltas.append(abs(a - b))
        delta_det = max(deltas) if deltas else None

        k_mip_heur = _k_mip(phis_h1)
        k_mip_exac = _k_mip(phis_ex)
        k_mip_hist = HISTORICO[num]["k_mip"]

        # Veredicto basado en el EXACTO (ground truth)
        if k_mip_exac is None:
            veredicto = "INDETERMINADO"
        elif k_mip_exac == 2:
            veredicto = "ARTEFACTO"
        elif k_mip_exac == k_mip_hist:
            veredicto = "GENUINO"
        else:
            veredicto = "CAMBIO"

        coincide_hist = (k_mip_exac == k_mip_hist)

        print(f"  modo_heur={modo_h}  modo_exac={modo_ex}  "
              f"t_heur(x2)={t_heur:.1f}s  t_exac={t_exac:.1f}s")
        print(f"  {'k':>3} {'phi_heur':>12} {'phi_exacto':>12} {'phi_historico':>14}")
        for k in K_VALUES:
            ph = phis_h1.get(k)
            pe = phis_ex.get(k)
            hh = HISTORICO[num].get(k)
            ph_s = f"{ph:.6f}" if ph is not None else "—"
            pe_s = f"{pe:.6f}" if pe is not None else "—"
            hh_s = f"{hh:.3f}" if hh is not None else "—"
            marca_e = " <== MIP_exacto" if k == k_mip_exac else ""
            print(f"  {k:>3} {ph_s:>12} {pe_s:>12} {hh_s:>14}{marca_e}")
        print(f"  k_mip: heur={k_mip_heur}  exacto={k_mip_exac}  historico={k_mip_hist}  "
              f"delta_determinismo={delta_det:.2e}" if delta_det is not None
              else f"  k_mip: heur={k_mip_heur}  exacto={k_mip_exac}  historico={k_mip_hist}")
        print(f"  VEREDICTO: {veredicto}  (coincide con historico: {coincide_hist})")

        fila = {
            "prueba":      num,
            "phi_k2":      round(phis_ex.get(2), 6) if phis_ex.get(2) is not None else None,
            "phi_k3":      round(phis_ex.get(3), 6) if phis_ex.get(3) is not None else None,
            "phi_k4":      round(phis_ex.get(4), 6) if phis_ex.get(4) is not None else None,
            "phi_k5":      round(phis_ex.get(5), 6) if phis_ex.get(5) is not None else None,
            "k_opt_actual_exacto":  k_mip_exac,
            "k_opt_actual_heur":    k_mip_heur,
            "k_opt_reportado_antes": k_mip_hist,
            "coincide":    coincide_hist,
            "delta_entre_corridas": delta_det,
            "veredicto":   veredicto,
            # phi heuristica para trazabilidad
            "phi_heur_k2": round(phis_h1.get(2), 6) if phis_h1.get(2) is not None else None,
            "phi_heur_k3": round(phis_h1.get(3), 6) if phis_h1.get(3) is not None else None,
            "phi_heur_k4": round(phis_h1.get(4), 6) if phis_h1.get(4) is not None else None,
            "phi_heur_k5": round(phis_h1.get(5), 6) if phis_h1.get(5) is not None else None,
        }
        filas.append(fila)

    # ── Exportar ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(filas)
    ruta = GEOMIP_ROOT / "results" / "revalidacion_mip_k_simetrico_n10.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta, index=False)

    print("\n" + "=" * 100)
    print("TABLA RESUMEN (phi = EXACTO ground truth)")
    print("=" * 100)
    cols = ["prueba", "phi_k2", "phi_k3", "phi_k4", "phi_k5",
            "k_opt_actual_exacto", "k_opt_reportado_antes", "coincide",
            "delta_entre_corridas", "veredicto"]
    print(df[cols].to_string(index=False))
    print(f"\nResultados guardados en: {ruta}")


if __name__ == "__main__":
    main()