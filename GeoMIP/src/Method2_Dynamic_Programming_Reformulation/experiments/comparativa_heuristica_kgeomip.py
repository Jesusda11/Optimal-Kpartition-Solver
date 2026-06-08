"""
Paso 6c — Comparativa KGeoMIP Exacto vs KGeoMIPHeuristica.

Valida la calidad de la heuristica geometrica comparando:
  - phi_exacto:     KGeometricSIA (enumeracion S(n,k) completa)
  - phi_heuristica: KGeoMIPHeuristica (candidatos guiados por tabla_transiciones)
  - delta_phi:      phi_heuristica - phi_exacto  (0 = heuristica encontro el optimo)
  - n_candidatos:   cuantos candidatos evaluo la heuristica
  - n_stirling:     cuantos candidatos evalua el exacto (sum S(n,k))
  - ratio:          n_candidatos / n_stirling (fraccion del espacio evaluada)

Como ejecutar (desde el directorio raiz de Method2):
    python experiments/comparativa_heuristica_kgeomip.py

El Excel se guarda en:
    GeoMIP/results/comparativa_heuristica_kgeomip.xlsx
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Resolucion de rutas ───────────────────────────────────────────────────────
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
from src.controllers.strategies.kgeometric import KGeometricSIA
from src.controllers.strategies.kgeometric_heuristica import KGeoMIPHeuristica
from src.funcs.decay import decay_exponencial
from src.funcs.partitions import contar_stirling

profiler_manager.enabled = False

K_MAX = 4
# La heuristica fuerza modo heuristico en todos los casos poniendo el umbral en 0.
# Para ver el fallback exacto, usar N_MAX_EXHAUSTIVO = 6.
N_MAX_EXHAUSTIVO = 0  # fuerza siempre heuristico para medir calidad

CASOS: list[tuple] = [
    ("N3A_01", "A", "100",    "111",    "111",    "111"),
    ("N3A_02", "A", "100",    "111",    "011",    "111"),
    ("N3A_03", "A", "100",    "111",    "111",    "011"),
    ("N3B_01", "B", "100",    "111",    "111",    "111"),
    ("N4A_01", "A", "1000",   "1111",   "1111",   "1111"),
    ("N4A_02", "A", "1000",   "1111",   "0111",   "1111"),
    ("N4A_03", "A", "1000",   "1111",   "1010",   "1111"),
    ("N4B_01", "B", "1000",   "1111",   "1111",   "1111"),
    ("N5A_01", "A", "10000",  "11111",  "11111",  "11111"),
    ("N5A_02", "A", "10000",  "11111",  "10101",  "11111"),
    ("N5B_01", "B", "10000",  "11111",  "11111",  "11111"),
    ("N6A_01", "A", "100000", "111111", "111111", "111111"),
    ("N6A_02", "A", "100000", "111111", "101010", "111111"),
]


def _resolver_tpm(estado_inicial: str, pagina: str) -> Path:
    nombre = f"N{len(estado_inicial)}{pagina}.csv"
    for base in (
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT / "data" / "samples",
    ):
        if (base / nombre).exists():
            return base / nombre
    raise FileNotFoundError(f"No se encontro '{nombre}'")


def ejecutar_exacto(caso, tpm) -> dict:
    id_caso, pagina, estado_inicial, condicion, alcance, mecanismo = caso
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)
    sia = KGeometricSIA(gestor, k_max=K_MAX, decay_fn=decay_exponencial)
    t0 = time.perf_counter()
    try:
        res = sia.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
        phi, error = res.perdida, None
    except Exception as exc:
        phi, error = None, str(exc)
    return {"phi_exacto": phi, "t_exacto_s": round(time.perf_counter() - t0, 6), "error_exacto": error}


def ejecutar_heuristica(caso, tpm) -> dict:
    id_caso, pagina, estado_inicial, condicion, alcance, mecanismo = caso
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)
    sia = KGeoMIPHeuristica(
        gestor, k_max=K_MAX,
        n_max_exhaustivo=N_MAX_EXHAUSTIVO,
        decay_fn=decay_exponencial,
    )
    t0 = time.perf_counter()
    try:
        res = sia.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
        phi, error = res.perdida, None
        n_eval = sia._n_candidatos_evaluados
        modo = sia._modo_usado
    except Exception as exc:
        phi, error, n_eval, modo = None, str(exc), 0, "error"
    return {
        "phi_heuristica": phi,
        "t_heuristica_s": round(time.perf_counter() - t0, 6),
        "n_candidatos": n_eval,
        "modo": modo,
        "error_heuristica": error,
    }


def main():
    print("=" * 70)
    print("Paso 6c  KGeoMIP Exacto vs KGeoMIPHeuristica")
    print(f"k_max={K_MAX}  n_max_exhaustivo={N_MAX_EXHAUSTIVO}  casos={len(CASOS)}")
    print("=" * 70)

    filas = []

    for i, caso in enumerate(CASOS, 1):
        id_caso, pagina, estado_inicial = caso[0], caso[1], caso[2]
        n = len(estado_inicial)
        n_stirling = sum(contar_stirling(n, k) for k in range(2, min(K_MAX, n) + 1))
        print(f"\n[{i:>2}/{len(CASOS)}] {id_caso}  n={n}  S_total={n_stirling}")

        try:
            tpm = np.genfromtxt(_resolver_tpm(estado_inicial, pagina), delimiter=",")
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        print("  Exacto      ... ", end="", flush=True)
        r_exacto = ejecutar_exacto(caso, tpm)
        if r_exacto["error_exacto"]:
            print(f"ERROR: {r_exacto['error_exacto']}")
        else:
            print(f"phi={r_exacto['phi_exacto']:.6f}  t={r_exacto['t_exacto_s']:.4f}s")

        print("  Heuristica  ... ", end="", flush=True)
        r_heur = ejecutar_heuristica(caso, tpm)
        if r_heur["error_heuristica"]:
            print(f"ERROR: {r_heur['error_heuristica']}")
        else:
            delta = None
            if r_exacto["phi_exacto"] is not None and r_heur["phi_heuristica"] is not None:
                delta = r_heur["phi_heuristica"] - r_exacto["phi_exacto"]
            ratio = r_heur["n_candidatos"] / n_stirling if n_stirling > 0 else None
            delta_str = f"  delta={delta:+.6f}" if delta is not None else ""
            print(
                f"phi={r_heur['phi_heuristica']:.6f}  "
                f"t={r_heur['t_heuristica_s']:.4f}s  "
                f"candidatos={r_heur['n_candidatos']}/{n_stirling}"
                f"{delta_str}"
            )

        delta_phi = None
        ratio = None
        if r_exacto["phi_exacto"] is not None and r_heur["phi_heuristica"] is not None:
            delta_phi = round(r_heur["phi_heuristica"] - r_exacto["phi_exacto"], 8)
        if n_stirling > 0 and r_heur["n_candidatos"] is not None:
            ratio = round(r_heur["n_candidatos"] / n_stirling, 4)

        filas.append({
            "id_caso":         id_caso,
            "n_bits":          n,
            "n_stirling_total":n_stirling,
            "phi_exacto":      r_exacto["phi_exacto"],
            "phi_heuristica":  r_heur["phi_heuristica"],
            "delta_phi":       delta_phi,
            "optimo_encontrado": delta_phi is not None and abs(delta_phi) < 1e-6,
            "n_candidatos":    r_heur["n_candidatos"],
            "ratio_espacio":   ratio,
            "t_exacto_s":      r_exacto["t_exacto_s"],
            "t_heuristica_s":  r_heur["t_heuristica_s"],
            "speedup":         round(r_exacto["t_exacto_s"] / r_heur["t_heuristica_s"], 2)
                               if r_heur["t_heuristica_s"] > 0 else None,
            "error":           r_exacto["error_exacto"] or r_heur["error_heuristica"],
        })

    df = pd.DataFrame(filas)
    ruta = GEOMIP_ROOT / "results" / "comparativa_heuristica_kgeomip.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta, index=False)
    print(f"\nResultados guardados en: {ruta}")

    validos = df[df["delta_phi"].notna()]
    if len(validos):
        optimos = validos["optimo_encontrado"].sum()
        print(f"\n── Resumen ──────────────────────────────────────────────────")
        print(f"  Optimo encontrado: {optimos}/{len(validos)} casos")
        print(f"  Delta phi max:     {validos['delta_phi'].abs().max():.6f}")
        print(f"  Ratio espacio medio: {validos['ratio_espacio'].mean():.4f}")
        print(f"  Speedup medio:     {validos['speedup'].mean():.2f}x")
        print(
            df[["id_caso","n_bits","phi_exacto","phi_heuristica",
                "delta_phi","n_candidatos","n_stirling_total","speedup"]]
            .to_string(index=False)
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
