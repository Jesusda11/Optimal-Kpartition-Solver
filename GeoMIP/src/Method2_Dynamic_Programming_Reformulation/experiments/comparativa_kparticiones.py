"""
Paso 5 — Comparativa empirica de KGeometricSIA vs GeometricSIA.

Ejecuta KGeometricSIA (k_max=4) y GeometricSIA (biparticion heuristica) sobre
los mismos 13 casos de prueba (N3-N6) con decay exponencial y distancia hamming.
Exporta resultados a GeoMIP/results/comparativa_kparticiones.xlsx.

Metricas de interes:
  - phi_geo: mejor phi segun GeometricSIA (biparticion heuristica)
  - phi_kgeo: mejor phi segun KGeometricSIA (k-particion exacta)
  - delta_phi: phi_kgeo - phi_geo (negativo => KGeo encuentra mejor MIP)
  - total_candidatos: S(n,2)+S(n,3)+... candidatos evaluados por KGeo

Como ejecutar (desde el directorio raiz de Method2):
    python experiments/comparativa_kparticiones.py

El Excel se guarda en:
    GeoMIP/results/comparativa_kparticiones.xlsx
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

# ── Silenciar logging ─────────────────────────────────────────────────────────
logging.disable(logging.CRITICAL)

# ── Parche nulo de SafeLogger ─────────────────────────────────────────────────
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

# ── Importaciones del proyecto ────────────────────────────────────────────────
from src.models.base.application import aplicacion
from src.middlewares.profile import profiler_manager
from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.kgeometric import KGeometricSIA
from src.funcs.decay import decay_exponencial
from src.funcs.partitions import contar_stirling

profiler_manager.enabled = False

# ── Configuracion ─────────────────────────────────────────────────────────────
K_MAX = 4

CASOS: list[tuple] = [
    # N3 (3 variables, 8 estados)
    ("N3A_01", "A", "100", "111", "111", "111"),
    ("N3A_02", "A", "100", "111", "011", "111"),
    ("N3A_03", "A", "100", "111", "111", "011"),
    ("N3B_01", "B", "100", "111", "111", "111"),

    # N4 (4 variables, 16 estados)
    ("N4A_01", "A", "1000", "1111", "1111", "1111"),
    ("N4A_02", "A", "1000", "1111", "0111", "1111"),
    ("N4A_03", "A", "1000", "1111", "1010", "1111"),
    ("N4B_01", "B", "1000", "1111", "1111", "1111"),

    # N5 (5 variables, 32 estados)
    ("N5A_01", "A", "10000", "11111", "11111", "11111"),
    ("N5A_02", "A", "10000", "11111", "10101", "11111"),
    ("N5B_01", "B", "10000", "11111", "11111", "11111"),

    # N6 (6 variables, 64 estados)
    ("N6A_01", "A", "100000", "111111", "111111", "111111"),
    ("N6A_02", "A", "100000", "111111", "101010", "111111"),
]


def _resolver_tpm_path(estado_inicial: str, pagina: str) -> Path:
    nombre = f"N{len(estado_inicial)}{pagina}.csv"
    candidatos = (
        METHOD2_ROOT / "src" / ".samples" / nombre,
        METHOD2_ROOT / ".samples" / nombre,
        GEOMIP_ROOT / "data" / "samples" / nombre,
    )
    for c in candidatos:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No se encontro '{nombre}'. Buscado en:\n"
        + "\n".join(f"  {c}" for c in candidatos)
    )


def ejecutar_geometric(caso: tuple, tpm: np.ndarray) -> dict:
    id_caso, pagina, estado_inicial, condiciones, alcance, mecanismo = caso
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)
    sia = GeometricSIA(gestor, decay_fn=decay_exponencial)
    t0 = time.perf_counter()
    try:
        res = sia.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        phi = res.perdida
        particion = res.particion
        error = None
    except Exception as exc:
        phi = None
        particion = None
        error = str(exc)
    t1 = time.perf_counter()
    return {
        "id_caso":    id_caso,
        "n_bits":     len(estado_inicial),
        "estrategia": "GeometricSIA",
        "phi":        phi,
        "tiempo_s":   round(t1 - t0, 6),
        "particion":  particion,
        "error":      error,
    }


def ejecutar_kgeometric(caso: tuple, tpm: np.ndarray, k_max: int) -> dict:
    id_caso, pagina, estado_inicial, condiciones, alcance, mecanismo = caso
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)
    sia = KGeometricSIA(gestor, k_max=k_max, decay_fn=decay_exponencial)
    t0 = time.perf_counter()
    try:
        res = sia.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        phi = res.perdida
        particion = res.particion
        error = None
    except Exception as exc:
        phi = None
        particion = None
        error = str(exc)
    t1 = time.perf_counter()
    n = len(estado_inicial)
    total_candidatos = sum(contar_stirling(n, k) for k in range(2, min(k_max, n) + 1))
    return {
        "id_caso":          id_caso,
        "n_bits":           n,
        "estrategia":       "KGeometricSIA",
        "k_max":            k_max,
        "total_candidatos": total_candidatos,
        "phi":              phi,
        "tiempo_s":         round(t1 - t0, 6),
        "particion":        particion,
        "error":            error,
    }


def construir_excel(df_geo: pd.DataFrame, df_kgeo: pd.DataFrame, ruta_salida: Path) -> None:
    df_comp = df_geo[["id_caso", "n_bits", "phi", "tiempo_s"]].merge(
        df_kgeo[["id_caso", "phi", "tiempo_s", "total_candidatos"]],
        on="id_caso",
        suffixes=("_geo", "_kgeo"),
    )
    df_comp["delta_phi"] = df_comp["phi_kgeo"] - df_comp["phi_geo"]
    df_comp["speedup"] = df_comp["tiempo_s_geo"] / df_comp["tiempo_s_kgeo"]

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        df_geo.to_excel(writer, sheet_name="GeometricSIA", index=False)
        df_kgeo.to_excel(writer, sheet_name="KGeometricSIA", index=False)
        df_comp.to_excel(writer, sheet_name="Comparativa", index=False)


def main() -> None:
    print("=" * 60)
    print("Paso 5 — KGeometricSIA vs GeometricSIA")
    print(f"k_max: {K_MAX}  |  Casos: {len(CASOS)}")
    print("=" * 60)

    filas_geo: list[dict] = []
    filas_kgeo: list[dict] = []

    for i, caso in enumerate(CASOS, 1):
        id_caso, pagina, estado_inicial = caso[0], caso[1], caso[2]
        n = len(estado_inicial)
        stirling_str = ", ".join(
            f"S({n},{k})={contar_stirling(n,k)}" for k in range(2, min(K_MAX, n) + 1)
        )
        print(f"\n[{i:>2}/{len(CASOS)}] {id_caso}  n={n}  candidatos: {stirling_str}")

        try:
            tpm_path = _resolver_tpm_path(estado_inicial, pagina)
            tpm = np.genfromtxt(tpm_path, delimiter=",")
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            filas_geo.append({
                "id_caso": id_caso, "n_bits": n, "estrategia": "GeometricSIA",
                "phi": None, "tiempo_s": None, "particion": None, "error": str(exc),
            })
            filas_kgeo.append({
                "id_caso": id_caso, "n_bits": n, "estrategia": "KGeometricSIA",
                "k_max": K_MAX, "total_candidatos": None,
                "phi": None, "tiempo_s": None, "particion": None, "error": str(exc),
            })
            continue

        print("  GeometricSIA ... ", end="", flush=True)
        fila_geo = ejecutar_geometric(caso, tpm)
        filas_geo.append(fila_geo)
        if fila_geo["error"]:
            print(f"ERROR: {fila_geo['error']}")
        else:
            print(f"phi={fila_geo['phi']:.6f}  t={fila_geo['tiempo_s']:.4f}s")

        print(f"  KGeometricSIA  ... ", end="", flush=True)
        fila_kgeo = ejecutar_kgeometric(caso, tpm, K_MAX)
        filas_kgeo.append(fila_kgeo)
        if fila_kgeo["error"]:
            print(f"ERROR: {fila_kgeo['error']}")
        else:
            delta = (fila_kgeo["phi"] - fila_geo["phi"]) if (fila_geo["phi"] is not None and fila_kgeo["phi"] is not None) else None
            delta_str = f"  delta={delta:+.6f}" if delta is not None else ""
            print(f"phi={fila_kgeo['phi']:.6f}  t={fila_kgeo['tiempo_s']:.4f}s{delta_str}")

    df_geo = pd.DataFrame(filas_geo)
    df_kgeo = pd.DataFrame(filas_kgeo)

    ruta_salida = GEOMIP_ROOT / "results" / "comparativa_kparticiones.xlsx"
    construir_excel(df_geo, df_kgeo, ruta_salida)
    print(f"\nResultados guardados en: {ruta_salida}")

    print("\n── Resumen comparativo ──────────────────────────────────")
    df_comp = df_geo[["id_caso", "n_bits", "phi", "tiempo_s"]].merge(
        df_kgeo[["id_caso", "phi", "tiempo_s", "total_candidatos"]],
        on="id_caso", suffixes=("_geo", "_kgeo"),
    )
    df_comp["delta_phi"] = df_comp["phi_kgeo"] - df_comp["phi_geo"]
    print(
        df_comp[["id_caso", "n_bits", "phi_geo", "phi_kgeo", "delta_phi",
                 "tiempo_s_geo", "tiempo_s_kgeo", "total_candidatos"]]
        .to_string(index=False)
    )
    print()


if __name__ == "__main__":
    main()
