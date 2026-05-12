"""
Paso 3 — Benchmark de paralelizacion de calcular_costos_nivel.

Ejecuta cada caso de prueba dos veces (parallel=True y parallel=False) con la
variante exponencial y mide el tiempo de ejecucion de aplicar_estrategia.
Exporta los resultados a GeoMIP/results/benchmark_paralelizacion.xlsx.

Como ejecutar (desde el directorio raiz de Method2):
    python experiments/benchmark_paralelizacion.py
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

# ── Parche nulo de SafeLogger (ver CAMBIOS_PASO2 para explicacion completa) ───
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
from src.funcs.decay import decay_exponencial

profiler_manager.enabled = False

# ── Casos de prueba ───────────────────────────────────────────────────────────
# Solo se usa la variante exponencial porque el objetivo es medir el impacto
# de la paralelizacion, no comparar funciones de decay.
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

N_REPETICIONES = 3  # Repeticiones por (caso, modo) para reducir ruido de medicion


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
    raise FileNotFoundError(f"No se encontro '{nombre}'.")


def ejecutar_una_vez(caso, parallel: bool, tpm: np.ndarray) -> float:
    """Ejecuta un caso con el modo indicado y retorna el tiempo de algo (s)."""
    id_caso, pagina, estado_inicial, condiciones, alcance, mecanismo = caso
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)
    sia = GeometricSIA(gestor, decay_fn=decay_exponencial, parallel=parallel)
    t0 = time.perf_counter()
    sia.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
    return time.perf_counter() - t0


def main() -> None:
    print("=" * 60)
    print("Paso 3 - Benchmark de paralelizacion BFS")
    print(f"Casos: {len(CASOS)}  |  Repeticiones por modo: {N_REPETICIONES}")
    print(f"Total de ejecuciones: {len(CASOS) * 2 * N_REPETICIONES}")
    print("=" * 60)

    filas: list[dict] = []

    for caso in CASOS:
        id_caso, pagina, estado_inicial, condiciones, alcance, mecanismo = caso
        try:
            tpm = np.genfromtxt(_resolver_tpm_path(estado_inicial, pagina), delimiter=",")
        except FileNotFoundError as exc:
            print(f"  [SKIP] {id_caso}: {exc}")
            continue

        for parallel, modo in [(False, "secuencial"), (True, "paralelo")]:
            tiempos = []
            for rep in range(N_REPETICIONES):
                t = ejecutar_una_vez(caso, parallel, tpm)
                tiempos.append(t)
                print(
                    f"  {id_caso:<12} {modo:<12} rep={rep+1}  t={t:.4f}s",
                    flush=True,
                )
            filas.append({
                "id_caso":       id_caso,
                "n_bits":        len(estado_inicial),
                "modo":          modo,
                "t_min_s":       round(min(tiempos), 6),
                "t_media_s":     round(sum(tiempos) / len(tiempos), 6),
                "t_max_s":       round(max(tiempos), 6),
            })

    df = pd.DataFrame(filas)

    # Tabla pivote: filas = caso, columnas = (secuencial | paralelo) x metrica
    pivot = df.pivot_table(
        values=["t_min_s", "t_media_s"],
        index="id_caso",
        columns="modo",
        aggfunc="first",
    )

    # Calcular speedup: t_secuencial / t_paralelo (usando t_media)
    sec = df[df["modo"] == "secuencial"].set_index("id_caso")["t_media_s"]
    par = df[df["modo"] == "paralelo"].set_index("id_caso")["t_media_s"]
    speedup = (sec / par).rename("speedup").reset_index()
    speedup.columns = ["id_caso", "speedup"]
    speedup["speedup"] = speedup["speedup"].round(3)

    ruta_salida = GEOMIP_ROOT / "results" / "benchmark_paralelizacion.xlsx"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Resultados_Crudos", index=False)
        pivot.to_excel(writer, sheet_name="Comparativa_Tiempos")
        speedup.to_excel(writer, sheet_name="Speedup", index=False)

    print(f"\nResultados guardados en: {ruta_salida}")
    print("\n── Speedup por caso (secuencial / paralelo) ──────────────")
    print(speedup.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
