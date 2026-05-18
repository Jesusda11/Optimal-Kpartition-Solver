"""
Paso 4 — Comparativa empirica de variantes de metrica de distancia en GeometricSIA.

Ejecuta las 4 variantes de distancia (hamming, hamming_normalizado, jaccard, causal)
sobre un conjunto fijo de casos de prueba (N3-N6) con decay exponencial fijo y
exporta resultados a GeoMIP/results/comparativa_distancias.xlsx.

El decay se fija en exponencial para aislar el efecto de la metrica de distancia
y mantener comparabilidad con los resultados del Paso 2 (comparativa_decay.py).

Como ejecutar (desde el directorio raiz de Method2):
    python experiments/comparativa_distancias.py

El Excel de salida se guarda en:
    GeoMIP/results/comparativa_distancias.xlsx
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Resolucion de rutas del proyecto ─────────────────────────────────────────
METHOD2_ROOT = Path(__file__).resolve().parents[1]
GEOMIP_ROOT  = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHOD2_ROOT))

# ── Silenciar logging ANTES de importar el proyecto ──────────────────────────
logging.disable(logging.CRITICAL)

# ── Parche nulo de SafeLogger ─────────────────────────────────────────────────
# Debe ocurrir antes de importar geometric.py/sia.py para evitar que
# __setup_logger abra FileHandlers por cada instancia de GeometricSIA.
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
from src.models.base.sia import SIA
from src.funcs.decay import decay_exponencial
from src.funcs.distances import DISTANCE_VARIANTS

profiler_manager.enabled = False

# ── Variante de decay fija ────────────────────────────────────────────────────
# Se fija en exponencial para aislar el efecto de la metrica de distancia.
DECAY_FN = decay_exponencial
DECAY_NOMBRE = "exponencial"

# ── Variantes de distancia a comparar ────────────────────────────────────────
#
#   hamming:             baseline (comportamiento original)
#   hamming_normalizado: d_H / n,  rango [0, 1]
#   jaccard:             diferencia / union activa, rango [0, 1]
#   causal:              L1 entre distribuciones de transicion, rango [0, 1]
#
VARIANTES: dict[str, str] = {
    "hamming":             "hamming",
    "hamming_normalizado": "hamming_normalizado",
    "jaccard":             "jaccard",
    "causal":              "causal",
}

VARIANTE_BASE = "hamming"

# ── Casos de prueba ───────────────────────────────────────────────────────────
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
        f"No se encontro '{nombre}'. Se busco en:\n"
        + "\n".join(f"  {c}" for c in candidatos)
    )


def ejecutar_caso(
    caso: tuple,
    dist_nombre: str,
    tpm: np.ndarray,
) -> dict:
    id_caso, pagina, estado_inicial, condiciones, alcance, mecanismo = caso

    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)
    sia = GeometricSIA(gestor, decay_fn=DECAY_FN, dist_fn=dist_nombre)

    t0 = time.perf_counter()
    try:
        resultado = sia.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        phi      = resultado.perdida
        particion = resultado.particion
        error    = None
    except Exception as exc:
        phi      = None
        particion = None
        error    = str(exc)
    t1 = time.perf_counter()

    return {
        "id_caso":        id_caso,
        "pagina":         pagina,
        "n_bits":         len(estado_inicial),
        "estado_inicial": estado_inicial,
        "condiciones":    condiciones,
        "alcance":        alcance,
        "mecanismo":      mecanismo,
        "decay":          DECAY_NOMBRE,
        "distancia":      dist_nombre,
        "phi":            phi,
        "tiempo_s":       round(t1 - t0, 6),
        "particion":      particion,
        "error":          error,
    }


def agregar_coincidencia(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna 'coincide_con_hamming' comparando con la variante base."""
    base = (
        df[df["distancia"] == VARIANTE_BASE][["id_caso", "particion"]]
        .rename(columns={"particion": "particion_base"})
    )
    df2 = df.merge(base, on="id_caso", how="left")
    df["coincide_con_hamming"] = df2["particion"] == df2["particion_base"]
    return df


def construir_excel(df: pd.DataFrame, ruta_salida: Path) -> None:
    pivot_phi = df.pivot_table(
        values="phi",
        index="id_caso",
        columns="distancia",
        aggfunc="first",
    )

    pivot_tiempo = df.pivot_table(
        values="tiempo_s",
        index="id_caso",
        columns="distancia",
        aggfunc="first",
    )

    pivot_coincidencia = df.pivot_table(
        values="coincide_con_hamming",
        index="id_caso",
        columns="distancia",
        aggfunc="first",
    )

    resumen = (
        df.groupby("distancia")
        .agg(
            phi_media=("phi", "mean"),
            phi_min=("phi", "min"),
            phi_max=("phi", "max"),
            phi_desv=("phi", "std"),
            tiempo_medio_s=("tiempo_s", "mean"),
            tiempo_total_s=("tiempo_s", "sum"),
            tasa_coincidencia=("coincide_con_hamming", "mean"),
            n_errores=("error", lambda x: x.notna().sum()),
        )
        .round(6)
    )

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Resultados_Crudos", index=False)
        pivot_phi.to_excel(writer, sheet_name="Phi_por_Distancia")
        pivot_tiempo.to_excel(writer, sheet_name="Tiempo_por_Distancia")
        pivot_coincidencia.to_excel(writer, sheet_name="Coincidencia")
        resumen.to_excel(writer, sheet_name="Resumen_Estadistico")


def main() -> None:
    print("=" * 60)
    print("Paso 4 - Comparativa de variantes de metrica de distancia")
    print(f"Decay fijo: {DECAY_NOMBRE}")
    print(f"Casos: {len(CASOS)}  |  Variantes: {len(VARIANTES)}")
    print(f"Total de ejecuciones: {len(CASOS) * len(VARIANTES)}")
    print("=" * 60)

    filas: list[dict] = []
    total = len(CASOS) * len(VARIANTES)
    n = 0

    for caso in CASOS:
        id_caso, pagina, estado_inicial, condiciones, alcance, mecanismo = caso
        try:
            tpm_path = _resolver_tpm_path(estado_inicial, pagina)
            tpm = np.genfromtxt(tpm_path, delimiter=",")
        except FileNotFoundError as exc:
            print(f"  [SKIP] {id_caso}: {exc}")
            for dist_nombre in VARIANTES:
                n += 1
                filas.append({
                    "id_caso": id_caso, "pagina": pagina,
                    "n_bits": len(estado_inicial), "estado_inicial": estado_inicial,
                    "condiciones": condiciones, "alcance": alcance,
                    "mecanismo": mecanismo, "decay": DECAY_NOMBRE,
                    "distancia": dist_nombre,
                    "phi": None, "tiempo_s": None,
                    "particion": None, "error": str(exc),
                })
            continue

        for dist_nombre in VARIANTES:
            n += 1
            print(
                f"[{n:>3}/{total}] {id_caso:<12} | {dist_nombre:<22}",
                flush=True,
            )
            fila = ejecutar_caso(caso, dist_nombre, tpm)
            filas.append(fila)

            if fila["error"]:
                print(f"    ERROR: {fila['error']}")
            else:
                print(f"    phi={fila['phi']:.6f}  t={fila['tiempo_s']:.4f}s")

    print()
    df = pd.DataFrame(filas)
    df = agregar_coincidencia(df)

    ruta_salida = GEOMIP_ROOT / "results" / "comparativa_distancias.xlsx"
    construir_excel(df, ruta_salida)

    print(f"Resultados guardados en: {ruta_salida}")
    print()

    resumen = (
        df.groupby("distancia")
        .agg(
            phi_media=("phi", "mean"),
            tiempo_medio_s=("tiempo_s", "mean"),
            tasa_coincidencia=("coincide_con_hamming", "mean"),
        )
        .round(6)
    )
    print("── Resumen estadistico por metrica de distancia ──────────")
    print(resumen.to_string())
    print()


if __name__ == "__main__":
    main()
