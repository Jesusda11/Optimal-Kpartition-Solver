"""
Paso 6b — Comparativa empirica de decay y distancia en KGeoMIP Simetrico.

Ejecuta KGeoMIP (k_max=4) con todas las combinaciones de funcion de decaimiento
(exponencial, polinomial, logaritmico) y metrica de distancia (hamming,
hamming_normalizado, jaccard, causal) sobre los 13 casos de prueba N3-N6.

Total: 3 decays x 4 distancias x 13 casos = 156 ejecuciones.

Hipotesis: en el modo exhaustivo de KGeoMIP, decay_fn y dist_fn NO afectan
el phi resultante, porque la enumeracion S(n,k) calcula phi directamente via
k_bipartir() + emd_efecto() sin recurrir a la tabla de transiciones geometrica.
Por tanto, se espera coincidencia total (100%) entre todas las combinaciones.

Este experimento verifica formalmente esa hipotesis y documenta el comportamiento
como baseline para comparar con la fase de heuristicas (Paso 6c), donde decay
y distancia SI afectaran los resultados al influir en la tabla_transiciones.

Como ejecutar (desde el directorio raiz de Method2):
    python experiments/comparativa_decay_distancias_kgeomip.py

El Excel se guarda en:
    GeoMIP/results/comparativa_decay_distancias_kgeomip.xlsx
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
from src.controllers.strategies.kgeometric import KGeometricSIA
from src.funcs.decay import DECAY_VARIANTS

profiler_manager.enabled = False

# ── Configuracion ─────────────────────────────────────────────────────────────
K_MAX = 4

# Variantes de decaimiento
DECAYS: dict[str, callable] = DECAY_VARIANTS  # exponencial, polinomial, logaritmico

# Variantes de distancia (causal se resuelve internamente en GeometricSIA)
DISTANCIAS: list[str] = ["hamming", "hamming_normalizado", "jaccard", "causal"]

# Combinacion de referencia para medir coincidencia
DECAY_BASE    = "exponencial"
DISTANCIA_BASE = "hamming"

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


def ejecutar_caso(
    caso: tuple,
    tpm: np.ndarray,
    decay_nombre: str,
    dist_nombre: str,
) -> dict:
    id_caso, pagina, estado_inicial, condicion, alcance, mecanismo = caso
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)
    sia = KGeometricSIA(
        gestor,
        k_max=K_MAX,
        decay_fn=DECAYS[decay_nombre],
        dist_fn=dist_nombre,
    )

    t0 = time.perf_counter()
    try:
        res   = sia.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
        phi   = res.perdida
        particion = res.particion
        error = None
    except Exception as exc:
        phi   = None
        particion = None
        error = str(exc)
    t1 = time.perf_counter()

    return {
        "id_caso":    id_caso,
        "n_bits":     len(estado_inicial),
        "decay":      decay_nombre,
        "distancia":  dist_nombre,
        "phi":        phi,
        "tiempo_s":   round(t1 - t0, 6),
        "particion":  particion,
        "error":      error,
    }


def agregar_coincidencia(df: pd.DataFrame) -> pd.DataFrame:
    """Compara phi y particion de cada combinacion contra la base (exponencial+hamming)."""
    base = (
        df[(df["decay"] == DECAY_BASE) & (df["distancia"] == DISTANCIA_BASE)]
        [["id_caso", "phi", "particion"]]
        .rename(columns={"phi": "phi_base", "particion": "particion_base"})
    )
    df2 = df.merge(base, on="id_caso", how="left")
    df["coincide_phi"]       = (df2["phi"] - df2["phi_base"]).abs() < 1e-6
    df["coincide_particion"] = df2["particion"] == df2["particion_base"]
    return df


def construir_excel(df: pd.DataFrame, ruta: Path) -> None:
    df["combo"] = df["decay"] + "+" + df["distancia"]

    pivot_phi = df.pivot_table(
        values="phi", index="id_caso", columns="combo", aggfunc="first"
    )
    pivot_tiempo = df.pivot_table(
        values="tiempo_s", index="id_caso", columns="combo", aggfunc="first"
    )
    pivot_phi_ok = df.pivot_table(
        values="coincide_phi", index="id_caso", columns="combo", aggfunc="first"
    )

    resumen = (
        df.groupby("combo")
        .agg(
            phi_media=("phi", "mean"),
            phi_min=("phi", "min"),
            phi_max=("phi", "max"),
            tiempo_medio_s=("tiempo_s", "mean"),
            tasa_coincidencia_phi=("coincide_phi", "mean"),
            tasa_coincidencia_part=("coincide_particion", "mean"),
            n_errores=("error", lambda x: x.notna().sum()),
        )
        .round(6)
    )

    ruta.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Resultados_Crudos", index=False)
        pivot_phi.to_excel(writer, sheet_name="Phi_por_Combinacion")
        pivot_tiempo.to_excel(writer, sheet_name="Tiempo_por_Combinacion")
        pivot_phi_ok.to_excel(writer, sheet_name="Coincidencia_Phi")
        resumen.to_excel(writer, sheet_name="Resumen_Estadistico")


def main() -> None:
    combos = [(d, dist) for d in DECAYS for dist in DISTANCIAS]
    total  = len(CASOS) * len(combos)

    print("=" * 68)
    print("Paso 6b  Comparativa decay x distancia en KGeoMIP Simetrico")
    print(f"Decays: {list(DECAYS)}  Distancias: {DISTANCIAS}")
    print(f"Casos: {len(CASOS)}  |  Combinaciones: {len(combos)}  |  Total: {total}")
    print("=" * 68)

    filas: list[dict] = []
    n = 0

    for caso in CASOS:
        id_caso, pagina, estado_inicial = caso[0], caso[1], caso[2]
        try:
            tpm_path = _resolver_tpm(estado_inicial, pagina)
            tpm = np.genfromtxt(tpm_path, delimiter=",")
        except FileNotFoundError as exc:
            print(f"\n[SKIP] {id_caso}: {exc}")
            for decay_n, dist_n in combos:
                n += 1
                filas.append({
                    "id_caso": id_caso, "n_bits": len(estado_inicial),
                    "decay": decay_n, "distancia": dist_n,
                    "phi": None, "tiempo_s": None,
                    "particion": None, "error": str(exc),
                })
            continue

        for decay_n, dist_n in combos:
            n += 1
            print(
                f"[{n:>3}/{total}] {id_caso:<10} | decay={decay_n:<12} | dist={dist_n:<22}",
                end="", flush=True,
            )
            fila = ejecutar_caso(caso, tpm, decay_n, dist_n)
            filas.append(fila)
            if fila["error"]:
                print(f"  ERROR: {fila['error']}")
            else:
                print(f"  phi={fila['phi']:.6f}  t={fila['tiempo_s']:.4f}s")

    df = pd.DataFrame(filas)
    df = agregar_coincidencia(df)

    ruta = GEOMIP_ROOT / "results" / "comparativa_decay_distancias_kgeomip.xlsx"
    construir_excel(df, ruta)
    print(f"\nResultados guardados en: {ruta}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    resumen = (
        df.groupby(["decay", "distancia"])
        .agg(
            phi_media=("phi", "mean"),
            tasa_coincidencia_phi=("coincide_phi", "mean"),
            tasa_coincidencia_part=("coincide_particion", "mean"),
        )
        .round(6)
    )
    print("\n── Resumen por combinacion decay+distancia ──────────────────")
    print(resumen.to_string())

    tasa_global = df["coincide_phi"].mean()
    print(f"\nTasa de coincidencia global con base ({DECAY_BASE}+{DISTANCIA_BASE}): {tasa_global:.4f}")
    if tasa_global == 1.0:
        print("  KGeoMIP Simetrico es INVARIANTE a decay y distancia en modo exhaustivo.")
    else:
        print("  ADVERTENCIA: se detectaron diferencias. Revisar casos con coincide_phi=False.")
    print("=" * 68)


if __name__ == "__main__":
    main()
