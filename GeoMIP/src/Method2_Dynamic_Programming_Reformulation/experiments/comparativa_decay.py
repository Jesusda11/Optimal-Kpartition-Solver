"""
Paso 2 — Comparativa empirica de variantes de decay function en GeometricSIA.

Ejecuta las 4 variantes de decay (exponencial, polinomial, logaritmico, adaptativo)
sobre un conjunto fijo de casos de prueba (N3-N6) y exporta resultados a Excel.

Como ejecutar (desde el directorio raiz de Method2):
    python experiments/comparativa_decay.py

El Excel de salida se guarda en:
    GeoMIP/results/comparativa_decay.xlsx
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

# ── Parche nulo de SafeLogger (DEBE ocurrir antes de importar geometric/sia) ──
#
# SafeLogger.__setup_logger abre 2 FileHandlers por instancia aunque
# logging.disable este activo. Con 52 corridas eso genera I/O innecesaria.
# Reemplazamos la clase entera con una implementacion nula que no toca disco.
# "from src.middlewares.slogger import SafeLogger" en geometric.py y sia.py
# captura la clase en el momento de la importacion, por eso este parche DEBE
# ejecutarse antes de que esos modulos sean importados por primera vez.
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
from src.funcs.decay import (
    decay_exponencial,
    decay_polinomial,
    decay_logaritmico,
    make_decay_adaptativo,
)

# ── Deshabilitar profiling ────────────────────────────────────────────────────
# ProfilingManager captura aplicacion.profiler_habilitado como argumento por
# defecto al importarse (True). Hay que forzar el flag en la instancia global.
profiler_manager.enabled = False

# ── Monkey-patches de diagnostico ────────────────────────────────────────────
#
# Envuelven sia_preparar_subsistema y find_mip para medir su duracion
# individual y contar cuantas veces se llama calcular_costo. Esto permite
# localizar exactamente el cuello de botella sin modificar el codigo fuente.

_orig_preparar = SIA.sia_preparar_subsistema

def _timed_preparar(self, *args, **kwargs):
    t = time.perf_counter()
    result = _orig_preparar(self, *args, **kwargs)
    dur = time.perf_counter() - t
    dims = getattr(self.sia_subsistema, "dims_ncubos", "?")
    n_cubos = len(getattr(self.sia_subsistema, "ncubos", []))
    print(
        f"      >> preparar={dur:.4f}s  ncubos={n_cubos}  "
        f"dims_presente={dims}",
        flush=True,
    )
    return result

SIA.sia_preparar_subsistema = _timed_preparar


_orig_find_mip = GeometricSIA.find_mip
_costo_calls = [0]

def _timed_find_mip(self):
    _costo_calls[0] = 0
    t = time.perf_counter()
    result = _orig_find_mip(self)
    dur = time.perf_counter() - t
    n_estados = sum(len(v) for v in self.caminos.values())
    n_mem = len(self.memoria_particiones)
    print(
        f"      >> find_mip={dur:.4f}s  estados_bfs={n_estados}  "
        f"costo_calls={_costo_calls[0]}  mem_partic={n_mem}",
        flush=True,
    )
    return result

GeometricSIA.find_mip = _timed_find_mip


_orig_calcular_costo = GeometricSIA.calcular_costo

def _counted_calcular_costo(self, *args, **kwargs):
    _costo_calls[0] += 1
    return _orig_calcular_costo(self, *args, **kwargs)

GeometricSIA.calcular_costo = _counted_calcular_costo

# ── Variantes de decay a comparar ─────────────────────────────────────────────
#
#   Ordenadas de mayor a menor "lentitud" de decrecimiento para d >= 2:
#     adaptativo(0.5) > logaritmico > polinomial > exponencial
#
#   Valores de gamma para distancias d = 1, 2, 3:
#     exponencial:    0.500 | 0.250 | 0.125
#     polinomial:     0.500 | 0.333 | 0.250
#     logaritmico:    0.631 | 0.500 | 0.431
#     adaptativo_a05: 0.707 | 0.577 | 0.500
#
VARIANTES: dict[str, callable] = {
    "exponencial":    decay_exponencial,
    "polinomial":     decay_polinomial,
    "logaritmico":    decay_logaritmico,
    "adaptativo_a05": make_decay_adaptativo(0.5),
}

# Variante usada como referencia para medir coincidencia de particion
VARIANTE_BASE = "exponencial"

# ── Casos de prueba ───────────────────────────────────────────────────────────
#
# Formato: (id_caso, pagina, estado_inicial, condiciones, alcance, mecanismo)
#
#   id_caso:       identificador unico legible
#   pagina:        letra del archivo CSV (ej. "A" -> N3A.csv)
#   estado_inicial: cadena binaria que define el tamano del sistema
#   condiciones:   bits en "1" se mantienen; bits en "0" se condicionan
#   alcance:       bits en "1" se mantienen en el futuro; "0" se marginalizan
#   mecanismo:     bits en "1" se mantienen en el presente; "0" se marginalizan
#
CASOS: list[tuple] = [
    # N3 (3 variables, 8 estados) — muy rapido
    ("N3A_01", "A", "100", "111", "111", "111"),   # sistema completo
    ("N3A_02", "A", "100", "111", "011", "111"),   # una variable futura removida
    ("N3A_03", "A", "100", "111", "111", "011"),   # una variable presente removida
    ("N3B_01", "B", "100", "111", "111", "111"),   # red alternativa N3B

    # N4 (4 variables, 16 estados) — rapido
    ("N4A_01", "A", "1000", "1111", "1111", "1111"),  # sistema completo
    ("N4A_02", "A", "1000", "1111", "0111", "1111"),  # primera variable futura removida
    ("N4A_03", "A", "1000", "1111", "1010", "1111"),  # dos variables futuras alternas
    ("N4B_01", "B", "1000", "1111", "1111", "1111"),  # red alternativa N4B

    # N5 (5 variables, 32 estados) — moderado
    ("N5A_01", "A", "10000", "11111", "11111", "11111"),  # sistema completo
    ("N5A_02", "A", "10000", "11111", "10101", "11111"),  # variables futuras alternas
    ("N5B_01", "B", "10000", "11111", "11111", "11111"),  # red alternativa N5B

    # N6 (6 variables, 64 estados) — mas lento pero manejable
    ("N6A_01", "A", "100000", "111111", "111111", "111111"),  # sistema completo
    ("N6A_02", "A", "100000", "111111", "101010", "111111"),  # variables futuras alternas
]


def _resolver_tpm_path(estado_inicial: str, pagina: str) -> Path:
    """Localiza el archivo CSV del TPM para el sistema dado."""
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
    variante_nombre: str,
    decay_fn,
    tpm: np.ndarray,
) -> dict:
    """
    Ejecuta GeometricSIA con una variante de decay sobre un caso de prueba.

    Retorna un diccionario con todos los datos del resultado, incluyendo
    phi, tiempo de ejecucion, particion encontrada y si hubo error.
    """
    id_caso, pagina, estado_inicial, condiciones, alcance, mecanismo = caso

    t_mgr = time.perf_counter()
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)

    t_geo = time.perf_counter()
    sia = GeometricSIA(gestor, decay_fn=decay_fn)

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

    # Timing breakdown — ayuda a diagnosticar bottlenecks
    print(
        f"      mgr={t_geo-t_mgr:.3f}s  "
        f"init={t0-t_geo:.3f}s  "
        f"algo={t1-t0:.3f}s",
        flush=True,
    )

    return {
        "id_caso":                id_caso,
        "pagina":                 pagina,
        "n_bits":                 len(estado_inicial),
        "estado_inicial":         estado_inicial,
        "condiciones":            condiciones,
        "alcance":                alcance,
        "mecanismo":              mecanismo,
        "variante":               variante_nombre,
        "phi":                    phi,
        "tiempo_s":               round(t1 - t0, 6),
        "particion":              particion,
        "error":                  error,
    }


def agregar_coincidencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la columna 'coincide_con_exponencial' al DataFrame.

    Para cada fila, indica si la particion encontrada es identica
    a la encontrada por la variante base (exponencial) en el mismo caso.
    La propia variante base siempre devuelve True.
    """
    base = (
        df[df["variante"] == VARIANTE_BASE][["id_caso", "particion"]]
        .rename(columns={"particion": "particion_base"})
    )
    df2 = df.merge(base, on="id_caso", how="left")
    df["coincide_con_exponencial"] = df2["particion"] == df2["particion_base"]
    return df


def construir_excel(df: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Construye el archivo Excel con cinco hojas:

    1. Resultados_Crudos       — una fila por (caso, variante)
    2. Phi_por_Variante        — tabla pivote de phi
    3. Tiempo_por_Variante     — tabla pivote de tiempo de ejecucion
    4. Coincidencia            — tabla pivote de coincidencia con exponencial
    5. Resumen_Estadistico     — estadisticas agregadas por variante
    """
    pivot_phi = df.pivot_table(
        values="phi",
        index="id_caso",
        columns="variante",
        aggfunc="first",
    )

    pivot_tiempo = df.pivot_table(
        values="tiempo_s",
        index="id_caso",
        columns="variante",
        aggfunc="first",
    )

    pivot_coincidencia = df.pivot_table(
        values="coincide_con_exponencial",
        index="id_caso",
        columns="variante",
        aggfunc="first",
    )

    resumen = (
        df.groupby("variante")
        .agg(
            phi_media=("phi", "mean"),
            phi_min=("phi", "min"),
            phi_max=("phi", "max"),
            phi_desv=("phi", "std"),
            tiempo_medio_s=("tiempo_s", "mean"),
            tiempo_total_s=("tiempo_s", "sum"),
            tasa_coincidencia=("coincide_con_exponencial", "mean"),
            n_errores=("error", lambda x: x.notna().sum()),
        )
        .round(6)
    )

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Resultados_Crudos", index=False)
        pivot_phi.to_excel(writer, sheet_name="Phi_por_Variante")
        pivot_tiempo.to_excel(writer, sheet_name="Tiempo_por_Variante")
        pivot_coincidencia.to_excel(writer, sheet_name="Coincidencia")
        resumen.to_excel(writer, sheet_name="Resumen_Estadistico")


def main() -> None:
    print("=" * 60)
    print("Paso 2 - Comparativa de variantes de decay")
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
            for variante_nombre in VARIANTES:
                n += 1
                filas.append({
                    "id_caso": id_caso, "pagina": pagina,
                    "n_bits": len(estado_inicial), "estado_inicial": estado_inicial,
                    "condiciones": condiciones, "alcance": alcance,
                    "mecanismo": mecanismo, "variante": variante_nombre,
                    "phi": None, "tiempo_s": None,
                    "particion": None, "error": str(exc),
                })
            continue

        for variante_nombre, decay_fn in VARIANTES.items():
            n += 1
            print(
                f"[{n:>3}/{total}] {id_caso:<12} | {variante_nombre:<16}",
                flush=True,
            )
            fila = ejecutar_caso(caso, variante_nombre, decay_fn, tpm)
            filas.append(fila)

            if fila["error"]:
                print(f"    ERROR: {fila['error']}")
            else:
                print(f"    phi={fila['phi']:.6f}  t={fila['tiempo_s']:.4f}s")

    print()
    df = pd.DataFrame(filas)
    df = agregar_coincidencia(df)

    ruta_salida = GEOMIP_ROOT / "results" / "comparativa_decay.xlsx"
    construir_excel(df, ruta_salida)

    print(f"Resultados guardados en: {ruta_salida}")
    print()

    resumen = (
        df.groupby("variante")
        .agg(
            phi_media=("phi", "mean"),
            tiempo_medio_s=("tiempo_s", "mean"),
            tasa_coincidencia=("coincide_con_exponencial", "mean"),
        )
        .round(6)
    )
    print("── Resumen estadistico por variante ──────────────────────")
    print(resumen.to_string())
    print()


if __name__ == "__main__":
    main()
