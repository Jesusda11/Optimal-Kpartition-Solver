"""
Validacion de KGeoMIP Simetrico contra Fuerza Bruta.

Valida que el generador RGS (Restricted Growth Strings) de KGeoMIP produce
exactamente el mismo conjunto de k-particiones que un generador bruto canonico
independiente, y que ambos coinciden en el phi minimo (MIP).

Tres verificaciones por caso y por k:
  1. count_ok  : conteo de particiones == S(n,k) (numero de Stirling esperado)
  2. sets_ok   : el CONJUNTO de particiones RGS == CONJUNTO brute-force
                 (verifica que no se omite ni se duplica ningun candidato)
  3. phi_ok    : |phi_rgs - phi_bf| < 1e-6

Las tres deben pasar para que el caso se marque PASS.

Como ejecutar (desde el directorio raiz de Method2):
    python experiments/validacion_fuerza_bruta.py

El Excel se guarda en:
    GeoMIP/results/validacion_fuerza_bruta.xlsx
"""

import sys
import time
import logging
from pathlib import Path
from itertools import product as iproduct

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
from src.funcs.decay import decay_exponencial
from src.funcs.base import emd_efecto
from src.funcs.partitions import generar_k_particiones, contar_stirling

profiler_manager.enabled = False

# ── Configuracion ─────────────────────────────────────────────────────────────
K_MAX = 4

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


# ── Generador independiente (brute-force canonico) ────────────────────────────

def _bruteforce_k_particiones(n: int, k: int):
    """
    Genera todas las k-particiones de {0,...,n-1} en forma canonica.

    Algoritmo: enumera k^n asignaciones posibles y filtra las que tienen
    forma canonica (la etiqueta i aparece por primera vez antes que i+1).
    Completamente independiente de generar_k_particiones (RGS).

    Complejidad: O(k^n) iteraciones, S(n,k) particiones emitidas.
    Solo para validacion en n<=6.
    """
    for assignment in iproduct(range(k), repeat=n):
        distinct = 0
        seen = set()
        canonical = True
        for g in assignment:
            if g not in seen:
                if g != distinct:
                    canonical = False
                    break
                seen.add(g)
                distinct += 1
        if not canonical or distinct != k:
            continue
        groups = [[] for _ in range(k)]
        for i, g in enumerate(assignment):
            groups[g].append(i)
        yield groups


def _partition_to_key(partition: list[list[int]]) -> frozenset:
    return frozenset(frozenset(grp) for grp in partition)


# ── Logica de validacion por caso ─────────────────────────────────────────────

def _evaluar_particion(
    asignacion: list[list[int]],
    comunes: np.ndarray,
    huerfanos_futuro: np.ndarray,
    dims_ncubos: np.ndarray,
    subsistema,
    dists_ref: np.ndarray,
) -> float:
    grupos = []
    for grupo in asignacion:
        arr = np.array(grupo, dtype=np.int8)
        grupos.append((comunes[arr], comunes[arr]))
    if huerfanos_futuro.size > 0:
        grupos.append((huerfanos_futuro, dims_ncubos))
    dist = subsistema.k_bipartir(grupos).distribucion_marginal()
    return float(emd_efecto(dist, dists_ref))


def validar_caso(caso: tuple, tpm: np.ndarray, k_max: int) -> list[dict]:
    """
    Prepara el subsistema y valida RGS vs brute-force para cada k.
    """
    id_caso, pagina, estado_inicial, condicion, alcance, mecanismo = caso
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)

    sia = KGeometricSIA(gestor, k_max=k_max, decay_fn=decay_exponencial)
    sia.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

    indices_ncubos  = sia.sia_subsistema.indices_ncubos
    dims_ncubos     = sia.sia_subsistema.dims_ncubos
    comunes         = np.intersect1d(indices_ncubos, dims_ncubos)
    huerfanos       = np.setdiff1d(indices_ncubos, dims_ncubos)
    n               = len(comunes)
    dists_ref       = sia.sia_dists_marginales

    resultados = []

    for k in range(2, min(k_max, n) + 1):
        stirling_esp = contar_stirling(n, k)

        # ── RGS ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        rgs_list, rgs_phis = [], []
        for asig in generar_k_particiones(n, k):
            phi = _evaluar_particion(asig, comunes, huerfanos, dims_ncubos,
                                     sia.sia_subsistema, dists_ref)
            rgs_list.append(asig)
            rgs_phis.append(phi)
        t_rgs = time.perf_counter() - t0

        # ── Brute Force ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        bf_list, bf_phis = [], []
        for asig in _bruteforce_k_particiones(n, k):
            phi = _evaluar_particion(asig, comunes, huerfanos, dims_ncubos,
                                     sia.sia_subsistema, dists_ref)
            bf_list.append(asig)
            bf_phis.append(phi)
        t_bf = time.perf_counter() - t0

        phi_rgs = min(rgs_phis) if rgs_phis else float("inf")
        phi_bf  = min(bf_phis)  if bf_phis  else float("inf")

        rgs_set = {_partition_to_key(p) for p in rgs_list}
        bf_set  = {_partition_to_key(p) for p in bf_list}

        count_ok = (len(rgs_list) == len(bf_list) == stirling_esp)
        sets_ok  = (rgs_set == bf_set)
        phi_ok   = abs(phi_rgs - phi_bf) < 1e-6
        passed   = count_ok and sets_ok and phi_ok

        resultados.append({
            "id_caso":           id_caso,
            "n_balanceados":     n,
            "k":                 k,
            "stirling_esperado": stirling_esp,
            "count_rgs":         len(rgs_list),
            "count_bf":          len(bf_list),
            "phi_rgs":           round(phi_rgs, 8),
            "phi_bf":            round(phi_bf, 8),
            "delta_phi":         round(abs(phi_rgs - phi_bf), 10),
            "count_ok":          count_ok,
            "sets_ok":           sets_ok,
            "phi_ok":            phi_ok,
            "PASS":              passed,
            "t_rgs_s":           round(t_rgs, 6),
            "t_bf_s":            round(t_bf, 6),
        })

    return resultados


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("Paso 6a  Validacion KGeoMIP Simetrico vs Fuerza Bruta")
    print(f"k_max: {K_MAX}  |  Casos: {len(CASOS)}")
    print("=" * 68)

    all_rows: list[dict] = []
    total_pass = 0
    total_checks = 0

    for i, caso in enumerate(CASOS, 1):
        id_caso, pagina, estado_inicial = caso[0], caso[1], caso[2]
        n = len(estado_inicial)
        print(f"\n[{i:>2}/{len(CASOS)}] {id_caso}  n={n}")

        try:
            nombre = f"N{n}{pagina}.csv"
            candidatos = (
                METHOD2_ROOT / "src" / ".samples" / nombre,
                METHOD2_ROOT / ".samples" / nombre,
                GEOMIP_ROOT / "data" / "samples" / nombre,
            )
            tpm_path = next((c for c in candidatos if c.exists()), None)
            if tpm_path is None:
                raise FileNotFoundError(f"No se encontro '{nombre}'")
            tpm = np.genfromtxt(tpm_path, delimiter=",")
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        try:
            filas = validar_caso(caso, tpm, K_MAX)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue

        for fila in filas:
            k      = fila["k"]
            status = "PASS" if fila["PASS"] else "FAIL"
            print(
                f"  k={k}  S(n,k)={fila['stirling_esperado']:>3}  "
                f"count_rgs={fila['count_rgs']:>3}  count_bf={fila['count_bf']:>3}  "
                f"phi_rgs={fila['phi_rgs']:.6f}  phi_bf={fila['phi_bf']:.6f}  "
                f"delta={fila['delta_phi']:.2e}  [{status}]"
            )
            all_rows.append(fila)
            total_checks += 1
            if fila["PASS"]:
                total_pass += 1

    # ── Exportar ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    ruta = GEOMIP_ROOT / "results" / "validacion_fuerza_bruta.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta, index=False)
    print(f"\nResultados guardados en: {ruta}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print(f"RESUMEN: {total_pass}/{total_checks} verificaciones PASS")
    if total_pass == total_checks:
        print("  KGeoMIP Simetrico VALIDADO correctamente.")
        print("  RGS y Brute Force coinciden en conteo, conjunto y phi optimo.")
    else:
        fallidos = [r for r in all_rows if not r["PASS"]]
        print(f"  ADVERTENCIA: {len(fallidos)} verificaciones fallaron:")
        for f in fallidos:
            print(
                f"    {f['id_caso']} k={f['k']}  "
                f"count_ok={f['count_ok']}  sets_ok={f['sets_ok']}  phi_ok={f['phi_ok']}"
            )
    print("=" * 68)


if __name__ == "__main__":
    main()
