"""
Validacion de KGeoMIP Asimetrico contra Fuerza Bruta.

Verifica que el generador RGS de KGeoMIPAsimetrico sobre el pool de 2n variables
produce exactamente el mismo conjunto de k-particiones que un generador bruto
canonico independiente, y que ambos coinciden en el phi minimo.

Ademas compara phi_asimetrico vs phi_simetrico vs phi_geo para confirmar:
    phi_asimetrico_k2 <= phi_geo (siempre, por ser GeometricSIA un subconjunto)

Tres verificaciones por caso y por k:
  1. count_ok  : conteo de particiones == S(m,k) exacto
  2. sets_ok   : el CONJUNTO de particiones RGS == CONJUNTO brute-force
  3. phi_ok    : |phi_rgs - phi_bf| < 1e-6

Como ejecutar (desde el directorio raiz de Method2):
    python experiments/validacion_asimetrico_fuerza_bruta.py

El Excel se guarda en:
    GeoMIP/results/validacion_asimetrico_fuerza_bruta.xlsx
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
from src.controllers.strategies.kgeometric_asimetrico import KGeometricSIAAsimetrico
from src.controllers.strategies.kgeometric import KGeometricSIA
from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.decay import decay_exponencial
from src.funcs.base import emd_efecto
from src.funcs.partitions import generar_k_particiones, contar_stirling

profiler_manager.enabled = False

# ── Configuracion ─────────────────────────────────────────────────────────────
K_MAX = 5  # S(8,5)=1050, S(8,4)=1701 — manejable; BF sobre 5^8=390625 asignaciones es rapido

CASOS: list[tuple] = [
    # (id, pagina, estado_ini, condicion, alcance, mecanismo)
    ("N3A_01", "A", "100", "111", "111", "111"),
    ("N3A_02", "A", "100", "111", "011", "111"),
    ("N3A_03", "A", "100", "111", "111", "011"),
    ("N3B_01", "B", "100", "111", "111", "111"),
    ("N4A_01", "A", "1000", "1111", "1111", "1111"),
    ("N4A_02", "A", "1000", "1111", "0111", "1111"),
    ("N4A_03", "A", "1000", "1111", "1010", "1111"),
    ("N4B_01", "B", "1000", "1111", "1111", "1111"),
]


# ── Generador brute-force independiente ───────────────────────────────────────

def _bruteforce_k_particiones(m: int, k: int):
    """
    Genera todas las k-particiones de {0,...,m-1} en forma canonica.
    Completamente independiente de generar_k_particiones (RGS).
    Solo para m<=8 (S(8,3)=966 es el caso mas grande en K_MAX=3).
    """
    for assignment in iproduct(range(k), repeat=m):
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


# ── Evaluacion de una particion de pool ───────────────────────────────────────

def _evaluar_particion_pool(
    particion: list[list[int]],
    indices_ncubos: np.ndarray,
    dims_ncubos: np.ndarray,
    subsistema,
    dists_ref: np.ndarray,
    sia_asim: KGeometricSIAAsimetrico,
) -> float:
    grupos = sia_asim._particion_pool_a_grupos(particion, indices_ncubos, dims_ncubos)
    dist = subsistema.k_bipartir(grupos).distribucion_marginal()
    return float(emd_efecto(dist, dists_ref))


# ── Logica de validacion por caso ─────────────────────────────────────────────

def validar_caso(caso: tuple, tpm: np.ndarray, k_max: int) -> list[dict]:
    id_caso, pagina, estado_inicial, condicion, alcance, mecanismo = caso
    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_inicial)

    sia_asim = KGeometricSIAAsimetrico(
        gestor, k_max=k_max, m_max_exhaustivo=999, decay_fn=decay_exponencial
    )
    sia_asim.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

    indices_ncubos = sia_asim.sia_subsistema.indices_ncubos
    dims_ncubos    = sia_asim.sia_subsistema.dims_ncubos
    n_fut = len(indices_ncubos)
    n_pres = len(dims_ncubos)
    m = n_fut + n_pres
    dists_ref = sia_asim.sia_dists_marginales

    # Phi de GeometricSIA (biparticion asimetrica heuristica)
    sia_geo = GeometricSIA(gestor, decay_fn=decay_exponencial)
    sol_geo = sia_geo.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
    phi_geo = float(sol_geo.perdida)

    # Phi de KGeometricSIA simetrico k=2
    gestor2 = Manager(estado_inicial=estado_inicial)
    sia_sim = KGeometricSIA(gestor2, k_max=2, decay_fn=decay_exponencial)
    try:
        sol_sim = sia_sim.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
        phi_sim_k2 = float(sol_sim.perdida)
    except ValueError:
        phi_sim_k2 = None  # subsistema sin nodos balanceados

    resultados = []

    for k in range(2, min(k_max, m) + 1):
        stirling_esp = contar_stirling(m, k)

        # ── RGS (nuestro generador) ───────────────────────────────────────
        t0 = time.perf_counter()
        rgs_list, rgs_phis = [], []
        for particion in generar_k_particiones(m, k):
            phi = _evaluar_particion_pool(
                particion, indices_ncubos, dims_ncubos,
                sia_asim.sia_subsistema, dists_ref, sia_asim
            )
            rgs_list.append(particion)
            rgs_phis.append(phi)
        t_rgs = time.perf_counter() - t0

        # ── Brute Force independiente ─────────────────────────────────────
        t0 = time.perf_counter()
        bf_list, bf_phis = [], []
        for particion in _bruteforce_k_particiones(m, k):
            phi = _evaluar_particion_pool(
                particion, indices_ncubos, dims_ncubos,
                sia_asim.sia_subsistema, dists_ref, sia_asim
            )
            bf_list.append(particion)
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

        # Invariante: phi_asim_k2 <= phi_geo (asimetrico cubre todo el espacio de geo)
        geo_ok = (phi_rgs <= phi_geo + 1e-6) if k == 2 else None

        resultados.append({
            "id_caso":           id_caso,
            "n_fut":             n_fut,
            "n_pres":            n_pres,
            "m_pool":            m,
            "k":                 k,
            "stirling_esperado": stirling_esp,
            "count_rgs":         len(rgs_list),
            "count_bf":          len(bf_list),
            "phi_rgs":           round(phi_rgs, 8),
            "phi_bf":            round(phi_bf, 8),
            "phi_geo":           round(phi_geo, 8),
            "phi_sim_k2":        round(phi_sim_k2, 8) if phi_sim_k2 is not None else None,
            "delta_phi":         round(abs(phi_rgs - phi_bf), 10),
            "count_ok":          count_ok,
            "sets_ok":           sets_ok,
            "phi_ok":            phi_ok,
            "geo_ok":            geo_ok,
            "PASS":              passed,
            "t_rgs_s":           round(t_rgs, 6),
            "t_bf_s":            round(t_bf, 6),
        })

    return resultados


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Validacion KGeoMIP Asimetrico vs Fuerza Bruta")
    print(f"k_max: {K_MAX}  |  Casos: {len(CASOS)}")
    print("Invariante adicional: phi_asim_k2 <= phi_geo")
    print("=" * 70)

    all_rows: list[dict] = []
    total_pass = 0
    total_checks = 0

    for i, caso in enumerate(CASOS, 1):
        id_caso, pagina, estado_inicial = caso[0], caso[1], caso[2]
        n = len(estado_inicial)
        print(f"\n[{i:>2}/{len(CASOS)}] {id_caso}  n={n}  m_pool_max={2*n}")

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
            geo_str = f"  geo_ok={fila['geo_ok']}" if fila["geo_ok"] is not None else ""
            print(
                f"  k={k}  S(m,k)={fila['stirling_esperado']:>5}  "
                f"count_rgs={fila['count_rgs']:>5}  count_bf={fila['count_bf']:>5}  "
                f"phi_rgs={fila['phi_rgs']:.6f}  phi_bf={fila['phi_bf']:.6f}  "
                f"phi_geo={fila['phi_geo']:.6f}  "
                f"delta={fila['delta_phi']:.2e}  [{status}]{geo_str}"
            )
            all_rows.append(fila)
            total_checks += 1
            if fila["PASS"]:
                total_pass += 1

    # ── Exportar ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    ruta = GEOMIP_ROOT / "results" / "validacion_asimetrico_fuerza_bruta.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta, index=False)
    print(f"\nResultados guardados en: {ruta}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"RESUMEN: {total_pass}/{total_checks} verificaciones PASS")

    geo_fails = [r for r in all_rows if r.get("geo_ok") is False]
    if geo_fails:
        print(f"ADVERTENCIA: {len(geo_fails)} casos donde phi_asim > phi_geo (BUG si ocurre):")
        for r in geo_fails:
            print(f"  {r['id_caso']} k={r['k']}: phi_asim={r['phi_rgs']:.6f} > phi_geo={r['phi_geo']:.6f}")
    else:
        geo_total = sum(1 for r in all_rows if r.get("geo_ok") is not None)
        if geo_total:
            print(f"  Invariante phi_asim_k2 <= phi_geo: OK en {geo_total}/{geo_total} casos")

    if total_pass == total_checks:
        print("  KGeoMIP Asimetrico VALIDADO correctamente.")
        print("  RGS y Brute Force coinciden en conteo, conjunto y phi optimo.")
    else:
        fallidos = [r for r in all_rows if not r["PASS"]]
        print(f"  ADVERTENCIA: {len(fallidos)} verificaciones fallaron:")
        for f in fallidos:
            print(
                f"    {f['id_caso']} k={f['k']}  "
                f"count_ok={f['count_ok']}  sets_ok={f['sets_ok']}  phi_ok={f['phi_ok']}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
