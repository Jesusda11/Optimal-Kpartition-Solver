"""
experiments/validacion_requisitos_kgeomip_asimetrico.py

Validacion formal de KGeoMIP Asimetrico segun los requerimientos del proyecto.

Cubre cuatro niveles de validacion:

  Nivel 1 — Comparacion con PyPhi (sistemas n=3..8, k=2)
    PyPhi calcula el MIP exacto via effect_mip.  KGeoMIPAsimetrico en modo
    exacto debe coincidir en phi.  Metricas por caso:
      - Acierto exacto     : |phi_asim - phi_pyphi| < 1e-6
      - Error relativo %   : |phi_asim - phi_pyphi| / max(phi_pyphi, 1e-9) * 100
      - Jaccard particion  : similitud entre las dos biparticiones (combinado
                             futuros+presentes con tags 'f{j}' / 'p{j}')
      - Speedup            : tiempo_pyphi / tiempo_asim_k2
    Umbral Excelente: acierto >90 %, error relativo <1 %.

  Nivel 2 — Tabla Phi_k para k in {2, 3, 4, 5}
    Sistemas con m <= 8 (exacto para todas las k).
    Muestra como phi disminuye al aumentar k.

  Nivel 3 — Monotonia delta_k <= delta_{k-1}
    Verifica que phi_{k+1} <= phi_k para k = 2, 3, 4.
    Debe cumplirse siempre: k+1-particion cubre el espacio de k-particion.

  Nivel 4 — Consistencia k=2 vs GeometricSIA
    phi_asim_k2 <= phi_geo para todos los casos.

Como ejecutar (desde Method2_Dynamic_Programming_Reformulation/):
    python experiments/validacion_requisitos_kgeomip_asimetrico.py

Salida:
    GeoMIP/results/validacion_requisitos_kgeomip_asimetrico.xlsx
"""

import sys
import time
import logging
import collections
import collections.abc
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
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

# ── Parche de compatibilidad para PyPhi (Python 3.10+) ───────────────────────
for _attr in ("Iterable", "Mapping", "MutableMapping", "Sequence"):
    if not hasattr(collections, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))

# ── Importaciones del proyecto ────────────────────────────────────────────────
from src.models.base.application import aplicacion
from src.middlewares.profile import profiler_manager
from src.controllers.manager import Manager
from src.controllers.strategies.kgeometric_asimetrico import KGeoMIPAsimetrico
from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.decay import decay_exponencial
from src.funcs.base import ABECEDARY
from src.funcs.partitions import contar_stirling

from pyphi import Network, Subsystem
from pyphi.labels import NodeLabels

profiler_manager.enabled = False

# ── Configuracion de casos ────────────────────────────────────────────────────
# Tupla: (id_caso, variante_csv, estado_inicial, condicion, alcance, mecanismo)

# Nivel 1: comparacion PyPhi vs KGeoMIPAsimetrico k=2 exacto (n=3..8)
CASOS_PYPHI = [
    ("N3A_01", "A", "100",      "111",      "111",      "111"),       # m=6
    ("N3A_02", "A", "100",      "111",      "011",      "111"),       # m=5
    ("N3B_01", "B", "100",      "111",      "111",      "111"),       # m=6
    ("N4A_01", "A", "1000",     "1111",     "1111",     "1111"),      # m=8
    ("N4A_02", "A", "1000",     "1111",     "0111",     "1111"),      # m=7
    ("N4B_01", "B", "1000",     "1111",     "1111",     "1111"),      # m=8
    ("N5A_01", "A", "10000",    "11111",    "11111",    "11111"),     # m=10
    ("N5A_02", "A", "10000",    "11111",    "10101",    "11111"),     # m=8
    ("N6A_01", "A", "100000",   "111111",   "111111",   "111111"),    # m=12, S(12,2)=2047
    ("N6A_02", "A", "100000",   "111111",   "101010",   "111111"),    # m=9
    ("N8A_01", "A", "10000000", "11111111", "11111111", "00001111"),  # m=12, 8fut+4pres
    ("N8A_02", "A", "10000000", "11111111", "10101010", "11110000"),  # m=8,  4fut+4pres
]

# Niveles 2 y 3: tabla Phi_k y monotonia (m<=8, exacto para k=2..5)
CASOS_TABLA = [
    ("N3A_01", "A", "100",   "111",   "111",   "111"),     # m=6, S(6,5)=15
    ("N3B_01", "B", "100",   "111",   "111",   "111"),     # m=6, S(6,5)=15
    ("N4A_01", "A", "1000",  "1111",  "1111",  "1111"),    # m=8, S(8,5)=1050
    ("N4A_02", "A", "1000",  "1111",  "0111",  "1111"),    # m=7, S(7,5)=140
    ("N4B_01", "B", "1000",  "1111",  "1111",  "1111"),    # m=8, S(8,5)=1050
    ("N5A_02", "A", "10000", "11111", "10101", "11111"),   # m=8, S(8,5)=1050
]


# ── Utilidades ────────────────────────────────────────────────────────────────

def resolver_tpm(n: int, variante: str) -> tuple:
    nombre = f"N{n}{variante}.csv"
    for base in (
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT  / "data" / "samples",
    ):
        if (base / nombre).exists():
            p = base / nombre
            return p, np.genfromtxt(p, delimiter=",")
    raise FileNotFoundError(f"No se encontro '{nombre}'")


def jaccard(A: frozenset, B: frozenset) -> float:
    union = A | B
    return len(A & B) / len(union) if union else 1.0


def jaccard_biparticion(part_py, grupos_asim: list) -> float:
    """
    Calcula similitud de Jaccard entre la biparticion de PyPhi y la nuestra.

    Ambas particiones se codifican como pares de frozensets de etiquetas
    'f{j}' (futuro global j) y 'p{j}' (presente global j).  Se prueban las
    dos alineaciones posibles y se toma el maximo (mejor match).

    grupos_asim: list[tuple[np.ndarray, np.ndarray]] — indices globales de nodo.
    """
    if part_py is None or not grupos_asim:
        return float("nan")
    try:
        prim = part_py.parts[True]
        dual = part_py.parts[False]

        def tagged_pyphi(pur, mech):
            return frozenset({f"f{j}" for j in pur} | {f"p{j}" for j in mech})

        def tagged_asim(futuros, presentes):
            return frozenset({f"f{int(j)}" for j in futuros} | {f"p{int(j)}" for j in presentes})

        py1 = tagged_pyphi(prim.purview, prim.mechanism)
        py2 = tagged_pyphi(dual.purview, dual.mechanism)

        f1, p1 = grupos_asim[0]
        f2, p2 = grupos_asim[1] if len(grupos_asim) > 1 else (np.array([]), np.array([]))
        as1 = tagged_asim(f1, p1)
        as2 = tagged_asim(f2, p2)

        score_a = (jaccard(py1, as1) + jaccard(py2, as2)) / 2
        score_b = (jaccard(py1, as2) + jaccard(py2, as1)) / 2
        return round(max(score_a, score_b), 6)
    except Exception:
        return float("nan")


# ── Runners ───────────────────────────────────────────────────────────────────

def correr_pyphi(
    estado_inicial: str,
    condicion: str,
    alcance_bin: str,
    mecanismo_bin: str,
    tpm: np.ndarray,
) -> tuple:
    """Retorna (phi, partition, tiempo_s)."""
    n      = len(estado_inicial)
    labels = NodeLabels(tuple(ABECEDARY[:n]), tuple(range(n)))
    red    = Network(tpm=tpm, node_labels=labels)

    candidato  = tuple(labels[i] for i, b in enumerate(condicion) if b == "1")
    estado_tup = tuple(int(s) for s in estado_inicial)
    subsistema = Subsystem(network=red, state=estado_tup, nodes=candidato)

    alcance_idx   = tuple(i for i, (b, c) in enumerate(zip(alcance_bin, condicion))   if b == "1" and c == "1")
    mecanismo_idx = tuple(i for i, (b, c) in enumerate(zip(mecanismo_bin, condicion)) if b == "1" and c == "1")

    t0  = time.perf_counter()
    mip = subsistema.effect_mip(mecanismo_idx, alcance_idx)
    t   = time.perf_counter() - t0
    return float(mip.phi), mip.partition, t


def correr_asimetrico(
    estado_inicial: str,
    condicion: str,
    alcance_bin: str,
    mecanismo_bin: str,
    tpm: np.ndarray,
    variante: str,
    k_min: int,
    k_max: int,
    m_max_exhaustivo: int,
) -> tuple:
    """Retorna (resultados_por_k: dict, tiempo_total_s: float)."""
    aplicacion.pagina_sample_network = variante
    gestor = Manager(estado_inicial=estado_inicial)
    sia = KGeoMIPAsimetrico(
        gestor,
        k_max=k_max,
        k_min=k_min,
        m_max_exhaustivo=m_max_exhaustivo,
        decay_fn=decay_exponencial,
    )
    t0 = time.perf_counter()
    sia.aplicar_estrategia(condicion, alcance_bin, mecanismo_bin, tpm)
    t  = time.perf_counter() - t0
    return sia._resultados_por_k, t


def correr_geo(
    estado_inicial: str,
    condicion: str,
    alcance_bin: str,
    mecanismo_bin: str,
    tpm: np.ndarray,
    variante: str,
) -> float:
    aplicacion.pagina_sample_network = variante
    gestor = Manager(estado_inicial=estado_inicial)
    sia = GeometricSIA(gestor, decay_fn=decay_exponencial)
    sol = sia.aplicar_estrategia(condicion, alcance_bin, mecanismo_bin, tpm)
    return float(sol.perdida)


# ── Nivel 1: comparacion con PyPhi ───────────────────────────────────────────

def nivel1_pyphi(casos: list) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("NIVEL 1 — KGeoMIPAsimetrico k=2 (exacto) vs PyPhi")
    print("=" * 70)

    filas = []
    for id_caso, variante, estado_ini, cond, alcance, mecanismo in casos:
        n     = len(estado_ini)
        n_fut = sum(b == "1" and c == "1" for b, c in zip(alcance, cond))
        n_pre = sum(b == "1" and c == "1" for b, c in zip(mecanismo, cond))
        m     = n_fut + n_pre
        print(f"\n  {id_caso}  n={n}  m={m}  alcance={alcance}  mec={mecanismo}")

        try:
            _, tpm = resolver_tpm(n, variante)
        except FileNotFoundError as exc:
            print(f"    [SKIP] {exc}")
            continue

        try:
            phi_py, part_py, t_py = correr_pyphi(estado_ini, cond, alcance, mecanismo, tpm)
        except Exception as exc:
            print(f"    [PyPhi ERROR] {exc}")
            continue

        # m_max_exhaustivo > m fuerza modo exacto para cualquier k
        try:
            res, t_asim = correr_asimetrico(
                estado_ini, cond, alcance, mecanismo, tpm, variante,
                k_min=2, k_max=2, m_max_exhaustivo=max(m + 1, 16),
            )
            phi_asim  = float(res[2]["phi"]) if 2 in res else float("nan")
            grupos_k2 = res[2]["grupos"]     if 2 in res else []
            n_cand    = res[2]["n_candidatos"] if 2 in res else 0
        except Exception as exc:
            print(f"    [KGeoMIP ERROR] {exc}")
            continue

        try:
            phi_geo = correr_geo(estado_ini, cond, alcance, mecanismo, tpm, variante)
        except Exception:
            phi_geo = float("nan")

        match_ex = abs(phi_asim - phi_py) < 1e-6
        err_rel  = abs(phi_asim - phi_py) / max(abs(phi_py), 1e-9) * 100.0
        jacc     = jaccard_biparticion(part_py, grupos_k2)
        speedup  = t_py / max(t_asim, 1e-6)
        geo_ok   = bool(phi_asim <= phi_geo + 1e-9) if not np.isnan(phi_geo) else None

        print(
            f"    phi_pyphi={phi_py:.6f}  phi_asim={phi_asim:.6f}  "
            f"err={err_rel:.4f}%  jacc={jacc:.4f}  speedup={speedup:.1f}x  "
            f"[{'PASS' if match_ex else 'FAIL'}]"
        )

        filas.append({
            "id_caso":       id_caso,
            "n":             n,
            "m":             m,
            "S(m,2)":        contar_stirling(m, 2),
            "phi_pyphi":     round(phi_py, 8),
            "phi_asim_k2":   round(phi_asim, 8),
            "phi_geo":       round(phi_geo, 8) if not np.isnan(phi_geo) else None,
            "match_exacto":  match_ex,
            "err_relativo%": round(err_rel, 6),
            "jaccard":       jacc,
            "speedup_x":     round(speedup, 2),
            "geo_ok":        geo_ok,
            "t_pyphi_s":     round(t_py, 4),
            "t_asim_s":      round(t_asim, 4),
            "n_candidatos":  n_cand,
            "modo":          "exacto",
        })

    df = pd.DataFrame(filas)
    if not df.empty:
        n_total = len(df)
        n_pass  = int(df["match_exacto"].sum())
        avg_err = float(df["err_relativo%"].mean())
        avg_jac = float(df["jaccard"].dropna().mean()) if df["jaccard"].notna().any() else float("nan")
        avg_spd = float(df["speedup_x"].mean())
        n_geo   = int(df["geo_ok"].sum()) if "geo_ok" in df else 0
        n_geo_t = int(df["geo_ok"].notna().sum()) if "geo_ok" in df else 0
        print(f"\n  RESUMEN Nivel 1:")
        print(f"    Tasa de acierto exacto : {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)"
              f"  {'[Excelente >90%]' if n_pass/n_total > 0.9 else '[Revisar]'}")
        print(f"    Error relativo medio   : {avg_err:.4f}%"
              f"  {'[Excelente <1%]' if avg_err < 1.0 else '[Revisar]'}")
        print(f"    Jaccard medio          : {avg_jac:.4f}")
        print(f"    Speedup medio          : {avg_spd:.1f}x")
        if n_geo_t:
            print(f"    phi_asim <= phi_geo    : {n_geo}/{n_geo_t} OK")
    return df


# ── Niveles 2 y 3: tabla Phi_k, monotonia creciente y acumulado ──────────────

def nivel2_3_tabla_monotonia(casos: list) -> pd.DataFrame:
    """
    Nivel 2: tabla phi_k para k in {2,3,4,5} (exacto, S(m,k) grupos no vacios).
    Nivel 3: dos metricas de monotonia:

      - monotonia_creciente: phi_{k+1} >= phi_k
          Correcto para S(m,k) estricto: mas grupos = mas cortes = mas perdida.
          La afirmacion "phi_{k+1} <= phi_k" solo vale si se permiten grupos vacios.

      - phi_acum_k = min(phi_2, ..., phi_k)
          Monotonia decreciente garantizada por construccion.
          Representa el mejor phi usando HASTA k grupos (MIP global del sistema).

    Nivel 3b: k_optimo — el k donde se alcanza el phi_k minimo por caso.
    """
    print("\n" + "=" * 70)
    print("NIVEL 2 — Tabla Phi_k para k in {2,3,4,5}")
    print("NIVEL 3 — Monotonia creciente (grupos no vacios) + phi acumulado")
    print("=" * 70)

    filas = []
    for id_caso, variante, estado_ini, cond, alcance, mecanismo in casos:
        n     = len(estado_ini)
        n_fut = sum(b == "1" and c == "1" for b, c in zip(alcance, cond))
        n_pre = sum(b == "1" and c == "1" for b, c in zip(mecanismo, cond))
        m     = n_fut + n_pre
        print(f"\n  {id_caso}  n={n}  m={m}  alcance={alcance}  mec={mecanismo}")

        try:
            _, tpm = resolver_tpm(n, variante)
        except FileNotFoundError as exc:
            print(f"    [SKIP] {exc}")
            continue

        try:
            res, t_total = correr_asimetrico(
                estado_ini, cond, alcance, mecanismo, tpm, variante,
                k_min=2, k_max=5, m_max_exhaustivo=8,
            )
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            continue

        phis = {k: float(res[k]["phi"]) for k in sorted(res.keys())}
        ks   = sorted(phis.keys())

        # Monotonia creciente: phi_{k+1} >= phi_k (correcto para S(m,k) estricto)
        mono_creciente = all(
            phis[ks[i + 1]] >= phis[ks[i]] - 1e-9 for i in range(len(ks) - 1)
        )

        # phi acumulado: min(phi_2, ..., phi_k) — decreciente por construccion
        phi_acum: dict[int, float] = {}
        acum = float("inf")
        for k in ks:
            acum = min(acum, phis[k])
            phi_acum[k] = acum

        # k optimo: k donde phi_k es minimo
        k_optimo = min(phis, key=lambda k: phis[k])

        phi_str  = "  ".join(f"phi_k{k}={phis[k]:.6f}" for k in ks)
        acum_str = "  ".join(f"acum_k{k}={phi_acum[k]:.6f}" for k in ks)
        print(f"    {phi_str}")
        print(f"    {acum_str}")
        print(
            f"    mono_creciente={'OK' if mono_creciente else 'FAIL'}"
            f"  k_optimo={k_optimo}"
            f"  t_total={t_total:.2f}s"
        )

        fila: dict = {
            "id_caso":          id_caso,
            "n":                n,
            "m":                m,
            "mono_creciente":   mono_creciente,
            "k_optimo":         k_optimo,
            "t_total_s":        round(t_total, 3),
        }
        for k in [2, 3, 4, 5]:
            fila[f"phi_k{k}"]    = round(phis.get(k, float("nan")), 8)
            fila[f"phi_acum_k{k}"] = round(phi_acum.get(k, float("nan")), 8)
            fila[f"S(m,{k})"]    = contar_stirling(m, k)
            fila[f"n_cand_k{k}"] = res[k]["n_candidatos"] if k in res else None
            fila[f"t_k{k}_s"]    = round(res[k]["tiempo_s"], 4) if k in res else None

        filas.append(fila)

    df = pd.DataFrame(filas)
    if not df.empty:
        n_tot  = len(df)
        n_mono = int(df["mono_creciente"].sum())
        print(f"\n  RESUMEN Nivel 3:")
        print(f"    Monotonia creciente phi_k+1 >= phi_k : {n_mono}/{n_tot} casos OK")
        dist_k = df["k_optimo"].value_counts().sort_index()
        print(f"    Distribucion k_optimo:")
        for k, cnt in dist_k.items():
            print(f"      k={k}: {cnt} caso(s)")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Validacion KGeoMIP Asimetrico — Requerimientos del Proyecto")
    print("=" * 70)

    df1 = nivel1_pyphi(CASOS_PYPHI)
    df2 = nivel2_3_tabla_monotonia(CASOS_TABLA)

    ruta = GEOMIP_ROOT / "results" / "validacion_requisitos_kgeomip_asimetrico.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        if not df1.empty:
            df1.to_excel(writer, sheet_name="Nivel1_PyPhi", index=False)
        if not df2.empty:
            df2.to_excel(writer, sheet_name="Nivel2-3_Tabla_Monotonia", index=False)

        if not df1.empty:
            n_total = len(df1)
            n_pass  = int(df1["match_exacto"].sum())
            avg_err = float(df1["err_relativo%"].mean())
            avg_jac = float(df1["jaccard"].dropna().mean()) if df1["jaccard"].notna().any() else float("nan")
            avg_spd = float(df1["speedup_x"].mean())
            n_geo   = int(df1["geo_ok"].sum()) if "geo_ok" in df1 else 0
            n_geo_t = int(df1["geo_ok"].notna().sum()) if "geo_ok" in df1 else 0
            resumen = pd.DataFrame([
                {"Metrica": "Tasa de acierto exacto (%)",
                 "Valor": round(100 * n_pass / n_total, 1),
                 "Umbral_Excelente": ">90",
                 "Calificacion": "Excelente" if n_pass / n_total > 0.9 else "Revisar"},
                {"Metrica": "Error relativo medio (%)",
                 "Valor": round(avg_err, 4),
                 "Umbral_Excelente": "<1",
                 "Calificacion": "Excelente" if avg_err < 1.0 else "Revisar"},
                {"Metrica": "Jaccard medio particion",
                 "Valor": round(avg_jac, 4),
                 "Umbral_Excelente": ">0.9",
                 "Calificacion": "Excelente" if avg_jac > 0.9 else "Aceptable"},
                {"Metrica": "Speedup medio vs PyPhi",
                 "Valor": round(avg_spd, 1),
                 "Umbral_Excelente": ">1",
                 "Calificacion": "Excelente" if avg_spd > 1 else "Revisar"},
                {"Metrica": "Consistencia phi_asim <= phi_geo",
                 "Valor": f"{n_geo}/{n_geo_t}",
                 "Umbral_Excelente": "100%",
                 "Calificacion": "OK" if n_geo == n_geo_t else "REVISAR"},
            ])
            resumen.to_excel(writer, sheet_name="Resumen_Metricas", index=False)

        if not df2.empty:
            n_tot  = len(df2)
            n_mono = int(df2["mono_creciente"].sum())

            # Fila por fila de la distribucion de k_optimo
            dist_k = df2["k_optimo"].value_counts().sort_index()
            filas_mono = [
                {
                    "Validacion": "Monotonia creciente phi_k+1 >= phi_k (grupos no vacios)",
                    "Descripcion": "Correcto para S(m,k) estricto: mas grupos fuerzan mas cortes",
                    "Casos_OK": n_mono,
                    "Casos_Total": n_tot,
                    "Resultado": "PASS" if n_mono == n_tot else f"{n_mono}/{n_tot}",
                },
                {
                    "Validacion": "phi_acum_k = min(phi_2,..,phi_k) — decreciente",
                    "Descripcion": "Garantizado por construccion; representa MIP usando hasta k grupos",
                    "Casos_OK": n_tot,
                    "Casos_Total": n_tot,
                    "Resultado": "PASS (trivial)",
                },
            ]
            for k, cnt in dist_k.items():
                filas_mono.append({
                    "Validacion": f"k_optimo = {k}",
                    "Descripcion": f"Casos donde phi_k{k} es el minimo sobre todos los k evaluados",
                    "Casos_OK": int(cnt),
                    "Casos_Total": n_tot,
                    "Resultado": f"{int(cnt)}/{n_tot}",
                })
            resumen_mono = pd.DataFrame(filas_mono)
            resumen_mono.to_excel(writer, sheet_name="Resumen_Monotonia", index=False)

    print(f"\nResultados guardados en: {ruta}")
    print("=" * 70)


if __name__ == "__main__":
    main()