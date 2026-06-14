"""
validacion_fase2.py — Validacion de la Fase 2 de KQNodes (k > 2) con DESGLOSE del gap.

KQNodes(k>2) construye la k-particion por biparticiones sucesivas (greedy) usando
el motor de Queyranne como oraculo de 2-particion. Es una HEURISTICA. El error
total respecto al optimo tiene DOS fuentes que aqui se separan explicitamente:

  1. phi_KQNodes        : greedy jerarquico sobre el oraculo QNodes (Queyranne).
  2. phi_greedy_ideal   : el MISMO greedy jerarquico pero con un oraculo de
                          biparticion EXACTO (fuerza bruta de la biparticion en
                          cada paso, en lugar de Queyranne). Mide lo mejor que el
                          greedy podria lograr con un oraculo perfecto.
  3. phi_optimo_exacto  : fuerza bruta sobre TODAS las k-particiones
                          (generar_k_particiones). Optimo global real.

Desglose del gap:
  gap_total   = phi_KQNodes      - phi_optimo_exacto
  gap_greedy  = phi_greedy_ideal - phi_optimo_exacto   (culpa de la estructura greedy)
  gap_oraculo = phi_KQNodes      - phi_greedy_ideal     (culpa del oraculo Queyranne)
  => gap_total = gap_greedy + gap_oraculo   (por definicion)

Tambien valida:
  - Invariante: phi_optimo_exacto <= phi_greedy_ideal <= ... y phi_KQNodes >= phi_optimo.
  - Determinismo: dos corridas de KQNodes dan phi identico por k (delta 0).
  - Registro por k con metadatos (modo_usado, n_candidatos) para k in {2,3,4,5}.

Ejecutar desde KQNodes/:
    <python> validacion_fase2.py
Salida: tabla por consola + Excel en KQNodes/results/validacion_fase2_kqnodes.xlsx
"""

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.strategies.kqnodes import KQNodes
from src.funcs.iit import emd_efecto
from src.funcs.partitions import generar_k_particiones, contar_stirling
from src.constants.base import ACTUAL, EFFECT

TOL = 1e-6

# (id, estado_inicial, condicion, alcance, mecanismo)
CASOS: list[tuple] = [
    ("N3_full", "100",  "111",  "111",  "111"),
    ("N3_mec",  "100",  "111",  "111",  "110"),
    ("N4_full", "1000", "1111", "1111", "1111"),
    ("N4_mec",  "1000", "1111", "1111", "1101"),
]
K_OBJETIVO = [2, 3, 4, 5]


# ── Helpers de evaluacion ─────────────────────────────────────────────────────

def _vertices_de(sub) -> list:
    futuro = [(EFFECT, int(f)) for f in sub.indices_ncubos]
    presente = [(ACTUAL, int(p)) for p in sub.dims_ncubos]
    return presente + futuro


def _grupos_reales(grupos_vert: list) -> list:
    out = []
    for g in grupos_vert:
        fut = sorted(idx for (t, idx) in g if t == EFFECT)
        pre = sorted(idx for (t, idx) in g if t == ACTUAL)
        out.append((np.array(fut, dtype=np.int8), np.array(pre, dtype=np.int8)))
    return out


def _phi(sub, dists, grupos_vert: list) -> float:
    dist = sub.k_bipartir(_grupos_reales(grupos_vert)).distribucion_marginal()
    return float(emd_efecto(dist, dists))


def phi_optimo_exacto(sub, dists, V: list, k: int) -> float:
    """Minimo phi sobre TODAS las k-particiones del conjunto de vertices V."""
    mejor = float("inf")
    for asignacion in generar_k_particiones(len(V), k):
        grupos_vert = [[V[p] for p in grupo] for grupo in asignacion]
        phi = _phi(sub, dists, grupos_vert)
        if phi < mejor:
            mejor = phi
    return mejor


def _biparticiones_propias(grupo: list):
    """Todas las biparticiones propias de `grupo`, cada una exactamente una vez
    (ancla el primer elemento en el lado A para no duplicar A/B)."""
    g = sorted(grupo)
    ancla, resto = g[0], g[1:]
    for r in range(len(resto) + 1):
        for combo in combinations(resto, r):
            a = [ancla] + list(combo)
            b = [x for x in g if x not in set(a)]
            if a and b:
                yield a, b


def phi_greedy_ideal(sub, dists, V: list, k: int) -> float:
    """
    Mismo greedy jerarquico que KQNodes, pero con ORACULO EXACTO: en cada paso, para
    cada grupo se prueban TODAS sus biparticiones (fuerza bruta) y se aplica la que
    minimiza el phi de la k-particion completa. Determinista (orden estable, '<').
    """
    grupos = [list(V)]
    phi_actual = None
    for _ in range(k - 1):
        mejor = None  # (phi, nuevos)
        for i, G in enumerate(grupos):
            if len(G) < 2:
                continue
            for a, b in _biparticiones_propias(G):
                nuevos = grupos[:i] + [a, b] + grupos[i + 1:]
                phi = _phi(sub, dists, nuevos)
                if mejor is None or phi < mejor[0]:
                    mejor = (phi, nuevos)
        if mejor is None:
            break
        phi_actual, grupos = mejor
    if phi_actual is None:
        phi_actual = _phi(sub, dists, grupos)
    return phi_actual


def correr_kqnodes_por_k(tpm, est, cond, alc, mec):
    """Corre KQNodes una vez para k=2..5 y devuelve {k: (phi, modo, n_cand)} + instancia."""
    kq = KQNodes(tpm, k_min=2, k_max=5)
    kq.aplicar_estrategia(est, cond, alc, mec)
    por_k = {
        k: (float(d["phi"]), d["modo_usado"], d["n_candidatos"])
        for k, d in kq._resultados_por_k.items()
    }
    return por_k, kq


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 110)
    print("VALIDACION FASE 2 — Desglose del gap: KQNodes vs greedy-ideal vs optimo exacto")
    print(f"tol = {TOL}")
    print("=" * 110)

    filas = []
    n_pass = 0
    n_total = 0

    for id_caso, est, cond, alc, mec in CASOS:
        n = len(est)
        aplicacion.set_pagina_red_muestra("A")
        try:
            tpm = Manager(est).cargar_red()
        except Exception as exc:
            print(f"[ERROR carga] {id_caso}: {exc}")
            continue

        # KQNodes x2 (determinismo) + metadatos por k
        por_k, kq = correr_kqnodes_por_k(tpm, est, cond, alc, mec)
        por_k_2, _ = correr_kqnodes_por_k(tpm, est, cond, alc, mec)

        sub = kq.sia_subsistema
        dists = kq.sia_dists_marginales
        V = _vertices_de(sub)
        k_top = min(max(K_OBJETIVO), len(V))

        print(f"\n[{id_caso}] n={n}  |V|={len(V)}  (futuros={sub.indices_ncubos.size}, "
              f"presentes={sub.dims_ncubos.size})")

        for k in K_OBJETIVO:
            if k > k_top:
                continue
            n_total += 1

            phi_kq, modo, n_cand = por_k[k]
            phi_kq_2 = por_k_2[k][0]
            delta_det = abs(phi_kq - phi_kq_2)
            determinista = (delta_det == 0.0)

            phi_ideal = phi_greedy_ideal(sub, dists, V, k)
            phi_exact = phi_optimo_exacto(sub, dists, V, k)

            gap_total   = phi_kq - phi_exact
            gap_greedy  = phi_ideal - phi_exact
            gap_oraculo = phi_kq - phi_ideal

            invariante = (phi_kq >= phi_exact - 1e-9) and (phi_ideal >= phi_exact - 1e-9)
            passed = invariante and determinista
            if passed:
                n_pass += 1
            veredicto = "PASS" if passed else "FAIL"

            print(
                f"  k={k} [{veredicto}] modo={modo:<20} evals={n_cand:<3}  "
                f"phi: KQ={phi_kq:.5f}  ideal={phi_ideal:.5f}  exacto={phi_exact:.5f}  "
                f"|  gap_total={gap_total:+.5f} = greedy({gap_greedy:+.5f}) + "
                f"oraculo({gap_oraculo:+.5f})  det(Δ={delta_det:.0e})"
            )

            filas.append({
                "caso": id_caso, "n": n, "k": k, "n_vertices": len(V),
                "phi_KQNodes": round(phi_kq, 8),
                "phi_greedy_ideal": round(phi_ideal, 8),
                "phi_optimo_exacto": round(phi_exact, 8),
                "gap_total": round(gap_total, 8),
                "gap_greedy": round(gap_greedy, 8),
                "gap_oraculo": round(gap_oraculo, 8),
                "modo_usado": modo,
                "n_candidatos": n_cand,
                "stirling_total": contar_stirling(len(V), k),
                "invariante_ok": invariante,
                "delta_determinismo": delta_det,
                "determinista": determinista,
                "veredicto": veredicto,
            })

    df = pd.DataFrame(filas)
    ruta = Path("results") / "validacion_fase2_kqnodes.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta, index=False)

    print("\n" + "=" * 110)
    print("TABLA RESUMEN — phi por fuente + desglose del gap")
    print("=" * 110)
    cols = ["caso", "n", "k", "phi_KQNodes", "phi_greedy_ideal", "phi_optimo_exacto",
            "gap_total", "gap_greedy", "gap_oraculo", "determinista", "veredicto"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    # Atribucion agregada del gap
    if not df.empty:
        tot = df["gap_total"].sum()
        gre = df["gap_greedy"].sum()
        ora = df["gap_oraculo"].sum()
        print(f"\nATRIBUCION AGREGADA del gap (suma sobre casos):")
        print(f"  gap_total  = {tot:+.5f}")
        if abs(tot) > 1e-9:
            print(f"  gap_greedy = {gre:+.5f}  ({100*gre/tot:.0f}% del total)")
            print(f"  gap_oraculo= {ora:+.5f}  ({100*ora/tot:.0f}% del total)")
        else:
            print(f"  gap_greedy = {gre:+.5f}   gap_oraculo = {ora:+.5f}  (total ~0)")

    print(f"\nGATE FASE 2 (invariante + determinismo): {n_pass}/{n_total} PASS")
    print(f"Excel: {ruta.resolve()}")
    if n_pass == n_total:
        print("FASE 2 OK — invariante respetado, determinista, gap desglosado.")
    else:
        print("FASE 2 CON FALLOS — revisar casos FAIL.")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()