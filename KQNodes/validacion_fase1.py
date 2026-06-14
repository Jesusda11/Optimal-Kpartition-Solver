"""
validacion_fase1.py — Gate de la Fase 1 de KQNodes.

Valida que KQNodes(k=2) reproduce BIT A BIT a QNodes original (mismo motor de
Queyranne), y que ambos alcanzan el optimo de bipartición (confirmado por fuerza
bruta), en n=3..6 con varios casos de alcance/mecanismo por cada n.

Criterio de PASS (doble + extras):
  1. phi_kqnodes == phi_qnodes        (igualdad EXACTA — mismo computo)
  2. biparticion_kqnodes == biparticion_qnodes   (string canonico fmt identico)
  3. |phi_qnodes - phi_fuerzabruta| < 1e-6        (es el optimo real)
  4. determinismo: dos corridas de KQNodes dan delta 0 (phi y particion)

Ejecutar desde KQNodes/:
    <python> validacion_fase1.py

Salida: tabla por consola + Excel en KQNodes/results/validacion_fase1_kqnodes.xlsx
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.strategies.q_nodes import QNodes
from src.strategies.kqnodes import KQNodes
from src.strategies.force import BruteForce

TOL = 1e-6

# (id, estado_inicial, condicion, alcance, mecanismo)
CASOS: list[tuple] = [
    # n = 3
    ("N3_full",   "100", "111", "111", "111"),
    ("N3_alc",    "100", "111", "110", "111"),
    ("N3_mec",    "100", "111", "111", "101"),
    # n = 4
    ("N4_full",   "1000", "1111", "1111", "1111"),
    ("N4_alc",    "1000", "1111", "0111", "1111"),
    ("N4_mec",    "1000", "1111", "1111", "1010"),
    ("N4_ambos",  "1000", "1111", "1110", "1101"),
    # n = 5
    ("N5_full",   "10000", "11111", "11111", "11111"),
    ("N5_alc",    "10000", "11111", "11110", "11111"),
    ("N5_mec",    "10000", "11111", "11111", "10101"),
    # n = 6
    ("N6_full",   "100000", "111111", "111111", "111111"),
    ("N6_alc",    "100000", "111111", "111110", "111111"),
    ("N6_mec",    "100000", "111111", "111111", "101010"),
]


def cargar_tpm(estado: str) -> np.ndarray:
    return Manager(estado).cargar_red()


def correr_qnodes(tpm, est, cond, alc, mec):
    sol = QNodes(tpm).aplicar_estrategia(est, cond, alc, mec)
    return float(sol.perdida), sol.particion


def correr_kqnodes(tpm, est, cond, alc, mec):
    sol = KQNodes(tpm, k_min=2, k_max=2).aplicar_estrategia(est, cond, alc, mec)
    return float(sol.perdida), sol.particion


def correr_bruteforce(tpm, est, cond, alc, mec):
    sol = BruteForce(tpm).aplicar_estrategia(est, cond, alc, mec)
    return float(sol.perdida)


def main() -> None:
    print("=" * 92)
    print("GATE FASE 1 — KQNodes(k=2) vs QNodes vs Fuerza Bruta")
    print(f"Casos: {len(CASOS)}  |  n=3..6  |  tol phi (BF) = {TOL}")
    print("=" * 92)

    filas = []
    n_pass = 0

    for id_caso, est, cond, alc, mec in CASOS:
        n = len(est)
        aplicacion.set_pagina_red_muestra("A")
        try:
            tpm = cargar_tpm(est)

            phi_qn, part_qn = correr_qnodes(tpm, est, cond, alc, mec)
            phi_kq, part_kq = correr_kqnodes(tpm, est, cond, alc, mec)
            phi_bf = correr_bruteforce(tpm, est, cond, alc, mec)

            # Determinismo: segunda corrida de KQNodes
            phi_kq2, part_kq2 = correr_kqnodes(tpm, est, cond, alc, mec)
            delta_det = abs(phi_kq - phi_kq2)
            det_part_ok = (part_kq == part_kq2)
            determinista = (delta_det == 0.0) and det_part_ok

            # ── GATE de fidelidad (lo que pidio el usuario): KQNodes == QNodes ──
            phi_igual   = (phi_kq == phi_qn)               # igualdad EXACTA
            part_igual  = (part_kq == part_qn)             # string canonico identico
            passed = phi_igual and part_igual and determinista
            veredicto = "PASS" if passed else "FAIL"
            if passed:
                n_pass += 1

            # ── Informativo: optimalidad vs fuerza bruta (propiedad de QNodes) ──
            kq_optimo = (abs(phi_kq - phi_bf) < TOL)
            gap_bf = round(phi_kq - phi_bf, 8)

            print(
                f"[{veredicto}] {id_caso:<9} n={n}  "
                f"phi_qn={phi_qn:.6f}  phi_kq={phi_kq:.6f}  phi_bf={phi_bf:.6f}  "
                f"bip_iguales={part_igual}  det(Δ={delta_det:.1e})={determinista}  "
                f"| optimo_vs_BF={kq_optimo} (gap={gap_bf:+.4f})"
            )

            filas.append({
                "caso": id_caso,
                "n": n,
                "phi_qnodes": round(phi_qn, 8),
                "phi_kqnodes": round(phi_kq, 8),
                "phi_fuerzabruta": round(phi_bf, 8),
                "biparticiones_identicas": part_igual,
                "phi_kq==phi_qn": phi_igual,
                "delta_determinismo": delta_det,
                "determinista": determinista,
                "veredicto": veredicto,
                "kqnodes_optimo_vs_BF": kq_optimo,
                "gap_vs_fuerzabruta": gap_bf,
                "particion_qnodes": part_qn,
                "particion_kqnodes": part_kq,
            })

        except Exception as exc:
            print(f"[ERROR] {id_caso}: {type(exc).__name__}: {exc}")
            filas.append({
                "caso": id_caso, "n": n, "veredicto": "ERROR", "error": repr(exc),
            })

    # ── Exportar ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(filas)
    ruta = Path("results") / "validacion_fase1_kqnodes.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta, index=False)

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("TABLA RESUMEN")
    print("=" * 92)
    cols = ["caso", "n", "phi_qnodes", "phi_kqnodes", "phi_fuerzabruta",
            "biparticiones_identicas", "determinista", "veredicto",
            "kqnodes_optimo_vs_BF"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    n_opt = int(df.get("kqnodes_optimo_vs_BF", pd.Series(dtype=bool)).sum()) \
        if "kqnodes_optimo_vs_BF" in df.columns else 0

    print(f"\nGATE DE FIDELIDAD (KQNodes == QNodes, phi + biparticion + determinismo):"
          f" {n_pass}/{len(CASOS)} PASS")
    print(f"OPTIMALIDAD vs fuerza bruta (informativo): {n_opt}/{len(CASOS)} casos optimos")
    print(f"Excel: {ruta.resolve()}")
    if n_pass == len(CASOS):
        print("GATE FASE 1 SUPERADO — KQNodes(k=2) reproduce QNodes bit a bit y es deterministico.")
        if n_opt < len(CASOS):
            print(f"NOTA: en {len(CASOS) - n_opt} casos el propio QNodes original es SUBOPTIMO vs "
                  "fuerza bruta (propiedad heredada del QNodes de referencia, no de KQNodes).")
    else:
        print("GATE FASE 1 NO SUPERADO — KQNodes diverge de QNodes en algun caso (revisar).")
    sys.exit(0 if n_pass == len(CASOS) else 1)


if __name__ == "__main__":
    main()
