"""
Comparacion de resultados exactos vs heuristicos de KGeoMIP Simetrico.

Lee los archivos de resultados generados por ejecutar_kgeomip.py y compara
el phi exacto (exhaustivo S(n,k)) contra el phi heuristico para cada caso y k.

Como ejecutar (desde Method2_Dynamic_Programming_Reformulation/):
    python experiments/comparar_exacto_vs_heuristica.py --hoja 10A-Elementos --k 2
    python experiments/comparar_exacto_vs_heuristica.py --hoja 10A-Elementos --k 3
    python experiments/comparar_exacto_vs_heuristica.py --hoja 10A-Elementos --k_min 2 --k_max 3

Los archivos que lee son:
    results/resultados_kgeomip_simetrico_{hoja}_k{n}.xlsx         (heuristico)
    results/resultados_kgeomip_simetrico_{hoja}_k{n}_exacto.xlsx  (exacto)

Guarda el reporte en:
    results/comparacion_exacto_heuristica_{hoja}_k{n}.xlsx
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GEOMIP_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = GEOMIP_ROOT / "results"

TOL = 1e-5  # tolerancia para considerar que el heuristico encontro el optimo


def parse_phi(valor) -> float | None:
    """Convierte string con coma decimal o float a float."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    try:
        return float(str(valor).replace(",", "."))
    except (ValueError, TypeError):
        return None


def comparar_k(hoja_clean: str, k: int) -> pd.DataFrame | None:
    """Lee ambos archivos para una k y retorna DataFrame de comparacion."""
    ruta_heur  = RESULTS_DIR / f"resultados_kgeomip_simetrico_{hoja_clean}_k{k}.xlsx"
    ruta_exact = RESULTS_DIR / f"resultados_kgeomip_simetrico_{hoja_clean}_k{k}_exacto.xlsx"

    if not ruta_heur.exists():
        print(f"  [FALTA] {ruta_heur.name}")
        return None
    if not ruta_exact.exists():
        print(f"  [FALTA] {ruta_exact.name}")
        return None

    df_h = pd.read_excel(ruta_heur)
    df_e = pd.read_excel(ruta_exact)

    col_phi  = f"k{k}_Perdida"
    col_cand = f"k{k}_N_candidatos"
    col_t    = f"k{k}_Tiempo_s"

    merged = df_h[["#Prueba", "Alcance", "Mecanismo", col_phi, col_cand, col_t]].merge(
        df_e[["#Prueba", col_phi, col_cand, col_t]],
        on="#Prueba",
        suffixes=("_heur", "_exact"),
    )

    merged["phi_heur"]  = merged[f"{col_phi}_heur"].apply(parse_phi)
    merged["phi_exact"] = merged[f"{col_phi}_exact"].apply(parse_phi)

    merged["delta_phi"] = merged.apply(
        lambda r: round(r["phi_heur"] - r["phi_exact"], 8)
        if r["phi_heur"] is not None and r["phi_exact"] is not None
        else None,
        axis=1,
    )
    merged["optimo"] = merged["delta_phi"].apply(
        lambda d: (abs(d) <= TOL) if d is not None else None
    )
    merged["k"] = k

    # Tiempo de parse para mostrar
    def pt(v):
        return parse_phi(v)

    merged["t_heur_s"]  = merged[f"{col_t}_heur"].apply(pt)
    merged["t_exact_s"] = merged[f"{col_t}_exact"].apply(pt)
    merged["cand_heur"]  = merged[f"{col_cand}_heur"]
    merged["cand_exact"] = merged[f"{col_cand}_exact"]

    return merged[["k", "#Prueba", "Alcance", "Mecanismo",
                   "phi_exact", "phi_heur", "delta_phi", "optimo",
                   "cand_exact", "cand_heur", "t_exact_s", "t_heur_s"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparacion exacto vs heuristico KGeoMIP")
    parser.add_argument("--hoja", required=True, help="Nombre de hoja (ej. '10A-Elementos')")

    k_group = parser.add_mutually_exclusive_group()
    k_group.add_argument("--k",     type=int, help="k especifica")
    k_group.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max",  type=int, default=None)

    args = parser.parse_args()

    if args.k is not None:
        ks = [args.k]
    else:
        k_min = args.k_min if args.k_min is not None else 2
        k_max = args.k_max if args.k_max is not None else 5
        ks = list(range(k_min, k_max + 1))

    hoja_clean = args.hoja.strip().replace(" ", "_")

    print("=" * 70)
    print(f"Comparacion exacto vs heuristico — {args.hoja}  k={ks}")
    print("=" * 70)

    frames = []
    for k in ks:
        print(f"\n── k={k} ──")
        df = comparar_k(hoja_clean, k)
        if df is None:
            continue
        frames.append(df)

        validos = df[df["delta_phi"].notna()]
        if len(validos) == 0:
            print("  Sin casos comparables.")
            continue

        optimos   = validos["optimo"].sum()
        delta_max = validos["delta_phi"].abs().max()
        delta_med = validos["delta_phi"].abs().mean()
        fallidos  = validos[validos["optimo"] == False]

        print(f"  Optimo encontrado: {optimos}/{len(validos)}")
        print(f"  Delta phi max:     {delta_max:.8f}")
        print(f"  Delta phi medio:   {delta_med:.8f}")

        if len(fallidos) > 0:
            print(f"\n  Casos donde heuristica NO encontro el optimo ({len(fallidos)}):")
            print(
                fallidos[["#Prueba", "Alcance", "Mecanismo",
                           "phi_exact", "phi_heur", "delta_phi",
                           "cand_heur", "cand_exact"]]
                .to_string(index=False)
            )

    if not frames:
        print("\nNo hay datos para comparar. Verifica que existan ambos archivos.")
        return

    df_total = pd.concat(frames, ignore_index=True)

    # ── Resumen global ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMEN GLOBAL")
    validos_tot = df_total[df_total["delta_phi"].notna()]
    if len(validos_tot):
        optimos_tot = validos_tot["optimo"].sum()
        print(f"  Optimo encontrado: {optimos_tot}/{len(validos_tot)} ({100*optimos_tot/len(validos_tot):.1f}%)")
        print(f"  Delta phi max:     {validos_tot['delta_phi'].abs().max():.8f}")
        print(f"  Delta phi medio:   {validos_tot['delta_phi'].abs().mean():.8f}")

        por_k = validos_tot.groupby("k").agg(
            optimos=("optimo", "sum"),
            total=("optimo", "count"),
            delta_max=("delta_phi", lambda x: x.abs().max()),
        )
        por_k["pct"] = (por_k["optimos"] / por_k["total"] * 100).round(1)
        print("\n  Por k:")
        print(por_k[["optimos", "total", "pct", "delta_max"]].to_string())

    # ── Guardar Excel ─────────────────────────────────────────────────────────
    ks_str    = str(ks[0]) if len(ks) == 1 else f"{ks[0]}-{ks[-1]}"
    ruta_out  = RESULTS_DIR / f"comparacion_exacto_heuristica_{hoja_clean}_k{ks_str}.xlsx"
    df_total.to_excel(ruta_out, index=False)
    print(f"\nReporte guardado en: {ruta_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
