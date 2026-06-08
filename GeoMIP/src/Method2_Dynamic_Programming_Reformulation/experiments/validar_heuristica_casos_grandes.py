"""
Validacion de la heuristica de KGeoMIP sobre los casos que corrieron en modo
heuristico (n_balanceados > n_max_exacto).

Lee el archivo de resultados heuristicos existente, identifica los casos con
modo='heuristico', y los corre exhaustivamente con enumeracion exacta S(n,k)
para comparar phi_exacto vs phi_heuristico.

Como ejecutar (desde Method2_Dynamic_Programming_Reformulation/):

    python experiments/validar_heuristica_casos_grandes.py --hoja 10A-Elementos
    python experiments/validar_heuristica_casos_grandes.py --hoja 10A-Elementos --k_max 3
    python experiments/validar_heuristica_casos_grandes.py --hoja 10A-Elementos --k_max 4

Salida:
    GeoMIP/results/validacion_heuristica_{hoja}_k2-{k_max}.xlsx
"""

import sys
import os
import re
import time
import logging
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.controllers.strategies.kgeometric_heuristica import KGeoMIPHeuristica
from src.funcs.decay import decay_exponencial
from src.funcs.partitions import contar_stirling

profiler_manager.enabled = False

TOL         = 1e-5
TIMEOUT_S   = int(os.getenv("KGEOMIP_TIMEOUT_S", "86400"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def parsear_hoja(sheet_name: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([A-Z])", sheet_name.strip())
    if not m:
        raise ValueError(f"No se pudo parsear '{sheet_name}'")
    return int(m.group(1)), m.group(2)


def resolver_tpm(n: int, variante: str) -> tuple[Path, np.ndarray]:
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


def convertir_a_binario(texto: str, n_bits: int) -> str:
    posiciones = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto.upper():
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)


def parse_phi(valor) -> float | None:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    try:
        return float(str(valor).replace(",", "."))
    except (ValueError, TypeError):
        return None


def leer_resultados_heuristicos(hoja_clean: str) -> pd.DataFrame | None:
    """Busca el archivo de resultados heuristicos en varias convenciones de nombre."""
    candidatos = [
        GEOMIP_ROOT / "results" / f"resultados_kgeomip_simetrico_{hoja_clean}_k2-5.xlsx",
        GEOMIP_ROOT / "results" / f"resultados_kgeomip_simetrico_{hoja_clean}.xlsx",
    ]
    for ruta in candidatos:
        if ruta.exists():
            print(f"  Resultados heuristicos: {ruta.name}")
            return pd.read_excel(ruta)
    return None


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker_exacto(
    estado_inicial: str,
    condiciones:    str,
    alcance:        str,
    mecanismo:      str,
    tpm:            np.ndarray,
    pagina:         str,
    k_min:          int,
    k_max:          int,
    cola:           multiprocessing.Queue,
) -> None:
    try:
        aplicacion.pagina_sample_network = pagina
        gestor = Manager(estado_inicial=estado_inicial)
        sia = KGeoMIPHeuristica(
            gestor,
            k_max=k_max,
            k_min=k_min,
            n_max_exhaustivo=999,   # siempre exacto
            decay_fn=decay_exponencial,
        )
        sia.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        resultado = {
            "por_k": {
                k: {"phi": datos["phi"], "n_candidatos": datos["n_candidatos"],
                    "tiempo_s": datos["tiempo_s"]}
                for k, datos in sia._resultados_por_k.items()
            },
            "error": None,
        }
    except Exception as exc:
        resultado = {"por_k": {}, "error": str(exc)}
    cola.put(resultado)


# ── Logica principal ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Valida heuristica KGeoMIP en casos grandes")
    parser.add_argument("--hoja",    required=True, help="Nombre de hoja (ej. '10A-Elementos')")
    parser.add_argument("--k_max",   type=int, default=3,
                        help="k maximo a evaluar (default 3; k=4 ~14 min; k=5 ~75 min para n=10)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S,
                        help=f"Timeout por caso (default {TIMEOUT_S}s)")
    args = parser.parse_args()

    n, variante   = parsear_hoja(args.hoja)
    hoja_clean    = args.hoja.strip().replace(" ", "_")
    k_max         = args.k_max
    estado_inicial = "1" + "0" * (n - 1)
    condiciones    = "1" * n

    # ── Cargar TPM ────────────────────────────────────────────────────────────
    tpm_path, tpm = resolver_tpm(n, variante)

    # ── Leer resultados heuristicos ───────────────────────────────────────────
    df_heur = leer_resultados_heuristicos(hoja_clean)
    if df_heur is None:
        print(f"[ERROR] No se encontro archivo de resultados heuristicos para {args.hoja}")
        print("  Busco en:")
        print(f"    results/resultados_kgeomip_simetrico_{hoja_clean}_k2-5.xlsx")
        print(f"    results/resultados_kgeomip_simetrico_{hoja_clean}.xlsx")
        return

    casos_heur = df_heur[df_heur["Modo"] == "heuristico"].copy()
    if casos_heur.empty:
        print("No hay casos en modo heuristico en el archivo. Nada que validar.")
        return

    # Stirling max para el aviso de tiempo
    stirling_info = {k: contar_stirling(n, k) for k in range(2, k_max + 1)}

    print("=" * 68)
    print(f"Validacion heuristica — {args.hoja}  k=2..{k_max}")
    print(f"n={n}  TPM: {tpm_path.name}")
    print(f"Casos a validar: {len(casos_heur)}  (modo heuristico en el archivo de resultados)")
    print(f"Candidatos maximos por caso: { {k: f'{v:,}' for k,v in stirling_info.items()} }")
    print(f"Timeout: {args.timeout}s")
    print("=" * 68)

    filas = []

    for _, row in casos_heur.iterrows():
        prueba_num  = int(row["#Prueba"])
        alcance_str = str(row["Alcance"]).strip()
        mec_str     = str(row["Mecanismo"]).strip()
        alcance     = convertir_a_binario(alcance_str, n)
        mecanismo   = convertir_a_binario(mec_str, n)

        # Phi heuristicos del archivo existente
        phi_heur = {k: parse_phi(row.get(f"k{k}_Perdida")) for k in range(2, k_max + 1)}
        cand_heur = {k: row.get(f"k{k}_N_candidatos") for k in range(2, k_max + 1)}

        print(f"\n[{prueba_num:>3}] Alcance={alcance_str!r:<28} Mecanismo={mec_str!r}")
        for k in range(2, k_max + 1):
            if phi_heur[k] is not None:
                print(f"       phi_heur k={k}: {phi_heur[k]:.6f}  (cand={cand_heur[k]})")

        # ── Correr exacto ─────────────────────────────────────────────────────
        cola    = multiprocessing.Queue()
        proceso = multiprocessing.Process(
            target=_worker_exacto,
            args=(estado_inicial, condiciones, alcance, mecanismo,
                  tpm, variante, 2, k_max, cola),
        )
        t_ini = time.perf_counter()
        proceso.start()
        proceso.join(timeout=args.timeout)
        t_total = round(time.perf_counter() - t_ini, 4)

        if proceso.is_alive():
            print(f"  [TIMEOUT {args.timeout}s]")
            proceso.terminate()
            proceso.join()
            por_k_exact = {}
            error = f"Timeout ({args.timeout}s)"
        elif cola.empty():
            por_k_exact = {}
            error = "Sin resultado"
        else:
            res         = cola.get()
            por_k_exact = res.get("por_k", {})
            error       = res.get("error")

        fila = {
            "#Prueba":   prueba_num,
            "Alcance":   alcance_str,
            "Mecanismo": mec_str,
            "T_total_s": t_total,
            "Error":     error,
        }

        for k in range(2, k_max + 1):
            phi_e = por_k_exact.get(k, {}).get("phi") if k in por_k_exact else None
            phi_h = phi_heur[k]
            delta = round(phi_e - phi_h, 8) if (phi_e is not None and phi_h is not None) else None
            optimo = abs(delta) <= TOL if delta is not None else None

            fila[f"k{k}_phi_exact"]   = phi_e
            fila[f"k{k}_phi_heur"]    = phi_h
            fila[f"k{k}_delta"]       = delta
            fila[f"k{k}_optimo"]      = optimo
            fila[f"k{k}_cand_exact"]  = por_k_exact.get(k, {}).get("n_candidatos")
            fila[f"k{k}_cand_heur"]   = cand_heur[k]
            fila[f"k{k}_t_exact_s"]   = por_k_exact.get(k, {}).get("tiempo_s")

            if phi_e is not None:
                status = "PASS" if optimo else f"FAIL delta={delta:+.6f}"
                print(f"  k={k}: exact={phi_e:.6f}  heur={phi_h:.6f}  [{status}]  cand={fila[f'k{k}_cand_exact']}")
        if error:
            print(f"  ERROR: {error}")

        filas.append(fila)

    # ── Guardar y resumir ─────────────────────────────────────────────────────
    df_out  = pd.DataFrame(filas)
    ruta    = GEOMIP_ROOT / "results" / f"validacion_heuristica_{hoja_clean}_k2-{k_max}.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(ruta, index=False)

    print(f"\n{'=' * 68}")
    print(f"Reporte guardado en: {ruta}")

    for k in range(2, k_max + 1):
        col  = f"k{k}_optimo"
        col_d = f"k{k}_delta"
        if col not in df_out.columns:
            continue
        validos  = df_out[df_out[col].notna()]
        if validos.empty:
            continue
        optimos  = validos[col].sum()
        delta_max = df_out[col_d].dropna().abs().max()
        print(f"  k={k}: optimo {optimos}/{len(validos)}  delta_max={delta_max:.8f}")

    print("=" * 68)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
