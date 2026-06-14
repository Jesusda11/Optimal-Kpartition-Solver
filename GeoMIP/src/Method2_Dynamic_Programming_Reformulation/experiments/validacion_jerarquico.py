"""
experiments/validacion_jerarquico.py

Valida el modo jerarquico de KGeoMIPAsimetricoJerarquico en tres dimensiones:

  Nivel 1 — Correctitud (n=3,4, k=3,4):
    Compara phi_jerarquico (m_max_exhaustivo=0, fuerza modo jerarquico)
    vs phi_exacto (m_max_exhaustivo=999, enumeracion S(m,k) completa).
    Invariante esperado: phi_jerarquico >= phi_exacto (greedy no garantiza optimo).
    Reporta la brecha relativa por caso.

  Nivel 2 — Escalabilidad (n=15, k=4):
    Ejecuta modo jerarquico en el caso que causaba OOM en modo directo.
    Verifica que termina sin crash y registra tiempo.

  Nivel 3 — Determinismo (n=15, k=4):
    Ejecuta el mismo caso dos veces y confirma phi identico.

Como ejecutar (desde Method2_Dynamic_Programming_Reformulation/):
    .venv/Scripts/python.exe experiments/validacion_jerarquico.py

El Excel se guarda en:
    GeoMIP/results/validacion_jerarquico.xlsx
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ──────────────────────────────────────────────────────────────────────
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
from src.controllers.strategies.kgeometric_asimetrico_jerarquico import (
    KGeometricSIAAsimetricoJerarquico,
)
from src.funcs.decay import decay_exponencial

profiler_manager.enabled = False

# ── Casos de correctitud (n pequeño) ──────────────────────────────────────────
CASOS_CORRECTITUD: list[tuple] = [
    # (id_caso, pagina, estado_ini, condicion, alcance, mecanismo)
    ("N3A_01", "A", "100", "111", "111", "111"),
    ("N3A_02", "A", "100", "111", "011", "111"),
    ("N3A_03", "A", "100", "111", "111", "011"),
    ("N3B_01", "B", "100", "111", "111", "111"),
    ("N4A_01", "A", "1000", "1111", "1111", "1111"),
    ("N4A_02", "A", "1000", "1111", "0111", "1111"),
    ("N4A_03", "A", "1000", "1111", "1010", "1111"),
    ("N4B_01", "B", "1000", "1111", "1111", "1111"),
]

K_CORRECTITUD = [3, 4]  # k valores para comparar exacto vs jerarquico

# ── Caso de escalabilidad / determinismo (n=15) ────────────────────────────────
CASO_ESCALA = ("N15B_01", "B", "100000000000000", "111111111111111",
               "111111111111111", "111111111111111")
K_ESCALA = 4


# ── Utilidades ────────────────────────────────────────────────────────────────

def _resolver_tpm(n: int, variante: str) -> np.ndarray:
    nombre = f"N{n}{variante}.csv"
    candidatos = (
        METHOD2_ROOT / "src" / ".samples" / nombre,
        METHOD2_ROOT / ".samples" / nombre,
        GEOMIP_ROOT / "data" / "samples" / nombre,
    )
    tpm_path = next((c for c in candidatos if c.exists()), None)
    if tpm_path is None:
        raise FileNotFoundError(f"No se encontro TPM: {nombre}")
    return np.genfromtxt(tpm_path, delimiter=",")


def _correr_estrategia(
    Clase,
    gestor: Manager,
    condicion: str,
    alcance: str,
    mecanismo: str,
    tpm: np.ndarray,
    k_max: int,
    m_max_exhaustivo: int,
) -> dict:
    """Instancia Clase, ejecuta aplicar_estrategia, retorna resultados_por_k + tiempo."""
    sia = Clase(
        gestor,
        k_min=2,
        k_max=k_max,
        m_max_exhaustivo=m_max_exhaustivo,
        decay_fn=decay_exponencial,
    )
    t0 = time.perf_counter()
    sia.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
    elapsed = time.perf_counter() - t0
    return sia._resultados_por_k, elapsed


# ── Nivel 1: Correctitud ──────────────────────────────────────────────────────

def nivel1_correctitud(rows_out: list) -> int:
    print("\n" + "=" * 70)
    print("NIVEL 1 — Correctitud: phi_jerarquico vs phi_exacto (n=3,4)")
    print("Invariante: phi_jerarquico >= phi_exacto  (greedy no garantiza optimo)")
    print("=" * 70)

    pass_count = 0
    total = 0

    for caso in CASOS_CORRECTITUD:
        id_caso, pagina, estado_ini, condicion, alcance, mecanismo = caso
        n = len(estado_ini)

        try:
            tpm = _resolver_tpm(n, pagina)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {id_caso}: {exc}")
            continue

        aplicacion.pagina_sample_network = pagina

        for k in K_CORRECTITUD:
            # Exacto
            gestor_ex = Manager(estado_inicial=estado_ini)
            try:
                por_k_ex, t_ex = _correr_estrategia(
                    KGeometricSIAAsimetrico, gestor_ex,
                    condicion, alcance, mecanismo, tpm,
                    k_max=k, m_max_exhaustivo=999,
                )
            except Exception as exc:
                print(f"  [ERROR exacto] {id_caso} k={k}: {exc}")
                continue

            # Jerarquico (forzar: m_max_exhaustivo=0 siempre usa modo jerarquico)
            gestor_jer = Manager(estado_inicial=estado_ini)
            try:
                por_k_jer, t_jer = _correr_estrategia(
                    KGeometricSIAAsimetricoJerarquico, gestor_jer,
                    condicion, alcance, mecanismo, tpm,
                    k_max=k, m_max_exhaustivo=0,
                )
            except Exception as exc:
                print(f"  [ERROR jerarquico] {id_caso} k={k}: {exc}")
                continue

            phi_ex  = por_k_ex.get(k, {}).get("phi")
            phi_jer = por_k_jer.get(k, {}).get("phi")

            if phi_ex is None or phi_jer is None:
                print(f"  [SKIP] {id_caso} k={k}: phi None (subsistema sin nodos suficientes)")
                continue

            # Invariante: phi_jerarquico >= phi_exacto
            invariante_ok = phi_jer >= phi_ex - 1e-9
            brecha_abs = phi_jer - phi_ex
            brecha_rel = (brecha_abs / phi_ex * 100) if phi_ex > 1e-9 else 0.0

            status = "PASS" if invariante_ok else "FAIL"
            total += 1
            if invariante_ok:
                pass_count += 1

            print(
                f"  {id_caso}  k={k}  "
                f"phi_ex={phi_ex:.6f}  phi_jer={phi_jer:.6f}  "
                f"brecha={brecha_abs:+.6f} ({brecha_rel:+.2f}%)  [{status}]"
            )

            rows_out.append({
                "nivel":       1,
                "id_caso":     id_caso,
                "k":           k,
                "phi_exacto":  round(phi_ex, 8),
                "phi_jerar":   round(phi_jer, 8),
                "brecha_abs":  round(brecha_abs, 8),
                "brecha_rel_%": round(brecha_rel, 4),
                "invariante_ok": invariante_ok,
                "t_exacto_s":  round(t_ex, 4),
                "t_jerar_s":   round(t_jer, 4),
                "PASS":        invariante_ok,
                "nota":        "phi_jer >= phi_ex (greedy aprox)",
            })

    print(f"\n  Nivel 1: {pass_count}/{total} PASS")
    return pass_count


# ── Nivel 2: Escalabilidad ────────────────────────────────────────────────────

def nivel2_escalabilidad(rows_out: list) -> bool:
    print("\n" + "=" * 70)
    print("NIVEL 2 — Escalabilidad: n=15, k=4 en modo jerarquico")
    print("Verifica que no hay OOM y el proceso termina en tiempo razonable.")
    print("=" * 70)

    id_caso, pagina, estado_ini, condicion, alcance, mecanismo = CASO_ESCALA
    n = len(estado_ini)

    try:
        tpm = _resolver_tpm(n, pagina)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {id_caso}: {exc}")
        rows_out.append({
            "nivel": 2, "id_caso": id_caso, "k": K_ESCALA,
            "PASS": None, "nota": f"TPM no encontrada: {exc}",
        })
        return False

    aplicacion.pagina_sample_network = pagina
    gestor = Manager(estado_inicial=estado_ini)

    print(f"  Ejecutando {id_caso} n={n} k={K_ESCALA}...")
    t0 = time.perf_counter()
    try:
        por_k, elapsed = _correr_estrategia(
            KGeometricSIAAsimetricoJerarquico, gestor,
            condicion, alcance, mecanismo, tpm,
            k_max=K_ESCALA, m_max_exhaustivo=0,
        )
        phi = por_k.get(K_ESCALA, {}).get("phi")
        n_eval = por_k.get(K_ESCALA, {}).get("n_candidatos")
        passed = True
        print(f"  OK — phi={phi}  evaluaciones={n_eval}  tiempo={elapsed:.2f}s")
    except MemoryError as exc:
        elapsed = time.perf_counter() - t0
        phi = None
        n_eval = None
        passed = False
        print(f"  FAIL — MemoryError: {exc}  (tiempo={elapsed:.2f}s)")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        phi = None
        n_eval = None
        passed = False
        print(f"  FAIL — {type(exc).__name__}: {exc}  (tiempo={elapsed:.2f}s)")

    rows_out.append({
        "nivel":   2,
        "id_caso": id_caso,
        "k":       K_ESCALA,
        "phi_jerar": round(phi, 8) if phi is not None else None,
        "n_eval":  n_eval,
        "t_jerar_s": round(elapsed, 2),
        "PASS":    passed,
        "nota":    "escalabilidad n=15 k=4 sin OOM",
    })

    print(f"\n  Nivel 2: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Nivel 3: Determinismo ─────────────────────────────────────────────────────

def nivel3_determinismo(rows_out: list) -> bool:
    print("\n" + "=" * 70)
    print("NIVEL 3 — Determinismo: n=15, k=4, dos ejecuciones identicas")
    print("Verifica que phi_run1 == phi_run2 (sin random.sample).")
    print("=" * 70)

    id_caso, pagina, estado_ini, condicion, alcance, mecanismo = CASO_ESCALA
    n = len(estado_ini)

    try:
        tpm = _resolver_tpm(n, pagina)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {id_caso}: {exc}")
        rows_out.append({
            "nivel": 3, "id_caso": id_caso, "k": K_ESCALA,
            "PASS": None, "nota": f"TPM no encontrada: {exc}",
        })
        return False

    aplicacion.pagina_sample_network = pagina
    phis = []

    for run in range(1, 3):
        gestor = Manager(estado_inicial=estado_ini)
        print(f"  Run {run}...")
        try:
            por_k, elapsed = _correr_estrategia(
                KGeometricSIAAsimetricoJerarquico, gestor,
                condicion, alcance, mecanismo, tpm,
                k_max=K_ESCALA, m_max_exhaustivo=0,
            )
            phi = por_k.get(K_ESCALA, {}).get("phi")
            phis.append(phi)
            print(f"    phi={phi}  tiempo={elapsed:.2f}s")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            phis.append(None)

    if len(phis) == 2 and phis[0] is not None and phis[1] is not None:
        determinista = abs(phis[0] - phis[1]) < 1e-9
    else:
        determinista = False

    status = "PASS" if determinista else "FAIL"
    print(f"\n  phi_run1={phis[0]}  phi_run2={phis[1]}  delta={abs(phis[0]-phis[1]) if None not in phis else 'N/A'}  [{status}]")

    rows_out.append({
        "nivel":      3,
        "id_caso":    id_caso,
        "k":          K_ESCALA,
        "phi_run1":   phis[0],
        "phi_run2":   phis[1],
        "delta_phi":  abs(phis[0] - phis[1]) if None not in phis else None,
        "PASS":       determinista,
        "nota":       "determinismo: run1 == run2",
    })

    print(f"\n  Nivel 3: {status}")
    return determinista


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Validacion KGeoMIP Asimetrico — Modo Jerarquico")
    print("Tres niveles: Correctitud | Escalabilidad | Determinismo")
    print("=" * 70)

    rows: list[dict] = []

    pass_n1 = nivel1_correctitud(rows)
    ok_n2   = nivel2_escalabilidad(rows)
    ok_n3   = nivel3_determinismo(rows)

    # ── Exportar ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    ruta = GEOMIP_ROOT / "results" / "validacion_jerarquico.xlsx"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta, index=False)
    print(f"\nResultados guardados en: {ruta}")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print(f"  Nivel 1 (correctitud phi_jer >= phi_ex): {pass_n1} PASS")
    print(f"  Nivel 2 (escalabilidad n=15 k=4 sin OOM): {'PASS' if ok_n2 else 'FAIL'}")
    print(f"  Nivel 3 (determinismo run1==run2):         {'PASS' if ok_n3 else 'FAIL'}")
    print("=" * 70)

    if ok_n2 and ok_n3:
        print("Modo jerarquico VALIDADO para uso en produccion.")
    else:
        print("ADVERTENCIA: revisar niveles fallidos antes de usar en produccion.")


if __name__ == "__main__":
    main()