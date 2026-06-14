"""
experiments/probe_determinismo_simetrico.py

Probe DIAGNOSTICO (read-only, no modifica nada) para responder UNA pregunta:

  Para las pruebas 10,24,31,34,38,39 de la hoja 10A-Elementos, en modo heuristico
  simetrico, ¿el numero de candidatos UNICOS por k supera m_max_candidatos=2000?

Si lo supera, entonces el branch de truncado no determinista (random.sample sin
seed, kgeometric_heuristica.py L166-170) SE ACTIVA y los resultados historicos
de esos casos NO son reproducibles.

Para contar los candidatos pre-truncado se instancia la estrategia con
m_max_candidatos = 10**9 (truncado nunca se dispara) y se lee len(candidatos).

Como ejecutar (desde Method2_Dynamic_Programming_Reformulation/):
    .venv/Scripts/python.exe experiments/probe_determinismo_simetrico.py
"""

import sys
import logging
from pathlib import Path

import numpy as np

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
from src.controllers.strategies.kgeometric_heuristica import KGeometricSIAHeuristica
from src.funcs.decay import decay_exponencial

# Reutilizar helpers del ejecutor oficial
from src.ejecutar_kgeomip import (
    convertir_a_binario, parsear_hoja, resolver_tpm, leer_excel, INPUT_XLSX,
)

profiler_manager.enabled = False

HOJA = "10A-Elementos"
PRUEBAS_OBJETIVO = {10, 24, 31, 34, 38, 39}
K_VALUES = [2, 3, 4, 5]
M_MAX_DEFAULT = 2000


def main() -> None:
    n, variante = parsear_hoja(HOJA)
    estado_inicial, todos_casos = leer_excel(INPUT_XLSX, HOJA)
    tpm_path, tpm = resolver_tpm(n, variante)
    condiciones = "1" * len(estado_inicial)

    print("=" * 78)
    print(f"PROBE determinismo simetrico — Hoja {HOJA}  n={len(estado_inicial)}")
    print(f"Estado inicial: {estado_inicial}")
    print(f"m_max_candidatos default = {M_MAX_DEFAULT}  (truncado random.sample si se supera)")
    print("=" * 78)

    casos = [c for c in todos_casos if c[0] in PRUEBAS_OBJETIVO]

    aplicacion.pagina_sample_network = variante

    print(f"\n{'Prueba':>6} {'k':>3} {'cand_unicos':>12} {'>2000?':>8} {'truncado_no_det':>16}")
    print("-" * 78)

    for prueba_num, alcance_str, mecanismo_str in casos:
        alcance   = convertir_a_binario(alcance_str,   len(estado_inicial))
        mecanismo = convertir_a_binario(mecanismo_str, len(estado_inicial))

        gestor = Manager(estado_inicial=estado_inicial)
        sia = KGeometricSIAHeuristica(
            gestor,
            k_max=5,
            k_min=2,
            n_max_exhaustivo=6,        # fuerza modo heuristico para n=10
            m_max_candidatos=10**9,    # nunca trunca -> conteo real pre-truncado
            decay_fn=decay_exponencial,
        )
        sia.sia_preparar_subsistema(condiciones, alcance, mecanismo, tpm)
        sia._flat_data = [nc.data.ravel() for nc in sia.sia_subsistema.ncubos]
        sia._setup_dist_fn()

        indices = sia.sia_subsistema.indices_ncubos
        dims    = sia.sia_subsistema.dims_ncubos
        comunes = np.intersect1d(indices, dims)
        n_bal   = len(comunes)

        sia._construir_tabla_transiciones()

        for k in K_VALUES:
            if k > n_bal:
                print(f"{prueba_num:>6} {k:>3} {'-':>12} {'-':>8}  (k>n_bal={n_bal})")
                continue
            cands = sia._generar_candidatos_heuristica(n_bal, k, comunes)
            n_cand = len(cands)
            supera = n_cand > M_MAX_DEFAULT
            flag = "SI -> NO DET" if supera else "no"
            print(f"{prueba_num:>6} {k:>3} {n_cand:>12} {str(supera):>8} {flag:>16}")

    print("-" * 78)
    print("Si alguna fila marca 'SI -> NO DET', el resultado historico de ese (prueba,k)")
    print("se genero con truncado aleatorio sin seed => NO reproducible.")


if __name__ == "__main__":
    main()