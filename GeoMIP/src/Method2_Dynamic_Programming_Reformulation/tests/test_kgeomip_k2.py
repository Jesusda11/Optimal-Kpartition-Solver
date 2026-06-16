"""Consistencia k=2 de KGeoMIP (lo que exige el enunciado).

- KGeoMIPAsimetrico(k=2) exacto encuentra el OPTIMO de biparticion == fuerza bruta
  independiente sobre `bipartir`.
- Invariante: phi_asim_k2 <= phi_geo (GeometricSIA, la heuristica original).
- Determinismo: dos corridas dan phi y particion identicos.
"""

from itertools import chain, combinations

import numpy as np
import pytest

from src.controllers.manager import Manager
from src.controllers.strategies.kgeometric_asimetrico import KGeoMIPAsimetrico
from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.base import emd_efecto
from src.funcs.decay import decay_exponencial
from src.models.base.application import aplicacion

CASO = ("100", "111", "111", "111")  # n=3 full


def _subconjuntos(xs):
    return chain.from_iterable(combinations(xs, r) for r in range(len(xs) + 1))


def _bf_biparticion_optima(sub, dists) -> float:
    """Minimo phi sobre TODAS las biparticiones propias via `bipartir` (independiente)."""
    F = [int(x) for x in sub.indices_ncubos]
    P = [int(x) for x in sub.dims_ncubos]
    mejor = float("inf")
    for af in _subconjuntos(F):
        for mp in _subconjuntos(P):
            if len(af) == 0 and len(mp) == 0:                 # primal vacio (trivial)
                continue
            if len(af) == len(F) and len(mp) == len(P):       # dual vacio (trivial)
                continue
            dist = sub.bipartir(
                np.array(af, dtype=np.int8), np.array(mp, dtype=np.int8)
            ).distribucion_marginal()
            mejor = min(mejor, float(emd_efecto(dist, dists)))
    return mejor


def _correr_asimetrico_k2(tpm):
    aplicacion.pagina_sample_network = "A"
    sia = KGeoMIPAsimetrico(
        Manager(estado_inicial=CASO[0]), k_min=2, k_max=2,
        m_max_exhaustivo=999, decay_fn=decay_exponencial,
    )
    sol = sia.aplicar_estrategia(CASO[1], CASO[2], CASO[3], tpm)
    return sia, sol


def test_asimetrico_k2_es_optimo_de_biparticion(tpm_n3):
    sia, sol = _correr_asimetrico_k2(tpm_n3)
    phi_bf = _bf_biparticion_optima(sia.sia_subsistema, sia.sia_dists_marginales)
    assert float(sol.perdida) == pytest.approx(phi_bf, abs=1e-9)


def test_asimetrico_k2_no_peor_que_geometric(tpm_n3):
    _, sol_asim = _correr_asimetrico_k2(tpm_n3)
    aplicacion.pagina_sample_network = "A"
    geo = GeometricSIA(Manager(estado_inicial=CASO[0]), decay_fn=decay_exponencial)
    sol_geo = geo.aplicar_estrategia(CASO[1], CASO[2], CASO[3], tpm_n3)
    assert float(sol_asim.perdida) <= float(sol_geo.perdida) + 1e-9


def test_asimetrico_k2_determinista(tpm_n3):
    _, sol1 = _correr_asimetrico_k2(tpm_n3)
    _, sol2 = _correr_asimetrico_k2(tpm_n3)
    assert float(sol1.perdida) == float(sol2.perdida)
    assert sol1.particion == sol2.particion
