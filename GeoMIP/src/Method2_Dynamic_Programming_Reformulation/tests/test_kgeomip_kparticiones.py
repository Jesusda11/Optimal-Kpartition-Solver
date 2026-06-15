"""Evaluacion correcta de k-particiones en KGeoMIP (modo exacto).

- El optimo por k del KGeoMIPAsimetrico exacto coincide con la fuerza bruta
  independiente sobre generar_k_particiones + k_bipartir.
- Monotonia con grupos no vacios: phi_{k+1} >= phi_k.
- Determinismo por k.
"""

import numpy as np
import pytest

from src.controllers.manager import Manager
from src.controllers.strategies.kgeometric_asimetrico import KGeoMIPAsimetrico
from src.funcs.base import emd_efecto
from src.funcs.partitions import generar_k_particiones
from src.funcs.decay import decay_exponencial
from src.models.base.application import aplicacion

CASO = ("100", "111", "111", "111")  # n=3 full -> pool m=6
K_MAX = 4


def _bf_kparticion(sub, dists, k: int) -> float:
    """Minimo phi sobre TODAS las k-particiones del pool (decodificacion independiente)."""
    F = [int(x) for x in sub.indices_ncubos]
    P = [int(x) for x in sub.dims_ncubos]
    n_fut = len(F)
    m = n_fut + len(P)
    mejor = float("inf")
    for part in generar_k_particiones(m, k):
        grupos = []
        for g in part:
            fut = [F[i] for i in g if i < n_fut]
            pres = [P[i - n_fut] for i in g if i >= n_fut]
            grupos.append((np.array(fut, dtype=np.int8), np.array(pres, dtype=np.int8)))
        dist = sub.k_bipartir(grupos).distribucion_marginal()
        mejor = min(mejor, float(emd_efecto(dist, dists)))
    return mejor


def _correr_exacto(tpm):
    aplicacion.pagina_sample_network = "A"
    sia = KGeoMIPAsimetrico(
        Manager(estado_inicial=CASO[0]), k_min=2, k_max=K_MAX,
        m_max_exhaustivo=999, decay_fn=decay_exponencial,
    )
    sia.aplicar_estrategia(CASO[1], CASO[2], CASO[3], tpm)
    return sia


def test_optimo_por_k_igual_fuerza_bruta(tpm_n3):
    sia = _correr_exacto(tpm_n3)
    sub, dists = sia.sia_subsistema, sia.sia_dists_marginales
    for k in range(2, K_MAX + 1):
        phi_strategy = float(sia._resultados_por_k[k]["phi"])
        phi_bf = _bf_kparticion(sub, dists, k)
        assert phi_strategy == pytest.approx(phi_bf, abs=1e-9), f"discrepancia en k={k}"


def test_monotonia_creciente_en_k(tpm_n3):
    sia = _correr_exacto(tpm_n3)
    phis = {k: float(sia._resultados_por_k[k]["phi"]) for k in range(2, K_MAX + 1)}
    for k in range(2, K_MAX):
        assert phis[k + 1] >= phis[k] - 1e-9, f"phi_{k+1}={phis[k+1]} < phi_{k}={phis[k]}"


def test_determinismo_por_k(tpm_n3):
    sia1 = _correr_exacto(tpm_n3)
    sia2 = _correr_exacto(tpm_n3)
    for k in range(2, K_MAX + 1):
        assert float(sia1._resultados_por_k[k]["phi"]) == float(sia2._resultados_por_k[k]["phi"])
