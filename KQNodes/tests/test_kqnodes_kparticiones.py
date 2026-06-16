"""KQNodes k>2 (jerarquico): invariante de aproximacion y determinismo.

KQNodes(k>2) es heuristico (greedy + oraculo Queyranne), asi que el invariante de
correccion es phi_KQNodes >= phi_optimo (fuerza bruta sobre k-particiones de vertices).
"""

import numpy as np
import pytest

from src.strategies.kqnodes import KQNodes
from src.funcs.iit import emd_efecto
from src.funcs.partitions import generar_k_particiones
from src.constants.base import ACTUAL, EFFECT
from src.models.base.application import aplicacion

CASO = ("100", "111", "111", "111")  # n=3 full -> |V|=6


def _vertices(sub):
    futuro = [(EFFECT, int(f)) for f in sub.indices_ncubos]
    presente = [(ACTUAL, int(p)) for p in sub.dims_ncubos]
    return presente + futuro


def _grupos_reales(grupos_vert):
    out = []
    for g in grupos_vert:
        fut = sorted(i for (t, i) in g if t == EFFECT)
        pre = sorted(i for (t, i) in g if t == ACTUAL)
        out.append((np.array(fut, dtype=np.int8), np.array(pre, dtype=np.int8)))
    return out


def _bf_kparticion(sub, dists, V, k):
    mejor = float("inf")
    for part in generar_k_particiones(len(V), k):
        gv = [[V[i] for i in g] for g in part]
        dist = sub.k_bipartir(_grupos_reales(gv)).distribucion_marginal()
        mejor = min(mejor, float(emd_efecto(dist, dists)))
    return mejor


def _run(tpm, k):
    aplicacion.set_pagina_red_muestra("A")
    kq = KQNodes(tpm, k_min=k, k_max=k)
    kq.aplicar_estrategia(*CASO)
    return kq


@pytest.mark.parametrize("k", [3, 4])
def test_kqnodes_kgt2_respeta_invariante(tpm_n3, k):
    kq = _run(tpm_n3, k)
    sub, dists = kq.sia_subsistema, kq.sia_dists_marginales
    V = _vertices(sub)
    phi_kq = float(kq._resultados_por_k[k]["phi"])
    phi_bf = _bf_kparticion(sub, dists, V, k)
    assert phi_kq >= phi_bf - 1e-9          # nunca mejor que el optimo


@pytest.mark.parametrize("k", [3, 4])
def test_kqnodes_kgt2_determinista(tpm_n3, k):
    k1 = _run(tpm_n3, k)
    k2 = _run(tpm_n3, k)
    assert float(k1._resultados_por_k[k]["phi"]) == float(k2._resultados_por_k[k]["phi"])
    assert k1._resultados_por_k[k]["particion"] == k2._resultados_por_k[k]["particion"]
