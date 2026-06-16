"""Consistencia k=2 de KQNodes: reproduce QNodes original bit a bit (phi + particion).

Es el gate de Fase 1 expresado como test unitario formal.
"""

import pytest

from src.strategies.kqnodes import KQNodes
from src.strategies.q_nodes import QNodes
from src.models.base.application import aplicacion

# (estado_inicial, condicion, alcance, mecanismo)
CASOS = {
    "n3": ("100", "111", "111", "111"),
    "n4": ("1000", "1111", "1111", "1111"),
}


def _kqnodes_k2(tpm, caso):
    aplicacion.set_pagina_red_muestra("A")
    return KQNodes(tpm, k_min=2, k_max=2).aplicar_estrategia(*caso)


def _qnodes(tpm, caso):
    aplicacion.set_pagina_red_muestra("A")
    return QNodes(tpm).aplicar_estrategia(*caso)


@pytest.mark.parametrize("fixture_name,key", [("tpm_n3", "n3"), ("tpm_n4", "n4")])
def test_kqnodes_k2_reproduce_qnodes(fixture_name, key, request):
    tpm = request.getfixturevalue(fixture_name)
    caso = CASOS[key]
    sol_kq = _kqnodes_k2(tpm, caso)
    sol_qn = _qnodes(tpm, caso)
    assert float(sol_kq.perdida) == float(sol_qn.perdida)   # phi identico (exacto)
    assert sol_kq.particion == sol_qn.particion             # misma biparticion


@pytest.mark.parametrize("fixture_name,key", [("tpm_n3", "n3"), ("tpm_n4", "n4")])
def test_kqnodes_k2_determinista(fixture_name, key, request):
    tpm = request.getfixturevalue(fixture_name)
    caso = CASOS[key]
    s1 = _kqnodes_k2(tpm, caso)
    s2 = _kqnodes_k2(tpm, caso)
    assert float(s1.perdida) == float(s2.perdida)
    assert s1.particion == s2.particion
