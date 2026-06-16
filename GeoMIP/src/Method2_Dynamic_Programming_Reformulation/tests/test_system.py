"""Tests unitarios de System: k_bipartir, bipartir y distribucion_marginal.

Test central: k_bipartir con 2 grupos reproduce EXACTAMENTE bipartir (el invariante
maxdiff=0), que es la base de la 'evaluacion correcta de k-particiones'.
"""

import numpy as np
import pytest

from src.funcs.base import emd_efecto


def _k_bipartir_como_biparticion(sub, alcance_fut, mecanismo_pres):
    """Construye los 2 grupos de k_bipartir equivalentes a bipartir(alcance, mecanismo).

    grupo1 = (futuros en alcance, presentes en mecanismo)
    grupo2 = (futuros restantes, presentes restantes)
    """
    F = [int(x) for x in sub.indices_ncubos]
    P = [int(x) for x in sub.dims_ncubos]
    a_f = [f for f in F if f in set(alcance_fut)]
    a_p = [p for p in P if p in set(mecanismo_pres)]
    b_f = [f for f in F if f not in set(alcance_fut)]
    b_p = [p for p in P if p not in set(mecanismo_pres)]
    return [
        (np.array(a_f, dtype=np.int8), np.array(a_p, dtype=np.int8)),
        (np.array(b_f, dtype=np.int8), np.array(b_p, dtype=np.int8)),
    ]


@pytest.mark.parametrize("fixture_name", ["subsistema_n3", "subsistema_n4"])
def test_k_bipartir_dos_grupos_igual_bipartir(fixture_name, request):
    sub, _ = request.getfixturevalue(fixture_name)
    F = [int(x) for x in sub.indices_ncubos]
    P = [int(x) for x in sub.dims_ncubos]

    # Varias biparticiones (primera mitad de futuros / primera mitad de presentes, etc.)
    casos = [
        (F[: len(F) // 2], P[: len(P) // 2]),
        (F[:1], P),
        (F, P[:1]),
        (F[:1], []),
    ]
    for alcance_fut, mecanismo_pres in casos:
        dist_bip = sub.bipartir(
            np.array(alcance_fut, dtype=np.int8),
            np.array(mecanismo_pres, dtype=np.int8),
        ).distribucion_marginal()

        grupos = _k_bipartir_como_biparticion(sub, alcance_fut, mecanismo_pres)
        dist_kbip = sub.k_bipartir(grupos).distribucion_marginal()

        maxdiff = float(np.max(np.abs(dist_bip - dist_kbip)))
        assert maxdiff == pytest.approx(0.0, abs=1e-12), (
            f"k_bipartir != bipartir para alcance={alcance_fut} mecanismo={mecanismo_pres} "
            f"(maxdiff={maxdiff})"
        )


@pytest.mark.parametrize("fixture_name", ["subsistema_n3", "subsistema_n4"])
def test_particion_trivial_tiene_phi_cero(fixture_name, request):
    """Un unico grupo con todos los nodos == subsistema sin cortar => phi = 0."""
    sub, dists = request.getfixturevalue(fixture_name)
    F = np.array([int(x) for x in sub.indices_ncubos], dtype=np.int8)
    P = np.array([int(x) for x in sub.dims_ncubos], dtype=np.int8)
    dist = sub.k_bipartir([(F, P)]).distribucion_marginal()
    assert emd_efecto(dist, dists) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("fixture_name", ["subsistema_n3", "subsistema_n4"])
def test_distribucion_marginal_propiedades(fixture_name, request):
    sub, dists = request.getfixturevalue(fixture_name)
    assert dists.shape[0] == sub.indices_ncubos.size       # un valor por futuro
    assert np.all(dists >= -1e-9) and np.all(dists <= 1 + 1e-9)


@pytest.mark.parametrize("fixture_name", ["subsistema_n3", "subsistema_n4"])
def test_k_bipartir_grupos_con_partes_vacias_no_rompe(fixture_name, request):
    """Grupos con futuros vacios o presentes vacios deben evaluarse sin error."""
    sub, dists = request.getfixturevalue(fixture_name)
    F = [int(x) for x in sub.indices_ncubos]
    P = [int(x) for x in sub.dims_ncubos]
    # grupo1 sin futuros (solo presentes), grupo2 con todos los futuros y sin presentes
    grupos = [
        (np.array([], dtype=np.int8), np.array(P, dtype=np.int8)),
        (np.array(F, dtype=np.int8), np.array([], dtype=np.int8)),
    ]
    dist = sub.k_bipartir(grupos).distribucion_marginal()
    assert dist.shape[0] == len(F)
    assert np.isfinite(emd_efecto(dist, dists))
