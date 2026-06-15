"""Tests unitarios de los generadores de k-particiones (src/funcs/partitions.py)."""

import pytest

from src.funcs.partitions import generar_k_particiones, contar_stirling


# ── contar_stirling vs valores conocidos ──────────────────────────────────────

@pytest.mark.parametrize("n,k,esperado", [
    (4, 2, 7), (5, 3, 25), (6, 4, 65), (8, 4, 1701),
    (3, 2, 3), (4, 3, 6), (6, 2, 31), (6, 3, 90),
])
def test_stirling_valores_conocidos(n, k, esperado):
    assert contar_stirling(n, k) == esperado


@pytest.mark.parametrize("n,k,esperado", [
    (0, 0, 1),   # convencion S(0,0)=1
    (5, 0, 0),   # k=0 con n>0
    (3, 5, 0),   # k>n
    (5, 1, 1),   # un solo grupo
    (5, 5, 1),   # todos singletons
])
def test_stirling_casos_borde(n, k, esperado):
    assert contar_stirling(n, k) == esperado


# ── generar_k_particiones ─────────────────────────────────────────────────────

@pytest.mark.parametrize("n,k", [(3, 2), (4, 2), (4, 3), (5, 3), (6, 4), (5, 5)])
def test_conteo_coincide_con_stirling(n, k):
    particiones = list(generar_k_particiones(n, k))
    assert len(particiones) == contar_stirling(n, k)


@pytest.mark.parametrize("n,k", [(3, 2), (4, 3), (5, 3), (6, 4)])
def test_particiones_cubren_y_son_disjuntas(n, k):
    for particion in generar_k_particiones(n, k):
        assert len(particion) == k
        assert all(len(g) > 0 for g in particion)              # k grupos no vacios
        plano = sorted(x for grupo in particion for x in grupo)
        assert plano == list(range(n))                          # cubre {0..n-1} sin solapar


@pytest.mark.parametrize("n,k", [(4, 2), (4, 3), (5, 3), (6, 3)])
def test_sin_duplicados(n, k):
    particiones = list(generar_k_particiones(n, k))
    claves = {frozenset(frozenset(g) for g in p) for p in particiones}
    assert len(claves) == len(particiones)


@pytest.mark.parametrize("n,k", [(4, 2), (5, 3), (6, 4)])
def test_forma_canonica_elemento0_en_grupo0(n, k):
    for particion in generar_k_particiones(n, k):
        assert 0 in particion[0]


def test_k_mayor_que_n_vacio():
    assert list(generar_k_particiones(3, 5)) == []


def test_k_igual_n_es_todo_singletons():
    parts = list(generar_k_particiones(4, 4))
    assert len(parts) == 1
    assert sorted(len(g) for g in parts[0]) == [1, 1, 1, 1]
