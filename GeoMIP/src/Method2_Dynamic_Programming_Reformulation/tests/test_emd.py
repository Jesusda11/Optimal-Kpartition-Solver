"""Tests unitarios de la metrica de perdida emd_efecto (src/funcs/base.py).

emd_efecto(u, v) = sum(|u_i - v_i|), por lo que sus valores son verificables a mano.
"""

import numpy as np
import pytest

from src.funcs.base import emd_efecto


def test_distribuciones_iguales_da_cero():
    u = np.array([0.5, 0.5, 0.2], dtype=np.float32)
    assert emd_efecto(u, u) == pytest.approx(0.0)


def test_valor_a_mano_simple():
    u = np.array([1.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 1.0], dtype=np.float32)
    assert emd_efecto(u, v) == pytest.approx(2.0)


def test_valor_a_mano_asimetrico():
    u = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    v = np.array([0.5, 0.1, 0.5], dtype=np.float32)
    # |0.2-0.5| + |0.8-0.1| + |0.5-0.5| = 0.3 + 0.7 + 0.0 = 1.0
    assert emd_efecto(u, v) == pytest.approx(1.0)


def test_simetria():
    u = np.array([0.1, 0.9, 0.3], dtype=np.float32)
    v = np.array([0.4, 0.4, 0.7], dtype=np.float32)
    assert emd_efecto(u, v) == pytest.approx(emd_efecto(v, u))


def test_no_negatividad():
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = rng.random(5).astype(np.float32)
        v = rng.random(5).astype(np.float32)
        assert emd_efecto(u, v) >= 0.0
