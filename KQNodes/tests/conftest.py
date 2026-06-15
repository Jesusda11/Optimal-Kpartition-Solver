"""
Configuracion de pytest para la suite de tests unitarios de KQNodes.

- Anade el root del arbol KQNodes a sys.path para importar `src.*`.
- Silencia SafeLogger (tests limpios).
- Expone fixtures con TPMs pequenas (N3A, N4A) y subsistemas ya preparados.

Ejecutar desde el root de KQNodes:  ..\\QNodes\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]          # .../KQNodes
SAMPLES = ROOT / "src" / ".samples"
sys.path.insert(0, str(ROOT))

logging.disable(logging.CRITICAL)

import src.middlewares.slogger as _slogger_mod


class _NullSafeLogger:
    def __init__(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def error(self, *a, **k): pass
    def critic(self, *a, **k): pass
    def fatal(self, *a, **k): pass
    def set_log(self, *a, **k): pass


_slogger_mod.SafeLogger = _NullSafeLogger

# Desactivar el profiler (pyinstrument) para tests limpios y rapidos.
from src.middlewares.profile import gestor_perfilado
gestor_perfilado.enabled = False

from src.models.base.application import aplicacion


def _cargar_tpm(n: int, variante: str = "A") -> np.ndarray:
    ruta = SAMPLES / f"N{n}{variante}.csv"
    if not ruta.exists():
        pytest.skip(f"TPM no disponible: {ruta}")
    return np.genfromtxt(ruta, delimiter=",")


@pytest.fixture(scope="session")
def tpm_n3() -> np.ndarray:
    return _cargar_tpm(3, "A")


@pytest.fixture(scope="session")
def tpm_n4() -> np.ndarray:
    return _cargar_tpm(4, "A")


def _preparar_subsistema(estado, condicion, alcance, mecanismo, tpm):
    """Devuelve (System, dists) de un subsistema preparado (sin correr algoritmo)."""
    from src.strategies.kqnodes import KQNodes
    aplicacion.set_pagina_red_muestra("A")
    sia = KQNodes(tpm, k_min=2, k_max=2)
    sia.sia_preparar_subsistema(estado, condicion, alcance, mecanismo)
    return sia.sia_subsistema, sia.sia_dists_marginales


@pytest.fixture
def subsistema_n3(tpm_n3):
    return _preparar_subsistema("100", "111", "111", "111", tpm_n3)


@pytest.fixture
def subsistema_n4(tpm_n4):
    return _preparar_subsistema("1000", "1111", "1111", "1111", tpm_n4)
