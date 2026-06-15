"""
Configuracion de pytest para la suite de tests unitarios de KGeoMIP (Method2).

- Anade el root del arbol Method2 a sys.path para poder importar `src.*`.
- Silencia SafeLogger y el profiling (tests rapidos y limpios).
- Expone fixtures con TPMs pequenas (N3A, N4A) y subsistemas ya preparados.

Las TPMs se cargan desde GeoMIP/data/samples (no se generan archivos de salida).
Ejecutar desde el root de Method2:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]              # .../Method2_Dynamic_Programming_Reformulation
SAMPLES = ROOT.parent.parent / "data" / "samples"       # GeoMIP/data/samples
sys.path.insert(0, str(ROOT))

# ── Silenciar logging y SafeLogger antes de importar estrategias ──────────────
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

from src.middlewares.profile import profiler_manager
profiler_manager.enabled = False

from src.models.base.application import aplicacion
from src.controllers.manager import Manager
from src.funcs.decay import decay_exponencial


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


def _preparar_subsistema(estado: str, condicion: str, alcance: str, mecanismo: str, tpm):
    """Devuelve (System, dists_marginales) de un subsistema ya preparado.

    Usa KGeoMIPAsimetrico solo como vehiculo para llamar a sia_preparar_subsistema
    (heredado de SIA); no ejecuta ningun algoritmo.
    """
    from src.controllers.strategies.kgeometric_asimetrico import KGeoMIPAsimetrico
    aplicacion.pagina_sample_network = "A"
    sia = KGeoMIPAsimetrico(
        Manager(estado_inicial=estado), k_max=4, m_max_exhaustivo=999,
        decay_fn=decay_exponencial,
    )
    sia.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
    return sia.sia_subsistema, sia.sia_dists_marginales


@pytest.fixture
def subsistema_n3(tpm_n3):
    return _preparar_subsistema("100", "111", "111", "111", tpm_n3)


@pytest.fixture
def subsistema_n4(tpm_n4):
    return _preparar_subsistema("1000", "1111", "1111", "1111", tpm_n4)
