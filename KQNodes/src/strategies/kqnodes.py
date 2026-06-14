"""
src/strategies/kqnodes.py

KQNodes — extension de la estrategia QNodes (algoritmo de Queyranne) al caso de
k-particiones con k >= 2. Nomenclatura oficial del proyecto (Manual Tecnico/Usuario
KQMIP): la clase principal de "QNodes K-particiones" debe llamarse exactamente
`KQNodes`.

ESTADO: Fase 1 (caso base k=2). KQNodes hereda de QNodes y ENVUELVE su motor de
Queyranne sin modificarlo: reutiliza tal cual `algorithm`, `funcion_submodular`,
`definir_clave` y `nodes_complement`. Para k=2, `aplicar_estrategia` ejecuta el
mismo Queyranne que QNodes (misma construccion de vertices, misma llamada a
`self.algorithm`) y solo cambia la etiqueta de la Solution a KQNODES_LABEL y
registra el resultado en `_resultados_por_k`. Por construccion el resultado es
identico al de QNodes; la Fase 1 lo valida empiricamente (phi y biparticion).

La extension a k>2 (Fase 2) sera JERARQUICA por biparticiones sucesivas usando el
mismo motor como oraculo de 2-particion; por ahora k>2 lanza NotImplementedError.

Determinismo: el motor de Queyranne es deterministico (sin random; los desempates
toman el primer indice en orden; `fmt_parte_q` ordena por indice -> string
canonico). No se introduce ninguna fuente de aleatoriedad en KQNodes.
"""

import time

import numpy as np

from src.strategies.q_nodes import QNodes
from src.funcs.format import fmt_biparticion_q
from src.models.core.solution import Solution
from src.middlewares.slogger import SafeLogger
from src.constants.models import (
    KQNODES_LABEL,
    KQNODES_STRAREGY_TAG,
)
from src.constants.base import ACTUAL, EFFECT


class KQNodes(QNodes):
    """
    Estrategia KQNodes (QNodes extendido a k-particiones). Hereda de QNodes.

    Args:
        tpm (np.ndarray): Matriz de Probabilidad de Transicion del sistema completo
            (convencion del arbol: SIA(tpm)).
        k_min (int): k minimo a evaluar (>= 2). Default 2.
        k_max (int): k maximo a evaluar. Default 2 (Fase 1: solo k=2).

    Atributos:
        _resultados_por_k (dict[int, dict]): por cada k, {phi, particion, dist, modo_usado}.
        _modo_usado (str): identificador del modo de busqueda empleado.
    """

    def __init__(self, tpm: np.ndarray, k_min: int = 2, k_max: int = 2):
        super().__init__(tpm)
        self.k_min: int = max(2, k_min)
        self.k_max: int = k_max
        self.logger = SafeLogger(KQNODES_STRAREGY_TAG)

        self._resultados_por_k: dict[int, dict] = {}
        self._modo_usado: str = "pendiente"

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        """
        Punto de entrada (misma firma que QNodes de este arbol).

        Fase 1: solo k=2 (biparticion exacta via Queyranne). Si k_max > 2 se lanza
        NotImplementedError porque la extension jerarquica k>2 es Fase 2.
        """
        if self.k_max > 2:
            raise NotImplementedError(
                "KQNodes Fase 1 solo implementa k=2. La extension k>2 "
                "(biparticiones sucesivas) es Fase 2."
            )

        biparticion = self._resolver_biparticion(
            estado_inicial, condicion, alcance, mecanismo
        )
        mip, perdida, dist, fmt_mip = biparticion

        self._modo_usado = "queyranne-k2"
        self._resultados_por_k = {
            2: {
                "phi": perdida,
                "particion": fmt_mip,
                "dist": dist,
                "modo_usado": self._modo_usado,
            }
        }

        return Solution(
            estrategia=KQNODES_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def _resolver_biparticion(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        """
        Ejecuta el motor de Queyranne heredado sobre el subsistema completo y
        devuelve la biparticion optima.

        Replica la construccion de vertices de QNodes.aplicar_estrategia (para
        capturar la clave `mip` exacta que retorna `self.algorithm`, lo que el
        camino de early-return del motor impide re-derivar a posteriori) y reusa
        `self.algorithm` / `self.nodes_complement` sin modificarlos.

        Returns:
            tuple: (mip, perdida, dist_marginal, particion_formateada)
                - mip: clave de un lado de la biparticion (vertices (tiempo,indice)).
                - perdida: phi de la biparticion.
                - dist_marginal: distribucion marginal de la particion.
                - particion_formateada: string canonico (fmt_biparticion_q).
        """
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        futuro = tuple(
            (EFFECT, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )

        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)

        mip = self.algorithm(vertices)

        fmt_mip = fmt_biparticion_q(list(mip), self.nodes_complement(mip))
        perdida_mip, dist_marginal_mip = self.memoria_grupo_candidato[mip]

        return mip, perdida_mip, dist_marginal_mip, fmt_mip
