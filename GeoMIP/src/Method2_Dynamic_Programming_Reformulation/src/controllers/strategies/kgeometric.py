"""
src/controllers/strategies/kgeometric.py

KGeometricSIA: extension de GeometricSIA a k-particiones (k en {2, ..., k_max}).

Cada nodo j asignado al grupo i aporta su variable futura Y su variable presente
al mismo grupo (definicion completa S(n, k), numeros de Stirling del 2do tipo).

Diferencia clave con GeometricSIA (biparticion):
  - GeometricSIA: una parte ve TODOS los presentes, la otra ve NINGUNO.
  - KGeometricSIA: cada parte ve solo sus propios presentes (cut simetrico).

Complejidad: sum_{k=2}^{k_max} S(n,k) evaluaciones exactas de EMD.
Para n<=6 y k_max=4 esto es manejable. Para n>6 se planea heuristica (Paso 2).

La tabla_transiciones de GeometricSIA se construye internamente y queda
disponible como self.tabla_transiciones para la futura heuristica de Paso 2.
"""

import time
from typing import Optional, Callable

import numpy as np

from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.manager import Manager
from src.funcs.base import emd_efecto
from src.funcs.partitions import generar_k_particiones, contar_stirling
from src.funcs.format import fmt_k_particion
from src.models.core.solution import Solution
from src.constants.models import KGEOMETRIC_LABEL, KGEOMETRIC_ANALYSIS_TAG
from src.middlewares.profile import profile
from src.constants.base import TYPE_TAG


class KGeometricSIA(GeometricSIA):
    """
    Extension de GeometricSIA a k-particiones (k = 2, ..., k_max).

    Particiona los n nodos del subsistema en k grupos no vacios. La asignacion
    es en forma canonica (restricted growth strings): el nodo 0 siempre
    pertenece al grupo 0, eliminando duplicados por permutacion de etiquetas.

    El numero total de candidatos evaluados es sum_{k=2}^{k_max} S(n,k).

    Args:
        gestor:   Manager con el estado inicial y pagina de la red.
        k_max:    Maximo numero de grupos (inclusive). Default 4.
        decay_fn: Funcion de decrecimiento gamma (heredada de GeometricSIA).
        parallel: Paralelizacion BFS interna (heredada de GeometricSIA).
        dist_fn:  Metrica de distancia para gamma (heredada de GeometricSIA).
    """

    def __init__(
        self,
        gestor: Manager,
        k_max: int = 4,
        decay_fn: Optional[Callable] = None,
        parallel: bool = True,
        dist_fn=None,
    ):
        super().__init__(gestor, decay_fn=decay_fn, parallel=parallel, dist_fn=dist_fn)
        self.k_max = k_max

    @profile(context={TYPE_TAG: KGEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ):
        """
        Encuentra la k-particion optima (minima perdida phi) para k en {2,...,k_max}.

        Flujo:
          1. Preparar subsistema (condicion, alcance, mecanismo, tpm).
          2. Calcular nodos balanceados (presentes en alcance Y mecanismo) y futuros
             huerfanos (en alcance pero no en mecanismo).
          3. Enumerar todas las k-particiones de los nodos balanceados.
          4. Para cada particion: k_bipartir -> distribucion_marginal -> emd_efecto.
             Los futuros huerfanos se incluyen como grupo fijo sin restriccion de presentes.
          5. Retornar la particion con menor phi.

        Subsistemas no balanceados (|alcance| != |mecanismo|):
          Los nodos que aparecen solo en el alcance (futuros huerfanos) no tienen
          variable presente propia. Se tratan como un grupo fijo que condiciona
          sobre todos los presentes disponibles (sin corte). Esto garantiza que
          k_bipartir nunca reciba un indice fuera de rango.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        self._flat_data = [ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos]
        self._setup_dist_fn()

        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos = self.sia_subsistema.dims_ncubos

        # Nodos con variable futura Y presente: sobre estos se genera la particion S(n,k).
        comunes = np.intersect1d(indices_ncubos, dims_ncubos)
        n = len(comunes)

        # Futuros sin presente propio: no se particionan, ven todos los presentes.
        huerfanos_futuro = np.setdiff1d(indices_ncubos, dims_ncubos)

        if n < 2:
            raise ValueError(
                f"KGeometricSIA requiere al menos 2 nodos balanceados "
                f"(en alcance Y mecanismo); se encontraron {n}."
            )

        mejor_phi: float = float("inf")
        mejor_asignacion: list[list[int]] = []
        mejor_dist: Optional[np.ndarray] = None

        for k in range(2, min(self.k_max, n) + 1):
            for asignacion in generar_k_particiones(n, k):
                grupos = []
                for grupo in asignacion:
                    arr = np.array(grupo, dtype=np.int8)
                    grupos.append((comunes[arr], comunes[arr]))

                # Futuros huerfanos: condicionan sobre todos los presentes (sin restriccion).
                if huerfanos_futuro.size > 0:
                    grupos.append((huerfanos_futuro, dims_ncubos))

                dist = self.sia_subsistema.k_bipartir(grupos).distribucion_marginal()
                phi = emd_efecto(dist, self.sia_dists_marginales)

                if phi < mejor_phi:
                    mejor_phi = phi
                    mejor_asignacion = asignacion
                    mejor_dist = dist

        grupos_fmt = []
        for grupo in mejor_asignacion:
            arr = np.array(grupo, dtype=int)
            nodos = [int(x) for x in comunes[arr]]
            grupos_fmt.append((nodos, nodos))
        if huerfanos_futuro.size > 0:
            grupos_fmt.append(
                ([int(x) for x in huerfanos_futuro], [int(x) for x in dims_ncubos])
            )

        return Solution(
            estrategia=KGEOMETRIC_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_k_particion(grupos_fmt),
        )

    def get_stirling_counts(self) -> dict[int, int]:
        """
        Retorna el numero de k-particiones a evaluar por cada k.
        Requiere haber llamado sia_preparar_subsistema primero.
        """
        n = len(self.sia_subsistema.indices_ncubos)
        return {k: contar_stirling(n, k) for k in range(2, min(self.k_max, n) + 1)}


# Alias oficial conforme a la nomenclatura del proyecto K-QGMIP.
# KGeoMIP y KGeometricSIA son exactamente la misma clase; ambos nombres
# pueden usarse indistintamente en imports y en el manual.
KGeoMIP = KGeometricSIA
