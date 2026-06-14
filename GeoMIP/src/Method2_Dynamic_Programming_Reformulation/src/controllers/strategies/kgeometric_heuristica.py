"""
src/controllers/strategies/kgeometric_heuristica.py

KGeoMIPHeuristica: KGeoMIP Simetrico con heuristica geometrica para n > n_max_exhaustivo.

Para n <= n_max_exhaustivo (default 6): enumeracion exacta S(n,k), identico a KGeometricSIA.
Para n > n_max_exhaustivo: genera candidatos guiados por la tabla_transiciones heredada de
GeometricSIA y evalua solo esos, haciendo el algoritmo escalable a n=22 y mas.

Estrategia de generacion de candidatos:
  1. Cortes en ranking: se ordena los n nodos por su costo acumulado en la tabla_transiciones
     y se generan todos los C(n-1, k-1) particiones por cortes en ese ranking.
  2. Candidatos por nivel BFS: para cada nivel del recorrido BFS, se toma el estado de
     menor costo y se genera el mejor corte de ese nivel.

Complejidad heuristica:
  k=2: C(n-1,1) + n/2  = n-1 + n/2  candidatos (~1.5n, igual que GeometricSIA)
  k=3: C(n-1,2) + n/2  candidatos
  k=4: C(n-1,3) + n/2  candidatos

Parametro m_max_candidatos limita el total si C(n-1,k-1) es muy grande.

Alias oficial: KGeoMIPHeuristica = KGeometricSIAHeuristica
"""

import time
from itertools import combinations
from typing import Optional, Callable

import numpy as np

from src.controllers.strategies.kgeometric import KGeometricSIA
from src.controllers.manager import Manager
from src.funcs.base import emd_efecto
from src.funcs.partitions import generar_k_particiones
from src.funcs.format import fmt_k_particion
from src.models.core.solution import Solution
from src.constants.models import (
    KGEOMETRIC_HEURISTICA_LABEL,
    KGEOMETRIC_HEURISTICA_ANALYSIS_TAG,
)
from src.middlewares.profile import profile
from src.constants.base import TYPE_TAG


class KGeometricSIAHeuristica(KGeometricSIA):
    """
    KGeoMIP Simetrico con heuristica geometrica para subsistemas grandes.

    Hereda toda la infraestructura de KGeometricSIA (y GeometricSIA):
    tabla_transiciones, calcular_costos_nivel, _setup_dist_fn, etc.

    Para n <= n_max_exhaustivo usa la enumeracion exacta de KGeometricSIA.
    Para n > n_max_exhaustivo construye tabla_transiciones y genera O(C(n,k))
    candidatos guiados por los costos acumulados.

    Args:
        n_max_exhaustivo: Umbral para usar exacto vs heuristico. Default 6.
        m_max_candidatos: Limite de candidatos por k en modo heuristico. Default 2000.
        Resto: identicos a KGeometricSIA.
    """

    def __init__(
        self,
        gestor: Manager,
        k_max: int = 4,
        k_min: int = 2,
        n_max_exhaustivo: int = 6,
        m_max_candidatos: int = 2000,
        decay_fn: Optional[Callable] = None,
        parallel: bool = True,
        dist_fn=None,
    ):
        super().__init__(
            gestor,
            k_max=k_max,
            decay_fn=decay_fn,
            parallel=parallel,
            dist_fn=dist_fn,
        )
        self.k_min = max(2, k_min)
        self.n_max_exhaustivo = n_max_exhaustivo
        self.m_max_candidatos = m_max_candidatos
        self._modo_usado: str = "exacto"
        self._n_candidatos_evaluados: int = 0
        # Mejor resultado por k: k -> {phi, asignacion, dist, n_candidatos, tiempo_s}
        self._resultados_por_k: dict[int, dict] = {}

    # ── Construccion de tabla_transiciones ────────────────────────────────────

    def _construir_tabla_transiciones(self) -> None:
        """
        Construye tabla_transiciones recorriendo el hipercubo BFS nivel a nivel.
        Equivale a la primera mitad de GeometricSIA.find_mip() sin evaluar phi.
        """
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final   = 1 - self.estado_inicial
        self.idx_ncubos     = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos        = {0: [self.estado_inicial.tolist()]}
        self.tabla_transiciones = {}
        self.tabla_transiciones[
            (tuple(self.caminos[0][0]), tuple(self.caminos[0][0]))
        ] = [0.0] * len(self.sia_subsistema.indices_ncubos)
        for nivel in range(1, len(self.estado_inicial) + 1):
            self.calcular_costos_nivel(self.estado_final, nivel)

    # ── Generacion de candidatos heuristicos ──────────────────────────────────

    def _generar_candidatos_heuristica(
        self, n: int, k: int, comunes: np.ndarray
    ) -> list[list[list[int]]]:
        """
        Genera candidatos de k-particion de {0,...,n-1} guiados por costos.

        Los candidatos usan indices 0..n-1 que se mapean a comunes en el llamador.

        Estrategia 1 — Cortes en ranking final:
          Extrae costos solo para los nodos en `comunes` (descarta huerfanos),
          ordena 0..n-1 por ese costo y genera todas las C(n-1, k-1) particiones
          por cortes posibles en ese ranking.

        Estrategia 2 — Todos los estados de cada nivel BFS:
          Para cada nivel del BFS y cada estado en ese nivel, extrae los costos
          de los nodos comunes y genera todos los C(n-1, k-1) cortes para ese
          estado. Esto aumenta cobertura para n grandes.

        Si el total supera m_max_candidatos se trunca de forma DETERMINISTA
        (orden por clave estable, sin aleatoriedad ni dependencia de PYTHONHASHSEED).
        """
        indices_ncubos = self.sia_subsistema.indices_ncubos
        # Posiciones de comunes dentro de indices_ncubos (para indexar costos)
        comunes_pos = [int(np.where(indices_ncubos == c)[0][0]) for c in comunes]

        def _ranking_desde_costos(costos_all: list) -> list[int]:
            costos_comunes = [costos_all[p] for p in comunes_pos]
            return list(np.argsort(costos_comunes))

        def _agregar_cortes(ranking: list[int]) -> None:
            for cortes in combinations(range(1, n), k - 1):
                grupos: list[frozenset] = []
                prev = 0
                for c in sorted(cortes):
                    grupos.append(frozenset(ranking[prev:c]))
                    prev = c
                grupos.append(frozenset(ranking[prev:]))
                if all(len(g) > 0 for g in grupos):
                    candidatos_set.add(frozenset(grupos))

        def _clave_estable(particion: frozenset) -> tuple:
            # Ordena grupos y elementos dentro de cada grupo; completamente
            # independiente de hash y PYTHONHASHSEED.
            return tuple(sorted(tuple(sorted(g)) for g in particion))

        candidatos_set: set[frozenset] = set()

        # Estrategia 1: cortes en ranking del nivel final
        key_final = (tuple(self.caminos[0][0]), tuple(self.estado_final))
        costos_final = self.tabla_transiciones.get(key_final, [0.0] * len(indices_ncubos))
        _agregar_cortes(_ranking_desde_costos(costos_final))

        # Estrategia 2: todos los estados de cada nivel BFS
        mitad = max(1, len(self.caminos) // 2)
        for nivel in range(1, mitad + 1):
            for estado in self.caminos.get(nivel, []):
                key = (tuple(self.caminos[0][0]), tuple(estado))
                costos_nivel = self.tabla_transiciones.get(key)
                if costos_nivel is not None:
                    _agregar_cortes(_ranking_desde_costos(costos_nivel))

        # Cap determinista: ordenar por clave estable y tomar los primeros N.
        # Sin aleatoriedad: mismo input => mismo subset en cualquier corrida.
        if len(candidatos_set) > self.m_max_candidatos:
            candidatos_set = set(
                sorted(candidatos_set, key=_clave_estable)[: self.m_max_candidatos]
            )

        # Convertir a listas de listas canonicas (grupo con nodo 0 primero)
        result: list[list[list[int]]] = []
        for frozen in candidatos_set:
            grupos = [sorted(list(g)) for g in frozen]
            for i, g in enumerate(grupos):
                if 0 in g:
                    grupos[0], grupos[i] = grupos[i], grupos[0]
                    break
            result.append(grupos)

        return result

    # ── Estrategia principal ──────────────────────────────────────────────────

    @profile(context={TYPE_TAG: KGEOMETRIC_HEURISTICA_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ):
        """
        Encuentra la k-particion optima (minima phi) usando exacto o heuristico.

        Modo exacto  (n <= n_max_exhaustivo): identico a KGeometricSIA.
        Modo heuristico (n > n_max_exhaustivo):
          1. Construir tabla_transiciones via BFS geometrico.
          2. Generar candidatos por cortes en ranking de costos.
          3. Evaluar phi para cada candidato con k_bipartir + emd_efecto.
          4. Retornar la particion de menor phi.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
        self._flat_data = [ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos]
        self._setup_dist_fn()

        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos    = self.sia_subsistema.dims_ncubos
        comunes        = np.intersect1d(indices_ncubos, dims_ncubos)
        n              = len(comunes)
        huerfanos      = np.setdiff1d(indices_ncubos, dims_ncubos)

        if n < 2:
            raise ValueError(
                f"KGeoMIPHeuristica requiere al menos 2 nodos balanceados; "
                f"se encontraron {n}."
            )

        mejor_phi: float = float("inf")
        mejor_asignacion: list[list[int]] = []
        mejor_dist: Optional[np.ndarray] = None
        total_evaluados: int = 0

        self._resultados_por_k = {}

        if n <= self.n_max_exhaustivo:
            # ── Modo exacto: mismo flujo que KGeometricSIA ────────────────
            self._modo_usado = "exacto"
            for k in range(self.k_min, min(self.k_max, n) + 1):
                mejor_phi_k: float = float("inf")
                mejor_asig_k: list = []
                mejor_dist_k = None
                n_cand_k = 0
                t0_k = time.time()
                for asignacion in generar_k_particiones(n, k):
                    grupos = []
                    for grupo in asignacion:
                        arr = np.array(grupo, dtype=np.int8)
                        grupos.append((comunes[arr], comunes[arr]))
                    if huerfanos.size > 0:
                        grupos.append((huerfanos, dims_ncubos))
                    dist = self.sia_subsistema.k_bipartir(grupos).distribucion_marginal()
                    phi = emd_efecto(dist, self.sia_dists_marginales)
                    total_evaluados += 1
                    n_cand_k += 1
                    if phi < mejor_phi_k:
                        mejor_phi_k = phi
                        mejor_asig_k = asignacion
                        mejor_dist_k = dist
                    if phi < mejor_phi:
                        mejor_phi = phi
                        mejor_asignacion = asignacion
                        mejor_dist = dist
                self._resultados_por_k[k] = {
                    "phi": mejor_phi_k,
                    "asignacion": mejor_asig_k,
                    "dist": mejor_dist_k,
                    "n_candidatos": n_cand_k,
                    "tiempo_s": round(time.time() - t0_k, 6),
                }

        else:
            # ── Modo heuristico ───────────────────────────────────────────
            self._modo_usado = "heuristico"
            self._construir_tabla_transiciones()

            for k in range(self.k_min, min(self.k_max, n) + 1):
                mejor_phi_k: float = float("inf")
                mejor_asig_k: list = []
                mejor_dist_k = None
                t0_k = time.time()
                candidatos = self._generar_candidatos_heuristica(n, k, comunes)
                for asignacion in candidatos:
                    grupos = []
                    for grupo in asignacion:
                        arr = np.array(grupo, dtype=np.int8)
                        grupos.append((comunes[arr], comunes[arr]))
                    if huerfanos.size > 0:
                        grupos.append((huerfanos, dims_ncubos))
                    dist = self.sia_subsistema.k_bipartir(grupos).distribucion_marginal()
                    phi = emd_efecto(dist, self.sia_dists_marginales)
                    total_evaluados += 1
                    if phi < mejor_phi_k:
                        mejor_phi_k = phi
                        mejor_asig_k = asignacion
                        mejor_dist_k = dist
                    if phi < mejor_phi:
                        mejor_phi = phi
                        mejor_asignacion = asignacion
                        mejor_dist = dist
                self._resultados_por_k[k] = {
                    "phi": mejor_phi_k,
                    "asignacion": mejor_asig_k,
                    "dist": mejor_dist_k,
                    "n_candidatos": len(candidatos),
                    "tiempo_s": round(time.time() - t0_k, 6),
                }

        self._n_candidatos_evaluados = total_evaluados

        # Formatear resultado
        grupos_fmt = []
        for grupo in mejor_asignacion:
            arr = np.array(grupo, dtype=int)
            nodos = [int(x) for x in comunes[arr]]
            grupos_fmt.append((nodos, nodos))
        if huerfanos.size > 0:
            grupos_fmt.append(
                ([int(x) for x in huerfanos], [int(x) for x in dims_ncubos])
            )

        return Solution(
            estrategia=KGEOMETRIC_HEURISTICA_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_k_particion(grupos_fmt),
        )


# Alias oficial
KGeoMIPHeuristica = KGeometricSIAHeuristica
