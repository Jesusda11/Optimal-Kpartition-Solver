"""
src/controllers/strategies/kgeometric_asimetrico.py

KGeometricSIAAsimetrico: k-MIP asimetrico sobre las 2n variables independientes.

== Contexto ==

Este modulo implementa el segundo de los dos enfoques de k-particion del proyecto
K-QGMIP. El primero (KGeometricSIA / kgeometric.py) es el enfoque SIMETRICO: cada
nodo j lleva su futuro J(t+1) y su presente j(t) al mismo grupo, generando
S(n,k) candidatos sobre n nodos.

Este modulo implementa el enfoque ASIMETRICO: se tratan las n variables futuras
{J0(t+1),...,Jn-1(t+1)} y las n variables presentes {j0(t),...,jn-1(t)} como 2n
entidades INDEPENDIENTES. Una k-particion asimetrica asigna cada variable a un
grupo libremente — el futuro de un nodo y el presente del mismo nodo pueden ir a
grupos distintos. Esto genera S(m,k) candidatos donde m = n_fut + n_pres = 2n
(para sistemas balanceados).

Ejemplos de crecimiento:
  n=3  m=6:  S(6,2)=31    vs S(3,2)=3   simetrico
  n=4  m=8:  S(8,2)=127   vs S(4,2)=7
  n=5  m=10: S(10,2)=511  vs S(5,2)=15
  n=10 m=20: S(20,2)=524287 vs S(10,2)=511

Relacion con GeometricSIA (biparticion asimetrica original):
  GeometricSIA usa biparticiones en las que un grupo de futuros ve TODOS los
  presentes y el otro ve NINGUNO. Eso es un caso especial de nuestras S(2n,2)
  biparticiones donde los presentes se distribuyen 0/all. KGeoMIPAsimetrico con
  k=2 explora TODAS las biparticiones de 2n variables, por lo que puede encontrar
  phi <= phi_geo (la misma o mejor solucion). Si en algun caso phi_asim > phi_geo,
  hay un bug.

Herencia: KGeometricSIAAsimetrico -> KGeometricSIAHeuristica -> KGeometricSIA
  -> GeometricSIA -> SIA

  Se hereda de KGeometricSIAHeuristica para reutilizar:
    - _construir_tabla_transiciones(): BFS sobre el hipercubo de presentes
    - self.caminos, self.tabla_transiciones: infraestructura de costos geometricos
    - calcular_costos_nivel() / calcular_costo(): calculo del costo de transicion
  Se SOBREESCRIBE completamente aplicar_estrategia() para operar sobre 2n variables.

Atributos propios:
    m_max_exhaustivo: umbral de tamano del pool (m=n_fut+n_pres) para usar
                      enumeracion exacta. Para m>umbral usa heuristica.
    _resultados_por_k: dict k -> {phi, grupos, dist, n_candidatos, tiempo_s}
                       donde 'grupos' es lista de (futuros_i, presentes_i)
                       con indices reales de nodos (listos para k_bipartir y fmt).
    _modo_usado: "exacto" o "heuristico"
"""

import time
from itertools import combinations
from typing import Optional, Callable

import numpy as np

from src.controllers.strategies.kgeometric_heuristica import KGeometricSIAHeuristica
from src.controllers.manager import Manager
from src.funcs.base import emd_efecto
from src.funcs.partitions import generar_k_particiones
from src.funcs.format import fmt_k_particion
from src.models.core.solution import Solution
from src.constants.models import (
    KGEOMETRIC_ASIMETRICO_LABEL,
    KGEOMETRIC_ASIMETRICO_ANALYSIS_TAG,
)
from src.middlewares.profile import profile
from src.constants.base import TYPE_TAG


class KGeometricSIAAsimetrico(KGeometricSIAHeuristica):
    """
    KGeoMIP Asimetrico: k-particion de las 2n variables (futuros + presentes)
    tratadas como entidades independientes.

    A diferencia del enfoque simetrico, aqui el futuro J(t+1) y el presente j(t)
    de un mismo nodo j pueden quedar en grupos distintos. Esto genera S(m,k)
    candidatos donde m = len(indices_ncubos) + len(dims_ncubos).

    Para m <= m_max_exhaustivo: enumeracion exacta de todos los S(m,k) candidatos.
    Para m > m_max_exhaustivo: heuristica geometrica basada en tabla_transiciones.

    La heuristica combina costos de variables futuras (extraidos directamente de
    tabla_transiciones) con costos de variables presentes (calculados como
    sensibilidad promedio de los futuros a cada presente via datos del n-cubo).
    Los 2n costos se ordenan y se generan candidatos por cortes en ese ranking.

    Args:
        m_max_exhaustivo: Umbral de tamano del pool m=n_fut+n_pres para exacto.
                          Default 8 (m=8 => n=4, S(8,3)~966, manejable).
        m_max_candidatos: Limite de candidatos por k en modo heuristico. Default 2000.
        Resto: identicos a KGeometricSIAHeuristica.
    """

    def __init__(
        self,
        gestor: Manager,
        k_max: int = 4,
        k_min: int = 2,
        m_max_exhaustivo: int = 8,
        m_max_candidatos: int = 2000,
        decay_fn: Optional[Callable] = None,
        parallel: bool = True,
        dist_fn=None,
    ):
        super().__init__(
            gestor,
            k_max=k_max,
            k_min=k_min,
            n_max_exhaustivo=m_max_exhaustivo,
            m_max_candidatos=m_max_candidatos,
            decay_fn=decay_fn,
            parallel=parallel,
            dist_fn=dist_fn,
        )
        self.m_max_exhaustivo = m_max_exhaustivo

    # ── Conversiones de representacion ────────────────────────────────────────

    def _particion_pool_a_grupos(
        self,
        particion: list[list[int]],
        indices_ncubos: np.ndarray,
        dims_ncubos: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Convierte una particion de pool-indices a lista de (futuros, presentes).

        El pool codifica las 2n variables como:
          pool[0..n_fut-1]       -> indices_ncubos (variables futuras)
          pool[n_fut..m-1]       -> dims_ncubos    (variables presentes)

        Cada grupo de pool-indices se divide en futures e indices de dims.
        Los arrays resultantes contienen indices reales de nodos (no posiciones
        en el pool), listos para pasarse a k_bipartir y fmt_k_particion.

        Si un grupo tiene futuros vacios: k_bipartir lo maneja (ninguna caja
        se asigna a ese grupo; sus presentes no son vistos por ningun futuro).
        Si un grupo tiene presentes vacios: esa caja marginaliza sobre todos
        sus dims actuales (equivale a "ve ningun presente").
        """
        n_fut = len(indices_ncubos)
        grupos = []
        for grupo_pool in particion:
            fut_pos = [v for v in grupo_pool if v < n_fut]
            pres_pos = [v - n_fut for v in grupo_pool if v >= n_fut]
            futuros = (
                indices_ncubos[np.array(fut_pos, dtype=int)]
                if fut_pos
                else np.array([], dtype=np.int8)
            )
            presentes = (
                dims_ncubos[np.array(pres_pos, dtype=int)]
                if pres_pos
                else np.array([], dtype=np.int8)
            )
            grupos.append((futuros, presentes))
        return grupos

    # ── Costos de variables presentes ─────────────────────────────────────────

    def _costos_presentes(self) -> list[float]:
        """
        Calcula el costo de cada variable presente como la sensibilidad promedio
        de los futuros a dicha variable.

        Para la variable presente p (posicion axis en dims_ncubos):
          costo_p = mean_j |diff_axis P(J_j=0 | V_t=*)|

        Donde diff_axis es la diferencia al variar p de 0 a 1, promediada
        sobre todas las combinaciones de los demas presentes (np.diff(...).mean()).
        Promedia luego sobre todos los futuros j.

        Estos costos se usan en el modo heuristico para rankear las n_pres
        variables presentes junto con las n_fut futuras (que tienen costos
        directos de tabla_transiciones).
        """
        dims_ncubos = self.sia_subsistema.dims_ncubos
        costos = []
        for p_idx, p in enumerate(dims_ncubos):
            axis = int(np.where(dims_ncubos == p)[0][0])
            total = sum(
                float(np.abs(np.diff(cube.data, axis=axis)).mean())
                for cube in self.sia_subsistema.ncubos
            )
            costos.append(total / len(self.sia_subsistema.ncubos))
        return costos

    # ── Generacion de candidatos heuristicos ──────────────────────────────────

    def _generar_candidatos_asimetrico(
        self,
        m: int,
        k: int,
        indices_ncubos: np.ndarray,
        dims_ncubos: np.ndarray,
        costos_presentes: list[float],
    ) -> list[list[list[int]]]:
        """
        Genera candidatos de k-particion del pool de m variables guiados por costos.

        Separa estrictamente dos conjuntos:

        candidatos_obligatorios — singleton cuts (solo k=2).
          Siempre se evaluan en su totalidad. NUNCA entran al conteo del cap.
          Son ~2*n_fut candidatos: aislar un futuro solo, o un nodo completo
          (futuro+presente), del resto del sistema.

        candidatos_opcionales — estrategias 1 y 2 (cortes en rankings de costos).
          Solo a este conjunto se le aplica el cap m_max_candidatos.
          El truncado es DETERMINISTA: se ordenan por clave estructural estable
          (no depende de hash ni PYTHONHASHSEED) y se toman los primeros N.
          No se usa random ni ninguna semilla.

        Conjunto final evaluado = candidatos_obligatorios
                                  U primeros m_max_candidatos de candidatos_opcionales.

        Los candidatos se representan como listas de grupos de pool-indices
        (0..m-1), canonicos con el pool-indice 0 en el primer grupo.
        """
        n_fut = len(indices_ncubos)

        def _ranking_combined(costos_fut: list, costos_pres: list) -> list[int]:
            costos_all = list(costos_fut) + list(costos_pres)
            return list(np.argsort(costos_all))

        def _agregar_cortes(ranking: list[int], destino: set) -> None:
            for cortes in combinations(range(1, m), k - 1):
                grupos: list[frozenset] = []
                prev = 0
                for c in sorted(cortes):
                    grupos.append(frozenset(ranking[prev:c]))
                    prev = c
                grupos.append(frozenset(ranking[prev:]))
                if all(len(g) > 0 for g in grupos):
                    destino.add(frozenset(grupos))

        def _clave_estable(particion: frozenset) -> tuple:
            # Ordena grupos y elementos dentro de cada grupo; completamente
            # independiente de hash y PYTHONHASHSEED.
            return tuple(sorted(tuple(sorted(g)) for g in particion))

        # ── Candidatos garantizados (singletons, k=2) ─────────────────────────
        candidatos_obligatorios: set[frozenset] = set()
        if k == 2:
            todos = frozenset(range(m))
            nodo_a_pool_pres: dict[int, int] = {
                int(dims_ncubos[p]): n_fut + p for p in range(len(dims_ncubos))
            }
            for j in range(n_fut):
                singleton_fut = frozenset({j})
                if len(singleton_fut) < m:
                    candidatos_obligatorios.add(
                        frozenset({singleton_fut, todos - singleton_fut})
                    )
                nodo_real = int(indices_ncubos[j])
                if nodo_real in nodo_a_pool_pres:
                    singleton_nodo = frozenset({j, nodo_a_pool_pres[nodo_real]})
                    if len(singleton_nodo) < m:
                        candidatos_obligatorios.add(
                            frozenset({singleton_nodo, todos - singleton_nodo})
                        )

        # ── Candidatos opcionales (estrategias 1 y 2) ─────────────────────────
        candidatos_opcionales: set[frozenset] = set()

        key_final = (tuple(self.caminos[0][0]), tuple(self.estado_final))
        costos_fut_final = self.tabla_transiciones.get(key_final, [0.0] * n_fut)
        _agregar_cortes(
            _ranking_combined(costos_fut_final, costos_presentes), candidatos_opcionales
        )

        mitad = max(1, len(self.caminos) // 2)
        for nivel in range(1, mitad + 1):
            for estado in self.caminos.get(nivel, []):
                key = (tuple(self.caminos[0][0]), tuple(estado))
                costos_fut_nivel = self.tabla_transiciones.get(key)
                if costos_fut_nivel is not None:
                    _agregar_cortes(
                        _ranking_combined(costos_fut_nivel, costos_presentes),
                        candidatos_opcionales,
                    )

        # Excluir de opcionales los que ya son obligatorios (evita doble evaluacion)
        candidatos_opcionales -= candidatos_obligatorios

        # Cap determinista: ordenar por clave estable y tomar los primeros N.
        # Sin aleatoriedad: mismo input => mismo subset en cualquier corrida.
        if len(candidatos_opcionales) > self.m_max_candidatos:
            candidatos_opcionales = set(
                sorted(candidatos_opcionales, key=_clave_estable)[: self.m_max_candidatos]
            )

        # ── Conjunto final ─────────────────────────────────────────────────────
        candidatos_final = candidatos_obligatorios | candidatos_opcionales

        # Convertir a listas canonicas (pool-indice 0 en primer grupo)
        result: list[list[list[int]]] = []
        for frozen in candidatos_final:
            grupos = [sorted(list(g)) for g in frozen]
            for i, g in enumerate(grupos):
                if 0 in g:
                    grupos[0], grupos[i] = grupos[i], grupos[0]
                    break
            result.append(grupos)

        return result

    # ── Estrategia principal ──────────────────────────────────────────────────

    @profile(context={TYPE_TAG: KGEOMETRIC_ASIMETRICO_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ):
        """
        Encuentra la k-particion asimetrica optima (minima phi) para k en {k_min,...,k_max}.

        Flujo exacto (m <= m_max_exhaustivo):
          1. Construir pool de m = n_fut + n_pres variables.
          2. Enumerar todos los S(m,k) candidatos con generar_k_particiones(m,k).
          3. Para cada candidato: convertir pool -> grupos (futuros,presentes),
             llamar k_bipartir, calcular phi con emd_efecto.
          4. Retornar la particion de menor phi global.

        Flujo heuristico (m > m_max_exhaustivo):
          1. Construir tabla_transiciones via BFS (mismo que KGeoMIPHeuristica).
          2. Calcular costos de variables presentes (_costos_presentes).
          3. Generar candidatos por cortes en ranking combinado de 2n costos.
          4. Evaluar phi para cada candidato.

        Nota sobre grupos con futuros vacios:
          k_bipartir maneja correctamente grupos con futuros=[]: itera sobre
          todas las cajas y cada una encuentra su entrada en presentes_por_indice
          (ya que la particion S(m,k) cubre todos los pool-indices incluyendo los
          n_fut futuros). Los presentes de ese grupo simplemente no son vistos
          por ninguna caja.

        _resultados_por_k[k]["grupos"] almacena la lista de (futuros_i, presentes_i)
        con indices reales de nodos, lista para formateado con fmt_k_particion.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
        self._flat_data = [ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos]
        self._setup_dist_fn()

        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos = self.sia_subsistema.dims_ncubos
        n_fut = len(indices_ncubos)
        n_pres = len(dims_ncubos)
        m = n_fut + n_pres

        if m < 2:
            raise ValueError(
                f"KGeoMIPAsimetrico requiere m >= 2 (n_fut={n_fut}, n_pres={n_pres})."
            )

        mejor_phi: float = float("inf")
        mejor_grupos: list = []
        mejor_dist = None
        self._resultados_por_k = {}

        if m <= self.m_max_exhaustivo:
            # ── Modo exacto: enumerar todos los S(m,k) candidatos ─────────────
            self._modo_usado = "exacto"
            for k in range(self.k_min, min(self.k_max, m) + 1):
                mejor_phi_k: float = float("inf")
                mejor_grupos_k: list = []
                mejor_dist_k = None
                n_cand_k = 0
                t0_k = time.time()

                for particion in generar_k_particiones(m, k):
                    grupos = self._particion_pool_a_grupos(
                        particion, indices_ncubos, dims_ncubos
                    )
                    dist = self.sia_subsistema.k_bipartir(grupos).distribucion_marginal()
                    phi = emd_efecto(dist, self.sia_dists_marginales)
                    n_cand_k += 1

                    if phi < mejor_phi_k:
                        mejor_phi_k = phi
                        mejor_grupos_k = grupos
                        mejor_dist_k = dist
                    if phi < mejor_phi:
                        mejor_phi = phi
                        mejor_grupos = grupos
                        mejor_dist = dist

                self._resultados_por_k[k] = {
                    "phi": mejor_phi_k,
                    "grupos": mejor_grupos_k,
                    "dist": mejor_dist_k,
                    "n_candidatos": n_cand_k,
                    "tiempo_s": round(time.time() - t0_k, 6),
                }

        else:
            # ── Modo heuristico: costos combinados de 2n variables ────────────
            self._modo_usado = "heuristico"
            self._construir_tabla_transiciones()
            costos_pres = self._costos_presentes()

            for k in range(self.k_min, min(self.k_max, m) + 1):
                mejor_phi_k: float = float("inf")
                mejor_grupos_k: list = []
                mejor_dist_k = None
                t0_k = time.time()

                candidatos_pool = self._generar_candidatos_asimetrico(
                    m, k, indices_ncubos, dims_ncubos, costos_pres
                )

                for particion in candidatos_pool:
                    grupos = self._particion_pool_a_grupos(
                        particion, indices_ncubos, dims_ncubos
                    )
                    dist = self.sia_subsistema.k_bipartir(grupos).distribucion_marginal()
                    phi = emd_efecto(dist, self.sia_dists_marginales)

                    if phi < mejor_phi_k:
                        mejor_phi_k = phi
                        mejor_grupos_k = grupos
                        mejor_dist_k = dist
                    if phi < mejor_phi:
                        mejor_phi = phi
                        mejor_grupos = grupos
                        mejor_dist = dist

                self._resultados_por_k[k] = {
                    "phi": mejor_phi_k,
                    "grupos": mejor_grupos_k,
                    "dist": mejor_dist_k,
                    "n_candidatos": len(candidatos_pool),
                    "tiempo_s": round(time.time() - t0_k, 6),
                }

        self._n_candidatos_evaluados = sum(
            d["n_candidatos"] for d in self._resultados_por_k.values()
        )

        return Solution(
            estrategia=KGEOMETRIC_ASIMETRICO_LABEL,
            perdida=mejor_phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_k_particion(mejor_grupos) if mejor_grupos else "",
        )


# Alias oficial
KGeoMIPAsimetrico = KGeometricSIAAsimetrico
