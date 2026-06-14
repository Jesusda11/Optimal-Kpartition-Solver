"""
src/strategies/kqnodes.py

KQNodes — extension de la estrategia QNodes (algoritmo de Queyranne) al caso de
k-particiones con k >= 2. Nomenclatura oficial del proyecto (Manual Tecnico/Usuario
KQMIP): la clase principal de "QNodes K-particiones" debe llamarse exactamente
`KQNodes`.

DISENO
------
KQNodes hereda de `QNodes` y ENVUELVE su motor de Queyranne sin modificarlo
(reutiliza `algorithm`, `funcion_submodular`, `definir_clave`, `nodes_complement`).

- k = 2 (Fase 1): biparticion exacta. Ejecuta el mismo Queyranne que QNodes sobre
  el conjunto completo de vertices y devuelve resultado identico (phi + biparticion),
  formateado con `fmt_biparticion_q`.

- k > 2 (Fase 2): descomposicion JERARQUICA por biparticiones sucesivas (greedy).
  Se parte de un unico grupo con todos los vertices y, en cada uno de los k-1 pasos,
  se evalua dividir cada grupo actual con el oraculo de Queyranne (2-particion) y se
  aplica la division que minimiza el phi de la k-particion COMPLETA, evaluada con
  `System.k_bipartir` + `emd_efecto`. La salida se formatea con `fmt_k_particion`.

Vertices: cada variable es (tiempo, indice) con tiempo ACTUAL(0)=presente,
EFFECT(1)=futuro. El futuro y el presente de un mismo nodo son vertices separados y
pueden caer en grupos distintos (modelo asimetrico, igual que QNodes).

APROXIMACION
------------
Queyranne resuelve la biparticion optima, pero NO se extiende de forma natural a
k>2; ademas, el QNodes de referencia es suboptimo vs fuerza bruta en algunos casos
(ver Fase 1). Por tanto KQNodes(k>2) es una HEURISTICA: phi_KQNodes >= phi_optimo.

DETERMINISMO
------------
Sin aleatoriedad. Orden estable en todo: los grupos se recorren en orden de lista;
los vertices de cada grupo se ordenan con `sorted`; los desempates de phi conservan
el primero (`<`). El motor de Queyranne es determinista. Mismo input => mismo output.
"""

import time

import numpy as np

from src.strategies.q_nodes import QNodes
from src.funcs.iit import emd_efecto
from src.funcs.format import fmt_biparticion_q, fmt_k_particion
from src.models.core.solution import Solution
from src.middlewares.slogger import SafeLogger
from src.constants.models import (
    KQNODES_LABEL,
    KQNODES_STRAREGY_TAG,
)
from src.constants.base import ACTUAL, EFFECT, INFTY_POS


class KQNodes(QNodes):
    """
    Estrategia KQNodes (QNodes extendido a k-particiones). Hereda de QNodes.

    Args:
        tpm (np.ndarray): Matriz de Probabilidad de Transicion del sistema completo.
        k_min (int): k minimo a evaluar (>= 2). Default 2.
        k_max (int): k maximo a evaluar. Default 2.

    Atributos:
        _resultados_por_k (dict[int, dict]): por cada k, {phi, particion, dist,
            n_candidatos, modo_usado}.
        _modo_usado (str): modo de busqueda global empleado.
    """

    def __init__(self, tpm: np.ndarray, k_min: int = 2, k_max: int = 2):
        super().__init__(tpm)
        self.k_min: int = max(2, k_min)
        self.k_max: int = k_max
        self.logger = SafeLogger(KQNODES_STRAREGY_TAG)

        self._resultados_por_k: dict[int, dict] = {}
        self._modo_usado: str = "pendiente"

    # ── Punto de entrada ──────────────────────────────────────────────────────

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        """
        Resuelve la k-MIP para k en [k_min, k_max] y devuelve la mejor (menor phi).

        k=2 reproduce QNodes exactamente; k>2 usa biparticiones sucesivas (greedy).
        Registra el resultado de cada k en `_resultados_por_k`.
        """
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        vertices = self._construir_vertices()
        self._preparar_motor(vertices)

        self._resultados_por_k = {}
        mejor_phi = INFTY_POS
        mejor_dist = None
        mejor_fmt = ""

        k_top = min(self.k_max, len(vertices))
        for k in range(self.k_min, k_top + 1):
            t0_k = time.time()
            if k == 2:
                _, phi, dist, fmt = self._biparticion_full(vertices)
                n_eval, modo = 1, "queyranne-k2"
            else:
                grupos, phi, dist, n_eval = self._resolver_jerarquico_k(k, vertices)
                fmt = fmt_k_particion(self._vertices_a_grupos(grupos))
                modo = "jerarquico-queyranne"

            self._resultados_por_k[k] = {
                "phi": phi,
                "particion": fmt,
                "dist": dist,
                "n_candidatos": n_eval,
                "modo_usado": modo,
                "tiempo_s": round(time.time() - t0_k, 6),
            }
            if phi < mejor_phi:
                mejor_phi, mejor_dist, mejor_fmt = phi, dist, fmt

        self._modo_usado = "jerarquico-queyranne" if k_top > 2 else "queyranne-k2"

        return Solution(
            estrategia=KQNODES_LABEL,
            perdida=mejor_phi if mejor_phi != INFTY_POS else 0.0,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=mejor_fmt,
        )

    # ── Construccion de vertices y preparacion del motor ──────────────────────

    def _construir_vertices(self) -> list:
        """Vertices (tiempo, indice) del subsistema: presentes (ACTUAL) + futuros (EFFECT)."""
        futuro = [(EFFECT, int(f)) for f in self.sia_subsistema.indices_ncubos]
        presente = [(ACTUAL, int(p)) for p in self.sia_subsistema.dims_ncubos]
        return presente + futuro

    def _preparar_motor(self, vertices: list) -> None:
        """Fija los atributos que el motor heredado de QNodes espera encontrar."""
        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )
        self.vertices = set(vertices)

    # ── k=2: biparticion exacta (envuelve Queyranne, identico a QNodes) ───────

    def _biparticion_full(self, vertices: list):
        """
        Ejecuta el motor de Queyranne sobre el conjunto COMPLETO de vertices.

        Returns:
            (mip, phi, dist_marginal, particion_formateada) — identico a QNodes.
        """
        self._reset_motor()
        mip = self.algorithm(list(vertices))
        fmt = fmt_biparticion_q(list(mip), self.nodes_complement(mip))
        phi, dist = self.memoria_grupo_candidato[mip]
        return mip, float(phi), dist, fmt

    # ── k>2: biparticiones sucesivas (greedy jerarquico) ──────────────────────

    def _resolver_jerarquico_k(self, k: int, vertices: list):
        """
        Construye la k-particion via k-1 biparticiones greedy sucesivas.

        En cada paso evalua, para cada grupo divisible, la biparticion propuesta por
        el oraculo de Queyranne, y aplica la que minimiza el phi de la k-particion
        completa (evaluado con k_bipartir + emd_efecto).

        Returns:
            (grupos_vertices, phi, dist, n_evaluaciones)
        """
        grupos: list[set] = [set(vertices)]
        phi_actual = None
        dist_actual = None
        n_eval = 0

        for _ in range(k - 1):
            mejor = None  # (phi, nuevos_grupos, dist)

            for i, grupo in enumerate(grupos):
                if len(grupo) < 2:
                    continue
                lado_a, lado_b = self._oraculo_biparticion(grupo)
                nuevos = grupos[:i] + [lado_a, lado_b] + grupos[i + 1:]
                dist, phi = self._evaluar_kparticion(nuevos)
                n_eval += 1
                if mejor is None or phi < mejor[0]:
                    mejor = (phi, nuevos, dist)

            if mejor is None:
                break  # todos los grupos son unitarios

            phi_actual, grupos, dist_actual = mejor

        if phi_actual is None:  # no se pudo dividir (defensivo)
            dist_actual, phi_actual = self._evaluar_kparticion(grupos)

        return grupos, phi_actual, dist_actual, n_eval

    def _oraculo_biparticion(self, grupo: set):
        """
        Usa el motor de Queyranne como oraculo de 2-particion sobre un grupo de
        vertices. Devuelve (lado_a, lado_b) como conjuntos de vertices.

        Para |grupo| == 2 el split es trivial (los dos singletons). Para |grupo| >= 3
        se ejecuta el motor (reseteando su memoria) y se extraen los lados de forma
        robusta con `_flatten_vertices` (independiente de la estructura de la clave).
        """
        grupo_list = sorted(grupo)
        if len(grupo_list) == 2:
            return {grupo_list[0]}, {grupo_list[1]}

        self._reset_motor()
        mip = self.algorithm(list(grupo_list))

        conjunto = set(grupo_list)
        lado_a = self._flatten_vertices(mip) & conjunto
        lado_b = conjunto - lado_a

        if not lado_a or not lado_b:  # split degenerado: fallback determinista
            lado_a = {grupo_list[0]}
            lado_b = conjunto - lado_a

        return lado_a, lado_b

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _reset_motor(self) -> None:
        """Limpia la memoria mutable del motor de Queyranne para una nueva corrida."""
        self.memoria_grupo_candidato = {}
        self.memoria_delta = {}
        self.clave_submodular = [], []

    def _flatten_vertices(self, obj) -> set:
        """
        Aplana recursivamente una estructura (tuplas/listas anidadas) a un conjunto
        de vertices (tiempo, indice) normalizados a (int, int). Un vertice es una
        tupla de longitud 2 cuyos dos elementos son enteros.
        """
        if (
            isinstance(obj, tuple)
            and len(obj) == 2
            and isinstance(obj[0], (int, np.integer))
            and isinstance(obj[1], (int, np.integer))
        ):
            return {(int(obj[0]), int(obj[1]))}
        salida: set = set()
        for elemento in obj:
            salida |= self._flatten_vertices(elemento)
        return salida

    def _vertices_a_grupos(self, grupos_vertices: list) -> list:
        """
        Convierte una lista de grupos de vertices a la forma que espera k_bipartir:
        lista de (futuros, presentes) con indices reales de nodos por grupo.
        """
        grupos_reales = []
        for grupo in grupos_vertices:
            futuros = sorted(idx for (tiempo, idx) in grupo if tiempo == EFFECT)
            presentes = sorted(idx for (tiempo, idx) in grupo if tiempo == ACTUAL)
            grupos_reales.append(
                (np.array(futuros, dtype=np.int8), np.array(presentes, dtype=np.int8))
            )
        return grupos_reales

    def _evaluar_kparticion(self, grupos_vertices: list):
        """Evalua la k-particion completa: devuelve (dist_marginal, phi)."""
        grupos_reales = self._vertices_a_grupos(grupos_vertices)
        dist = self.sia_subsistema.k_bipartir(grupos_reales).distribucion_marginal()
        phi = float(emd_efecto(dist, self.sia_dists_marginales))
        return dist, phi