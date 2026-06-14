"""
src/controllers/strategies/kgeometric_asimetrico_jerarquico.py

KGeoMIP Asimetrico con descomposicion jerarquica para sistemas grandes (n>=15, k>=4).

Motivacion
----------
El modo directo de KGeometricSIAAsimetrico genera O(|BFS_estados| x C(m-1, k-1))
candidatos simultaneamente antes de aplicar m_max_candidatos.
Para n=15, k=4: ~22.818 estados BFS x C(29,3)=3654 = ~83 millones de frozensets
(~58 GB) -> OOM kill del SO, independientemente del tipo de TPM.

Enfoque: biparticiones sucesivas (greedy)
-----------------------------------------
Se construye la k-particion mediante k-1 biparticiones:

  1. Partir el pool (m variables) en 2 grupos usando cortes en ranking de costos.
  2. Para cada paso siguiente, evaluar TODAS las posibles biparticiones de TODOS
     los grupos existentes (incluyendo el resultado del paso anterior).
  3. Elegir la division que produce el menor phi del conjunto total de grupos.
  4. Repetir hasta tener k grupos.

Criterio de subdivision
-----------------------
Greedy exacto: en cada paso se evaluan todas las posibles biparticiones de todos
los grupos actuales y se elige la que minimiza phi global.  El numero de
evaluaciones por paso es sum(len(grupo)-1 for grupo in grupos) <= m - |grupos|.

Para n=15, k=4 con m=30: ~29 + ~28 + ~27 = ~84 evaluaciones totales vs 83M del
modo directo. Costo de memoria O(biparticion), nunca se materializa el espacio
completo de k-particiones.

Invariante de aproximacion
--------------------------
phi_jerarquico >= phi_exacto: el greedy puede no hallar el optimo global porque
el espacio no se cubre exhaustivamente.  Esto es CORRECTO y esperado: la clase
se documenta como aproximacion, no como exacto.

Integracion con el ejecutor
----------------------------
Activar con --modo jerarquico en src/ejecutar_kgeomip_asimetrico.py.
Los _resultados_por_k incluyen "modo_usado": "jerarquico" en cada entrada para
distinguir resultados de ambos modos en el Excel de salida.
"""

import time
from typing import Optional, Callable

import numpy as np

from src.controllers.strategies.kgeometric_asimetrico import KGeometricSIAAsimetrico
from src.controllers.manager import Manager
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_k_particion
from src.models.core.solution import Solution

KGEOMIP_JERARQUICO_LABEL = "KGeoMIPAsimetrico-Jerarquico"


class KGeoMIPAsimetricoJerarquico(KGeometricSIAAsimetrico):
    """
    KGeoMIP Asimetrico con modo jerarquico para k-particiones en sistemas grandes.

    Para m <= m_max_exhaustivo: delega en el modo exacto del padre (S(m,k) completo).
    Para m >  m_max_exhaustivo: usa biparticiones sucesivas greedy.

    Args:
        Identicos a KGeometricSIAAsimetrico.
    """

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _ranking_local(
        self,
        grupo: frozenset,
        n_fut: int,
        costos_pres: list,
    ) -> list:
        """
        Devuelve los pool-indices del grupo en orden ascendente de costo.

        Costo por pool-index p:
          p < n_fut  -> costo acumulado del futuro p en tabla_transiciones[(s_ini, s_fin)]
          p >= n_fut -> costo del presente (p - n_fut) de _costos_presentes()

        Determinista: sorted() es estable; empates se rompen por pool-index natural.
        """
        key_final = (tuple(self.caminos[0][0]), tuple(self.estado_final))
        costos_fut = self.tabla_transiciones.get(
            key_final, [0.0] * len(self.sia_subsistema.indices_ncubos)
        )

        def _costo(p: int) -> float:
            return costos_fut[p] if p < n_fut else costos_pres[p - n_fut]

        return sorted(grupo, key=_costo)

    def _evaluar_particion_pool(
        self,
        grupos_pool: list,
        indices_ncubos: np.ndarray,
        dims_ncubos: np.ndarray,
    ) -> tuple:
        """
        Convierte lista de frozensets de pool-indices a (grupos_reales, phi, dist).
        Llama a k_bipartir sobre el subsistema completo ya preparado.
        """
        grupos_reales = self._particion_pool_a_grupos(
            [sorted(g) for g in grupos_pool], indices_ncubos, dims_ncubos
        )
        dist = self.sia_subsistema.k_bipartir(grupos_reales).distribucion_marginal()
        phi  = float(emd_efecto(dist, self.sia_dists_marginales))
        return grupos_reales, phi, dist

    # ── Algoritmo jerarquico ──────────────────────────────────────────────────

    def _resolver_jerarquico_k(
        self,
        k: int,
        indices_ncubos: np.ndarray,
        dims_ncubos: np.ndarray,
        costos_pres: list,
    ) -> tuple:
        """
        Construye la k-particion optima (greedy) mediante k-1 biparticiones.

        En cada paso se evaluan todos los cortes de todos los grupos existentes
        y se elige el que produce el menor phi del conjunto resultante.

        Retorna (grupos_reales, phi, dist, n_evaluaciones).
        """
        m     = len(indices_ncubos) + len(dims_ncubos)
        n_fut = len(indices_ncubos)

        # Iniciar: todos los pool-indices en un unico grupo
        grupos_pool: list = [frozenset(range(m))]

        phi_actual:  float = float("inf")
        dist_actual        = None
        n_eval:      int   = 0

        for _paso in range(k - 1):
            mejor_grupos_pool = None
            mejor_phi         = float("inf")
            mejor_dist        = None

            for i, grupo in enumerate(grupos_pool):
                if len(grupo) < 2:
                    continue  # grupo unitario: no se puede subdividir

                ranking = self._ranking_local(grupo, n_fut, costos_pres)

                for cut in range(1, len(ranking)):
                    g1 = frozenset(ranking[:cut])
                    g2 = frozenset(ranking[cut:])

                    candidato = grupos_pool[:i] + [g1, g2] + grupos_pool[i + 1:]
                    _, phi, dist = self._evaluar_particion_pool(
                        candidato, indices_ncubos, dims_ncubos
                    )
                    n_eval += 1

                    if phi < mejor_phi:
                        mejor_phi         = phi
                        mejor_dist        = dist
                        mejor_grupos_pool = candidato

            if mejor_grupos_pool is None:
                # Todos los grupos son unitarios: no se puede alcanzar k
                # Calcular phi de la particion actual y detener
                _, phi_actual, dist_actual = self._evaluar_particion_pool(
                    grupos_pool, indices_ncubos, dims_ncubos
                )
                n_eval += 1
                break

            grupos_pool  = mejor_grupos_pool
            phi_actual   = mejor_phi
            dist_actual  = mejor_dist

        grupos_reales = self._particion_pool_a_grupos(
            [sorted(g) for g in grupos_pool], indices_ncubos, dims_ncubos
        )
        return grupos_reales, phi_actual, dist_actual, n_eval

    # ── Override de aplicar_estrategia ────────────────────────────────────────

    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ) -> Solution:
        """
        Punto de entrada.

        Para m <= m_max_exhaustivo: delega en el padre (exacto S(m,k)) y
          anota modo_usado en cada entrada de _resultados_por_k.
        Para m >  m_max_exhaustivo: usa biparticiones jerarquicas.

        Nota: sia_preparar_subsistema se llama aqui para leer m antes de
        decidir el modo; si se delega al padre, se llama una segunda vez
        dentro de super() — sin efecto observable para m pequeno.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
        n_fut  = len(self.sia_subsistema.indices_ncubos)
        n_pres = len(self.sia_subsistema.dims_ncubos)
        m      = n_fut + n_pres

        if m < 2:
            raise ValueError(
                f"KGeoMIPAsimetricoJerarquico requiere m >= 2 "
                f"(n_fut={n_fut}, n_pres={n_pres})."
            )

        # ── Modo exacto: delegar al padre ──────────────────────────────────
        if m <= self.m_max_exhaustivo:
            result = super().aplicar_estrategia(condicion, alcance, mecanismo, tpm)
            for datos in self._resultados_por_k.values():
                datos.setdefault("modo_usado", self._modo_usado)
            return result

        # ── Modo jerarquico ────────────────────────────────────────────────
        self._modo_usado = "jerarquico"
        self._flat_data  = [ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos]
        self._setup_dist_fn()
        self._construir_tabla_transiciones()
        costos_pres = self._costos_presentes()

        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos    = self.sia_subsistema.dims_ncubos

        self._resultados_por_k: dict = {}
        mejor_phi    = float("inf")
        mejor_grupos = []
        mejor_dist   = None

        for k in range(self.k_min, min(self.k_max, m) + 1):
            t0_k = time.time()
            grupos_k, phi_k, dist_k, n_cand = self._resolver_jerarquico_k(
                k, indices_ncubos, dims_ncubos, costos_pres
            )
            t_k = time.time() - t0_k

            self._resultados_por_k[k] = {
                "phi":          phi_k if phi_k != float("inf") else None,
                "grupos":       grupos_k,
                "dist":         dist_k,
                "n_candidatos": n_cand,
                "tiempo_s":     round(t_k, 6),
                "modo_usado":   "jerarquico",
            }

            if phi_k < mejor_phi:
                mejor_phi    = phi_k
                mejor_grupos = grupos_k
                mejor_dist   = dist_k

        self._n_candidatos_evaluados = sum(
            d["n_candidatos"] for d in self._resultados_por_k.values()
        )

        return Solution(
            estrategia=KGEOMIP_JERARQUICO_LABEL,
            perdida=mejor_phi if mejor_phi != float("inf") else 0.0,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_k_particion(mejor_grupos) if mejor_grupos else "",
        )


# Nomenclatura oficial: la clase principal es KGeoMIPAsimetricoJerarquico.
# KGeometricSIAAsimetricoJerarquico se conserva como alias de retrocompatibilidad.
KGeometricSIAAsimetricoJerarquico = KGeoMIPAsimetricoJerarquico