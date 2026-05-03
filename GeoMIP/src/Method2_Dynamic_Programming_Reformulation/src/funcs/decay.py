import math
from typing import Callable


# ---------------------------------------------------------------------------
# Funciones de decrecimiento γ(d)
#
# Todas reciben un entero d (distancia de Hamming ≥ 0) y retornan un float
# γ ∈ (0, 1] que pondera la contribución del costo de transición t(i,j).
# ---------------------------------------------------------------------------

def decay_exponencial(d: int) -> float:
    """
    Factor original de GeometricSIA: γ = 2^(-d).

    Es la función de referencia (baseline). Decrece a la mitad por cada
    unidad de distancia Hamming, lo que equivale a descartar la mitad
    de la influencia causal en cada salto del hipercubo.

    Propiedades notables:
      - decay_exponencial(0) = 1.0   (transición trivial, peso completo)
      - decay_exponencial(1) = 0.5
      - decay_exponencial(n) = 2^(-n) → 0 asintóticamente
    """
    return 1.0 / (2 ** d)


def decay_polinomial(d: int) -> float:
    """
    Factor polinomial: γ = 1 / (d + 1).

    Decrece más lentamente que el exponencial para distancias grandes,
    lo que da más peso relativo a transiciones lejanas en el hipercubo.
    Útil para sistemas con interacciones causales de largo alcance donde
    la influencia no decae tan abruptamente con la distancia topológica.

    Propiedades notables:
      - decay_polinomial(0) = 1.0
      - decay_polinomial(1) = 0.5   (igual que el exponencial en d=1)
      - decay_polinomial(2) ≈ 0.333 (vs 0.25 del exponencial)
      - decay_polinomial(n) ∼ 1/n   (decrecimiento lento)
    """
    return 1.0 / (d + 1)


def decay_logaritmico(d: int) -> float:
    """
    Factor logarítmico: γ = 1 / log2(d + 2).

    Es el más lento de todos los decrecimienos — incluso más lento que
    el polinomial. Asigna pesos casi uniformes para distancias moderadas,
    lo que equivale a considerar que la estructura causal del sistema
    tiene una «memoria larga» y las relaciones entre estados lejanos
    siguen siendo relevantes.

    Propiedades notables:
      - decay_logaritmico(0) = 1.0   (log2(2) = 1)
      - decay_logaritmico(1) ≈ 0.631 (log2(3) ≈ 1.585)
      - decay_logaritmico(2) = 0.5   (log2(4) = 2)
      - decay_logaritmico(n) ∼ 1/log2(n) → 0 muy lentamente
    """
    return 1.0 / math.log2(d + 2)


def make_decay_adaptativo(alpha: float) -> Callable[[int], float]:
    """
    Fábrica que genera un factor de decrecimiento de potencia ajustable:
    γ = 1 / (d + 1)^alpha.

    Cuando alpha=1 se recupera el decay_polinomial.
    Cuando alpha→∞ se aproxima al comportamiento del exponencial.
    Cuando alpha→0 el factor tiende a 1 (sin decrecimiento).

    Permite calibrar la sensibilidad del algoritmo a la distancia
    topológica según las características del sistema analizado.

    Args:
        alpha: Exponente de la potencia. Valor recomendado: (0.5, 2.0).

    Returns:
        Función γ(d) = 1 / (d + 1)^alpha.
    """
    def decay_adaptativo(d: int) -> float:
        return 1.0 / ((d + 1) ** alpha)
    decay_adaptativo.__name__ = f"decay_adaptativo(alpha={alpha})"
    return decay_adaptativo


# ---------------------------------------------------------------------------
# Registro centralizado de variantes disponibles
#
# Facilita la selección por nombre en scripts de comparativa y en exec.py.
# ---------------------------------------------------------------------------

DECAY_VARIANTS: dict[str, Callable[[int], float]] = {
    "exponencial": decay_exponencial,
    "polinomial":  decay_polinomial,
    "logaritmico": decay_logaritmico,
}


def get_decay(nombre: str, alpha: float = 1.0) -> Callable[[int], float]:
    """
    Retorna la función de decrecimiento por nombre.

    Args:
        nombre:  Clave del diccionario DECAY_VARIANTS o "adaptativo".
        alpha:   Solo relevante cuando nombre == "adaptativo".

    Returns:
        Función γ(d) correspondiente.

    Raises:
        KeyError: Si el nombre no está registrado.
    """
    if nombre == "adaptativo":
        return make_decay_adaptativo(alpha)
    return DECAY_VARIANTS[nombre]
