"""
src/funcs/distances.py

Funciones de distancia entre estados binarios del hipercubo n-dimensional.

Todas reciben dos listas de enteros {0,1} de igual longitud y retornan un
float >= 0 que representa la distancia entre los dos estados. El valor se
usa como argumento d de la funcion de decrecimiento gamma(d) en GeometricSIA.

La distancia causal (distancia_causal) requiere acceso a los datos de
transicion del sistema y se implementa como metodo de GeometricSIA directamente.
"""

from typing import Callable


# ---------------------------------------------------------------------------
# Funciones de distancia topologica
# ---------------------------------------------------------------------------

def distancia_hamming(a: list[int], b: list[int]) -> float:
    """
    Distancia de Hamming clasica: numero de posiciones en que difieren a y b.
    Retorna un entero en [0, n] (como float). Baseline del algoritmo original.

    Propiedades notables:
      - distancia_hamming([0,0,0], [0,0,0]) = 0.0  (estados identicos)
      - distancia_hamming([1,0,0], [0,1,0]) = 2.0
      - distancia_hamming([1,1,1], [0,0,0]) = n    (estados opuestos)
    """
    return float(sum(x != y for x, y in zip(a, b)))


def distancia_hamming_normalizada(a: list[int], b: list[int]) -> float:
    """
    Hamming normalizado por la longitud del vector: d_H(a, b) / n.
    Float en [0, 1]. Hace que el factor de decrecimiento sea independiente
    del tamano del sistema, permitiendo comparaciones entre sistemas de
    diferente numero de variables.

    Propiedades notables:
      - distancia_hamming_normalizada([1,0], [0,1]) = 1.0
      - Para n=4: d_H=2 -> 0.5, d_H=3 -> 0.75 (vs Hamming: 2, 3)
    """
    n = len(a)
    if n == 0:
        return 0.0
    return sum(x != y for x, y in zip(a, b)) / n


def distancia_jaccard(a: list[int], b: list[int]) -> float:
    """
    Distancia de Jaccard para vectores binarios.
    d_J(a, b) = |{i: a[i] != b[i]}| / |{i: a[i]=1 OR b[i]=1}|.
    Float en [0, 1].

    Mide la disimilitud relativa al soporte activo de los estados (posiciones
    con al menos un bit en 1). Dos estados que difieren solo en posiciones
    donde ambos tienen bits activos reciben mayor distancia que si difirieran
    en posiciones inactivas, capturando la estructura de activacion del sistema.

    Caso borde: si ambos estados son el vector cero (soporte vacio), retorna
    0.0 (estados identicos desde el punto de vista del soporte activo).

    Propiedades notables:
      - distancia_jaccard([1,1,0,0], [0,1,0,0]) = 1/2 = 0.5
        (1 diferente, 2 en union)
      - distancia_jaccard([1,0,0,0], [0,1,0,0]) = 2/2 = 1.0
        (2 diferentes, 2 en union — maxima disimilitud)
      - distancia_jaccard([0,0,0], [0,0,0]) = 0.0  (soporte vacio)
    """
    union = sum(1 for x, y in zip(a, b) if x == 1 or y == 1)
    if union == 0:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x != y) / union


# ---------------------------------------------------------------------------
# Registro centralizado de variantes disponibles
# ---------------------------------------------------------------------------

DISTANCE_VARIANTS: dict[str, Callable[[list[int], list[int]], float]] = {
    "hamming":             distancia_hamming,
    "hamming_normalizado": distancia_hamming_normalizada,
    "jaccard":             distancia_jaccard,
    # "causal" se resuelve como metodo de GeometricSIA (requiere _flat_data)
}


def get_distance(nombre: str) -> Callable[[list[int], list[int]], float]:
    """
    Retorna la funcion de distancia por nombre.

    Para la distancia causal, pasar la cadena "causal" directamente al
    constructor de GeometricSIA — no pasar por esta funcion.

    Args:
        nombre: Clave del diccionario DISTANCE_VARIANTS.

    Returns:
        Funcion d(a, b) correspondiente.

    Raises:
        KeyError: Si el nombre no esta registrado.
    """
    return DISTANCE_VARIANTS[nombre]
