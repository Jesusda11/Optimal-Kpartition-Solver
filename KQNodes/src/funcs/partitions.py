"""
src/funcs/partitions.py

Generacion de k-particiones de n elementos usando restricted growth strings (RGS).

PORTADO desde el arbol KGeoMIP (Method2_Dynamic_Programming_Reformulation) sin
cambios de logica: es Python puro sin dependencias del arbol de origen. Se usa
para enumeracion exacta (fuerza bruta) en la validacion de KQNodes en casos
pequenos.

Una k-particion de n elementos es una asignacion surjectiva de cada elemento
{0, ..., n-1} a uno de k grupos {0, ..., k-1}. Se generan en forma canonica
(el primer elemento siempre pertenece al grupo 0) para evitar duplicados por
permutacion de etiquetas de grupo.

El numero de k-particiones de n elementos es el numero de Stirling del segundo
tipo S(n, k). Ejemplos:
  S(3,2) = 3    S(4,2) = 7    S(4,3) = 6
  S(6,2) = 31   S(6,3) = 90   S(6,4) = 65
"""

from typing import Generator


def generar_k_particiones(n: int, k: int) -> Generator[list[list[int]], None, None]:
    """
    Genera todas las k-particiones de los indices {0, ..., n-1} en forma canonica.

    Usa restricted growth strings con poda: si los elementos restantes son
    insuficientes para llenar los grupos pendientes, descarta la rama.

    Args:
        n: Numero de elementos a particionar.
        k: Numero de grupos (todos deben ser no vacios).

    Yields:
        Lista de k listas; cada lista contiene los indices asignados a ese grupo.
        El indice 0 siempre pertenece al grupo 0.
    """
    if k < 1 or k > n:
        return

    assignment: list[int] = []

    def backtrack(pos: int, current_max: int):
        if pos == n:
            if current_max == k - 1:
                groups = [[] for _ in range(k)]
                for i, g in enumerate(assignment):
                    groups[g].append(i)
                yield [list(grp) for grp in groups]
            return
        remaining = n - pos
        labels_needed = k - 1 - current_max
        if remaining < labels_needed:
            return
        upper = min(current_max + 1, k - 1)
        for label in range(upper + 1):
            assignment.append(label)
            new_max = max(current_max, label)
            yield from backtrack(pos + 1, new_max)
            assignment.pop()

    yield from backtrack(0, -1)


def contar_stirling(n: int, k: int) -> int:
    """
    Calcula el numero de Stirling del segundo tipo S(n, k).

    S(n,k) = numero de formas de partir n elementos en k grupos no vacios.
    Recurrencia: S(n,k) = k*S(n-1,k) + S(n-1,k-1), con S(0,0)=1.
    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1]
    return dp[n][k]
