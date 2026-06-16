# Optimal-KPartition-Solver

> Búsqueda de **k-Particiones de Mínima Información (k-MIP)** en sistemas dinámicos binarios, mediante la extensión de dos estrategias clásicas de bipartición —**GeoMIP** (geométrica) y **QNodes** (minimización submodular)— al caso general de *k* particiones.

Proyecto final de **Análisis y Diseño de Algoritmos** · Universidad de Caldas · 2026

---

## Tabla de contenido

- [¿Qué resuelve?](#qué-resuelve)
- [Estrategias implementadas](#estrategias-implementadas)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Interpretación de resultados](#interpretación-de-resultados)
- [Pruebas](#pruebas)
- [Resultados destacados](#resultados-destacados)
- [Documentación](#documentación)
- [Autores](#autores)

---

## ¿Qué resuelve?

Dado un sistema de *n* variables binarias que evolucionan en el tiempo (descrito por una **matriz de probabilidad de transición**, TPM), el problema consiste en encontrar la forma de dividirlo en *k* partes que minimice la **pérdida de información φ** respecto a su dinámica original. Esa división óptima es la **k-Partición de Mínima Información (k-MIP)**.

La pérdida se mide con la *Earth Mover's Distance* sobre los repertorios de efecto, que bajo independencia condicional se reduce a la distancia L1:

```
φ = Σ |uᵢ − vᵢ|
```

Las estrategias originales del proyecto base solo resolvían el caso **k = 2** (bipartición). Este trabajo las generaliza a **k ∈ {2, 3, 4, 5}**.

---

## Estrategias implementadas

| Estrategia | Fundamento | Modelos | Modos de búsqueda |
|------------|------------|---------|-------------------|
| **KGeoMIP** | Tabla de costos geométrica sobre el hipercubo de estados | Simétrico `S(n,k)` y Asimétrico `S(2n,k)` | Exacto · Heurístico · Jerárquico |
| **KQNodes** | Algoritmo de Queyranne como oráculo de bipartición | — | Exacto (k=2) · Jerárquico (k>2) |

**Dos modelos de partición** (en KGeoMIP):
- **Simétrico** — particiona nodos; presente y futuro de cada nodo van juntos. Coherente con la partición de mecanismo de IIT.
- **Asimétrico** — particiona las `2n` variables de forma independiente. Generaliza la bipartición original de GeoMIP.

**Tres modos de búsqueda**, seleccionados automáticamente según la escala:
- **Exacto** — enumera todas las k-particiones (óptimo garantizado, solo viable para *n* pequeño).
- **Heurístico** — genera candidatos prometedores guiados por la tabla de costos.
- **Jerárquico** — construye la k-partición por biparticiones sucesivas (escalable a sistemas grandes).

---

## Estructura del repositorio

```
Optimal-Kpartition-Solver/
├── GeoMIP/
│   └── src/
│       └── Method2_Dynamic_Programming_Reformulation/   # Árbol KGeoMIP
│           ├── src/
│           │   ├── ejecutar_kgeomip.py                  # Ejecutor simétrico
│           │   ├── ejecutar_kgeomip_asimetrico.py       # Ejecutor asimétrico
│           │   └── controllers/strategies/              # Clases de estrategia
│           ├── pyproject.toml
│           └── tests/                                   # 51 pruebas unitarias
│
├── QNodes/                                              # Proyecto base (referencia)
│   └── .venv/                                           # Entorno compartido con KQNodes
│
└── KQNodes/                                             # Árbol KQNodes
    ├── ejecutar_kqnodes.py                              # Ejecutor KQNodes
    └── tests/                                           # 48 pruebas unitarias
```

> **Arquitectura de dos árboles independientes.** KGeoMIP y KQNodes tienen clases base, dependencias y versiones de Python distintas. Es una decisión deliberada de aislamiento: preserva intacto el proyecto base como referencia de validación. **KQNodes reutiliza el entorno virtual de QNodes.**

---

## Requisitos

| Software | Versión |
|----------|---------|
| Python (KGeoMIP) | ≥ 3.9.13 |
| Python (KQNodes) | ≥ 3.11 |
| [uv](https://docs.astral.sh/uv/) | Última estable |
| Git | Reciente |

`uv` gestiona los entornos virtuales y descarga las versiones de Python necesarias automáticamente.

---

## Instalación

```bash
git clone https://github.com/<usuario>/Optimal-Kpartition-Solver.git
cd Optimal-Kpartition-Solver
```

**KGeoMIP:**
```bash
cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
uv sync
```

**KQNodes** (instala el entorno en `QNodes/`, que KQNodes reutiliza):
```bash
cd QNodes
uv sync
```

---

## Uso rápido

Los comandos usan Windows (PowerShell). En Linux, reemplazar `.venv\Scripts\python.exe` por `.venv/bin/python`.

**KGeoMIP Simétrico** — desde `Method2_Dynamic_Programming_Reformulation/`:
```powershell
.venv\Scripts\python.exe src\ejecutar_kgeomip.py --hoja 10A-Elementos
```

**KGeoMIP Asimétrico:**
```powershell
.venv\Scripts\python.exe src\ejecutar_kgeomip_asimetrico.py --hoja 10A-Elementos --k 2
```

**KQNodes** — desde `KQNodes/` (usa el intérprete de QNodes explícitamente):
```powershell
..\QNodes\.venv\Scripts\python.exe ejecutar_kqnodes.py --hoja 10A-Elementos
```

Los datos de entrada se leen de `GeoMIP/tests/PruebasK-Particiones.xlsx` (una hoja por tamaño de sistema). Los resultados se escriben en archivos `.xlsx` dentro de las carpetas `results/`.

**Parámetros principales:** `--hoja` (hoja del Excel), `--k` / `--k_min` / `--k_max` (rango de particiones), `--inicio` y `--cantidad` (subconjunto de pruebas), `--timeout` (límite por prueba).

---

## Interpretación de resultados

Cada fila del Excel de salida contiene, por cada *k*, la **partición hallada**, su **pérdida φ** y el **tiempo**. La partición se expresa con grupos entre barras:

```
|A,B,C||D,E,F,G,H,I,J|      ← bipartición (k=2)
```

El **MIP global** de un subsistema es el menor φ entre todos los valores de *k*. El campo `Modo` indica si el resultado fue `exacto`, `heuristico` o `jerarquico`.

---

## Pruebas

Suite de **99 pruebas unitarias** (51 KGeoMIP + 48 KQNodes) con `pytest`.

```bash
# KGeoMIP — desde Method2_Dynamic_Programming_Reformulation/
uv pip install pytest
uv run pytest

# KQNodes — desde KQNodes/, con el intérprete de QNodes
uv pip install --python ..\QNodes\.venv\Scripts\python.exe pytest
..\QNodes\.venv\Scripts\python.exe -m pytest
```

> Algunas pruebas se omiten (*skip*) si faltan los archivos de TPM de muestra en `src/.samples/`. Es comportamiento esperado, no un fallo.

---

## Resultados destacados

- **Validación 100 %** frente a [PyPhi](https://github.com/wmayner/pyphi) (error relativo 0 %) y frente a un enumerador de fuerza bruta independiente (32/32).
- **Reproducción exacta** de las estrategias base en k=2 (KQNodes ≡ QNodes, 13/13).
- **Hallazgo principal:** en el modelo simétrico, el **12,2 %** de los casos (n=10) tienen su MIP en **k > 2**, lo que demuestra que explorar k-particiones aporta valor más allá de la bipartición.
- **Descomposición del error de KQNodes (k>2):** 64 % atribuible a la suboptimalidad del oráculo de Queyranne, 36 % a la estructura voraz.
- **Determinismo total:** misma entrada → mismo resultado, en todos los modos.

---

## Documentación

- **Manual Técnico** — fundamentos teóricos, arquitectura, diseño algorítmico, análisis de complejidad y resultados experimentales.
- **Manual de Usuario** — instalación, uso, interpretación de resultados, solución de problemas y glosario.

---

## Autores

| Integrante | Código |
|------------|--------|
| Juan Alejandro Betancourth | 40960 |
| Jesús Daniel Arias | 41056 |
| Nicolás Carvajal González | 36962 |

**Docente:** Luz Enith Guerrero Mendieta
**Asignatura:** Análisis y Diseño de Algoritmos — Universidad de Caldas, Manizales, Colombia
