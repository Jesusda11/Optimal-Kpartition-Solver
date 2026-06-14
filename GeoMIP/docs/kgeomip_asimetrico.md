# KGeoMIP Asimétrico — Documentación interna

Módulo: `src/controllers/strategies/kgeometric_asimetrico.py`  
Modo jerárquico: `src/controllers/strategies/kgeometric_asimetrico_jerarquico.py`  
Ejecutor: `src/ejecutar_kgeomip_asimetrico.py`

---

## 1. Codificación del pool

El pool agrupa `m = n_fut + n_pres` variables:

```
pool[0 .. n_fut-1]       → nodos futuros  (indices_ncubos, índices globales)
pool[n_fut .. m-1]       → nodos presentes (dims_ncubos, índices globales)
```

Una k-partición del pool es una asignación de los `m` pool-indices a `k` grupos no vacíos.  
El número de k-particiones exacto es `S(m, k)` (número de Stirling de 2ª especie).

---

## 2. Arquitectura

```
KGeometricSIAAsimetrico
├── sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
│     → configura sia_subsistema, sia_dists_marginales, caminos, estado_final
├── _construir_tabla_transiciones()   → tabla_transiciones[key] = costos por nodo futuro
├── _costos_presentes()               → lista de costos por nodo presente
├── _generar_candidatos_asimetrico()  → genera frozensets de pool-indices (modo directo)
├── _particion_pool_a_grupos()        → convierte lista de listas a (futuros_i, presentes_i)
└── aplicar_estrategia()              → punto de entrada, retorna Solution

KGeometricSIAAsimetricoJerarquico (hereda de KGeometricSIAAsimetrico)
├── _ranking_local(grupo, n_fut, costos_pres)   → pool-indices en orden ascendente de costo
├── _evaluar_particion_pool(grupos_pool, ...)   → (grupos_reales, phi, dist)
├── _resolver_jerarquico_k(k, ...)              → (grupos_reales, phi, dist, n_eval)
└── aplicar_estrategia()                        → override: exacto si m≤m_max_exhaustivo,
                                                   jerárquico si m>m_max_exhaustivo
```

---

## 3. Parámetros del constructor

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `gestor` | `Manager` | — | Gestor de estado del subsistema |
| `k_min` | `int` | `2` | k mínimo de bipartición |
| `k_max` | `int` | `5` | k máximo de bipartición |
| `m_max_exhaustivo` | `int` | `8` | Umbral de pool para modo exacto vs heurístico |
| `decay_fn` | `Callable` | — | Función de decaimiento para costo geométrico |
| `m_max_candidatos` | `int` | `2000` | Cap de candidatos heurísticos por k |

---

## 4. Diccionario `_resultados_por_k[k]`

```python
{
    "phi":          float | None,   # φ mínimo encontrado para exactamente k grupos
    "grupos":       list[tuple[np.ndarray, np.ndarray]],  # (futuros_i, presentes_i) globales
    "dist":         np.ndarray,     # distribución marginal de la mejor partición
    "n_candidatos": int,            # número de candidatos evaluados
    "tiempo_s":     float,          # tiempo en segundos para este k
    "modo_usado":   str,            # "exacto", "heuristico" o "jerarquico"
}
```

---

## 5. Métrica φ (phi)

```python
phi = emd_efecto(dist_particion, dist_subsistema_completo)
    = sum(|u_i - v_i|)   # Earth Mover's Distance sobre distribuciones marginales
```

### Monotonicidad con grupos no vacíos (S(m,k) estricto)

`φ_{k+1} ≥ φ_k`: más grupos implican más cortes forzados → más pérdida de información.  
La dirección inversa (`φ_{k+1} ≤ φ_k`) solo valdría si se permitieran grupos vacíos.

**Métrica acumulada correctamente decreciente:**
```python
phi_acum_k = min(φ_2, φ_3, ..., φ_k)   # trivialmente decreciente en k
```

---

## 6. Modo directo vs modo jerárquico

### Modo directo (default)

Enumera candidatos como frozensets de pool-indices usando generación RGS (Restricted Growth Strings).  
Para `m ≤ m_max_exhaustivo`: genera los `S(m,k)` candidatos exactos.  
Para `m > m_max_exhaustivo`: usa heurística con cap `m_max_candidatos`.

**Problema de OOM para n≥15, k≥4:**  
El cap se aplica *después* de materializar todos los candidatos. Para n=15, k=4, m=30:
- BFS genera ~22.818 estados × C(29,3)=3.654 cortes = **~83 millones de frozensets (~58 GB)**
- El SO mata el proceso antes de aplicar el cap.

### Modo jerárquico (`KGeometricSIAAsimetricoJerarquico`)

Construye la k-partición mediante **k-1 biparticiones greedy sucesivas**:

1. Iniciar con todos los pool-indices en un único grupo.
2. En cada paso, evaluar todas las biparticiones posibles de todos los grupos actuales.
3. Elegir la bipartición que minimiza φ del conjunto resultante.
4. Repetir hasta tener k grupos.

**Complejidad de memoria:** O(bipartición), nunca se materializa el espacio completo.  
**Evaluaciones para n=15, k=4:** ~84 evaluaciones vs 83M del modo directo.  
**Invariante de aproximación:** `φ_jerárquico ≥ φ_exacto` (greedy puede no hallar el óptimo global).

---

## 7. Flags CLI — `src/ejecutar_kgeomip_asimetrico.py`

| Flag | Variable de entorno | Default | Descripción |
|---|---|---|---|
| `--hoja` | — | *(requerido)* | Nombre de hoja del Excel (ej. `10A-Elementos`) |
| `--inicio` | — | `0` | Índice de inicio 0-based |
| `--cantidad` | — | `50` | Número de pruebas a ejecutar |
| `--timeout` | `KGEOMIP_TIMEOUT_S` | `86400` | Timeout por prueba en segundos |
| `--exacto` | — | `False` | Fuerza enumeración exhaustiva S(m,k) para todos los casos |
| `--k` | — | — | Evaluar solo esta k (mutuamente exclusivo con `--k_min`) |
| `--k_min` | — | `2` | k mínimo del rango |
| `--k_max` | — | `5` | k máximo del rango |
| `--m_max_candidatos` | `KGEOMIP_M_MAX_CANDIDATOS` | `2000` | Cap de candidatos heurísticos por k. Reducir a 500 alivia presión de RAM para n≥20 |
| `--modo` | — | `directo` | Modo de búsqueda: `directo` (S(m,k) exhaustivo/heurístico) o `jerarquico` (biparticiones sucesivas, recomendado para n≥15, k≥4) |
| `--etiqueta` | — | `""` | Sufijo adicional para el nombre del archivo de salida |

### Ejemplos de uso

```bash
# Modo directo (default):
python src/ejecutar_kgeomip_asimetrico.py --hoja 10A-Elementos

# Modo jerárquico para sistemas grandes:
python src/ejecutar_kgeomip_asimetrico.py --hoja 15B-Elementos --modo jerarquico

# Modo jerárquico solo k=4:
python src/ejecutar_kgeomip_asimetrico.py --hoja 15B-Elementos --modo jerarquico --k 4

# Con límite de candidatos reducido (n≥20):
python src/ejecutar_kgeomip_asimetrico.py --hoja 20A-Elementos --m_max_candidatos 500
```

---

## 8. Columnas Excel de salida

Para cada k∈{2,3,4,5}:

| Columna | Descripción |
|---|---|
| `k{k}_Particion` | Representación textual de la mejor k-partición |
| `k{k}_Perdida` | φ mínimo (formato europeo con coma decimal) |
| `k{k}_Tiempo_s` | Tiempo en segundos para este k |
| `k{k}_N_candidatos` | Candidatos evaluados |
| `k{k}_ModoUsado` | `"exacto"`, `"heuristico"` o `"jerarquico"` |

---

## 9. Historial de bugs relevantes

### Bug #1: cap aplicado post-materialización (OOM para n≥15, k≥4)

`_generar_candidatos_asimetrico` generaba todos los candidatos BFS × combinaciones antes de aplicar `m_max_candidatos`. Para n=15, k=4: ~83M frozensets → OOM.  
**Solución:** usar `--modo jerarquico` para estos casos.

### Bug #2: truncación no determinista (`random.sample`)

La versión heurística original usaba `random.sample` para el cap, produciendo resultados no reproducibles entre ejecuciones.  
**Solución:** `_clave_estable` — ordenamiento determinista por hash estable del frozenset.

### Bug #3: monotonicidad en dirección incorrecta

El script de validación original chequeaba `φ_{k+1} ≤ φ_k` (esperando que más grupos = menos pérdida). Incorrecto para S(m,k) estricto.  
**Solución:** invertir a `φ_{k+1} ≥ φ_k` y reportar `phi_acum_k = min(φ_2,...,φ_k)` como métrica decreciente.