"""
src/ejecutar_kgeomip.py

Ejecutor de KGeoMIP Simetrico para las pruebas del proyecto K-QGMIP.

Lee el Excel de pruebas (GeoMIP/tests/PruebasK-Particiones.xlsx), ejecuta
KGeoMIPHeuristica con k_max=5 para cada caso y guarda resultados en:
    GeoMIP/results/resultados_kgeomip_simetrico_{hoja}.xlsx

Estructura del Excel de entrada (por hoja):
  Fila 1 col B : Estado inicial (e.g. 1000000000 para n=10)
  Fila 2 col B : Sistema
  Fila 3 col B : Sistema candidato
  Fila 4       : Seccion (Biparticiones / 3-Particiones / ...)
  Fila 5       : Cabeceras (#Prueba | Alcance | Mecanismo | ...)
  Fila 6+      : Datos    (int | letras | letras)

Alcance y Mecanismo vienen como strings de letras (ej. "ABCDE") y se
convierten a binario usando la posicion en el abecedario.

NO modifica el Excel de entrada. El usuario copia los resultados manualmente.

Como ejecutar (desde el directorio Method2_Dynamic_Programming_Reformulation/):
    python src/ejecutar_kgeomip.py --hoja 10A-Elementos
    python src/ejecutar_kgeomip.py --hoja "25A-Elementos " --inicio 0 --cantidad 50
    python src/ejecutar_kgeomip.py --hoja 15B-Elementos --timeout 7200

Variables de entorno opcionales:
    KGEOMIP_INPUT_XLSX   : ruta al Excel de entrada
    KGEOMIP_N_MAX_EXACTO : umbral exacto/heuristico (default 8)
    KGEOMIP_TIMEOUT_S    : timeout por prueba en segundos (default 3600)
"""

import sys
import os
import re
import time
import logging
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
METHOD2_ROOT = Path(__file__).resolve().parents[1]
GEOMIP_ROOT  = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHOD2_ROOT))

# ── Silenciar logging ─────────────────────────────────────────────────────────
logging.disable(logging.CRITICAL)

import src.middlewares.slogger as _slogger_mod

class _NullSafeLogger:
    def __init__(self, name): pass
    def debug(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def error(self, *a, **k): pass
    def critic(self, *a, **k): pass
    def fatal(self, *a, **k): pass
    def set_log(self, *a, **k): pass

_slogger_mod.SafeLogger = _NullSafeLogger

# ── Importaciones del proyecto ────────────────────────────────────────────────
from src.models.base.application import aplicacion
from src.middlewares.profile import profiler_manager
from src.controllers.manager import Manager
from src.controllers.strategies.kgeometric_heuristica import KGeoMIPHeuristica
from src.funcs.decay import decay_exponencial
from src.funcs.format import fmt_k_particion

profiler_manager.enabled = False

# ── Configuracion ─────────────────────────────────────────────────────────────
K_MAX          = 5
N_MAX_EXACTO   = int(os.getenv("KGEOMIP_N_MAX_EXACTO", "8"))
TIMEOUT_S      = int(os.getenv("KGEOMIP_TIMEOUT_S",    "3600"))
INPUT_XLSX     = Path(os.getenv(
    "KGEOMIP_INPUT_XLSX",
    str(GEOMIP_ROOT / "tests" / "PruebasK-Particiones.xlsx"),
))


# ── Helpers ───────────────────────────────────────────────────────────────────

def convertir_a_binario(texto: str, n_bits: int) -> str:
    """Convierte string de letras ('ABCDE') a string binario de n_bits."""
    posiciones = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto.upper():
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)


def parsear_hoja(sheet_name: str) -> tuple[int, str]:
    """
    Extrae n y variante de un nombre de hoja como '10A-Elementos' o '25A-Elementos '.
    Retorna (n, variante), ej. (10, 'A') o (25, 'A').
    """
    m = re.match(r"(\d+)([A-Z])", sheet_name.strip())
    if not m:
        raise ValueError(f"No se pudo parsear la hoja '{sheet_name}'. Formato esperado: '10A-Elementos'.")
    return int(m.group(1)), m.group(2)


def resolver_tpm(n: int, variante: str) -> tuple[Path, np.ndarray]:
    """Busca NxV.csv y retorna (ruta, tpm) o lanza FileNotFoundError."""
    nombre = f"N{n}{variante}.csv"
    candidatos = (
        METHOD2_ROOT / "src" / ".samples" / nombre,
        METHOD2_ROOT / ".samples"          / nombre,
        GEOMIP_ROOT  / "data" / "samples"  / nombre,
    )
    for c in candidatos:
        if c.exists():
            return c, np.genfromtxt(c, delimiter=",")
    raise FileNotFoundError(
        f"No se encontro '{nombre}'. Buscado en: {', '.join(str(c) for c in candidatos)}"
    )


def leer_excel(ruta: Path, sheet_name: str) -> tuple[str, list[tuple]]:
    """
    Lee el Excel y retorna (estado_inicial, [(prueba, alcance_str, mecanismo_str), ...]).
    """
    from openpyxl import load_workbook
    wb = load_workbook(ruta, read_only=True, data_only=True)
    ws = wb[sheet_name]

    raw = ws["B1"].value
    estado_inicial = str(int(raw)) if isinstance(raw, (int, float)) else str(raw).strip()

    filas = []
    for row in ws.iter_rows(min_row=6, max_col=3, values_only=True):
        prueba, alcance_str, mecanismo_str = row
        if prueba is not None and alcance_str is not None and mecanismo_str is not None:
            filas.append((int(prueba), str(alcance_str).strip(), str(mecanismo_str).strip()))

    wb.close()
    return estado_inicial, filas


# ── Worker de multiprocessing ─────────────────────────────────────────────────

def _worker(
    estado_inicial: str,
    condiciones:    str,
    alcance:        str,
    mecanismo:      str,
    tpm:            np.ndarray,
    pagina:         str,
    n_max_exacto:   int,
    cola:           multiprocessing.Queue,
) -> None:
    """Proceso hijo: crea KGeoMIPHeuristica, ejecuta y pone resultados en la cola."""
    try:
        aplicacion.pagina_sample_network = pagina
        gestor = Manager(estado_inicial=estado_inicial)
        sia = KGeoMIPHeuristica(
            gestor,
            k_max=K_MAX,
            n_max_exhaustivo=n_max_exacto,
            decay_fn=decay_exponencial,
        )
        sia.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)

        resultado = {
            "modo":    sia._modo_usado,
            "por_k":  {},
            "error":  None,
        }
        for k, datos in sia._resultados_por_k.items():
            # Formatear la particion del mejor k
            asig = datos["asignacion"]
            if asig:
                import numpy as _np
                indices = sia.sia_subsistema.indices_ncubos
                dims    = sia.sia_subsistema.dims_ncubos
                comunes = _np.intersect1d(indices, dims)
                huerfanos = _np.setdiff1d(indices, dims)
                grupos_fmt = []
                for grupo in asig:
                    arr = _np.array(grupo, dtype=int)
                    nodos = [int(x) for x in comunes[arr]]
                    grupos_fmt.append((nodos, nodos))
                if huerfanos.size > 0:
                    grupos_fmt.append(
                        ([int(x) for x in huerfanos], [int(x) for x in dims])
                    )
                particion_str = fmt_k_particion(grupos_fmt)
            else:
                particion_str = None

            resultado["por_k"][k] = {
                "phi":          datos["phi"],
                "particion":    particion_str,
                "n_candidatos": datos["n_candidatos"],
                "tiempo_s":     datos["tiempo_s"],
            }

    except Exception as exc:
        resultado = {"modo": "error", "por_k": {}, "error": str(exc)}

    cola.put(resultado)


# ── Ejecutor principal ────────────────────────────────────────────────────────

def ejecutar_desde_excel(
    sheet_name: str,
    inicio:     int = 0,
    cantidad:   int = 50,
    pagina:     str = "A",
) -> None:
    n, variante = parsear_hoja(sheet_name)
    estado_inicial, todos_casos = leer_excel(INPUT_XLSX, sheet_name)

    # Usar variante del nombre de hoja para la TPM (salvo que pagina se sobreescriba)
    tpm_variante = variante
    tpm_path, tpm = resolver_tpm(n, tpm_variante)
    condiciones   = "1" * len(estado_inicial)

    filas = todos_casos[inicio : inicio + cantidad]

    print("=" * 70)
    print(f"KGeoMIP Simetrico — Hoja: {sheet_name}")
    print(f"Estado inicial: {estado_inicial}  n={len(estado_inicial)}")
    print(f"TPM: {tpm_path}")
    print(f"Pruebas: {inicio + 1} → {inicio + len(filas)}  |  k_max={K_MAX}  |  n_max_exacto={N_MAX_EXACTO}")
    print(f"Timeout por prueba: {TIMEOUT_S}s")
    print("=" * 70)

    resultados = []

    for prueba_num, alcance_str, mecanismo_str in filas:
        alcance   = convertir_a_binario(alcance_str,   len(estado_inicial))
        mecanismo = convertir_a_binario(mecanismo_str, len(estado_inicial))

        print(f"\n[{prueba_num:>3}] Alcance={alcance_str!r:<30} Mecanismo={mecanismo_str!r}")

        cola = multiprocessing.Queue()
        proceso = multiprocessing.Process(
            target=_worker,
            args=(estado_inicial, condiciones, alcance, mecanismo,
                  tpm, pagina, N_MAX_EXACTO, cola),
        )
        t_ini = time.perf_counter()
        proceso.start()
        proceso.join(timeout=TIMEOUT_S)
        t_total = round(time.perf_counter() - t_ini, 4)

        if proceso.is_alive():
            print(f"  [TIMEOUT {TIMEOUT_S}s] — prueba {prueba_num} terminada forzosamente.")
            proceso.terminate()
            proceso.join()
            por_k_data = {}
            modo = "timeout"
            error = f"Timeout ({TIMEOUT_S}s)"
        elif cola.empty():
            por_k_data = {}
            modo = "error"
            error = "Proceso termino sin resultado"
        else:
            res = cola.get()
            por_k_data = res.get("por_k", {})
            modo = res.get("modo", "?")
            error = res.get("error")

        fila: dict = {
            "#Prueba":   prueba_num,
            "Alcance":   alcance_str,
            "Mecanismo": mecanismo_str,
            "Modo":      modo,
            "T_total_s": t_total,
            "Error":     error,
        }

        for k in range(2, K_MAX + 1):
            prefix = f"k{k}"
            if k in por_k_data:
                d = por_k_data[k]
                phi_str = str(d["phi"]).replace(".", ",") if d["phi"] is not None else None
                fila[f"{prefix}_Particion"]    = d["particion"]
                fila[f"{prefix}_Perdida"]      = phi_str
                fila[f"{prefix}_Tiempo_s"]     = str(d["tiempo_s"]).replace(".", ",") if d["tiempo_s"] is not None else None
                fila[f"{prefix}_N_candidatos"] = d["n_candidatos"]
                if error is None:
                    print(
                        f"  k={k}: phi={d['phi']:.6f}  "
                        f"t={d['tiempo_s']:.3f}s  cand={d['n_candidatos']}"
                    )
            else:
                fila[f"{prefix}_Particion"]    = None
                fila[f"{prefix}_Perdida"]      = None
                fila[f"{prefix}_Tiempo_s"]     = None
                fila[f"{prefix}_N_candidatos"] = None

        if error:
            print(f"  ERROR: {error}")

        resultados.append(fila)

    # ── Guardar resultados ────────────────────────────────────────────────────
    df = pd.DataFrame(resultados)
    nombre_hoja = sheet_name.strip().replace(" ", "_")
    ruta_salida = GEOMIP_ROOT / "results" / f"resultados_kgeomip_simetrico_{nombre_hoja}.xlsx"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ruta_salida, index=False)

    print(f"\n{'=' * 70}")
    print(f"Resultados guardados en: {ruta_salida}")

    completadas = sum(1 for r in resultados if r["Error"] is None)
    print(f"Completadas: {completadas}/{len(resultados)}  |  Timeouts/Errores: {len(resultados)-completadas}")
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecutor KGeoMIP Simetrico")
    parser.add_argument("--hoja",     required=True,  help="Nombre de la hoja Excel (ej. '10A-Elementos')")
    parser.add_argument("--inicio",   type=int, default=0,  help="Indice de inicio (0-based)")
    parser.add_argument("--cantidad", type=int, default=50, help="Numero de pruebas a ejecutar")
    args = parser.parse_args()

    ejecutar_desde_excel(
        sheet_name = args.hoja,
        inicio     = args.inicio,
        cantidad   = args.cantidad,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
