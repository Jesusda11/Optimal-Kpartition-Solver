"""
smoke_fase0.py — Verificacion de la Fase 0 de KQNodes.

NO implementa algoritmo. Solo valida que:
  1. El arbol KQNodes importa (clase KQNodes + utilidades portadas).
  2. El QNodes ORIGINAL (copiado) corre end-to-end k=2 en el arbol nuevo.
  3. El `k_bipartir` portado reproduce EXACTAMENTE `bipartir` (chequeo semantico).
  4. generar_k_particiones / contar_stirling son consistentes (S(n,k)).
  5. fmt_k_particion produce salida no vacia.

Ejecutar desde el directorio KQNodes/:
    <python> smoke_fase0.py
"""

import sys
import numpy as np

# Evitar crash de consola cp1252 con caracteres unicode (⎛ ⎞ ∅).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

resultados = []

def check(nombre, cond, detalle=""):
    estado = "PASS" if cond else "FAIL"
    resultados.append((nombre, cond))
    print(f"  [{estado}] {nombre}" + (f"  -> {detalle}" if detalle else ""))


print("=" * 78)
print("SMOKE TEST — Fase 0 KQNodes")
print("=" * 78)

# ── 1. Imports ──────────────────────────────────────────────────────────────
print("\n[1] Imports del arbol KQNodes")
try:
    from src.models.base.application import aplicacion
    from src.controllers.manager import Manager
    from src.strategies.kqnodes import KQNodes
    from src.strategies.q_nodes import QNodes
    from src.funcs.partitions import generar_k_particiones, contar_stirling
    from src.funcs.format import fmt_k_particion
    from src.funcs.iit import emd_efecto
    check("import KQNodes + utilidades portadas + emd_efecto", True)
except Exception as exc:
    check("import KQNodes + utilidades portadas + emd_efecto", False, repr(exc))
    print("\nABORTADO: fallo de import.")
    sys.exit(1)

# ── 2. QNodes original corre k=2 end-to-end en el arbol nuevo ────────────────
print("\n[2] QNodes original (copiado) corre k=2 en KQNodes/")
aplicacion.set_pagina_red_muestra("A")
EST, COND, ALC, MEC = "1000", "1111", "1111", "1111"
try:
    gestor = Manager(EST)
    tpm = gestor.cargar_red()  # carga src/.samples/N4A.csv
    q = QNodes(tpm)
    sol = q.aplicar_estrategia(EST, COND, ALC, MEC)
    phi_q = float(sol.perdida)
    check("QNodes.aplicar_estrategia k=2 (N4A)", np.isfinite(phi_q), f"phi={phi_q:.6f}")
except Exception as exc:
    check("QNodes.aplicar_estrategia k=2 (N4A)", False, repr(exc))
    phi_q = None

# ── 3. Chequeo semantico: k_bipartir reproduce bipartir ─────────────────────
print("\n[3] k_bipartir portado == bipartir (semantica identica)")
try:
    kq = KQNodes(tpm)                       # usa SIA(tpm) heredado
    kq.sia_preparar_subsistema(EST, COND, ALC, MEC)
    sub = kq.sia_subsistema

    futuros = [int(x) for x in sub.indices_ncubos]
    presentes = [int(x) for x in sub.dims_ncubos]
    h_f = len(futuros) // 2
    h_p = len(presentes) // 2
    A = futuros[:h_f]                       # alcance (futuros del grupo 1)
    M = presentes[:h_p]                     # mecanismo (presentes del grupo 1)
    A_comp = [f for f in futuros if f not in A]
    M_comp = [p for p in presentes if p not in M]

    dist_bip = sub.bipartir(
        np.array(A, dtype=np.int8), np.array(M, dtype=np.int8)
    ).distribucion_marginal()

    dist_kbip = sub.k_bipartir(
        [(A, M), (A_comp, M_comp)]
    ).distribucion_marginal()

    iguales = np.allclose(dist_bip, dist_kbip, atol=1e-9)
    maxdiff = float(np.max(np.abs(dist_bip - dist_kbip))) if dist_bip.size else 0.0
    check("k_bipartir([(A,M),(A',M')]) == bipartir(A,M)", iguales, f"maxdiff={maxdiff:.2e}")

    # phi de esa misma particion via emd_efecto (consistencia con el pipeline)
    phi_kbip = float(emd_efecto(dist_kbip, kq.sia_dists_marginales))
    check("emd_efecto sobre k_bipartir finito", np.isfinite(phi_kbip), f"phi={phi_kbip:.6f}")
except Exception as exc:
    check("k_bipartir([(A,M),(A',M')]) == bipartir(A,M)", False, repr(exc))

# ── 4. Generadores de k-particiones ─────────────────────────────────────────
print("\n[4] generar_k_particiones / contar_stirling")
try:
    casos = [(4, 2, 7), (4, 3, 6), (5, 3, 25), (6, 4, 65)]
    ok_gen = True
    det = []
    for n, k, esperado in casos:
        cnt = sum(1 for _ in generar_k_particiones(n, k))
        stir = contar_stirling(n, k)
        ok = (cnt == stir == esperado)
        ok_gen = ok_gen and ok
        det.append(f"S({n},{k})={cnt}/{stir}/{esperado}{'' if ok else ' !!'}")
    check("conteos == Stirling esperado", ok_gen, "  ".join(det))

    # Cada k-particion cubre {0..n-1} sin solapamiento
    grupos = list(generar_k_particiones(5, 3))[0]
    plano = sorted(x for g in grupos for x in g)
    check("k-particion cubre {0..n-1} disjunta", plano == list(range(5)), str(grupos))
except Exception as exc:
    check("conteos == Stirling esperado", False, repr(exc))

# ── 5. fmt_k_particion ──────────────────────────────────────────────────────
print("\n[5] fmt_k_particion")
try:
    s = fmt_k_particion([([0], [0]), ([1], [1, 2])])
    check("fmt_k_particion devuelve str no vacio", isinstance(s, str) and len(s) > 0,
          f"len={len(s)}")
except Exception as exc:
    check("fmt_k_particion devuelve str no vacio", False, repr(exc))

# ── 6. KQNodes scaffold: aplicar_estrategia aun NO implementado ──────────────
print("\n[6] KQNodes scaffold (aplicar_estrategia debe seguir sin implementar)")
try:
    kq2 = KQNodes(tpm, k_min=2, k_max=5)
    raised = False
    try:
        kq2.aplicar_estrategia(EST, COND, ALC, MEC)
    except NotImplementedError:
        raised = True
    check("aplicar_estrategia lanza NotImplementedError (Fase 0)", raised,
          f"k_min={kq2.k_min} k_max={kq2.k_max}")
except Exception as exc:
    check("aplicar_estrategia lanza NotImplementedError (Fase 0)", False, repr(exc))

# ── Resumen ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
total = len(resultados)
passed = sum(1 for _, c in resultados if c)
print(f"RESUMEN: {passed}/{total} checks PASS")
if passed == total:
    print("FASE 0 OK — portacion limpia, arbol funcional, KQNodes lista para Fase 1.")
else:
    print("FASE 0 CON FALLOS — revisar arriba.")
print("=" * 78)
sys.exit(0 if passed == total else 1)
