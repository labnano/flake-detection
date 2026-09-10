"""
sweep.py  --  PASSO 3 de 3: varre combinacoes de limiares e mede o desempenho.

O que faz
---------
Junta candidatos.csv (score, size, max_prob) + rotulos.csv (o gabarito que voce
anotou/revisou) e, para cada combinacao de regras de filtro, calcula:

    precisao = dos candidatos ACEITOS, quantos sao flake de verdade
    recall   = dos flakes de verdade, quantos foram aceitos
    F1       = media harmonica das duas (o equilibrio)

Um candidato e ACEITO (previsto como flake) se passa em TODAS as condicoes:
    score >= s   E   size >= z   E   max_prob >= occ_min   E   max_prob <= occ_max

As tres primeiras sao os limiares nativos do MaskTerial (SCORE/SIZE/
MIN_CLASS_OCCUPANCY). A quarta -- occ_max, um TETO de ocupacao -- e a regra
extra sugerida pela sujeira de cola (que tem ocupacao ~1.0). Como tudo aqui e
offline, da pra testar essa regra mesmo o MaskTerial nao a tendo.

Nota sobre a validade: os rotulos cobrem TODOS os candidatos com score >= 0.015
(nao e amostra). Como todo limiar testado tambem e >= 0.015, precisao e recall
sao EXATOS -- sem vies de amostragem.

Uso:
    python sweep.py <caminho/para/candidatos.csv>
"""

import argparse
import csv
import os

import numpy as np

# --- GRADES varridas (mexa aqui) ---
SCORES  = [0.015, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
SIZES   = [0, 50, 100, 200, 400]          # 0 = sem filtro de tamanho; 50 = default
OCC_MIN = [0.0, 0.025]                     # piso de ocupacao; 0 = sem; 0.025 = default
OCC_MAX = [1.01, 0.99, 0.98, 0.97, 0.95, 0.92, 0.90, 0.85]  # 1.01 = sem teto

BETA = 1.0     # F-beta: 1=equilibrio; >1 favorece recall (nao perder flake); <1 favorece precisao
TOP_N = 15     # quantas melhores combinacoes mostrar

# Limiares que o microscopio usa HOJE (referencia para bater).
DEFAULT = dict(s=0.05, z=50, omin=0.025, omax=1.01)


def carregar(csv_path):
    """Junta candidatos + rotulos por cand_id; exclui os '?'. Devolve arrays."""
    base = os.path.dirname(csv_path)
    cand = {r["cand_id"]: r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}

    score, size, occ, y = [], [], [], []
    for r in csv.DictReader(open(os.path.join(base, "rotulos.csv"), encoding="utf-8")):
        if r["rotulo"] not in ("0", "1"):   # exclui duvidas irredutiveis
            continue
        c = cand[r["cand_id"]]
        score.append(float(c["score"]))
        size.append(int(c["size"]))
        occ.append(float(c["max_prob"]))
        y.append(int(r["rotulo"]))
    return (np.array(score), np.array(size, dtype=float),
            np.array(occ), np.array(y))


def avaliar(score, size, occ, y, s, z, omin, omax):
    """Aplica uma regra e devolve as metricas (vetorizado sobre todos os candidatos)."""
    pred = (score >= s) & (size >= z) & (occ >= omin) & (occ <= omax)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = BETA * BETA
    f = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) else 0.0
    return {"s": s, "z": z, "omin": omin, "omax": omax,
            "tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f": f}


CAB = f"{'score':>6} {'size':>4} {'occ>=':>5} {'occ<=':>5} | {'TP':>4} {'FP':>5} {'FN':>4} | {'prec':>5} {'rec':>5} {'F1':>5}"


def linha(r):
    return (f"{r['s']:>6.3f} {int(r['z']):>4} {r['omin']:>5.3f} {r['omax']:>5.2f} | "
            f"{r['tp']:>4} {r['fp']:>5} {r['fn']:>4} | "
            f"{r['prec']:>5.3f} {r['rec']:>5.3f} {r['f']:>5.3f}")


def main(csv_path):
    score, size, occ, y = carregar(csv_path)
    n_flakes = int(np.sum(y == 1))
    print(f"candidatos rotulados: {len(y)}  (flakes={n_flakes}, lixo={len(y)-n_flakes})\n")

    # (1) Referencia: o ajuste default de hoje.
    d = avaliar(score, size, occ, y, **DEFAULT)
    print("=== DEFAULT atual (referencia) ===")
    print(CAB)
    print(linha(d), "\n")

    # (2) Score sozinho: mostra o teto do score isolado (sem os outros filtros).
    print("=== score sozinho (size>=0, sem piso/teto de ocupacao) ===")
    print(CAB)
    for s in SCORES:
        print(linha(avaliar(score, size, occ, y, s, 0, 0.0, 1.01)))
    print()

    # (3) Grid completo -> melhores por F1 (ou F-beta).
    resultados = [avaliar(score, size, occ, y, s, z, omin, omax)
                  for s in SCORES for z in SIZES
                  for omin in OCC_MIN for omax in OCC_MAX]
    resultados.sort(key=lambda r: r["f"], reverse=True)

    print(f"=== TOP {TOP_N} combinacoes por F{'' if BETA==1 else BETA} ===")
    print(CAB)
    for r in resultados[:TOP_N]:
        print(linha(r))

    melhor = resultados[0]
    print(f"\nmelhor F1 = {melhor['f']:.3f}   vs   default = {d['f']:.3f}")
    if melhor["omax"] < 1.0:
        print(f"  -> a melhor regra USA teto de ocupacao (occ <= {melhor['omax']:.2f}): a pista da cola ajudou.")
    else:
        print("  -> a melhor regra NAO usa teto de ocupacao.")

    # (4) Salva o grid inteiro para voce explorar/plotar depois.
    saida = os.path.join(os.path.dirname(csv_path), "resultados_sweep.csv")
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["score", "size", "occ_min", "occ_max", "tp", "fp", "fn", "prec", "rec", "f"])
        for r in resultados:
            w.writerow([r["s"], int(r["z"]), r["omin"], r["omax"],
                        r["tp"], r["fp"], r["fn"],
                        f"{r['prec']:.4f}", f"{r['rec']:.4f}", f"{r['f']:.4f}"])
    print(f"\ngrid completo salvo em: {saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Caminho para o candidatos.csv")
    args = parser.parse_args()
    main(os.path.abspath(args.csv))
