"""
sweep_cor.py  --  sweep estendido: combina os limiares com o contraste de cor,
e mede o resultado com VALIDACAO CRUZADA (para nao se enganar com overfit).

O que mudou em relacao ao sweep.py
----------------------------------
- Aposentei `size` e `occ_min`: a analise mostrou que nao separam nada.
- Adicionei o contraste de cor como regra: `c_g <= G_MAX` (verde, o sinal forte)
  e `c_r <= R_MAX` (vermelho, sinal auxiliar). Contraste negativo = mais escuro
  que o substrato = cara de grafeno.
- Mantive `score` (piso) e `occ_max` (teto de ocupacao) como sinais auxiliares.

Regra: ACEITA se  score >= s  E  occ <= omax  E  c_g <= gmax  E  c_r <= rmax.

Por que validacao cruzada (a parte importante)
----------------------------------------------
Se a gente testa centenas de combinacoes e escolhe a de maior F1 sobre TODOS os
dados, o "melhor" pode ser sorte -- ajustado a ESTES candidatos, nao ao problema.
Quanto mais combinacoes, maior o risco. Isso e overfit da SELECAO.

A validacao cruzada (k-fold) contorna: divide os dados em k partes; para cada
parte, escolhe a melhor regra usando as OUTRAS k-1 (treino) e mede o F1 na parte
reservada (teste), que a regra nao viu. A media desses F1 de teste e um numero
HONESTO -- estima como a regra se sai em dados novos. O F1 "no conjunto todo" e
OTIMISTA; a diferenca entre os dois e o tamanho da ilusao.

(A validacao final de verdade e rodar numa OUTRA varredura -- isto aqui e a
melhor estimativa possivel com uma varredura so.)

Uso:
    python sweep_cor.py <caminho/para/candidatos.csv>
"""

import argparse
import csv
import os

import numpy as np

# --- GRADES (mexa aqui) ---
SCORES  = [0.015, 0.05, 0.10, 0.20, 0.40]
OCC_MAX = [1.01, 0.99, 0.95]                       # 1.01 = sem teto de ocupacao
G_MAX   = [0.02, 0.0, -0.02, -0.04, -0.06, -0.08, -0.10, -0.15]  # teto no verde
R_MAX   = [1.01, -0.05, -0.10, -0.20]             # 1.01 = sem filtro no vermelho

BETA = 1.0     # F-beta: 1=equilibrio; >1 favorece recall; <1 favorece precisao
K = 5          # numero de folds da validacao cruzada
SEED = 42
TOP_N = 12

COMBOS = [(s, omax, gmax, rmax)
          for s in SCORES for omax in OCC_MAX for gmax in G_MAX for rmax in R_MAX]


def carregar(csv_path):
    """Junta candidatos (score, occ) + contraste (c_g, c_r) + rotulos. Arrays."""
    base = os.path.dirname(csv_path)
    cand = {r["cand_id"]: r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    contr = {r["cand_id"]: r
             for r in csv.DictReader(open(os.path.join(base, "contraste.csv"), encoding="utf-8"))}

    score, occ, cg, cr, y = [], [], [], [], []
    for r in csv.DictReader(open(os.path.join(base, "rotulos.csv"), encoding="utf-8")):
        cid = r["cand_id"]
        if r["rotulo"] not in ("0", "1") or cid not in contr:
            continue
        score.append(float(cand[cid]["score"]))
        occ.append(float(cand[cid]["max_prob"]))
        cg.append(float(contr[cid]["c_g"]))
        cr.append(float(contr[cid]["c_r"]))
        y.append(int(r["rotulo"]))
    return tuple(np.array(a) for a in (score, occ, cg, cr, y))


def metrica(pred, y):
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = BETA * BETA
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0
    return f, p, r, tp, fp, fn


def regra(score, occ, cg, cr, s, omax, gmax, rmax):
    return (score >= s) & (occ <= omax) & (cg <= gmax) & (cr <= rmax)


def melhor_combo(score, occ, cg, cr, y, idx):
    """Escolhe a combinacao de maior F1 sobre o subconjunto idx (o 'treino')."""
    best_f, best_c = -1.0, None
    for c in COMBOS:
        f, *_ = metrica(regra(score[idx], occ[idx], cg[idx], cr[idx], *c), y[idx])
        if f > best_f:
            best_f, best_c = f, c
    return best_c


def main(csv_path):
    score, occ, cg, cr, y = carregar(csv_path)
    n = len(y)
    print(f"candidatos: {n}  (flakes={int(np.sum(y==1))}, lixo={int(np.sum(y==0))})\n")

    # (1) Referencias, no conjunto todo.
    print("=== referencias (conjunto todo) ===")
    for nome, pred in [("score>=0.05", score >= 0.05),
                       ("teto occ<=0.85", occ <= 0.85),
                       ("G<=-0.04 sozinho", cg <= -0.04)]:
        f, p, r, tp, fp, fn = metrica(pred, y)
        print(f"  {nome:<20} F1={f:.3f}  P={p:.3f}  R={r:.3f}  (TP={tp} FP={fp} FN={fn})")

    # (2) Sweep no conjunto todo -> F1 OTIMISTA (com vies de selecao).
    res = []
    for c in COMBOS:
        f, p, r, tp, fp, fn = metrica(regra(score, occ, cg, cr, *c), y)
        res.append((f, p, r, tp, fp, fn, c))
    res.sort(key=lambda x: x[0], reverse=True)

    print(f"\n=== TOP {TOP_N} no conjunto todo (F1 OTIMISTA) ===")
    print(f"  {'score':>5} {'occ<=':>5} {'G<=':>6} {'R<=':>6} |  F1     P     R    (TP/FP/FN)")
    for f, p, r, tp, fp, fn, c in res[:TOP_N]:
        print(f"  {c[0]:>5.3f} {c[1]:>5.2f} {c[2]:>+6.2f} {c[3]:>+6.2f} | "
              f"{f:.3f} {p:.3f} {r:.3f}  ({tp}/{fp}/{fn})")

    # (3) Validacao cruzada -> F1 HONESTO (fora da amostra).
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(n), K)
    f1s, escolhidos = [], []
    for i in range(K):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(K) if j != i])
        c = melhor_combo(score, occ, cg, cr, y, train)
        f, *_ = metrica(regra(score[test], occ[test], cg[test], cr[test], *c), y[test])
        f1s.append(f)
        escolhidos.append(c)

    print(f"\n=== validacao cruzada {K}-fold (F1 HONESTO) ===")
    print(f"  F1 medio fora-da-amostra = {np.mean(f1s):.3f}  (+/- {np.std(f1s):.3f})")
    print("  combo escolhido por fold (estabilidade):")
    for i, (c, f) in enumerate(zip(escolhidos, f1s)):
        print(f"    fold {i+1}: score>={c[0]:.3f} occ<={c[1]:.2f} G<={c[2]:+.2f} R<={c[3]:+.2f}"
              f"   -> F1_teste={f:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Caminho para o candidatos.csv")
    args = parser.parse_args()
    main(os.path.abspath(args.csv))
