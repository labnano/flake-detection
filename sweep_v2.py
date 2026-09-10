"""
sweep_v2.py  --  PASSO 3 do ajuste de parametros: escolher o ajuste e medir
quanto da para confiar nele.

O que mudou em relacao ao sweep.py
-----------------------------------
1) O VETO DO CLASSIFICADOR ENTRA NA CONTA -- E COMO PARAMETRO.
   O sweep antigo aplicava score/size/occupancy e ignorava o filtro que mais
   mata candidato, entao otimizava um pipeline que o microscopio nao executa.
   Aqui a regra reproduz a ordem real do MaskTerial:

       score > s   E   size > z   E   classe(t) != 0   E   ocupacao(t) >= o
       E   contraste verde <= g

   O 't' e o LIMIAR_MAHALANOBIS, que o extrator gravou para varios valores --
   por isso ele pode ser varrido aqui, offline, como qualquer outro.

2) A METRICA E OPERACIONAL, NAO F1.
   Tres vezes seguidas em agosto a regra de maior F1 venceu jogando fora os
   melhores flakes: eram minoria, e o F1 pesa todo flake igual. Aqui a pergunta
   e outra, e esta na moeda que importa -- o seu tempo de microscopio:

       "aceitando olhar N candidatos por 1000 tiles, que fracao dos BONS
        flakes eu capturo?"

   Por 1000 tiles, e nao por amostra, porque a extracao e uma fatia: a unidade
   precisa ser independente de quantos tiles foram sorteados.

3) OS ROTULOS SAO PESADOS.
   Cada rotulo veio de um sorteio estratificado e representa n_estrato/n_vistos
   candidatos da populacao. Sem esse peso, ter olhado proporcionalmente mais
   candidatos de score alto faria parecer que a populacao inteira e assim.

4) A VALIDACAO E ENTRE AMOSTRAS (leave-one-sample-out).
   A validacao cruzada de agosto sorteava candidatos ao acaso -- mas os cinco
   folds vinham da MESMA amostra, mesma cola, mesma exposicao. Ela media
   variancia amostral, nao generalizacao. Aqui cada fold e uma amostra INTEIRA:
   escolhe o ajuste com as outras, mede na que ficou de fora. E a unica forma
   de responder "isso vai funcionar na proxima amostra?".

Uso:
    python sweep_v2.py <pasta/avaliacao_lote> [--orcamento 40]
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

# --- GRADES varridas (mexa aqui) ---
MAHALANOBIS = ["10", "20", "30", "50", "100", "inf"]
SCORES = [0.015, 0.03, 0.05, 0.10, 0.20, 0.40]
SIZES = [0, 50, 200]
OCC_MIN = [0.0, 0.07, 0.30]
CG_MAX = [1.0, 0.0, -0.02, -0.04, -0.08, -0.15]   # 1.0 = sem filtro de cor

# Orcamentos (candidatos aceitos por 1000 tiles) para desenhar a curva.
ORCAMENTOS = [5, 10, 20, 40, 80, 160, 320]

# O ajuste que o microscopio usa HOJE (parametros.py), para comparacao.
HOJE = dict(t="30", s=0.015, z=50, o=0.07, g=-0.04)


def carregar(pasta, arquivo_rotulos="rotulos.csv"):
    """Junta candidatos + rotulos + tiles. Devolve um dicionario de arrays."""
    def ler(nome):
        with open(os.path.join(pasta, nome), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    cands = {r["cand_id"]: r for r in ler("candidatos.csv")}
    tiles = ler("tiles.csv")
    rotulos = ler(arquivo_rotulos)

    # Quantos tiles por amostra: e o denominador de "por 1000 tiles".
    n_tiles = defaultdict(int)
    for t in tiles:
        n_tiles[t["amostra"]] += 1

    dados = defaultdict(list)
    descartados = 0
    for r in rotulos:
        if r["rotulo"] not in ("2", "1", "0"):    # duvidas ficam de fora
            descartados += 1
            continue
        c = cands.get(r["cand_id"])
        if c is None:
            continue
        dados["amostra"].append(r["amostra"])
        dados["rotulo"].append(int(r["rotulo"]))
        dados["tipo"].append(r["tipo"])
        # peso: quantos candidatos da populacao este rotulo representa
        dados["peso"].append(int(r["n_estrato"]) / int(r["n_vistos"]))
        dados["score"].append(float(c["score"]))
        dados["size"].append(float(c["size_px"]))
        dados["c_g"].append(float(c["c_g"]))
        dados["c_r"].append(float(c["c_r"]))
        for t in MAHALANOBIS:
            dados[f"cls_{t}"].append(int(c[f"cls_{t}"]))
            dados[f"occ_{t}"].append(float(c[f"occ_{t}"]))

    out = {k: np.array(v) for k, v in dados.items()}
    out["n_tiles"] = dict(n_tiles)
    if descartados:
        print(f"({descartados} duvidas excluidas)")
    return out


def aceita(d, t, s, z, o, g):
    """Aplica a regra completa, na ordem em que o MaskTerial aplica.

    Os '>' sao estritos porque e assim no filter_instances do MaskTerial
    (scores > threshold, count_nonzero > size_threshold); o '>=' da ocupacao
    tambem vem de la (max_prob < min_class_occupancy descarta).
    """
    return ((d["score"] > s)
            & (d["size"] > z)
            & (d[f"cls_{t}"] != 0)
            & (d[f"occ_{t}"] >= o)
            & (d["c_g"] <= g))


def medir(d, pred, amostras):
    """Metricas por amostra, ja com os pesos aplicados.

    Devolve, para cada amostra: quantos candidatos por 1000 tiles a regra
    entrega, e que fracao dos flakes BONS (rotulo 2) ela captura.
    """
    linhas = {}
    for a in amostras:
        m = d["amostra"] == a
        w = d["peso"][m]
        aceito = pred[m]
        bom = d["rotulo"][m] == 2
        qualquer = d["rotulo"][m] >= 1

        total_bons = float(np.sum(w[bom]))
        aceitos = float(np.sum(w[aceito]))
        linhas[a] = {
            "por_mil": 1000.0 * aceitos / max(d["n_tiles"].get(a, 1), 1),
            "recall_bons": float(np.sum(w[aceito & bom])) / total_bons if total_bons else np.nan,
            "recall_todos": (float(np.sum(w[aceito & qualquer]))
                             / max(float(np.sum(w[qualquer])), 1e-9)),
            "precisao": (float(np.sum(w[aceito & qualquer])) / aceitos) if aceitos else 0.0,
            "bons_perdidos": float(np.sum(w[~aceito & bom])),
        }
    return linhas


def resume(linhas):
    """Agrega as amostras. Media SIMPLES entre amostras, nao juntando todos os
    candidatos num balaio so: cada amostra e uma unidade experimental, e a que
    tiver mais candidatos nao deve dominar a media."""
    def med(campo):
        v = [x[campo] for x in linhas.values() if not np.isnan(x[campo])]
        return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)
    return {"por_mil": med("por_mil")[0],
            "recall_bons": med("recall_bons")[0],
            "recall_bons_dp": med("recall_bons")[1],
            "recall_todos": med("recall_todos")[0],
            "precisao": med("precisao")[0]}


def combos():
    for t in MAHALANOBIS:
        for s in SCORES:
            for z in SIZES:
                for o in OCC_MIN:
                    for g in CG_MAX:
                        yield dict(t=t, s=s, z=z, o=o, g=g)


def texto(c):
    return (f"maha={c['t']:>3} score>{c['s']:<5.3f} size>{c['z']:<4.0f} "
            f"occ>={c['o']:<4.2f} G<={c['g']:+.2f}")


def melhor_sob_orcamento(orcamento, avaliados):
    """Entre as regras que cabem no orcamento, a que captura mais flakes bons.

    Devolve (regra, metricas) ou (None, None) se nenhuma regra couber -- o que
    acontece quando o orcamento e menor que o que a regra mais restritiva da
    grade ainda entrega.
    """
    melhor, melhor_r = None, None
    for c, r in avaliados:
        if np.isnan(r["recall_bons"]) or r["por_mil"] > orcamento:
            continue
        if melhor_r is None or r["recall_bons"] > melhor_r["recall_bons"]:
            melhor, melhor_r = c, r
    return melhor, melhor_r


def main(pasta, orcamento, arquivo_rotulos="rotulos.csv"):
    d = carregar(pasta, arquivo_rotulos)
    amostras = sorted(set(d["amostra"]))
    n = len(d["rotulo"])
    print(f"rotulos: {n} | amostras: {len(amostras)}")
    for a in amostras:
        m = d["amostra"] == a
        print(f"  {a:<32} {int(np.sum(m)):>4} rotulos | "
              f"bons={int(np.sum(m & (d['rotulo'] == 2))):>3} "
              f"marginais={int(np.sum(m & (d['rotulo'] == 1))):>3} "
              f"lixo={int(np.sum(m & (d['rotulo'] == 0))):>4} | "
              f"{d['n_tiles'].get(a, 0)} tiles")

    # --- (1) O ajuste de hoje, como referencia ---
    r_hoje = resume(medir(d, aceita(d, **HOJE), amostras))
    print(f"\n=== ajuste de HOJE ({texto(HOJE)}) ===")
    print(f"  entrega {r_hoje['por_mil']:.1f} candidatos/1000 tiles | "
          f"captura {100*r_hoje['recall_bons']:.0f}% dos bons | "
          f"precisao {100*r_hoje['precisao']:.0f}%")

    # --- (2) Avalia a grade inteira ---
    avaliados = [(c, resume(medir(d, aceita(d, **c), amostras))) for c in combos()]
    print(f"\n{len(avaliados)} combinacoes avaliadas.")

    # --- (3) A curva: o melhor recall de bons a cada orcamento ---
    print("\n=== curva de operacao ===")
    print(f"  {'orcamento':>9}  {'% dos bons':>10}  regra")
    for orc in ORCAMENTOS:
        c, r = melhor_sob_orcamento(orc, avaliados)
        if c is None:
            print(f"  {orc:>9}  {'--':>10}  (nenhuma regra cabe)")
            continue
        print(f"  {orc:>9}  {100*r['recall_bons']:>9.0f}%  {texto(c)}")

    # --- (4) Validacao leave-one-sample-out no orcamento pedido ---
    print(f"\n=== leave-one-sample-out (orcamento = {orcamento}/1000 tiles) ===")
    print("  escolhe o ajuste SEM ver a amostra, mede nela.\n")
    escolhidos, recalls = [], []
    for fora in amostras:
        treino = [a for a in amostras if a != fora]
        aval_treino = [(c, resume(medir(d, aceita(d, **c), treino))) for c in combos()]
        c, _ = melhor_sob_orcamento(orcamento, aval_treino)
        if c is None:
            print(f"  {fora:<32} nenhuma regra cabe no orcamento")
            continue
        r_teste = medir(d, aceita(d, **c), [fora])[fora]
        escolhidos.append(tuple(c.items()))
        recalls.append(r_teste["recall_bons"])
        print(f"  {fora:<32} {100*r_teste['recall_bons']:>5.0f}% dos bons | "
              f"{r_teste['por_mil']:>6.1f}/1000 | {texto(c)}")

    if recalls:
        val = [r for r in recalls if not np.isnan(r)]
        print(f"\n  recall de bons FORA DA AMOSTRA: {100*np.mean(val):.0f}% "
              f"(+/- {100*np.std(val):.0f})")
        mais_comum = max(set(escolhidos), key=escolhidos.count)
        print(f"  estabilidade: o mesmo ajuste venceu em "
              f"{escolhidos.count(mais_comum)}/{len(escolhidos)} folds")
        print(f"  -> {texto(dict(mais_comum))}")

    # --- (5) De que tipo e o que sobra de falso positivo ---
    c_final, _ = melhor_sob_orcamento(orcamento, avaliados)
    if c_final is not None:
        pred = aceita(d, **c_final)
        fp = pred & (d["rotulo"] == 0)
        print(f"\n=== falsos positivos que sobram (ajuste de {orcamento}/1000) ===")
        total = float(np.sum(d["peso"][fp]))
        if total <= 0:
            print("  nenhum -- entre os candidatos rotulados, essa regra nao "
                  "aceitou nada que voce marcou como lixo.")
        else:
            for tipo in sorted(set(d["tipo"][fp])):
                m = fp & (d["tipo"] == tipo)
                peso = float(np.sum(d["peso"][m]))
                nome = tipo if tipo else "(nao classificado)"
                print(f"  {nome:<20} {100*peso/total:>5.1f}%")

    # --- (6) Grade completa em disco, para plotar depois ---
    saida = os.path.join(pasta, "resultados_sweep_v2.csv")
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["maha", "score", "size", "occ_min", "cg_max",
                    "por_mil_tiles", "recall_bons", "recall_bons_dp",
                    "recall_todos", "precisao"])
        for c, r in avaliados:
            w.writerow([c["t"], c["s"], c["z"], c["o"], c["g"],
                        f"{r['por_mil']:.2f}", f"{r['recall_bons']:.4f}",
                        f"{r['recall_bons_dp']:.4f}",
                        f"{r['recall_todos']:.4f}", f"{r['precisao']:.4f}"])
    print(f"\ngrade completa: {saida}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pasta", help="Pasta com candidatos.csv, rotulos.csv e tiles.csv")
    p.add_argument("--orcamento", type=float, default=40,
                   help="Candidatos aceitos por 1000 tiles que voce topa olhar")
    p.add_argument("--rotulos", default="rotulos.csv",
                   help="nome do CSV de rotulos dentro da pasta "
                        "(use rotulos_consolidado.csv apos consolidar)")
    args = p.parse_args()
    main(os.path.abspath(args.pasta), args.orcamento, args.rotulos)
