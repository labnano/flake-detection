"""
fila_dirigida.py  --  PASSO 5: rotular onde o modelo decide, nao onde o sorteio
cai.

O problema que este script resolve
-----------------------------------
A curva de aprendizado de 27/08 saiu PLANA: entre 126 e 419 rotulos de treino, o
desempenho nao muda. Rotular mais candidatos do mesmo jeito nao compra
desempenho -- isso foi medido, nao suposto.

Mas o mesmo run mostrou que o desvio entre amostras fica em 27-37% em TODO ponto
da curva, e a causa e outra: 15 flakes bons espalhados em 6 amostras (duas com 2
cada, uma com zero). Nao da para decidir se 41% e melhor que 20% com essa
amostra.

Ou seja: **rotulo para treinar e rotulo para medir sao recursos diferentes.** O
treino saturou; a medicao nem comecou. Este script ataca a medicao.

A ideia
-------
O sorteio estratificado por score x veto encontrou 15 bons em 512 rotulos -- uns
3 bons a cada 100. Mas agora existe um modelo que ORDENA os candidatos, e a
precisao medida no topo do ranking foi de 48% a 73%. Rotulando o topo, a mesma
hora de trabalho rende de 15 a 25 vezes mais flakes bons.

Isso e active learning na forma mais simples que existe: usar o modelo atual
para decidir o que vale a pena olhar em seguida.

Um cuidado estatistico que NAO pode ser esquecido
--------------------------------------------------
Os rotulos novos virao de um estrato novo -- "topo do ranking do modelo" -- que
se SOBREPOE aos estratos antigos (um candidato pode ser 'alto|vivo' e tambem
estar no topo). Peso de estrato so funciona com estratos DISJUNTOS: com
sobreposicao ha dupla contagem e a populacao estimada fica errada.

Por isso os pesos gravados pelo anotador nesta fila NAO devem ser usados como
estao. Depois de rotular, a estratificacao inteira precisa ser recalculada
cruzando o esquema antigo com "topo/resto" -- 'consolidar_rotulos.py', que vem
depois. Aqui a fila e so a fila.

Uso:
    python fila_dirigida.py <pasta/avaliacao_lote> [--por-amostra 50]
                            [--features ...] [--C 1.0] [--peso-bom 3]
"""

import argparse
import csv
import os

import numpy as np

import treinar_filtro as tf


def carregar_todos(pasta):
    """Todos os candidatos do CSV, com as mesmas derivadas do treino.

    Diferente do tf.carregar(), que so devolve os que tem rotulo: aqui e a
    populacao inteira, porque e nela que o modelo vai pontuar.
    """
    caminho = os.path.join(pasta, "candidatos.csv")
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    d = {k: np.array([float(r[k]) for r in linhas]) for k in tf.COLUNAS}
    d["cand_id"] = np.array([r["cand_id"] for r in linhas])
    d["amostra"] = np.array([r["amostra"] for r in linhas])

    forma = np.array([tf.forma(r["contorno"]) for r in linhas], dtype=float)
    d["solidez"], d["arcarea"] = forma[:, 0], forma[:, 1]
    for k in ["solidez", "arcarea"]:
        faltando = np.isnan(d[k])
        if np.any(faltando):
            d[k][faltando] = np.nanmedian(d[k])

    d["mag"] = np.sqrt(d["c_b"] ** 2 + d["c_g"] ** 2 + d["c_r"] ** 2)
    d["alongamento"] = (np.maximum(d["w"], d["h"])
                        / np.maximum(np.minimum(d["w"], d["h"]), 1))

    # As MESMAS derivadas do tf.carregar(). Se divergirem, o modelo e treinado
    # com uma feature e pontuado com outra -- sem erro nenhum, so resultado pior.
    if "std_g" in d:
        for canal in "bgr":
            d[f"snr_{canal}"] = (np.abs(d[f"c_{canal}"])
                                 / np.maximum(d[f"std_{canal}"], 1e-4))

    # Saneamento: NaN/inf em qualquer feature derruba o sklearn com um erro que
    # nao diz de qual coluna veio. Mascaras de 1 pixel produzem std NaN em CSVs
    # gerados antes do conserto do extrator -- sao 0,01% dos candidatos e todos
    # abaixo do SIZE_THRESHOLD, mas basta um para invalidar a matriz.
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            ruim = ~np.isfinite(v)
            if np.any(ruim):
                v[ruim] = np.nanmedian(v[~ruim]) if np.any(~ruim) else 0.0

    return d, linhas


def main(pasta, por_amostra, features, C, peso_bom, saida):
    # 1) Treina com TUDO que ja foi rotulado. Aqui nao ha fold: o modelo nao
    #    esta sendo avaliado, esta sendo usado como instrumento de busca.
    d_rot = tf.carregar(pasta)
    X, y, w, nomes = tf.montar(d_rot, peso_bom, features)
    modelo = tf.novo_modelo(C)
    modelo.fit(X, y, logit__sample_weight=w)
    print(f"modelo treinado em {len(y)} rotulos ({int(np.sum(y))} positivos)")
    print(f"features: {', '.join(nomes)}")

    # 2) Pontua a populacao inteira.
    d_tudo, linhas = carregar_todos(pasta)
    X_tudo = np.column_stack([d_tudo[n] for n in nomes])
    prob = modelo.predict_proba(X_tudo)[:, 1]
    print(f"candidatos pontuados: {len(prob)}")

    # 3) Tira de cena o que ja tem rotulo -- inclusive as duvidas, que ja foram
    #    olhadas uma vez e nao vale reolhar.
    caminho_rot = os.path.join(pasta, "rotulos.csv")
    ja_vistos = set()
    if os.path.exists(caminho_rot):
        with open(caminho_rot, encoding="utf-8") as f:
            ja_vistos = {r["cand_id"] for r in csv.DictReader(f)}
    print(f"ja rotulados (excluidos da fila): {len(ja_vistos)}")

    # 4) Top-N por amostra. Por amostra, e nao no balaio geral, porque a
    #    probabilidade nao e comparavel entre amostras -- foi exatamente esse o
    #    achado que fez a calibracao virar 'por_amostra' no treinar_filtro.
    #    Um ranking global entregaria 300 candidatos de duas amostras so.
    escolhidos = []
    for amostra in sorted(set(d_tudo["amostra"])):
        m = (d_tudo["amostra"] == amostra)
        idx = np.where(m)[0]
        idx = np.array([i for i in idx if d_tudo["cand_id"][i] not in ja_vistos])
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-prob[idx])][:por_amostra]
        escolhidos.extend(idx.tolist())
        print(f"  {amostra:<32} {len(idx):>4} candidatos | "
              f"prob {prob[idx].min():.3f} a {prob[idx].max():.3f}")

    # 5) Grava a fila no MESMO formato do candidatos.csv, para o anotador_v2 ler
    #    sem modificacao, mais uma coluna `prob` para inspecao.
    os.makedirs(saida, exist_ok=True)
    destino = os.path.join(saida, "candidatos.csv")
    with open(os.path.join(pasta, "candidatos.csv"), encoding="utf-8") as f:
        cabecalho = next(csv.reader(f))

    with open(destino, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cabecalho + ["prob"])
        for i in escolhidos:
            wr.writerow([linhas[i][c] for c in cabecalho] + [f"{prob[i]:.4f}"])

    print(f"\nfila com {len(escolhidos)} candidatos: {destino}")
    print("\nagora rotule:")
    print(f'  python anotador_v2.py "{destino}"')
    print("\nO anotador vai gravar rotulos.csv NESTA pasta, separado do original.")
    print("Os estratos/pesos que ele gravar sao relativos a fila, nao a")
    print("populacao -- serao recalculados depois pelo consolidar_rotulos.py.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pasta", help="pasta com candidatos.csv e rotulos.csv")
    p.add_argument("--por-amostra", type=int, default=50)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--peso-bom", type=float, default=3.0)
    p.add_argument("--features", default=None,
                   help="mesma sintaxe do treinar_filtro.py")
    p.add_argument("--saida", default=None,
                   help="pasta da fila (padrao: <pasta>/dirigida)")
    a = p.parse_args()
    feats = [s.strip() for s in a.features.split(",")] if a.features else None
    main(a.pasta, a.por_amostra, feats, a.C, a.peso_bom,
         a.saida or os.path.join(a.pasta, "dirigida"))
