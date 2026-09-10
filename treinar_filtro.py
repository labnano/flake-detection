"""
treinar_filtro.py  --  PASSO 3 do treino: trocar a regra de limiares por um
modelo, sem trocar o modo de julgar.

A ideia
-------
O sweep_v2 escolhe UMA regra de 5 parametros por busca em grade. O resultado de
hoje mostrou o custo disso: 38% dos bons flakes dentro da amostra, 20% fora --
metade do desempenho era a grade encontrando ruido em ~500 rotulos.

E, pior que o overfitting, ha um limite de LINGUAGEM: a regra so sabe dizer
"G <= g", um teto. O flake fino nao mora abaixo de um teto, mora numa faixa --
e o bulk, que e o falso positivo que sobrou, e MAIS contrastado que ele, nao
menos. Nenhum valor de g resolve isso.

Aqui a funcao de decisao muda de familia: em vez de um teto por eixo, uma
fronteira com direcao livre no espaco de contraste, com os pesos escolhidos por
maxima verossimilhanca em vez de busca exaustiva. Tudo o mais -- os pesos de
estrato, a validacao entre amostras, a metrica operacional -- e igual ao
sweep_v2, de proposito: so assim a comparacao mede o modelo, e nao a metrica.

Duas vantagens estruturais do modelo sobre a grade, que valem entender:

1) A SAIDA E CONTINUA. A grade precisa ser reotimizada para cada orcamento
   (veja a "curva de operacao" do sweep_v2: uma regra diferente em cada linha).
   O modelo devolve uma probabilidade, e a curva inteira sai de um ajuste so --
   basta mover o limiar. Voce ganha o direito de escolher o orcamento DEPOIS.

2) OS COEFICIENTES SAO LEGIVEIS. Um modelo linear em features fisicas nao e uma
   caixa preta: o peso de cada canal diz quanto ele contribui para a decisao.
   Ler os coeficientes e o que transforma "o modelo deu 0,7" em conhecimento
   sobre a amostra.

Uso:
    python treinar_filtro.py <pasta/avaliacao_lote> [--orcamento 40] [--C 1.0]
                             [--peso-bom 3]
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# O que o sweep_v2 entregou FORA da amostra, por orcamento. Preencher rodando
# `python sweep_v2.py <pasta> --orcamento N` -- a comparacao so vale contra o
# MESMO orcamento. (A primeira versao daqui tinha um unico numero fixo, o de 40,
# e por isso anunciou "+64 pontos" ao ser comparada com o run de 160. Um
# baseline errado nao produz um erro: produz uma conclusao errada, que e pior.)
# ATENCAO: os pesos mudaram em 27/08. O rotulos.csv original gravava
# `n_vistos` = quantos foram SORTEADOS no estrato, nao quantos foram de fato
# rotulados -- e o sorteio (~1340) foi maior que a rotulagem (512). Resultado:
# a populacao estimada era 14.538 em vez de 38.133, um fator 2,6. Os recalls
# (proporcoes) sobreviveram razoavelmente; os custos "por 1000 tiles" nao.
# Os numeros abaixo com a marca (v1) sao do peso ERRADO e ficam so como
# historico. Use rotulos_consolidado.csv, que corrige isso.
BASELINE_LOSO = {
    # com rotulos_consolidado.csv:
    320: 0.40,
    # (v1, peso errado -- nao comparar): 40 -> 0.20, 160 -> 0.85
}


# ---------------------------------------------------------------------------
# CARREGAMENTO  (encanamento)
# ---------------------------------------------------------------------------

COLUNAS = ["c_b", "c_g", "c_r", "score", "size_px", "area",
           "d_p25", "d_p50", "d_p75", "frac_substrato", "occ_30", "w", "h",
           # features de uniformidade e contraste local (extrator v2, 27/08).
           # Se o seu candidatos.csv for anterior a elas, o carregamento avisa
           # e segue sem elas em vez de quebrar.
           "std_b", "std_g", "std_r", "cl_b", "cl_g", "cl_r"]


def carregar(pasta, arquivo_rotulos="rotulos.csv"):
    """Junta candidatos + rotulos + tiles. Arrays alinhados por posicao."""
    def ler(nome):
        with open(os.path.join(pasta, nome), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    cands = {r["cand_id"]: r for r in ler("candidatos.csv")}

    n_tiles = defaultdict(int)
    for t in ler("tiles.csv"):
        n_tiles[t["amostra"]] += 1

    col = defaultdict(list)
    for r in ler(arquivo_rotulos):
        if r["rotulo"] not in ("2", "1", "0"):      # duvidas fora
            continue
        c = cands.get(r["cand_id"])
        if c is None:
            continue
        col["amostra"].append(r["amostra"])
        col["rotulo"].append(int(r["rotulo"]))      # 2=bom 1=marginal 0=lixo
        col["tipo"].append(r["tipo"])
        col["peso"].append(int(r["n_estrato"]) / int(r["n_vistos"]))
        for k in COLUNAS:
            if k in c:
                col[k].append(float(c[k]))

        sol, arc = forma(c["contorno"])
        col["solidez"].append(sol)
        col["arcarea"].append(arc)

    d = {k: np.array(v) for k, v in col.items()}

    # Derivadas. `mag` e nao-linear nas tres cores, entao acrescenta informacao
    # a um modelo linear mesmo sendo funcao delas: e a INTENSIDADE do contraste,
    # a grandeza que cresce com o numero de camadas. O perfil mostrou que ela e
    # o que mais separa flake bom (0,19) de marginal (0,39).
    d["mag"] = np.sqrt(d["c_b"] ** 2 + d["c_g"] ** 2 + d["c_r"] ** 2)
    d["alongamento"] = (np.maximum(d["w"], d["h"])
                        / np.maximum(np.minimum(d["w"], d["h"]), 1))

    # Razao sinal/ruido do contraste: media dividida pela dispersao DENTRO da
    # mascara. Separa "contraste pequeno mas CONSISTENTE" (monocamada, que tem
    # espessura constante) de "contraste pequeno e ALEATORIO" (substrato vazio,
    # ruido da camera). As duas tem a mesma media -- foi o empate medido em
    # 27/08, que fez um monolayer bom e um vazio caírem no mesmo lugar -- e SNR
    # que difere por fator 8 (bom 5,8 contra vazio 0,7, quartis sem tocar).
    if "std_g" in d:
        d["snr_g"] = np.abs(d["c_g"]) / np.maximum(d["std_g"], 1e-4)
        d["snr_b"] = np.abs(d["c_b"]) / np.maximum(d["std_b"], 1e-4)
        d["snr_r"] = np.abs(d["c_r"]) / np.maximum(d["std_r"], 1e-4)

    # Imputacao: contornos degenerados (poucos pontos, area nula) viram NaN, e
    # o sklearn nao aceita NaN. Preencher com a mediana da coluna e a saida mais
    # conservadora -- o candidato fica "medio" naquela feature em vez de sumir.
    for k in ["solidez", "arcarea"]:
        faltando = np.isnan(d[k])
        if np.any(faltando):
            d[k][faltando] = np.nanmedian(d[k])
            print(f"({int(np.sum(faltando))} contornos degenerados: "
                  f"{k} preenchido com a mediana)")


    # Saneamento: NaN/inf em qualquer feature derruba o sklearn com um erro que
    # nao diz de qual coluna veio. Mascaras de 1 pixel produzem std NaN em CSVs
    # gerados antes do conserto do extrator -- sao 0,01% dos candidatos e todos
    # abaixo do SIZE_THRESHOLD, mas basta um para invalidar a matriz.
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            ruim = ~np.isfinite(v)
            if np.any(ruim):
                v[ruim] = np.nanmedian(v[~ruim]) if np.any(~ruim) else 0.0
    d["n_tiles"] = dict(n_tiles)
    return d


def forma(contorno_str):
    """Solidez e arcarea a partir do contorno gravado como 'x y x y ...'.

    Sao exatamente as duas features que o L2_model do MaskTerial usa para
    estimar falso positivo -- entao medi-las aqui e testar, offline, se vale
    plugar o modelo naquele slot.

    solidez = area / area do fecho convexo. Perto de 1 quando a forma e cheia
              e convexa (blob de cola); menor quando tem reentrancias.
    arcarea = perimetro / raiz(area). E adimensional -- nao muda se o objeto
              crescer -- e mede o quanto o contorno e esfarrapado: minimo ~3,5
              para um circulo, cresce com a irregularidade.
    """
    pts = np.fromstring(contorno_str, sep=" ", dtype=np.float32)
    if len(pts) < 6:                      # menos de 3 pontos: nao ha poligono
        return np.nan, np.nan
    c = pts.reshape(-1, 1, 2).astype(np.int32)
    area = cv2.contourArea(c)
    if area <= 0:
        return np.nan, np.nan
    area_hull = cv2.contourArea(cv2.convexHull(c))
    solidez = area / area_hull if area_hull > 0 else np.nan
    arcarea = cv2.arcLength(c, True) / area ** 0.5
    return solidez, arcarea


# ---------------------------------------------------------------------------
# A PECA QUE CARREGA A IDEIA  --  PEDRO ESCREVE
# ---------------------------------------------------------------------------

def montar(d, peso_bom, nomes=None):
    """Transforma os dados carregados na mesa de treino: X, y, w, nomes.

    Voce decide tres coisas aqui, e cada uma e uma escolha de modelagem:

    1) X -- QUAIS FEATURES.  Uma matriz (n_rotulos, n_features). O perfil
       mediu o sinal de cada candidata; use isso, nao o instinto de jogar
       todas. Lembre do orcamento: ESS dos positivos = 45, e a regra de bolso
       pede ~10 exemplos por parametro. Isso te da 4 features, talvez 5 --
       contando o intercepto, que o modelo adiciona sozinho.
       Descartadas por medida, nao por gosto: as razoes entre canais (instaveis
       quando c_g ~ 0, que e justamente o caso de cola e vazio).
       Monte com np.column_stack([...]).

    2) y -- O ALVO.  Binario: 1 para flake, 0 para lixo. Com 15 bons (ESS 12)
       nao da para treinar "bom flake" como classe propria -- entao bom e
       marginal entram juntos como positivo.

    3) w -- O PESO DE CADA ROTULO, e aqui esta o conserto da armadilha do F1.
       Sao DOIS pesos que se multiplicam:

         (a) d["peso"], o peso de estrato (n_estrato/n_vistos), que desfaz a
             distorcao do sorteio -- sem ele a populacao parece o que voce
             escolheu olhar;
         (b) o peso de importancia: um flake BOM (rotulo 2) custa mais caro
             de perder que um marginal. Multiplique os bons por `peso_bom`.

       Tres vezes em agosto a metrica subiu jogando fora os melhores flakes,
       porque F1 pesa todo flake igual. Este produto e onde a assimetria entra
       no aprendizado -- nao na metrica depois, mas na perda que o modelo
       minimiza.

    Devolva tambem `nomes`: a lista de nomes das features, na MESMA ordem das
    colunas de X. E o que permite imprimir os coeficientes e ler o modelo.

    Returns:
        X (n, k) float, y (n,) 0/1, w (n,) float, nomes list[str]
    """
    # 1) FEATURES. Quatro, escolhidas pelo que o perfil_classes.py mediu:
    #    c_b            -- separa BULK quase sozinha (AUC 0,977): o flake fino
    #                      tem contraste azul POSITIVO (+0,06), o bulk e opaco
    #                      e escuro em tudo (-0,75);
    #    c_g            -- separa COLA (flake ~-0,15, cola ~+0,06);
    #    c_r            -- completa o espaco de cor; deixa a fronteira escolher
    #                      livremente a direcao em BGR em vez de ficar presa a
    #                      cortes alinhados aos eixos, que era o limite da grade;
    #    frac_substrato -- separa VAZIO (flake 0,00, vazio 0,68), que a cor nao
    #                      toca porque vazio e substrato: contraste ~0 nos tres.
    #
    #    Fora, por medida e nao por gosto: as razoes entre canais (b/g, r/g).
    #    Elas pontuaram alto, mas o denominador c_g e quase nulo justamente em
    #    cola e vazio -- ali a razao vira ruido amplificado e nao sobrevive a
    #    proxima amostra com exposicao diferente.
    #
    #    Este conjunto e o PADRAO; --features troca na linha de comando. Ao
    #    acrescentar, lembre do orcamento: ESS dos positivos = 45, e cada
    #    feature e mais um parametro a estimar com os mesmos 45.
    if nomes is None:
        nomes = ["c_b", "c_g", "c_r", "frac_substrato"]

    # column_stack empilha os vetores como COLUNAS: 503 rotulos x 4 features.
    # A ordem aqui e a mesma de `nomes` -- e o que faz os coeficientes
    # impressos no fim corresponderem as grandezas certas.
    X = np.column_stack([d[nome] for nome in nomes])

    # 2) ALVO. Binario. Bom (2) e marginal (1) entram juntos como positivo:
    #    com 15 bons (ESS 12) nao ha como treinar "bom" como classe propria.
    #    A comparacao >= devolve True/False para todos os elementos de uma vez
    #    (operacao vetorizada); astype(int) converte para 1/0.
    y = (d["rotulo"] >= 1).astype(int)

    # 3) PESO. O produto de dois pesos com papeis diferentes:
    #
    #    (a) d["peso"] = n_estrato/n_vistos -- quantos candidatos da populacao
    #        cada rotulo representa. Sem ele, ter olhado proporcionalmente mais
    #        candidatos de score alto faria o modelo acreditar que a populacao
    #        inteira e assim.
    #
    #    (b) importancia: um flake BOM custa mais caro de perder. np.where
    #        escolhe elemento a elemento -- peso_bom onde o rotulo e 2, 1.0 no
    #        resto -- e a multiplicacao entre arrays em numpy tambem e elemento
    #        a elemento.
    #
    #    Este produto e o conserto da armadilha de agosto: a assimetria entra
    #    na PERDA que o modelo minimiza, nao numa metrica aplicada depois.
    importancia = np.where(d["rotulo"] == 2, peso_bom, 1.0)
    w = d["peso"] * importancia

    return X, y, w, nomes


# ---------------------------------------------------------------------------
# AVALIACAO  (encanamento -- espelha o sweep_v2 de proposito)
# ---------------------------------------------------------------------------

def novo_modelo(C):
    """Escalonamento + regressao logistica, nesta ordem e dentro do mesmo objeto.

    O StandardScaler nao e detalhe: a regularizacao L2 da logistica penaliza
    coeficientes grandes, e um coeficiente e grande ou pequeno em relacao a
    ESCALA da feature. Sem padronizar, size_px (~1000) e c_b (~0.06) sofrem
    penalidades incomparaveis e o modelo simplesmente ignora o contraste.

    Estar dentro de um Pipeline garante que a media e o desvio usados na
    padronizacao sejam calculados SO no fold de treino. Padronizar tudo de uma
    vez antes de dividir vaza informacao do teste para o treino -- e um dos
    vazamentos mais comuns e mais silenciosos que existem.
    """
    return Pipeline([
        ("escala", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=2000)),
    ])


def limiar_para_orcamento(prob, peso, n_tiles, alvo_por_mil):
    """Menor limiar de probabilidade que ainda cabe no orcamento.

    Orcamento = candidatos aceitos por 1000 tiles. Como cada rotulo representa
    `peso` candidatos da populacao, aceitar um rotulo custa `peso` unidades.
    Ordenamos por probabilidade decrescente (aceitar os mais promissores
    primeiro), acumulamos o custo, e cortamos quando estoura.
    """
    orcamento = alvo_por_mil * n_tiles / 1000.0
    ordem = np.argsort(-prob)
    acumulado = np.cumsum(peso[ordem])
    cabe = acumulado <= orcamento
    if not np.any(cabe):
        return np.inf                      # nem o melhor candidato cabe
    return float(prob[ordem][cabe][-1])


def medir(rotulo, peso, aceito, n_tiles):
    """Mesmas metricas do sweep_v2.medir, para a comparacao ser justa."""
    bom = rotulo == 2
    qualquer = rotulo >= 1
    total_bons = float(np.sum(peso[bom]))
    aceitos = float(np.sum(peso[aceito]))
    return {
        "por_mil": 1000.0 * aceitos / max(n_tiles, 1),
        "recall_bons": (float(np.sum(peso[aceito & bom])) / total_bons
                        if total_bons else np.nan),
        "recall_todos": (float(np.sum(peso[aceito & qualquer]))
                         / max(float(np.sum(peso[qualquer])), 1e-9)),
        "precisao": (float(np.sum(peso[aceito & qualquer])) / aceitos
                     if aceitos else 0.0),
    }


def subamostrar(y, treino_idx, fracao, rng):
    """Sorteia uma fracao do conjunto de treino, ESTRATIFICADA por classe.

    Sortear ao acaso seria mais simples e errado: com 14% de positivos, uma
    fracao pequena as vezes sairia sem positivo nenhum, e a curva mediria o
    azar do sorteio em vez do efeito do tamanho. Sorteando a mesma fracao
    dentro de cada classe, a proporcao se mantem em todos os pontos da curva.
    """
    if fracao >= 1.0:
        return treino_idx
    partes = []
    for classe in (0, 1):
        idx = treino_idx[y[treino_idx] == classe]
        n = max(int(round(fracao * len(idx))), 1)
        partes.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.concatenate(partes)


def avaliar_loso(d, X, y, w, orcamento, C, calibracao="por_amostra",
                 fracao_treino=1.0, semente=0):
    """Leave-one-sample-out: treina sem ver a amostra, mede nela.

    O MODELO nunca ve a amostra de teste -- isso e inegociavel. O limiar de
    decisao e outra historia, e o modo `calibracao` escolhe entre duas leituras
    do problema:

      "global"      -- limiar fixado nas amostras de treino e aplicado como esta.
                       Exige que a probabilidade signifique a mesma coisa em
                       toda amostra. O primeiro run mostrou que nao significa:
                       com alvo de 40/1000, os folds gastaram de 0 a 125. A
                       exposicao muda, a distribuicao de contraste escorrega, e
                       o limiar absoluto vai junto.

      "por_amostra" -- o limiar e o que gasta o orcamento NAQUELA amostra:
                       ordena os candidatos por probabilidade e pega os N
                       melhores. Mede RANQUEAMENTO, nao calibracao absoluta.

    Isso NAO e vazamento, e a distincao vale entender: usar dados do teste so e
    trapaca quando se usam os ROTULOS do teste. Aqui entram apenas as
    probabilidades e os pesos -- nenhuma informacao sobre quem e flake. E e
    exatamente o que o microscopio faz: roda o modelo em tudo, ordena, e voce
    olha os N primeiros. Nao ha rotulo nenhum disponivel nessa hora.
    """
    amostras = sorted(set(d["amostra"]))
    rng = np.random.default_rng(semente)
    linhas = {}

    for a in amostras:
        teste = d["amostra"] == a
        treino = subamostrar(y, np.where(~teste)[0], fracao_treino, rng)

        modelo = novo_modelo(C)
        modelo.fit(X[treino], y[treino], logit__sample_weight=w[treino])

        prob_teste = modelo.predict_proba(X[teste])[:, 1]

        if calibracao == "global":
            n_tiles_treino = sum(d["n_tiles"].get(x, 0) for x in amostras if x != a)
            limiar = limiar_para_orcamento(
                modelo.predict_proba(X[treino])[:, 1],
                d["peso"][treino], n_tiles_treino, orcamento)
        else:
            limiar = limiar_para_orcamento(
                prob_teste, d["peso"][teste],
                d["n_tiles"].get(a, 1), orcamento)

        aceito = prob_teste >= limiar
        linhas[a] = medir(d["rotulo"][teste], d["peso"][teste],
                          aceito, d["n_tiles"].get(a, 1))
        linhas[a]["limiar"] = limiar

        # Guarda o peso dos falsos positivos por tipo, para a decomposicao.
        # Comparar isso com a tabela equivalente do sweep_v2 responde uma
        # pergunta que o recall sozinho nao responde: o modelo erra o MESMO
        # que a regra, ou erra outra coisa? Se ele mata bulk e passa a errar em
        # cola, a proxima feature a caçar e outra.
        fp = aceito & (d["rotulo"][teste] == 0)
        linhas[a]["fp_por_tipo"] = {
            t: float(np.sum(d["peso"][teste][fp & (d["tipo"][teste] == t)]))
            for t in set(d["tipo"][teste][fp])
        }

    return linhas


def curva_de_operacao(d, X, y, w, C, orcamentos, calibracao):
    """Recall de bons fora da amostra em varios orcamentos.

    Um unico modelo por fold serve a curva inteira -- so o limiar muda. E o
    contraste direto com a grade, que reotimiza tudo a cada linha.
    """
    print(f"\n{'orcamento':>10}{'% dos bons (fora)':>20}{'dp':>8}"
          f"{'precisao':>11}{'gasto real':>13}")
    print("-" * 62)
    for orc in orcamentos:
        linhas = avaliar_loso(d, X, y, w, orc, C, calibracao)
        r = [v["recall_bons"] for v in linhas.values() if not np.isnan(v["recall_bons"])]
        p = [v["precisao"] for v in linhas.values()]
        g = [v["por_mil"] for v in linhas.values()]
        print(f"{orc:>10}{100*np.mean(r):>19.0f}%{100*np.std(r):>7.0f}%"
              f"{100*np.mean(p):>10.0f}%{np.mean(g):>13.1f}")


def curva_aprendizado(d, X, y, w, C, orcamento, calibracao,
                      fracoes=(0.3, 0.5, 0.7, 0.85, 1.0), repeticoes=5):
    """Desempenho em funcao de QUANTOS rotulos foram usados no treino.

    E o que transforma "rotule mais" de conselho em numero. Se a curva ainda
    sobe em 100%, cada hora no anotador ainda compra desempenho. Se ela achatou,
    o gargalo passou a ser outro -- features, ou o segmentador la atras -- e
    rotular mais e desperdicio.

    Cada ponto e a media de `repeticoes` sorteios diferentes, porque com 69
    positivos o resultado de UM sorteio e ruido puro. O desvio entre sorteios
    (dp_sorteio) mede a instabilidade do proprio treino: se ele for grande, o
    modelo esta a merce de quais rotulos calharam de entrar.
    """
    print(f"\n=== curva de aprendizado (orcamento {orcamento:.0f}/1000) ===")
    print("se ainda sobe em 100%, rotular mais compra desempenho\n")
    print(f"{'% rotulos':>10}{'n treino':>10}{'% dos bons':>13}"
          f"{'dp amostras':>14}{'dp sorteio':>13}")
    print("-" * 60)

    for f in fracoes:
        medias = []
        for s in range(repeticoes):
            linhas = avaliar_loso(d, X, y, w, orcamento, C, calibracao,
                                  fracao_treino=f, semente=s)
            r = [v["recall_bons"] for v in linhas.values()
                 if not np.isnan(v["recall_bons"])]
            medias.append((float(np.mean(r)), float(np.std(r))))

        recall = float(np.mean([m for m, _ in medias]))
        dp_amostras = float(np.mean([s for _, s in medias]))
        dp_sorteio = float(np.std([m for m, _ in medias]))
        n_treino = int(round(f * len(y) * 5 / 6))     # 5 dos 6 folds, em media
        print(f"{100*f:>9.0f}%{n_treino:>10}{100*recall:>12.0f}%"
              f"{100*dp_amostras:>13.0f}%{100*dp_sorteio:>12.0f}%")


def main(pasta, orcamento, C, peso_bom, calibracao, features,
         arquivo_rotulos="rotulos.csv"):
    d = carregar(pasta, arquivo_rotulos)
    X, y, w, nomes = montar(d, peso_bom, features)

    n_pos = int(np.sum(y == 1))
    print(f"rotulos: {len(y)} | positivos: {n_pos} | features: {len(nomes)}")
    print(f"features: {', '.join(nomes)}")
    print(f"C = {C} | peso do flake bom = {peso_bom}x | calibracao = {calibracao}")

    print(f"\n=== leave-one-sample-out (orcamento = {orcamento}/1000 tiles) ===")
    linhas = avaliar_loso(d, X, y, w, orcamento, C, calibracao)
    for a in sorted(linhas):
        v = linhas[a]
        rb = "  nan" if np.isnan(v["recall_bons"]) else f"{100*v['recall_bons']:>4.0f}%"
        print(f"  {a:<32} {rb} dos bons | {v['por_mil']:>6.1f}/1000 | "
              f"precisao {100*v['precisao']:>3.0f}%")

    r = [v["recall_bons"] for v in linhas.values() if not np.isnan(v["recall_bons"])]
    g = [v["por_mil"] for v in linhas.values()]
    media, dp = float(np.mean(r)), float(np.std(r))
    print(f"\n  recall de bons FORA DA AMOSTRA: {100*media:.0f}% (+/- {100*dp:.0f})")
    print(f"  gasto real: {np.mean(g):.1f}/1000 "
          f"(min {np.min(g):.0f}, max {np.max(g):.0f}) -- alvo era {orcamento:.0f}")

    base = BASELINE_LOSO.get(int(orcamento))
    if base is None:
        print(f"  sem baseline registrado para o orcamento {orcamento:.0f}: rode")
        print(f"    python sweep_v2.py <pasta> --orcamento {orcamento:.0f}")
        print("  e anote em BASELINE_LOSO. Comparar com outro orcamento nao vale.")
    else:
        delta = media - base
        print(f"  baseline (sweep_v2, MESMO orcamento): {100*base:.0f}%")
        print(f"  diferenca: {100*delta:+.0f} pontos"
              f"{'  -- dentro do ruido, nao conclua nada' if abs(delta) < dp else ''}")

    # Decomposicao dos falsos positivos, somada sobre os folds.
    total = defaultdict(float)
    for v in linhas.values():
        for t, p in v["fp_por_tipo"].items():
            total[t or "(nao classificado)"] += p
    soma = sum(total.values())
    if soma:
        print(f"\n=== falsos positivos que sobram (orcamento {orcamento:.0f}) ===")
        for t, p in sorted(total.items(), key=lambda kv: -kv[1]):
            print(f"  {t:<20} {100*p/soma:>5.1f}%")

    curva_de_operacao(d, X, y, w, C, [10, 20, 40, 80, 160, 320], calibracao)
    curva_aprendizado(d, X, y, w, C, orcamento, calibracao)

    # Modelo final, ajustado em TUDO -- so para LER, nunca para reportar
    # desempenho (medir onde se treinou nao mede nada).
    modelo = novo_modelo(C)
    modelo.fit(X, y, logit__sample_weight=w)
    coefs = modelo.named_steps["logit"].coef_[0]
    print("\n=== o que o modelo aprendeu ===")
    print("(coeficientes em unidades de desvio-padrao da feature:")
    print(" positivo = valor alto empurra para FLAKE)\n")
    for nome, c in sorted(zip(nomes, coefs), key=lambda t: -abs(t[1])):
        print(f"  {nome:<18}{c:>+8.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pasta", help="pasta com candidatos.csv, rotulos.csv, tiles.csv")
    p.add_argument("--orcamento", type=float, default=40)
    p.add_argument("--C", type=float, default=1.0,
                   help="inverso da regularizacao: MENOR = mais regularizado")
    p.add_argument("--peso-bom", type=float, default=3.0,
                   help="quantas vezes um flake bom vale mais que um marginal")
    p.add_argument("--calibracao", choices=["por_amostra", "global"],
                   default="por_amostra",
                   help="por_amostra: pega os N melhores de cada amostra "
                        "(mede ranqueamento). global: limiar fixo vindo do treino "
                        "(mede tambem se a probabilidade transfere entre amostras)")
    p.add_argument("--features", default=None,
                   help="lista separada por virgula. Disponiveis: c_b c_g c_r "
                        "mag score size_px area d_p25 d_p50 d_p75 "
                        "frac_substrato occ_30 solidez arcarea alongamento. "
                        "Padrao: c_b,c_g,c_r,frac_substrato")
    p.add_argument("--rotulos", default="rotulos.csv",
                   help="nome do CSV de rotulos dentro da pasta "
                        "(use rotulos_consolidado.csv apos consolidar)")
    a = p.parse_args()
    feats = [s.strip() for s in a.features.split(",")] if a.features else None
    main(a.pasta, a.orcamento, a.C, a.peso_bom, a.calibracao, feats, a.rotulos)
