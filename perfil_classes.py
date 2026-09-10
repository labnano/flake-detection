"""
perfil_classes.py  --  PASSO 2 do treino: olhar os dados ANTES de modelar.

Por que este passo existe
--------------------------
O sweep_v2 mostrou que, no orcamento de 40 candidatos por 1000 tiles, os falsos
positivos que sobram sao 78% BULK e 22% VAZIO -- e zero cola. O filtro de cor
resolveu a cola e revelou o inimigo seguinte.

Isso tem uma consequencia que nenhuma varredura de limiar consegue contornar.
O contraste optico cresce em modulo com o numero de camadas: cola fica perto de
zero, flake fino perto de -0,2, bulk mais negativo ainda. A regra do sweep e um
TETO (G <= g), que corta o que tem POUCO contraste. O flake fino nao mora abaixo
de um teto -- mora numa FAIXA. Nenhum valor de g isola uma faixa.

Treinar so vale se existir alguma grandeza em que flake e bulk de fato se
separem. Um modelo nao inventa sinal que nao esta nas features; ele so acha a
fronteira no sinal que existe. Entao a pergunta deste script e uma so:

    "em qual grandeza flake fino e bulk se separam -- e quanto?"

Se a resposta for 'em nenhuma', a conclusao nao e treinar um modelo maior: e
que faltam features (forma, textura, contraste por percentil) ou falta medir
outra coisa no microscopio.

O que o script mede
-------------------
1) PERFIL POR GRUPO. Para cada grandeza, a mediana e os quartis dentro de cada
   grupo (bom, marginal, cola, bulk, vazio). Da para ler a olho se as caixas se
   sobrepoem.

2) AUC. Uma medida de separacao entre dois grupos que nao depende de limiar:
   AUC = probabilidade de um positivo sorteado ao acaso ter valor MAIOR que um
   negativo sorteado ao acaso. 0,5 = nenhum sinal (moeda); 1,0 = separacao
   perfeita; 0,0 = separacao perfeita ao contrario (tao util quanto 1,0, basta
   inverter o sinal). E o resumo honesto de "quanto sinal ha nesta feature",
   porque varre todos os limiares possiveis de uma vez -- e por isso e o teste
   certo ANTES de escolher um.

3) TAMANHO EFETIVO DE AMOSTRA (ESS). Os rotulos sao ponderados: cada um
   representa n_estrato/n_vistos candidatos da populacao. Peso desigual custa
   precisao estatistica -- 69 rotulos com pesos muito diferentes valem menos que
   69 rotulos de peso igual. O ESS = (soma dos pesos)^2 / (soma dos pesos^2)
   diz quantos rotulos de peso igual valeriam o mesmo. E o numero que voce deve
   usar ao julgar se pode confiar numa diferenca.

Uso:
    python perfil_classes.py <pasta/avaliacao_lote>
"""

import argparse
import csv
import os

import numpy as np

# Limite abaixo do qual nao se calcula razao entre canais: dividir por um
# contraste quase nulo explode o valor e o ruido junto.
EPS_RAZAO = 5e-3

# Colunas de metadado: vem do rotulos.csv, nao do candidatos.csv, e nao sao
# numericas. Ficam de fora da conversao para float.
META = {"grupo", "amostra", "peso"}


def carregar(pasta, arquivo_rotulos="rotulos.csv"):
    """Junta candidatos + rotulos. Mesma logica do sweep_v2, sem os tiles.

    Devolve um dicionario com arrays alinhados: a posicao i de cada array e o
    mesmo candidato rotulado.
    """
    def ler(nome):
        with open(os.path.join(pasta, nome), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    cands = {r["cand_id"]: r for r in ler("candidatos.csv")}

    col = {k: [] for k in ["grupo", "peso", "amostra",
                           "c_b", "c_g", "c_r", "score", "size_px", "area",
                           "d_p25", "d_p50", "d_p75", "frac_substrato",
                           "occ_30", "w", "h",
                           "std_b", "std_g", "std_r", "cl_b", "cl_g", "cl_r"]}

    for r in ler(arquivo_rotulos):
        if r["rotulo"] not in ("2", "1", "0"):     # duvidas ficam de fora
            continue
        c = cands.get(r["cand_id"])
        if c is None:
            continue

        # O grupo junta rotulo e tipo: os negativos nao sao um balaio so.
        if r["rotulo"] == "2":
            grupo = "bom"
        elif r["rotulo"] == "1":
            grupo = "marginal"
        else:
            grupo = r["tipo"] if r["tipo"] else "lixo_s/tipo"

        col["grupo"].append(grupo)
        col["amostra"].append(r["amostra"])
        col["peso"].append(int(r["n_estrato"]) / int(r["n_vistos"]))

        # So as colunas NUMERICAS -- grupo/amostra/peso ja foram preenchidas
        # acima e nao vem do candidatos.csv. O `if k in c` deixa passar um CSV
        # de extrator antigo (sem std_*/cl_*) sem quebrar; a lista FEATURES e
        # podada mais abaixo para refletir o que de fato existe.
        for k in list(col):
            if k in META:
                continue
            if k in c:
                col[k].append(float(c[k]))

    d = {k: np.array(v) for k, v in col.items()}

    # --- features derivadas -------------------------------------------------
    # A magnitude do contraste e o candidato natural a separar bulk de fino:
    # e o que cresce com o numero de camadas, independente da direcao no
    # espaco de cor.
    d["mag"] = np.sqrt(d["c_b"] ** 2 + d["c_g"] ** 2 + d["c_r"] ** 2)

    # As RAZOES entre canais sao a assinatura espectral: independem da
    # intensidade e capturam a FORMA do espectro. Se bulk e flake fino diferem
    # so em magnitude, a razao nao separa. Se diferem em cor, ela separa e a
    # magnitude nao precisa entrar sozinha. Medir as duas responde qual e o caso.
    seguro = np.abs(d["c_g"]) > EPS_RAZAO
    for nome, num in [("b_sobre_g", d["c_b"]), ("r_sobre_g", d["c_r"])]:
        v = np.full(len(d["c_g"]), np.nan)
        v[seguro] = num[seguro] / d["c_g"][seguro]
        d[nome] = v

    # Alongamento da caixa: bulk costuma ser um bloco macico, flake fino tende
    # a ter forma irregular. E a feature de forma mais barata que existe --
    # as boas (solidez, arcarea) exigem parsear o contorno, fica pro passo 3.
    d["alongamento"] = np.maximum(d["w"], d["h"]) / np.maximum(np.minimum(d["w"], d["h"]), 1)

    # Razao sinal/ruido do contraste: a media dividida pela dispersao DENTRO da
    # mascara. E a grandeza que separa "contraste pequeno mas consistente"
    # (monocamada: espessura constante) de "contraste pequeno e aleatorio"
    # (substrato vazio, ruido da camera). As duas tem a mesma media -- que foi
    # exatamente o empate medido em 27/08 -- e SNR bem diferente.
    if "std_g" in d:
        d["snr_g"] = np.abs(d["c_g"]) / np.maximum(d["std_g"], 1e-4)
        d["snr_local_g"] = np.abs(d["cl_g"]) / np.maximum(d["std_g"], 1e-4)

    # Descarta as features que este CSV nao tem (extrator antigo).
    for k in list(FEATURES):
        if k not in d:
            FEATURES.remove(k)

    return d


FEATURES = ["c_b", "c_g", "c_r", "mag", "b_sobre_g", "r_sobre_g",
            "score", "size_px", "d_p25", "d_p50", "d_p75",
            "frac_substrato", "occ_30", "alongamento",
            # uniformidade e contraste local -- so existem em candidatos.csv
            # gerados pelo extrator de 27/08 em diante.
            "std_b", "std_g", "std_r", "cl_b", "cl_g", "cl_r",
            "snr_g", "snr_local_g"]

ORDEM_GRUPOS = ["bom", "marginal", "cola", "bulk", "vazio", "lixo_s/tipo"]


def quantil_ponderado(x, w, q):
    """Quantil q de x, com cada ponto contando com peso w.

    A receita: ordena os valores, acumula os pesos, e acha onde a soma
    acumulada cruza q * (peso total). Sem pesos isso vira o quantil comum.
    """
    ok = ~np.isnan(x)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.nan
    ordem = np.argsort(x)
    x, w = x[ordem], w[ordem]
    acum = np.cumsum(w) - 0.5 * w          # ponto medio de cada degrau
    acum /= np.sum(w)
    return float(np.interp(q, acum, x))


def ess(w):
    """Tamanho efetivo de amostra (Kish). Com pesos iguais devolve len(w)."""
    if len(w) == 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def auc_ponderada(x_pos, w_pos, x_neg, w_neg):
    """AUC = P(valor de um positivo > valor de um negativo), com pesos.

    Implementacao direta da definicao (Mann-Whitney): compara cada positivo
    com cada negativo e soma o produto dos pesos quando o positivo ganha.
    Empate conta meio ponto. Com ~500 rotulos a matriz e minuscula, entao nao
    vale a pena a versao esperta baseada em ordenacao.
    """
    ok_p, ok_n = ~np.isnan(x_pos), ~np.isnan(x_neg)
    x_pos, w_pos = x_pos[ok_p], w_pos[ok_p]
    x_neg, w_neg = x_neg[ok_n], w_neg[ok_n]
    if len(x_pos) == 0 or len(x_neg) == 0:
        return np.nan

    maior = (x_pos[:, None] > x_neg[None, :]).astype(float)
    empate = (x_pos[:, None] == x_neg[None, :]).astype(float)
    pesos = w_pos[:, None] * w_neg[None, :]

    return float(np.sum((maior + 0.5 * empate) * pesos) / np.sum(pesos))


def tabela_perfil(d):
    print("\n=== perfil por grupo: mediana [q25, q75], ponderado ===")
    grupos = [g for g in ORDEM_GRUPOS if np.any(d["grupo"] == g)]

    cab = f"{'feature':<15}" + "".join(f"{g:>26}" for g in grupos)
    print(cab)
    print("-" * len(cab))

    for f in FEATURES:
        linha = f"{f:<15}"
        for g in grupos:
            m = d["grupo"] == g
            med = quantil_ponderado(d[f][m], d["peso"][m], 0.50)
            q25 = quantil_ponderado(d[f][m], d["peso"][m], 0.25)
            q75 = quantil_ponderado(d[f][m], d["peso"][m], 0.75)
            linha += f"{med:>9.3f} [{q25:>6.2f},{q75:>6.2f}]"
        print(linha)


def tabela_auc(d):
    """Duas perguntas separadas, porque tem respostas diferentes:

    (1) flake vs BULK    -- o inimigo que sobrou no orcamento apertado
    (2) flake vs LIXO    -- o problema completo, como o pipeline o vive
    """
    flake = np.isin(d["grupo"], ["bom", "marginal"])
    bulk = d["grupo"] == "bulk"
    lixo = ~flake

    print("\n=== poder de separacao (AUC ponderada) ===")
    print("0,50 = nenhum sinal | distancia de 0,50 e o que importa")
    print("(AUC < 0,50 separa igualmente bem, com o sinal invertido)\n")

    linhas = []
    for f in FEATURES:
        a_bulk = auc_ponderada(d[f][flake], d["peso"][flake],
                               d[f][bulk], d["peso"][bulk])
        a_lixo = auc_ponderada(d[f][flake], d["peso"][flake],
                               d[f][lixo], d["peso"][lixo])
        linhas.append((abs(a_bulk - 0.5), f, a_bulk, a_lixo))

    linhas.sort(reverse=True)     # o que mais separa BULK vem primeiro
    print(f"{'feature':<15}{'vs BULK':>10}{'vs TODO LIXO':>15}")
    print("-" * 40)
    for _, f, a_bulk, a_lixo in linhas:
        print(f"{f:<15}{a_bulk:>10.3f}{a_lixo:>15.3f}")


def main(pasta, arquivo_rotulos="rotulos.csv"):
    d = carregar(pasta, arquivo_rotulos)

    print(f"rotulos carregados: {len(d['grupo'])}")
    print("\n=== quantos, e quanto valem ===")
    print(f"{'grupo':<14}{'n':>5}{'ESS':>8}{'peso total':>13}")
    print("-" * 40)
    for g in ORDEM_GRUPOS:
        m = d["grupo"] == g
        if not np.any(m):
            continue
        print(f"{g:<14}{int(np.sum(m)):>5}{ess(d['peso'][m]):>8.1f}"
              f"{float(np.sum(d['peso'][m])):>13.0f}")

    flake = np.isin(d["grupo"], ["bom", "marginal"])
    print(f"\npositivos (bom+marginal): n={int(np.sum(flake))} | "
          f"ESS={ess(d['peso'][flake]):.1f}")
    print("O ESS e o n que conta na hora de julgar uma diferenca.")

    tabela_perfil(d)
    tabela_auc(d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pasta", help="pasta com candidatos.csv e rotulos.csv")
    p.add_argument("--rotulos", default="rotulos.csv",
                   help="nome do CSV de rotulos dentro da pasta "
                        "(use rotulos_consolidado.csv apos consolidar)")
    a = p.parse_args()
    main(a.pasta, a.rotulos)
