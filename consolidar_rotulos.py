"""
consolidar_rotulos.py  --  junta os rotulos das duas campanhas e recalcula os
pesos de estrato para a populacao inteira.

Por que nao basta concatenar os dois CSVs
------------------------------------------
Cada rotulo carrega um peso `n_estrato / n_vistos`: quantos candidatos da
populacao ele representa. Esse peso so faz sentido se os estratos forem
DISJUNTOS -- cada candidato pertencendo a exatamente um. Se dois estratos se
sobrepoem, os candidatos da interseccao sao contados duas vezes e a populacao
estimada fica inflada; a precisao e o recall saem errados na mesma proporcao.

E e exatamente isso que aconteceu:

  campanha 1 (sorteio)   -> estratos {baixo,medio,alto} x {vivo,vetado}
  campanha 2 (dirigida)  -> estrato "topo do ranking do modelo"

Um candidato do topo tambem e 'alto|vivo'. Sobreposicao total.

A correcao e cruzar os dois esquemas, criando estratos disjuntos:

    {baixo, medio, alto} x {vivo, vetado} x {topo, resto}

O candidato que caiu na fila dirigida sai do seu estrato antigo e passa para o
estrato 'topo' correspondente. Os n_estrato dos estratos antigos encolhem na
mesma medida. Ninguem e contado duas vezes.

Um detalhe que muda o resultado: 'topo' e definido POR AMOSTRA, porque a fila
foi montada por amostra (top-50 de cada). O estrato de uma amostra nao diz nada
sobre a outra.

Duvidas ('?') sao excluidas do numerador E do denominador -- ou seja, tratadas
como se nao tivessem sido sorteadas. A alternativa (conta-las como vistas e
descarta-las depois) subestimaria a populacao, porque o peso cairia sem que
nenhum rotulo entrasse no lugar. Com ~12 duvidas em 812 a diferenca e pequena,
mas a escolha precisa ser deliberada.

Uso:
    python consolidar_rotulos.py <pasta/avaliacao_lote>
    # depois:
    python treinar_filtro.py <pasta> --rotulos rotulos_consolidado.csv
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

FAIXAS_SCORE = [0.05, 0.20]     # iguais as do anotador_v2
COLUNA_VETO = "cls_30"


def faixa_veto(linha):
    """Estrato antigo: faixa de score x sobrevive ao veto. Mesma regra do
    anotador_v2.estrato_de -- se mudar la, muda aqui."""
    s = float(linha["score"])
    faixa = ("baixo" if s < FAIXAS_SCORE[0]
             else "medio" if s < FAIXAS_SCORE[1] else "alto")
    veto = "vivo" if linha[COLUNA_VETO] != "0" else "vetado"
    return f"{faixa}|{veto}"


def ler(caminho):
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(pasta, dirigida, saida):
    populacao = ler(os.path.join(pasta, "candidatos.csv"))
    fila = ler(os.path.join(dirigida, "candidatos.csv"))
    ids_fila = {r["cand_id"] for r in fila}

    rotulos = []
    for caminho in [os.path.join(pasta, "rotulos.csv"),
                    os.path.join(dirigida, "rotulos.csv")]:
        if os.path.exists(caminho):
            n = len(rotulos)
            rotulos += ler(caminho)
            print(f"{os.path.relpath(caminho, pasta):<28} {len(rotulos)-n:>4} rotulos")

    # Descarta duvidas (ver docstring) e rotulos duplicados: se o mesmo
    # candidato aparecer nas duas campanhas, vale o mais recente.
    validos = {}
    duvidas = 0
    for r in rotulos:
        if r["rotulo"] not in ("2", "1", "0"):
            duvidas += 1
            continue
        validos[r["cand_id"]] = r
    print(f"\nvalidos: {len(validos)} | duvidas descartadas: {duvidas}")

    # --- ESTRATO NOVO, DISJUNTO ---------------------------------------------
    estrato = {}
    for linha in populacao:
        cid = linha["cand_id"]
        lugar = "topo" if cid in ids_fila else "resto"
        estrato[cid] = (linha["amostra"], f"{faixa_veto(linha)}|{lugar}")

    # n_estrato: quantos candidatos da POPULACAO caem em cada (amostra, estrato)
    n_estrato = Counter(estrato.values())

    # n_vistos: quantos deles ganharam rotulo valido
    n_vistos = Counter(estrato[cid] for cid in validos if cid in estrato)

    orfaos = [cid for cid in validos if cid not in estrato]
    if orfaos:
        print(f"[aviso] {len(orfaos)} rotulos sem candidato correspondente, "
              "descartados")

    # --- ESCRITA ------------------------------------------------------------
    destino = os.path.join(pasta, saida)
    escritos = 0
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cand_id", "amostra", "estrato", "n_estrato", "n_vistos",
                    "rotulo", "tipo"])
        for cid, r in validos.items():
            if cid not in estrato:
                continue
            amostra, nome = estrato[cid]
            w.writerow([cid, amostra, nome, n_estrato[(amostra, nome)],
                        n_vistos[(amostra, nome)], r["rotulo"], r["tipo"]])
            escritos += 1

    print(f"\nescrito: {destino} ({escritos} rotulos)")

    # --- DIAGNOSTICO --------------------------------------------------------
    # O peso e n_estrato/n_vistos. Pesos muito desiguais reduzem o tamanho
    # efetivo de amostra, entao vale olhar a dispersao antes de confiar.
    print(f"\n{'estrato':<26}{'n_pop':>9}{'vistos':>8}{'peso':>9}")
    print("-" * 52)
    por_estrato = defaultdict(lambda: [0, 0])
    for (amostra, nome), n in n_estrato.items():
        if n_vistos[(amostra, nome)] == 0:
            continue
        por_estrato[nome][0] += n
        por_estrato[nome][1] += n_vistos[(amostra, nome)]
    for nome in sorted(por_estrato):
        n, v = por_estrato[nome]
        print(f"{nome:<26}{n:>9}{v:>8}{n/v:>9.1f}")

    contagem = Counter((estrato[c][1].split("|")[-1], r["rotulo"])
                       for c, r in validos.items() if c in estrato)
    print(f"\n{'origem':<10}{'bons':>7}{'marginais':>12}{'lixo':>8}")
    print("-" * 37)
    for origem in ["topo", "resto"]:
        print(f"{origem:<10}{contagem[(origem,'2')]:>7}"
              f"{contagem[(origem,'1')]:>12}{contagem[(origem,'0')]:>8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pasta", help="pasta com candidatos.csv e rotulos.csv")
    p.add_argument("--dirigida", default=None,
                   help="pasta da fila dirigida (padrao: <pasta>/dirigida)")
    p.add_argument("--saida", default="rotulos_consolidado.csv")
    a = p.parse_args()
    main(a.pasta, a.dirigida or os.path.join(a.pasta, "dirigida"), a.saida)
