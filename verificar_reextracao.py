"""
verificar_reextracao.py  --  os cand_id novos apontam para os MESMOS candidatos?

Por que isso e obrigatorio antes de reaproveitar os rotulos
------------------------------------------------------------
O `cand_id` e um contador sequencial da extracao, nao um identificador do
objeto. Os 800 rotulos de rotulos_consolidado.csv guardam so esse numero. Se a
re-extracao mudar a ORDEM em que os candidatos aparecem -- outro sorteio de
tiles, outra ordem de amostras, uma instancia a mais vinda de nao-determinismo
da GPU -- entao o rotulo 4711 passa a descrever outro objeto.

E o pior tipo de erro: nada quebra. O treino roda, o sweep roda, os numeros
saem, e todos estao errados. Este script compara as colunas que NAO mudaram
(score, bbox, contraste) candidato a candidato e recusa o reaproveitamento se
algo nao bater.

Uso:
    python verificar_reextracao.py <csv_antigo> <csv_novo>
"""

import argparse
import csv
import sys

# Colunas que existiam antes e cujo valor a mudanca do extrator nao deveria
# alterar. Se todas baterem em todos os candidatos, os cand_id sao os mesmos.
CHAVES = ["amostra", "imagem", "score", "size_px", "x", "y", "w", "h",
          "c_b", "c_g", "c_r", "frac_substrato"]

TOLERANCIA = 1e-6      # float lido de texto: comparar com folga minima


def ler(caminho):
    with open(caminho, encoding="utf-8") as f:
        return {r["cand_id"]: r for r in csv.DictReader(f)}


def igual(a, b):
    try:
        return abs(float(a) - float(b)) <= TOLERANCIA * max(1.0, abs(float(a)))
    except ValueError:
        return a == b


def main(antigo, novo):
    A, B = ler(antigo), ler(novo)
    print(f"antigo: {len(A)} candidatos | novo: {len(B)} candidatos")

    if len(A) != len(B):
        print("\n[FALHOU] contagens diferentes -- a extracao nao foi reproduzida.")
        print("Confira: mesmas amostras, mesma ordem, mesmo --tiles, mesma SEMENTE.")
        return 1

    faltando = [k for k in A if k not in B]
    if faltando:
        print(f"\n[FALHOU] {len(faltando)} cand_id sumiram (ex: {faltando[:5]}).")
        return 1

    divergentes = []
    for cid, a in A.items():
        b = B[cid]
        ruins = [k for k in CHAVES if k in a and k in b and not igual(a[k], b[k])]
        if ruins:
            divergentes.append((cid, ruins))

    if divergentes:
        print(f"\n[FALHOU] {len(divergentes)} candidatos divergem.")
        for cid, ruins in divergentes[:5]:
            print(f"  cand_id {cid}: {', '.join(ruins)}")
            for k in ruins[:3]:
                print(f"      {k}: antigo={A[cid][k]}  novo={B[cid][k]}")
        print("\nNAO reaproveite rotulos_consolidado.csv com este CSV: os")
        print("cand_id descrevem objetos diferentes. Rotular de novo, ou achar")
        print("o que mudou na extracao.")
        return 1

    novas = [c for c in next(iter(B.values())) if c not in next(iter(A.values()))]
    print(f"\n[OK] todos os {len(A)} cand_id apontam para o mesmo candidato.")
    print(f"Colunas novas: {', '.join(novas) if novas else '(nenhuma)'}")
    print("Pode copiar rotulos_consolidado.csv e tiles.csv para a pasta nova.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("antigo", help="candidatos.csv da extracao original")
    p.add_argument("novo", help="candidatos.csv da re-extracao")
    a = p.parse_args()
    sys.exit(main(a.antigo, a.novo))
