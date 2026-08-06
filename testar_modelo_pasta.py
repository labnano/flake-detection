"""
Roda o modelo do MaskTerial em todas as imagens .tif de uma pasta do
computador, sem precisar do microscópio conectado.

Útil para conferir se o modelo está detectando flakes corretamente antes
de sair rodando uma varredura de verdade.

Rodar no terminal como 'python testar_modelo_pasta.py *caminho da pasta*' com o ambiente virtual aberto
"""

import argparse
import glob
import os

import cv2

from modelo import obter_predictor, display_results


def main(pasta):
    predictor = obter_predictor()

    caminhos = glob.glob(os.path.join(pasta, "**", "*.tif"), recursive=True)

    if not caminhos:
        print(f"Nenhuma imagem .tif encontrada em: {pasta}")
        return

    for caminho in caminhos:
        imagem = cv2.imread(caminho)
        print(f"\nTestando: {caminho}")
        #imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        flakes = predictor.predict(imagem)

        if flakes:
            print(f"  {len(flakes)} flake(s) encontrado(s).")
            display_results(imagem, flakes, show=True)
        else:
            print("  Nenhum flake encontrado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pasta", help="Pasta com as imagens .tif para testar")
    args = parser.parse_args()
    main(args.pasta)
