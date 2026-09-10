"""
stack_exposicao.py -- varre a exposição e salva uma imagem por valor.

Irmão do calibrar_razao_foco.py: lá o eixo varrido é o Z, aqui é o tempo de
exposição. Captura e salva, só isso -- a análise fica por sua conta, offline,
em cima dos arquivos.

Uso:
    python stack_exposicao.py
"""

import datetime
import os

import numpy as np
from pycromanager import Core
from tifffile import imwrite

from caminhos import DIRETORIO_SAIDA_BASE
from capturar_imagem import capturar_imagem

# Faixa varrida, em ms.
EXPOSICAO_MINIMA = 1.0
EXPOSICAO_MAXIMA = 60.0
PASSO = 2.0


def main():
    core = Core()
    camera = core.get_camera_device()

    if core.is_sequence_running():
        core.stop_sequence_acquisition()

    exposicoes = np.arange(EXPOSICAO_MINIMA, EXPOSICAO_MAXIMA + PASSO / 2, PASSO)

    print(f"{len(exposicoes)} exposições, de {exposicoes[0]:.2f} a {exposicoes[-1]:.2f} ms.")
    input("Posicione o estágio no campo desejado e pressione ENTER...")

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    diretorio = os.path.join(DIRETORIO_SAIDA_BASE, "Exposicao_" + timestamp)
    os.makedirs(diretorio, exist_ok=True)

    # Guardado para ser devolvido no fim: o script não deve deixar a câmera
    # numa exposição diferente da que encontrou.
    exposicao_original = core.get_exposure()

    try:
        for indice, exposicao in enumerate(exposicoes):
            core.set_exposure(float(exposicao))
            core.wait_for_device(camera)

            # Um quadro descartado depois de trocar a exposição: a câmera
            # costuma entregar ainda o quadro anterior, capturado com a
            # configuração antiga. Ele apareceria no stack como um ponto
            # fora da curva sem nenhuma causa física.
            capturar_imagem(core, camera)

            imagem = capturar_imagem(core, camera)

            nome_arquivo = f"exp_{indice:03d}_{exposicao:06.2f}ms.tif"
            imwrite(os.path.join(diretorio, nome_arquivo), imagem)

            print(f"  {nome_arquivo}")

    finally:
        print(f"\nDevolvendo a exposição para {exposicao_original:.2f} ms...")
        core.set_exposure(exposicao_original)
        core.wait_for_device(camera)

    print(f"\n{len(exposicoes)} imagens em: {diretorio}")


if __name__ == "__main__":
    main()
