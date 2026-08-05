"""
Mede o campo de visão real da câmera usando o
deslocamento conhecido do próprio estágio motorizado

Ideia: tira uma foto, move o estágio por uma distância pequena e
CONHECIDA, tira outra foto, e usa correlação de fase (cv2.phaseCorrelate)
para medir quantos pixels a imagem se deslocou entre as duas fotos. Daí:

    µm/pixel = distância_movida / deslocamento_medido_em_pixels
    FOV = tamanho_da_imagem_em_pixels * µm/pixel

Qualquer área com um pouco de textura/contraste serve (poeira, risco no substrato,
borda de um flake, etc.);
"""

from pycromanager import Core

import cv2
import numpy as np

from capturar_imagem import capturar_imagem

"""
 Distância a mover o estágio em cada teste, em µm. Precisa ser grande o
 bastante pra dar um deslocamento de vários pixels (medição confiável),
 mas pequena o bastante pra ainda sobrar bastante área em comum entre as
 duas fotos (senão a correlação de fase não acha o pico direito).
 Ajuste conforme a magnificação: comece pequeno e olhe a "confiança"
 impressa pelo script -- se estiver baixa, ou aumente esse valor (imagem
 de baixa magnificação/FOV grande) ou procure uma área com mais textura.
"""
DESLOCAMENTO_TESTE_UM = 10.0 #para magnificações menores, escolher tamanho também menores

# Quantas vezes repetir a medição em cada eixo, pra tirar a média
REPETICOES = 5

CONFIANCA_MINIMA = 0.3


def _para_correlacao(imagem):
    """Converte pra escala de cinza em float32 -- formato exigido pelo phaseCorrelate."""

    cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) if imagem.ndim == 3 else imagem
    return cinza.astype(np.float32)


def medir_deslocamento_em_pixels(imagem_a, imagem_b):
    """
    Usa correlação de fase pra medir quanto a imagem_b está deslocada em relação à imagem_a.

    Devolve ((deslocamento_x_px, deslocamento_y_px), confianca).
    """

    deslocamento_px, confianca = cv2.phaseCorrelate(
        _para_correlacao(imagem_a), _para_correlacao(imagem_b)
    )
    return deslocamento_px, confianca


def medir_um_por_pixel(core, camera, xy_stage, eixo):
    """
    Faz {REPETICOES} medições de µm/pixel movendo o estágio só no eixo
    indicado ('x' ou 'y'), voltando à posição original a cada repetição,
    e devolve a lista de medições (uma por repetição).
    """

    medicoes = []

    for rep in range(1, REPETICOES + 1):
        x0 = core.get_x_position(xy_stage)
        y0 = core.get_y_position(xy_stage)

        imagem_a = capturar_imagem(core, camera)

        if eixo == "x":
            core.set_xy_position(xy_stage, x0 + DESLOCAMENTO_TESTE_UM, y0)
        else:
            core.set_xy_position(xy_stage, x0, y0 + DESLOCAMENTO_TESTE_UM)
        core.wait_for_device(xy_stage)

        imagem_b = capturar_imagem(core, camera)

        # Volta pra posição original antes da próxima repetição.
        core.set_xy_position(xy_stage, x0, y0)
        core.wait_for_device(xy_stage)

        (deslocamento_x_px, deslocamento_y_px), confianca = medir_deslocamento_em_pixels(
            imagem_a, imagem_b
        )

        deslocamento_no_eixo_px = abs(deslocamento_x_px if eixo == "x" else deslocamento_y_px)
        deslocamento_perpendicular_px = abs(deslocamento_y_px if eixo == "x" else deslocamento_x_px)

        if deslocamento_no_eixo_px < 1e-6:
            print(f"  [{rep}/{REPETICOES}] deslocamento medido ~0 px -- descartando essa repetição.")
            continue

        um_por_pixel = DESLOCAMENTO_TESTE_UM / deslocamento_no_eixo_px
        medicoes.append(um_por_pixel)

        aviso_confianca = "" if confianca >= CONFIANCA_MINIMA else "  [Aviso] confiança baixa!"
        print(
            f"  [{rep}/{REPETICOES}] deslocamento no eixo: {deslocamento_no_eixo_px:6.2f} px | "
            f"perpendicular: {deslocamento_perpendicular_px:5.2f} px | "
            f"confiança: {confianca:.2f} | "
            f"µm/pixel: {um_por_pixel:.4f}{aviso_confianca}"
        )

    return medicoes


def main():
    core = Core()
    camera = core.get_camera_device()
    xy_stage = core.get_xy_stage_device()

    if core.is_sequence_running():
        core.stop_sequence_acquisition()

    input(
        "Mova o estágio para uma área com alguma textura/contraste "
        "(não precisa ser um flake -- poeira ou um risco no substrato já "
        "serve) e pressione ENTER..."
    )

    largura_px = core.get_image_width()
    altura_px = core.get_image_height()
    print(f"\nTamanho da imagem: {largura_px} x {altura_px} px")

    x_inicial = core.get_x_position(xy_stage)
    y_inicial = core.get_y_position(xy_stage)

    try:
        print("\n--- MEDINDO EIXO X ---")
        medicoes_x = medir_um_por_pixel(core, camera, xy_stage, "x")

        print("\n--- MEDINDO EIXO Y ---")
        medicoes_y = medir_um_por_pixel(core, camera, xy_stage, "y")
    finally:
        # Garante que o estágio volta pro ponto de partida mesmo se algo falhar acima.
        core.set_xy_position(xy_stage, x_inicial, y_inicial)
        core.wait_for_device(xy_stage)

    if not medicoes_x or not medicoes_y:
        print(
            "\n[Erro] Não foi possível medir um dos eixos (correlação de fase "
            "não encontrou deslocamento válido). Tente uma área com mais "
            "textura, ou aumente DESLOCAMENTO_TESTE_UM."
        )
        return

    um_por_pixel_x = float(np.mean(medicoes_x))
    um_por_pixel_y = float(np.mean(medicoes_y))

    fov_x = largura_px * um_por_pixel_x
    fov_y = altura_px * um_por_pixel_y

    print("\n--- RESULTADO FINAL ---")
    print(f"µm/pixel em X: {um_por_pixel_x:.4f}  (desvio padrão: {np.std(medicoes_x):.4f})")
    print(f"µm/pixel em Y: {um_por_pixel_y:.4f}  (desvio padrão: {np.std(medicoes_y):.4f})")
    print(f"\nFOV medido: {fov_x:.2f} x {fov_y:.2f} µm")
    print(
        "\nCopie esses valores para a linha "
        "\"fov_x, fov_y = ...\" no início do deteccao.py."
    )


if __name__ == "__main__":
    main()
