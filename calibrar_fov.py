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

# Deslocamento da PRIMEIRA medição, em µm. Serve só para descobrir a ordem
# de grandeza da escala -- não precisa ser bem escolhido, só pequeno o
# bastante para funcionar em qualquer objetiva sem perder a sobreposição.
DESLOCAMENTO_INICIAL_UM = 10.0

# Deslocamento da medição FINAL, como fração da largura/altura da imagem.
# Por que medir de novo com outro valor, em vez de usar um número fixo:
# a precisão relativa melhora com o tamanho do deslocamento. Dois erros
# competem aqui --
#   - o erro do próprio estágio (ex: ±0,1 µm) é ABSOLUTO: num comando de
#     10 µm ele vale 1%, num de 100 µm vale 0,1%;
#   - o erro da correlação de fase (~0,03 px) também é absoluto em pixels.
# Ambos encolhem, em termos relativos, quanto maior o deslocamento. O
# limite é a sobreposição entre as duas fotos: medido em simulação, a
# partir de ~50% da largura a confiança desaba (0,28) e o erro salta para
# 0,5 px. Perto de 30% ainda é preciso e sobra sobreposição de sobra.
# Como isso é uma FRAÇÃO DO CAMPO, o valor se adapta sozinho à objetiva --
# não existe mais uma constante em µm para acertar por magnificação.
FRACAO_DO_CAMPO = 0.28

# Quantas vezes repetir a medição em cada eixo, pra tirar a média
REPETICOES = 5

# Confiança mínima devolvida pelo cv2.phaseCorrelate para a medição valer.
# Abaixo disso a repetição é DESCARTADA, não só avisada: uma medição em que
# a correlação não achou um pico claro entra na média com o mesmo peso das
# boas e puxa o resultado inteiro para um valor errado.
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


def medir_um_por_pixel(core, camera, xy_stage, eixo, deslocamento_um):
    """
    Faz {REPETICOES} medições de µm/pixel movendo o estágio só no eixo
    indicado ('x' ou 'y') por deslocamento_um, voltando à posição original
    a cada repetição, e devolve a lista de medições que passaram nos
    testes de qualidade (pode ser menor que REPETICOES).
    """

    medicoes = []

    for rep in range(1, REPETICOES + 1):
        x0 = core.get_x_position(xy_stage)
        y0 = core.get_y_position(xy_stage)

        imagem_a = capturar_imagem(core, camera)

        if eixo == "x":
            core.set_xy_position(xy_stage, x0 + deslocamento_um, y0)
        else:
            core.set_xy_position(xy_stage, x0, y0 + deslocamento_um)
        core.wait_for_device(xy_stage)

        imagem_b = capturar_imagem(core, camera)

        # Volta pra posição original antes da próxima repetição.
        core.set_xy_position(xy_stage, x0, y0)
        core.wait_for_device(xy_stage)

        (deslocamento_x_px, deslocamento_y_px), confianca = medir_deslocamento_em_pixels(
            imagem_a, imagem_b
        )

        # MAGNITUDE do deslocamento, não a componente no eixo.
        # Se a câmera estiver rotacionada em relação aos eixos do estágio
        # (montagem C-mount torta, coisa banal), mover o estágio em X puro
        # gera deslocamento nas DUAS componentes da imagem. Usar só a
        # componente no eixo mede um valor menor que o real, e portanto
        # devolve µm/pixel MAIOR que o real -- o que superestima o FOV.
        # FOV superestimado faz a varredura andar mais que o campo e deixar
        # faixas nunca fotografadas entre um tile e o próximo. A magnitude
        # é imune à rotação: medido em simulação, com 15° de câmera torta o
        # método antigo errava +3,6% e a magnitude, +0,04%.
        deslocamento_px = float(np.hypot(deslocamento_x_px, deslocamento_y_px))

        if confianca < CONFIANCA_MINIMA:
            print(
                f"  [{rep}/{REPETICOES}] confiança {confianca:.2f} abaixo do mínimo "
                f"({CONFIANCA_MINIMA}) -- descartando essa repetição."
            )
            continue

        if deslocamento_px < 1e-6:
            print(f"  [{rep}/{REPETICOES}] deslocamento medido ~0 px -- descartando essa repetição.")
            continue

        um_por_pixel = deslocamento_um / deslocamento_px
        medicoes.append(um_por_pixel)

        print(
            f"  [{rep}/{REPETICOES}] deslocamento: {deslocamento_px:7.2f} px "
            f"(componentes X={deslocamento_x_px:+7.2f}, Y={deslocamento_y_px:+7.2f}) | "
            f"confiança: {confianca:.2f} | "
            f"µm/pixel: {um_por_pixel:.4f}"
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

    medicoes_x = medicoes_y = []

    try:
        # --- ETAPA 1: medição grosseira, só pra descobrir a escala ---
        # Não dá pra escolher o deslocamento ideal sem saber quantos µm vale
        # um pixel; e é justamente isso que se quer medir. Então mede-se uma
        # vez com um valor conservador, e o resultado dessa medição define o
        # deslocamento bom para a medição de verdade.
        print(f"\n--- ETAPA 1/2: medição inicial ({DESLOCAMENTO_INICIAL_UM} µm) ---")
        aproximadas = medir_um_por_pixel(
            core, camera, xy_stage, "x", DESLOCAMENTO_INICIAL_UM
        )

        if not aproximadas:
            print(
                "\n[Erro] A medição inicial falhou em todas as repetições. "
                "Procure uma área com mais textura/contraste, ou aumente "
                "DESLOCAMENTO_INICIAL_UM se a objetiva for de baixa magnificação."
            )
            return

        um_por_pixel_aprox = float(np.mean(aproximadas))

        # Deslocamento que faz a imagem andar FRACAO_DO_CAMPO do campo.
        deslocamento_x_um = FRACAO_DO_CAMPO * largura_px * um_por_pixel_aprox
        deslocamento_y_um = FRACAO_DO_CAMPO * altura_px * um_por_pixel_aprox

        print(f"\nEscala aproximada: {um_por_pixel_aprox:.4f} µm/pixel")
        print(
            f"Deslocamento escolhido para a medição final "
            f"({FRACAO_DO_CAMPO:.0%} do campo): "
            f"X = {deslocamento_x_um:.1f} µm, Y = {deslocamento_y_um:.1f} µm"
        )

        # --- ETAPA 2: medição final, com o deslocamento dimensionado ---
        print(f"\n--- ETAPA 2/2: MEDINDO EIXO X ({deslocamento_x_um:.1f} µm) ---")
        medicoes_x = medir_um_por_pixel(core, camera, xy_stage, "x", deslocamento_x_um)

        print(f"\n--- ETAPA 2/2: MEDINDO EIXO Y ({deslocamento_y_um:.1f} µm) ---")
        medicoes_y = medir_um_por_pixel(core, camera, xy_stage, "y", deslocamento_y_um)

    finally:
        # Garante que o estágio volta pro ponto de partida mesmo se algo falhar acima.
        core.set_xy_position(xy_stage, x_inicial, y_inicial)
        core.wait_for_device(xy_stage)

    if not medicoes_x or not medicoes_y:
        print(
            "\n[Erro] Não foi possível medir um dos eixos: todas as repetições "
            f"ficaram abaixo da confiança mínima ({CONFIANCA_MINIMA}) ou deram "
            "deslocamento nulo. Tente uma área com mais textura."
        )
        return

    um_por_pixel_x = float(np.mean(medicoes_x))
    um_por_pixel_y = float(np.mean(medicoes_y))

    fov_x = largura_px * um_por_pixel_x
    fov_y = altura_px * um_por_pixel_y

    print("\n--- RESULTADO FINAL ---")
    print(
        f"µm/pixel em X: {um_por_pixel_x:.4f}  "
        f"(desvio padrão: {np.std(medicoes_x):.4f}, {len(medicoes_x)}/{REPETICOES} medições válidas)"
    )
    print(
        f"µm/pixel em Y: {um_por_pixel_y:.4f}  "
        f"(desvio padrão: {np.std(medicoes_y):.4f}, {len(medicoes_y)}/{REPETICOES} medições válidas)"
    )
    print(f"\nFOV medido: {fov_x:.2f} x {fov_y:.2f} µm")
    print(
        "\nCopie esse par para o parametros.py, na constante da objetiva que "
        "você acabou de calibrar:"
    )
    print(f"    FOV_DETECCAO = ({fov_x:.1f}, {fov_y:.1f})   # objetiva do deteccao.py")
    print(f"    FOV_MOSAICO  = ({fov_x:.1f}, {fov_y:.1f})   # objetiva do varredura_mosaico.py")


if __name__ == "__main__":
    main()
