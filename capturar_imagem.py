"""
Função de captura de imagem compartilhada entre os scripts que falam com
o microscópio (deteccao.py, calibrar_camera.py, calibrar_fov.py,
varredura_mosaico.py).
"""

import numpy as np


def capturar_imagem(core, camera):
    core.snap_image()
    core.wait_for_device(camera)

    pixels_brutos = core.get_image()
    largura = core.get_image_width()
    altura = core.get_image_height()

    try:
        imagem_bgra = pixels_brutos.reshape((altura, largura, 4))
        imagem = np.ascontiguousarray(imagem_bgra[:, :, :3][:, :, ::-1]) #imagem em RGB
    except ValueError:
        # Câmera não devolveu 4 canais (ex: modo monocromático).
        bytes_por_pixel = core.get_bytes_per_pixel()
        tipo_dado = np.uint8 if bytes_por_pixel == 1 else np.uint16
        imagem = pixels_brutos.view(tipo_dado).reshape((altura, largura))

    return imagem
