"""Parâmetros de aquisição e detecção, centralizados num só lugar.

Depois de rodar calibrar_camera.py e calibrar_fov.py, copie os valores
encontrados para cá. Assim a calibração fica toda em um único arquivo, do
mesmo jeito que os caminhos das pastas ficam em caminhos.py.

Obs.: a varredura de detecção (deteccao.py) e a de mosaico
(varredura_mosaico.py) usam objetivas de magnificações diferentes, por
isso cada uma tem o seu próprio campo de visão e a sua própria exposição.
"""

# --- CAMPO DE VISÃO, em µm (vem de calibrar_fov.py) ---
FOV_DETECCAO = (93.3, 93.3)     # objetiva usada em deteccao.py
FOV_MOSAICO = (933.0, 933.0)    # objetiva usada em varredura_mosaico.py

# --- EXPOSIÇÃO, em ms (vem de calibrar_camera.py) ---
EXPOSICAO_DETECCAO = 20.0
EXPOSICAO_MOSAICO = 3.0

# --- BALANÇO DE BRANCO (vem de calibrar_camera.py) ---
WHITE_BALANCE_RED = 400
WHITE_BALANCE_BLUE = 240

# --- LIMIARES DE DETECÇÃO do MaskTerial (usados em modelo.py) ---
SCORE_THRESHOLD = 0.0005
MIN_CLASS_OCCUPANCY = 0.00025
SIZE_THRESHOLD = 50
