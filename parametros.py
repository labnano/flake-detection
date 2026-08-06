"""Parâmetros de aquisição e detecção, centralizados num só lugar.

Depois de rodar calibrar_camera.py e calibrar_fov.py, copie os valores
encontrados para cá. Assim a calibração fica toda em um único arquivo, do
mesmo jeito que os caminhos das pastas ficam em caminhos.py.

Obs.: a varredura de detecção (deteccao.py) e a de mosaico
(varredura_mosaico.py) usam objetivas de magnificações diferentes, por
isso cada uma tem o seu próprio campo de visão e a sua própria exposição.
"""

# --- CAMPO DE VISÃO, em µm (vem de calibrar_fov.py) ---
FOV_DETECCAO = (93.0, 93.0)     # objetiva usada em deteccao.py
FOV_MOSAICO = (933.0, 933.0)    # objetiva usada em varredura_mosaico.py

# --- SOBREPOSIÇÃO ENTRE TILES VIZINHOS ---
# Fração do campo que tiles vizinhos compartilham. 0.0 = tiles encostados
# borda com borda (o comportamento antigo); 0.10 = o estágio avança só 90%
# do campo, sobrando uma faixa de 10% em comum.
#
# Por que não deixar em zero: um flake que cai em cima da divisa entre dois
# tiles é cortado ao meio e cada pedaço é medido separadamente, com área
# errada. Sem sobreposição isso acontece com ~20% dos flakes de 10 µm e
# ~38% dos de 20 µm (campo de 93 µm). Com sobreposição, qualquer flake
# MENOR que a faixa compartilhada cabe inteiro em pelo menos um tile.
#
# Perder o flake de vez é raro e não é o motivo daqui: para sumir, os dois
# pedaços teriam que ficar abaixo de SIZE_THRESHOLD, o que exige área total
# menor que ~0,45 µm de lado -- praticamente o limite óptico.
#
# Custo: o número de imagens cresce com 1/(1-s)², nos dois eixos.
#   0.05 -> +6% de tiles, protege flakes até ~4,7 µm
#   0.10 -> +19%,          protege até ~9,3 µm   <-- escolhido
#   0.20 -> +54%,          protege até ~18,6 µm
# Flakes maiores que a faixa continuam podendo ser cortados, mas aparecem
# inteiros no mosaico montado -- que é onde se olha para eles mesmo.
SOBREPOSICAO = 0.10

# --- EXPOSIÇÃO, em ms (vem de calibrar_camera.py) ---
EXPOSICAO_DETECCAO = 25.0
EXPOSICAO_MOSAICO = 3.0

# --- BALANÇO DE BRANCO (vem de calibrar_camera.py) ---
WHITE_BALANCE_RED = 88
WHITE_BALANCE_BLUE = 163

# --- LIMIARES DE DETECÇÃO do MaskTerial (usados em modelo.py) ---
SCORE_THRESHOLD = 0.05
MIN_CLASS_OCCUPANCY = 0.025
SIZE_THRESHOLD = 50

# --- SCORE MINIMO PARA FOCO ---
SCORE_FOCO = 1