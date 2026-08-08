"""Parâmetros de aquisição e detecção centralizados num só lugar.

Depois de rodar calibrar_camera.py e calibrar_fov.py, copie os valores
encontrados para cá. Assim a calibração fica toda em um único arquivo, do
mesmo jeito que os caminhos das pastas ficam em caminhos.py.

"""

# --- CAMPO DE VISÃO, em µm (vem de calibrar_fov.py) ---
FOV_DETECCAO = (93.0, 93.0)     # objetiva usada em deteccao.py
FOV_MOSAICO = (933.0, 933.0)    # objetiva usada em varredura_mosaico.py

"""--- SOBREPOSIÇÃO ENTRE IMAGENS VIZINHAS ---
Fração do campo que imagens vizinhas compartilham. 

Por que não deixar em zero: um flake que cai em cima da divisa entre dois
tiles é cortado ao meio e cada pedaço é medido separadamente, com área
errada. Com sobreposição, qualquer flake MENOR que a faixa compartilhada cabe
inteiro em pelo menos uma imagem.

Custo: o número de imagens cresce com 1/(1-s)², nos dois eixos.
    0.05 ->  +6%,          protege flakes até ~4,7 µm
    0.10 -> +19%,          protege até ~9,3 µm
    0.20 -> +54%,          protege até ~18,6 µm

Flakes maiores que a faixa continuam podendo ser cortados, mas aparecem
inteiros no mosaico montado -- que é onde se olha para eles mesmo. """

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

# --- BANDAS DE ESCALA DA MÉTRICA DE FOCO (ver medir_foco_relativo) ---
#
# Em pixels. A ideia: desfocar é um filtro passa-baixa -- destrói o detalhe
# fino e quase não toca na estrutura grossa. Medindo as duas faixas e
# tomando a RAZÃO, a quantidade de conteúdo do campo (que multiplica as
# duas igualmente) se cancela, e sobra só o efeito do foco.
#
# O vão entre SIGMA_FINA_MAX e SIGMA_GROSSA_MIN é proposital: é a folga que
# desacopla o denominador do numerador. Se a banda grossa começasse em 4 px
# ela também cairia com o desfoque e cancelaria parte do sinal -- foi por
# isso que dividir pela medir_textura (que usa sigma 2) não serve.
#
# CALIBRAÇÃO: dependem da profundidade de campo da objetiva e de quantos
# pixels o borrão ocupa. Os valores abaixo são ponto de partida.
SIGMA_FINA_MIN = 1.0
SIGMA_FINA_MAX = 4.0
SIGMA_GROSSA_MIN = 8.0
SIGMA_GROSSA_MAX = 32.0

# Fator de redução aplicado ANTES de medir a banda grossa.
# Por que é seguro: essa banda não contém, por construção, nada menor que
# SIGMA_GROSSA_MIN = 8 px. Reduzir por 4 leva o menor detalhe relevante a
# 2 px -- ainda acima do limite de Nyquist, então nada de útil se perde.
# Por que vale a pena: o custo de um blur cresce com o tamanho do kernel
# (~6*sigma). Na resolução total, sigma 32 pede kernel de ~193 px; na
# imagem reduzida o mesmo sigma físico vira 8, kernel de ~49 px, sobre 16x
# menos pixels. O custo cai quase duas ordens de grandeza -- e é isso que
# permite medir em TODA posição da varredura, e não só nos checkpoints.
REDUCAO_BANDA_GROSSA = 4