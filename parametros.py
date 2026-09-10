"""Parâmetros de aquisição e detecção centralizados num só lugar.

Depois de rodar calibrar_camera.py, calibrar_fov.py e calibrar_razao_foco.py,
copie os valores encontrados para cá. Assim a calibração fica toda em um único
arquivo, do mesmo jeito que os caminhos das pastas ficam em caminhos.py.

ORGANIZAÇÃO: por OBJETIVA, não por script
------------------------------------------
A versão antiga deste arquivo tinha FOV_DETECCAO/FOV_MOSAICO,
EXPOSICAO_DETECCAO/EXPOSICAO_MOSAICO -- ou seja, os parâmetros estavam
nomeados pelo SCRIPT que os usava. Isso embute uma hipótese que não se
sustenta: a de que cada script sempre usa a mesma objetiva.

Campo de visão, exposição e balanço de branco são propriedades da OBJETIVA
(e da iluminação), não do script. Quem escolhe a objetiva é você, na hora,
girando o revólver -- então quem pergunta é o script, e a resposta indexa
este dicionário.
"""

# =====================================================================
#  OBJETIVAS
# =====================================================================
#
# COMO PREENCHER: todo campo com None ainda não foi calibrado (é um "?").
# O script se recusa a varrer com uma objetiva cujos campos de aquisição
# estejam faltando, e diz exatamente quais -- em vez de varrer com um
# valor errado e você descobrir isso depois de horas de microscópio.
#
# De onde vem cada campo:
#
#   fov               -> calibrar_fov.py       (par (x, y) em µm)
#   exposicao         -> calibrar_camera.py    (ms)
#   wb_red / wb_blue  -> calibrar_camera.py    (inteiros do WhiteBalance)
#   autofoco_faixa_um -> calibrar_razao_foco.py (o z-stack; ver abaixo)
#   autofoco_passo_um -> calibrar_razao_foco.py
#   textura_minima    -> autofoco.csv de uma varredura curta (ver abaixo)
#
# --- Como tirar faixa e passo do z-stack ---
#
# A profundidade de campo (DOF) cai com o quadrado da abertura numérica
# (DOF ~ 1/NA²), então esses dois números MUDAM MUITO entre objetivas --
# é justamente por isso que eles moram aqui dentro e não no script.
# Um valor bom para o 50x é péssimo para o 5x, nos dois sentidos:
#
#   faixa curta demais -> o foco real cai fora do alcance da busca, o
#       autofoco bate na borda e é descartado em toda posição. O log fica
#       cheio de "melhor Z na borda da faixa" e o modelo nunca melhora.
#   passo grosso demais -> o pico de nitidez cai entre duas amostras e o
#       autofoco escolhe um Z sistematicamente errado.
#   passo fino demais   -> só desperdício: cada passo é uma foto, e o
#       autofoco roda dezenas de vezes por varredura.
#
# Do gráfico que o calibrar_razao_foco.py gera, meça a LARGURA DO PICO de
# nitidez (a 90% da altura, digamos). Ela é a sua DOF efetiva. Então:
#
#   autofoco_passo_um  ~  DOF / 4      (>= 2 amostras dentro da DOF; é o
#                                       Nyquist da busca -- com menos que
#                                       isso o pico passa despercebido)
#   autofoco_faixa_um  ~  2 a 3 x DOF  (larga o bastante pra abraçar o
#                                       erro do modelo de foco, estreita
#                                       o bastante pra não custar caro:
#                                       o custo é 2*faixa/passo fotos)
#
# --- Como achar a textura_minima ---
#
# É o portão que decide se vale rodar autofoco num campo: abaixo dela o
# campo é considerado vazio, onde a nitidez mede o ruído do sensor e o
# autofoco devolveria um Z aleatório (que contamina o modelo de foco).
#
# ATENÇÃO -- ela NÃO é transferível entre objetivas. A medir_textura
# devolve um desvio padrão em níveis de cinza de uma imagem de 8 bits, ou
# seja, ESCALA COM O BRILHO: com metade da exposição, o mesmo campo dá
# metade da textura. Duas objetivas com exposições diferentes têm
# limiares diferentes, necessariamente.
#
# Para medir: rode uma varredura curta com a textura_minima em 0.0 (nunca
# pula) e abra o autofoco.csv que ela grava na pasta da amostra. A coluna
# Textura mostra o valor de cada campo. Ponha o limiar entre o que os
# campos vazios deram e o que os campos com estrutura deram.

OBJETIVAS = {
    "5x": {
        "fov": (938.59, 946.82),                # ? (µm, µm)  -- calibrar_fov.py
        "exposicao": 12,          # ? ms        -- calibrar_camera.py
        "wb_red": 79,             # ?           -- calibrar_camera.py
        "wb_blue": 181,            # ?           -- calibrar_camera.py
        "autofoco_faixa_um": None,  # ? µm        -- z-stack
        "autofoco_passo_um": None,  # ? µm        -- z-stack
        "textura_minima": 4.57,     # ?           -- autofoco.csv
    },
    "10x": {
        "fov": (472.81, 480.19),                # ?
        "exposicao": 18,          # ?
        "wb_red": 91,             # ?
        "wb_blue": 186,            # ?
        "autofoco_faixa_um": 71.25,  # ?
        "autofoco_passo_um": 7.125,  # ?
        "textura_minima": 3.28,     # ?
    },
    "20x": {
        "fov": (239.79, 244.25),                # ?
        "exposicao": 36,          # ?
        "wb_red": 87,             # ?
        "wb_blue": 191,            # ?
        "autofoco_faixa_um": 30.00,  # ?  (z-stack de 10/08: pico largo e raso)
        "autofoco_passo_um": 3.00,  # ?
        "textura_minima": 2.61,     # ?
    },
    "50x": {
        "fov": (96.89, 98.9),                # ?
        "exposicao": 83,          # ?
        "wb_red": 75,             # ?
        "wb_blue": 188,            # ?
        "autofoco_faixa_um": 10.00,  # ?  (z-stack de 10/08: pico de ~5 µm a
        "autofoco_passo_um": 1.00,  # ?   90% -> a faixa antiga, 3.0, era
        "textura_minima": 1.20,     # ?   MENOR que a própria DOF)
    },
    "100x": {
        "fov": (49.99, 49.7),                # ?
        "exposicao": 60,          # ?
        "wb_red": 60,             # ?
        "wb_blue": 235,            # ?
        "autofoco_faixa_um": None,  # ?
        "autofoco_passo_um": None,  # ?
        "textura_minima": 1.05,     # ?
    },
}

# Sem esses, a varredura não sai do lugar: são o que define para onde o
# estágio anda e com que luz a foto é tirada. O script recusa a objetiva.
CAMPOS_AQUISICAO = ("fov", "exposicao", "wb_red", "wb_blue")

# Sem esses, dá pra varrer -- só que confiando exclusivamente no modelo de
# foco ajustado no início, sem correção durante a varredura. O script
# avisa e pergunta se você quer seguir assim.
CAMPOS_AUTOFOCO = ("autofoco_faixa_um", "autofoco_passo_um", "textura_minima")


"""--- SOBREPOSIÇÃO ENTRE IMAGENS VIZINHAS ---
Fração do campo que imagens vizinhas compartilham.

Por que não deixar em zero: um flake que cai em cima da divisa entre dois
tiles é cortado ao meio e cada pedaço é medido separadamente, com área
errada. Com sobreposição, qualquer flake MENOR que a faixa compartilhada cabe
inteiro em pelo menos uma imagem.

Custo: o número de imagens cresce com 1/(1-s)², nos dois eixos.
    0.05 ->  +6%
    0.10 -> +19%
    0.20 -> +54%

Fica fora do dicionário das objetivas de propósito: é uma FRAÇÃO do campo,
adimensional, então o mesmo valor significa a mesma coisa em qualquer
objetiva. O que muda com a objetiva é quantos µm essa fração representa --
e isso o script calcula na hora, a partir do fov.

Flakes maiores que a faixa continuam podendo ser cortados, mas aparecem
inteiros no mosaico montado -- que é onde se olha para eles mesmo. """

SOBREPOSICAO = 0.10


# =====================================================================
#  LEGADO -- em uso por deteccao.py e calibrar_razao_foco.py
# =====================================================================
# Esses são os valores que estavam no arquivo antes da reorganização por
# objetiva. Ficam aqui, isolados, por um motivo específico: o deteccao.py
# ainda não foi migrado, e apagá-los quebraria um script que hoje funciona.
#
# NÃO os copie para o dicionário acima. Não se sabe a qual objetiva eles
# pertencem -- o nome dizia só qual script os usava -- e o próprio FOV foi
# medido com a versão enviesada do calibrar_fov.py, antes da correção.
#
# Some daqui quando o deteccao.py passar a perguntar a objetiva.
FOV_DETECCAO = (93.0, 93.0)
EXPOSICAO_DETECCAO = 25.0
WHITE_BALANCE_RED = 88
WHITE_BALANCE_BLUE = 163

# --- LIMIARES DE DETECÇÃO do MaskTerial (usados em modelo.py) ---
SCORE_THRESHOLD = 0.015
MIN_CLASS_OCCUPANCY = 0.07
SIZE_THRESHOLD = 200

# --- FILTRO DE COR (pós-processamento, ver filtrar_por_cor em modelo.py) ---
# O contraste óptico separa grafeno de sujeira de cola quando os limiares acima
# não conseguem: o grafeno em SiO2 tem contraste NEGATIVO no verde (~550 nm),
# a cola tem contraste ~0. Mantemos só os flakes com contraste verde <= este
# valor (o canal verde do mean_contrast BGR de cada flake).
#
# Medido nesta amostra (3894 candidatos de uma varredura rotulados à mão):
#   -0.04  -> preserva 247/253 dos melhores flakes e remove ~2/3 da cola (padrão)
#    0.00  -> preserva TODOS os bons flakes, mas deixa passar mais cola
#   -0.08  -> remove mais cola ainda, mas já perde flakes de baixo contraste
# Para DESLIGAR o filtro, ponha um valor alto (ex. 10.0).
#
# APOSENTADO em 27/08/2026: o pipeline usa agora o modelo treinado
# (modelo.filtrar_por_modelo + LIMIAR_PROB_FLAKE abaixo). Este limiar segue
# aqui porque o filtrar_por_cor continua sendo o plano B quando o .joblib não
# está presente na máquina.
LIMIAR_CONTRASTE_G = -0.04


# --- LIMIAR DO MODELO TREINADO (modelo.filtrar_por_modelo) ---
# Probabilidade mínima para um candidato ser aceito como flake. O modelo é uma
# regressão logística sobre o contraste BGR médio, frac_substrato e snr_g,
# treinada em
# 800 candidatos rotulados à mão de 6 amostras (27/08/2026).
#
# Por que ele substitui o teto de contraste verde: o contraste cresce em módulo
# com o número de camadas, então um teto corta o que tem POUCO contraste (cola)
# e deixa passar o que tem MUITO (bulk). O flake fino mora numa FAIXA. O modelo
# usa os três canais e a fronteira que ele achou tem azul e verde com sinais
# OPOSTOS -- o que decide é a inclinação do espectro, não o quão escuro.
#
# A 5ª feature -- snr_g = |c_g| / std_g, a razão sinal/ruído do contraste -- foi
# acrescentada em 27/08 e é o que separa MONOCAMADA de substrato vazio. As duas
# têm contraste médio quase igual (um monolayer bom deu c_g -0,064 e um vazio
# -0,073, indistinguíveis), mas a monocamada tem espessura constante, logo
# contraste consistente, e o vazio é ruído do sensor: snr_g de 5,8 contra 0,7,
# quartis sem se tocar. Ela levou o recall de 83% para 90% e derrubou o desvio
# entre amostras de 25 para 16 pontos.
#
# Calibrado sobre a população de 38.133 candidatos (15.000 tiles):
#   0.55 -> ~200 candidatos por 1000 tiles  (mais seletivo)
#   0.31 -> ~300 por 1000 tiles             (padrão; o medido)
#   0.15 -> ~400 por 1000 tiles             (mais permissivo)
#   0.12 -> ~500 por 1000 tiles
#
# No ponto padrão, medido com leave-one-sample-out sobre 800 rótulos: captura
# 90% dos flakes BONS (±16) contra 40% da melhor regra de limiares no mesmo
# custo, com precisão 62%. O ajuste de produção anterior entregava 82% -- medido
# DENTRO da amostra, portanto mais generoso -- a 436 candidatos/1000 tiles e
# precisão 46%.
LIMIAR_PROB_FLAKE = 0.31


# --- DESENHO DE CONTORNOS UNIDOS (ver contornos_unidos/desenhar_contornos) ---
# O modelo às vezes fragmenta um flake em vários pedaços, ou detecta a mesma
# região em segmentos sobrepostos. Para o mosaico, a gente soma as máscaras e
# FECHA os vãos de até ~RAIO_FUSAO_PX pixels, virando um contorno só.
#   raio 0  -> une só o que se sobrepõe (não fecha fragmentação com gap)
#   raio maior -> fecha os gaps, mas raio demais funde flakes VIZINHOS distintos
# Ajuste vendo o mosaico.
RAIO_FUSAO_PX = 8

# Espessura do contorno no mosaico. Mais grosso que o display_results (2) de
# propósito: no mosaico reduzido, uma linha fina some.
ESPESSURA_CONTORNO = 2

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
# Ficam globais porque a razão hoje é só INSTRUMENTAÇÃO: ela é registrada,
# nunca decide nada. Se um dia virar portão (um RAZAO_MINIMA), o limiar
# terá de ir para o dicionário das objetivas -- a escala da razão acompanha
# a DOF, então um limiar único seria errado em quase todas.
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

# --- LIMIAR DO CLASSIFICADOR DE CONTRASTE (aplicado em modelo.py) ---
# Distância de Mahalanobis máxima, em desvios, entre o contraste de um pixel e o
# centro da classe mais próxima. Acima disso o pixel vira FUNDO (classe 0).
#
# Por que isso pesa mais que os três limiares acima: se a classe majoritária
# dentro de uma máscara for fundo, o MaskTerial DESCARTA o candidato lá dentro
# do predict() -- e SCORE/SIZE/OCCUPANCY só agem sobre o que sobrou. Medido
# nesta amostra: esse veto corta 91% do que passa por score>=0,015 e size>=50.
#
# 10 = o default do MaskTerial, que o load_models não repassa. Subir afrouxa o
# veto: mais pixels recebem classe, mais candidatos sobrevivem -- ao custo de
# aceitar mais lixo junto.
#
# MEDIDO em 27/08/2026 (800 rótulos, ponderado pela população). Fração dos
# flakes BONS que sobrevive ao veto, e fração do total que passa junto:
#
#     limiar    bons    todos          o que significa
#        10      39%      11%          o default: joga fora 61% dos bons
#        30      82%      42%          o valor anterior: ainda perde 18%
#        50      90%      57%   <--     recupera quase tudo que dá
#       100      91%      69%          +1 ponto de bom por +12 de lixo
#       inf      91%      72%          idem
#
# Escolhido 50: acima disso o ganho satura. Os ~9% de bons que morrem mesmo
# SEM veto de distância são a outra causa -- pixels mais parecidos com o
# substrato do que com qualquer material conhecido. Esses só voltam refazendo
# o classificador com dados desta montagem, não mexendo em limiar.
#
# Afrouxar o veto só é seguro porque o modelo treinado filtra depois
# (LIMIAR_PROB_FLAKE): entra mais lixo, e ele corta.
LIMIAR_MAHALANOBIS = 50.0
