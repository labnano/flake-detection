"""Funções de calibração de foco e geração de malha de varredura,
compartilhadas entre deteccao.py e varredura_mosaico.py."""

import json
import math
import os
import time
from contextlib import contextmanager

import cv2
import numpy as np

from capturar_imagem import capturar_imagem
from parametros import (
    SIGMA_FINA_MIN,
    SIGMA_FINA_MAX,
    SIGMA_GROSSA_MIN,
    SIGMA_GROSSA_MAX,
    REDUCAO_BANDA_GROSSA,
    OBJETIVAS,
    CAMPOS_AQUISICAO,
    CAMPOS_AUTOFOCO,
)
"""
Limite de condicionamento aceitável para os ajustes de foco.

O número de condição da matriz de projeto é o FATOR DE AMPLIFICAÇÃO do
erro dos pontos de calibração no resultado: com cond = 1e6, um erro
humano de 0,4 µm ao focar um ponto vira até 4e5 µm no Z previsto.
"""
LIMITE_CONDICIONAMENTO = 1e4

# Inclinação da amostra a partir da qual vale a pena parar e realinhar o
# chip no suporte. Em 3° a caixa contida já perde ~10% da área marcada.
ROTACAO_AVISO_GRAUS = 3.0

# Fração da área marcada que a caixa contida pode descartar sem reclamar.
PERDA_AVISO = 0.10

# Teto do offset escalar de foco, em MÚLTIPLOS da faixa de autofoco.
#
# Por que em múltiplos e não em µm fixos: a faixa já é proporcional à
# profundidade de campo da objetiva, então um teto escrito assim escala
# junto -- 5 µm seria frouxo demais no 5x e apertado demais no 100x.
#
# O offset só entra em cena quando o re-ajuste do modelo falha; ele é
# recalculado (não acumulado) a cada checkpoint, mas cada checkpoint pode
# deslocá-lo em até uma faixa. Sem teto, uma sequência de medidas ruins
# levaria o foco embora sem nada para segurar.
AJUSTE_Z_MAXIMO_EM_FAIXAS = 3.0

# =====================================================================
#  BUSCA DE FOCO -- quando parar de gastar fotos
# =====================================================================
#
# A busca antiga varria a faixa inteira ponto a ponto e ficava com o
# argmax: com faixa/passo = 10 (o que o parametros.py tem hoje em todas as
# objetivas), isso são 21 fotos por checkpoint, SEMPRE -- inclusive quando
# o modelo estava certo e o foco já estava no centro da faixa.
#
# A busca nova (buscar_foco) cresce do centro para fora e para assim que
# encontra um máximo com vizinho medido dos dois lados. Quando o modelo
# está bom ela custa 3 fotos; quando está ruim, degrada até o custo antigo.
# É a mesma faixa, o mesmo passo e o mesmo critério de borda -- muda só a
# ORDEM em que os pontos são visitados e a hora de parar.

# Fração mínima da queda ESPERADA que a queda observada nos vizinhos do
# pico precisa atingir para o pico ser considerado real.
#
# De onde sai a queda esperada: o parametros.py define faixa ~ 2,5 x DOF e
# a DOF como a largura do pico a 90% da altura. Modelando o topo do pico
# como parábola, a queda a uma distância d do máximo é
#       queda(d) ~ 0,10 * (d / (DOF/2))^2
# e, com DOF ~ faixa/2,5, medindo a um passo do pico (d = passo):
#       queda(passo) ~ 2,5 * (passo/faixa)^2
# Com passo/faixa = 0,1 isso dá ~2,5%: pouco, porque o passo atual é bem
# mais fino que a DOF. É por isso que o limiar é uma FRAÇÃO da queda
# esperada e não um número absoluto -- assim ele acompanha sozinho quando
# você mudar o passo, em vez de virar um limiar errado em silêncio.
#
# O que este teste pega: campo que passou no portão de textura mas cuja
# curva de nitidez é chata (estrutura focada em outro plano, sujeira na
# ótica, iluminação em gradiente). Ali o argmax é ruído, e sem este teste
# ele viraria ponto de calibração.
FRACAO_QUEDA_MINIMA = 0.33

# Deslocamento sub-passo, em frações do passo, a partir do qual vale a pena
# gastar UMA foto extra para registrar a imagem no vértice interpolado.
# Abaixo disso o ganho de nitidez não paga a foto.
DESLOCAMENTO_MINIMO_REFINO = 0.15

# =====================================================================
#  ROBUSTEZ DO MODELO DE FOCO
# =====================================================================
#
# Duas proteções para o mesmo problema: o modelo se re-ajusta com o que o
# autofoco mede, então uma medida ruim não estraga uma imagem -- estraga
# TODAS as posições seguintes, e para sempre.

# Um SALTO maior que isto é tratado como medida espúria (o autofoco travou
# num detrito alto, numa borda de flake, num risco na lâmina) e é
# descartado em vez de virar ponto de calibração.
#
# São dois critérios, e vale o MAIOR dos dois:
#   FATOR_OUTLIER_FAIXA * faixa -> piso absoluto. Com faixa ~ 2,5 x DOF,
#       0,5 deixa passar sem discussão qualquer salto de até ~1 DOF,
#       que é a escala do que o modelo erra legitimamente.
#   FATOR_OUTLIER_MAD * desvio  -> teto adaptativo, onde o desvio é medido
#       nos resíduos do próprio modelo (via MAD, que é robusto: um outlier
#       já dentro do conjunto não infla a mediana como inflaria o desvio
#       padrão). Numa amostra que se comporta bem isso aperta o critério;
#       numa amostra ruidosa, afrouxa. É o mesmo espírito do "3 sigma",
#       só que com um sigma que o próprio dado define.
#
# SALTO, e não a correção em si -- a diferença é o que separa este filtro
# de uma armadilha. Sob deriva térmica as correções legítimas crescem sem
# parar, e um filtro que as medisse contra ZERO passaria a recusá-las
# assim que a deriva acumulada superasse o limite. Aí o modelo pararia de
# ser corrigido, a deriva continuaria, a correção seguinte seria maior
# ainda: uma espiral que só termina com a amostra inteira desfocada. Medir
# contra a ÚLTIMA CORREÇÃO ACEITA desarma isso -- deriva dá incrementos
# pequenos e do mesmo sinal, detrito dá um salto isolado.
FATOR_OUTLIER_FAIXA = 0.5
FATOR_OUTLIER_MAD = 3.0

# Quantas rejeições seguidas por salto o sistema aceita antes de concluir
# que o errado é a referência, e não as medidas.
#
# É a segunda trava contra a mesma espiral. Se duas medições independentes,
# em posições diferentes, insistem que o foco está longe de onde o modelo
# acha, a hipótese "as duas caíram em detrito" fica menos plausível que
# "o modelo se perdeu" -- e a terceira é aceita, com aviso no console.
REJEICOES_ATE_ACEITAR = 2

# Quanto a faixa de busca é alargada depois de um checkpoint ter batido na
# borda -- ou seja, depois de o foco ter sido procurado e não encontrado.
#
# Bater na borda significa que o foco está FORA do alcance da busca. Sem
# alargar, o checkpoint seguinte procura no mesmo lugar, não acha de novo,
# e a varredura segue borrando até o fim sem nunca ter como voltar. É o
# mesmo raciocínio (e o mesmo fator) do autofoco de abertura de cada
# amostra, só que disparado pelo sintoma em vez de pelo calendário: custa
# fotos apenas quando alguma coisa de fato saiu do lugar.
FATOR_FAIXA_RECUPERACAO = 3.0

# Meia-vida, em NÚMERO DE PONTOS, do peso de um ponto medido pelo autofoco.
#
# Por que envelhecer os pontos: o re-ajuste antigo dava peso igual a todos.
# Como a lista só cresce, um ponto novo entrava com peso 1/N -- ou seja, a
# capacidade de correção do modelo DIMINUÍA ao longo da varredura, que é o
# oposto do que se quer quando existe deriva térmica (o foco andando com o
# TEMPO, não com a posição). Com meia-vida de 12 pontos, o que foi medido
# nos últimos ~12 checkpoints domina o ajuste e a deriva é seguida.
#
# Por que não menor: com meia-vida h, o número EFETIVO de pontos que pesam
# no ajuste é da ordem de h/ln2 ~ 1,4h. A quadrática tem 6 coeficientes, e
# ajustar 6 parâmetros com ~6 pontos efetivos é reproduzir o ruído de cada
# um deles. Com h = 12 sobram ~17 pontos efetivos, folga confortável; com
# h = 4 seriam ~6, e a quadrática ficaria no limite.
#
# Sensibilidade medida (ensaio simulado, 216 imagens, 11 checkpoints, erro
# de foco médio das imagens gravadas):
#     deriva          h=4     h=8     h=12    h=20
#     nenhuma        0,27    0,30    0,28    0,28  µm
#     ~25 µm/h       0,38    0,44    0,49    0,52  µm
#     ~50 µm/h       0,70    0,88    0,89    0,94  µm
# Ou seja: sem deriva o valor é indiferente, e com deriva realista todos
# ficam muito abaixo da DOF (~4 µm no 50x). Se os seus autofoco.csv
# mostrarem deriva forte -- Correcao crescendo com o Contador --, baixar
# para 6 ou 8 é a primeira coisa a tentar.
MEIA_VIDA_PONTOS = 12.0

# Piso do peso dos pontos coletados à mão na Fase 2 (as âncoras), usado
# apenas no PLANO B do re-ajuste -- ver reajustar_modelo_foco.
#
# Para que ele existe: as âncoras são os únicos pontos GARANTIDAMENTE bem
# espalhados pela amostra (a Fase 2 valida o condicionamento delas). Os
# pontos de autofoco recentes ficam todos juntos, porque a varredura
# progride localmente -- ajustar uma superfície só com eles seria
# extrapolar de uma vizinhança para a amostra inteira.
#
# Por que não é o padrão: segurar as âncoras segura a geometria, mas
# segura junto a ALTURA -- e sob deriva térmica isso prende o modelo à
# posição que a amostra tinha no começo, por mais que o autofoco meça o
# contrário. Medido no ensaio simulado: com o piso sempre ligado, o
# envelhecimento reduzia o erro de foco em 38%; com ele apenas como plano
# B, em 47%. Daí a ordem: primeiro tenta sem, e só recorre a ele quando o
# ajuste não fecha por falta de espalhamento.
PESO_MINIMO_ANCORA = 0.25


# =====================================================================
#  RE-ANCORAGEM -- o foco que anda enquanto a amostra espera a vez
# =====================================================================
#
# Os pontos de foco de TODAS as amostras são coletados na Fase 2, antes de
# qualquer imagem ser tirada. Numa rodada de cinco amostras isso quer dizer
# que a última é varrida horas depois de ter sido focada -- e nesse
# intervalo o foco anda.
#
# Medido na varredura de 17/08 (50x, 5 amostras, 8,9 h): ~35 µm/h, seguindo
# os Z que o autofoco mediu ao longo da amostra 1. As amostras 2 a 5 foram
# varridas de 2 a 7 h depois de calibradas, com uma busca de autofoco de
# ±10 µm. O foco nunca esteve ao alcance: nas ~35 mil imagens delas a
# nitidez ficou travada em 20,0 -- o piso de ruído de leitura do sensor,
# sem nenhum sinal óptico. Quatro amostras perdidas sem um único aviso.
#
# A re-ancoragem mede esse deslocamento IMEDIATAMENTE antes de varrer cada
# amostra e corrige com ele os pontos de calibração. Ela não substitui o
# autofoco durante a varredura: aquele continua acompanhando o que o foco
# anda ao longo das ~2 h de uma amostra. Esta resolve o outro problema, que
# o autofoco não tem como resolver -- o que andou ENQUANTO ela esperava.

# Passo da busca grossa, em frações da faixa de autofoco da objetiva.
#
# A busca grossa atravessa centenas de µm, então cada foto pesa -- mas um
# passo grosso demais passa por cima do pico sem enxergá-lo. O teto é o
# Nyquist da busca: com faixa ~ 2,5 x DOF (a receita do parametros.py), um
# passo de 0,2 faixa é ~ DOF/2, ou seja duas amostras dentro da
# profundidade de campo. Na 50x, com faixa 10 µm, dá 2 µm.
#
# A varredura de 17/08 sugere que a margem real é maior ainda: as quedas de
# nitidez medidas ali implicam um pico de ~9 µm de largura, contra os 4 µm
# que a faixa configurada supõe. Se isso se confirmar, o passo poderia
# subir -- mas errar para o lado fino aqui custa fotos, e errar para o
# grosso custa a amostra inteira.
FATOR_PASSO_GROSSO = 0.2

# Proeminência mínima do pico da busca grossa, em desvios robustos do
# próprio conjunto de medidas.
#
# É o que separa "achei o foco" de "varri 300 µm de imagem borrada e fiquei
# com o maior ruído". A variância do Laplaciano SATURA num piso longe do
# foco: na varredura de 17/08, 60 imagens sorteadas de uma amostra perdida
# deram todas entre 19,5 e 20,4. Nesse platô o argmax é sorteio.
#
# Medindo a altura do máximo acima da MEDIANA em unidades do desvio robusto
# (MAD) do conjunto, aquele caso dá ~2,7 e é recusado; um pico de verdade,
# que sobe de 20 para 60 ou 275, passa de longe. A mediana e o MAD são os
# estimadores certos aqui justamente porque a maior parte da faixa varrida
# ESTÁ fora de foco -- o pico é o outlier que se procura, e não pode ser o
# mesmo que define a régua.
PROEMINENCIA_MINIMA_GROSSA = 5.0

# Dispersão máxima entre os deslocamentos medidos nos pontos de conferência
# para que o desvio seja tratado como um OFFSET uniforme, em frações da
# faixa de autofoco da objetiva.
#
# A hipótese que se testa aqui é física: deriva térmica ou mecânica desloca
# o conjunto inteiro, e não reinclina o chip dentro do suporte. Se ela vale,
# os deslocamentos medidos em pontos distantes concordam entre si e basta
# somar um número a todas as âncoras -- a geometria, que a Fase 2 validou,
# é preservada de graça. Se não vale, a forma mudou e não há atalho: todas
# as âncoras têm de ser re-medidas.
#
# 0,3 da faixa é ~1 DOF: folgado o bastante para não disparar com o ruído da
# própria medida (o refino sub-passo entrega bem menos que isso), apertado o
# bastante para pegar uma mudança de forma que importe.
FATOR_DISPERSAO_OFFSET = 0.3

# Quantos pontos de conferência a re-ancoragem visita depois de medir o
# deslocamento no ponto de sondagem.
#
# Dois, mais o de sondagem, dão TRÊS medidas. Três é o mínimo para que a
# mediana identifique qual delas discordou: com duas dá para saber que houve
# desacordo, mas não qual das duas mentiu. E três pontos ainda determinam um
# plano, então se o desvio não for uniforme eles próprios dão a correção de
# primeira ordem.
QTD_PONTOS_CONFERENCIA = 2

# Quanto a faixa da CONFERÊNCIA é alargada em relação à da objetiva.
#
# A conferência existe para detectar que a superfície mudou de forma, e não
# só de altura. Se ela procurasse só na faixa normal, uma mudança de forma
# maior que essa faixa a faria bater na borda ou -- pior -- devolver um
# pico de ruído perto do centro, e a discordância que ela deveria denunciar
# passaria como concordância.
#
# Foi o que aconteceu num ensaio simulado com deriva de 150 µm mais uma
# inclinação nova: na faixa normal a conferência "confirmou" o deslocamento
# da sondagem e o modelo saiu com 16 µm de erro; com a faixa alargada ela
# encontrou o foco de verdade, a discordância apareceu, e a re-ancoragem
# caiu corretamente no caminho de re-medir todas as âncoras.
#
# Alargar aqui é seguro justamente porque o defeito do limiar de pico chato
# foi corrigido: a buscar_foco recebe a faixa NOMINAL como referência, então
# o critério de aceitação continua o mesmo por mais que a busca se estenda.
FATOR_FAIXA_CONFERENCIA = 3.0

# Quantos pontos diferentes a sondagem tenta antes de desistir.
#
# A sondagem pode falhar por um motivo puramente local: aquele campo não tem
# estrutura em que medir foco. Desistir na primeira tentativa custaria a
# amostra inteira (~2 h varrendo com o modelo velho); tentar outro ponto
# custa mais uma busca grossa (~2 min). A assimetria é grande o bastante
# para justificar tentar de novo, e há âncoras de sobra.
QTD_TENTATIVAS_SONDAGEM = 3


class Cronometro:
    """Acumula quanto tempo se gasta em cada etapa de um laço.

    Existe para responder uma pergunta que nenhuma otimização deveria ser
    feita sem responder antes: ONDE o tempo está indo. Numa varredura, os
    candidatos são o estágio XY, o estágio Z, a câmera e o disco -- e a
    diferença entre eles muda completamente o que vale a pena atacar.

    Uso:
        cronometro = Cronometro()

        with cronometro.medir("mover_xy"):
            core.set_xy_position(...)
            core.wait_for_device(...)

        print(cronometro.resumo())

    O `with` aqui é um GERENCIADOR DE CONTEXTO: o código antes do `yield`
    roda ao entrar no bloco, o de depois ao sair -- e o `finally` garante
    que o tempo é contabilizado mesmo se o corpo levantar exceção. É o
    mesmo mecanismo do `with open(...)`, que fecha o arquivo aconteça o
    que acontecer.

    Usa time.perf_counter() e não time.time(): o perf_counter é monotônico
    (nunca anda para trás) e tem resolução bem melhor. O time.time() marca
    data e hora do mundo, e pode dar salto se o relógio do sistema for
    acertado no meio da varredura -- o que produziria um intervalo negativo
    no meio do log."""

    def __init__(self):
        self.total = {}
        self.chamadas = {}

    @contextmanager
    def medir(self, nome):
        inicio = time.perf_counter()

        try:
            yield
        finally:
            decorrido = time.perf_counter() - inicio

            self.total[nome] = self.total.get(nome, 0.0) + decorrido
            self.chamadas[nome] = self.chamadas.get(nome, 0) + 1

    def somar(self, outro):
        """Incorpora as medidas de outro cronômetro (ex.: o de uma amostra
        no cronômetro do lote inteiro)."""

        for nome, valor in outro.total.items():
            self.total[nome] = self.total.get(nome, 0.0) + valor
            self.chamadas[nome] = self.chamadas.get(nome, 0) + outro.chamadas[nome]

    def resumo(self, indentacao="  "):
        """Tabela ordenada do mais caro para o mais barato, com o tempo
        médio por chamada e a fatia que cada etapa representa do total
        cronometrado. A fatia é o que responde 'o que vale a pena
        otimizar'; a média é o que se compara com a especificação do
        equipamento."""

        if not self.total:
            return f"{indentacao}(nada cronometrado)"

        soma = sum(self.total.values())
        linhas = [
            f"{indentacao}{'etapa':<14}{'total':>10}{'chamadas':>10}"
            f"{'média':>11}{'fatia':>8}"
        ]

        for nome, valor in sorted(self.total.items(), key=lambda par: -par[1]):
            chamadas = self.chamadas[nome]

            linhas.append(
                f"{indentacao}{nome:<14}{valor:>9.1f}s{chamadas:>10}"
                f"{valor / chamadas * 1000:>10.0f}ms{valor / soma:>8.0%}"
            )

        return "\n".join(linhas)

    def como_dicionario(self):
        """O mesmo conteúdo em forma serializável, para ir junto dos
        metadados da varredura -- comparar duas rodadas exige o número
        gravado, não o que passou no console."""

        return {
            nome: {
                "segundos": round(valor, 3),
                "chamadas": self.chamadas[nome],
                "media_ms": round(valor / self.chamadas[nome] * 1000, 2),
            }
            for nome, valor in sorted(self.total.items(), key=lambda par: -par[1])
        }


def campos_faltando(nome_objetiva, campos):
    """Devolve os campos de uma objetiva que ainda estão como None ("?").

    Existe para o script poder RECUSAR uma objetiva mal calibrada antes de
    você marcar cantos e focar pontos -- e não no meio da varredura, com um
    None chegando no core.set_exposure()."""

    config = OBJETIVAS[nome_objetiva]

    return [campo for campo in campos if config.get(campo) is None]


def escolher_objetiva(exigidos=CAMPOS_AQUISICAO):
    """Pergunta qual objetiva está montada e devolve (nome, config).

    `exigidos` são os campos sem os quais a objetiva é recusada. A varredura
    exige os quatro de aquisição; um script de calibração exige só o que ele
    de fato usa -- o calibrar_textura.py, por exemplo, precisa da exposição e
    do balanço de branco, mas não do fov, e exigir o fov ali só criaria uma
    ordem de calibração artificial.

    O Micro-Manager desta montagem NÃO sabe qual objetiva está no caminho
    óptico -- o revólver é girado à mão e nada reporta isso de volta. Ou
    seja: esta resposta é a ÚNICA fonte da verdade, e não há como o script
    conferi-la contra o hardware.

    A consequência prática de errar aqui é silenciosa e cara: um fov errado
    vira um passo de estágio errado, e a varredura inteira sai com buracos
    ou com faixas repetidas, sem nenhum sintoma visível durante a corrida.
    Daí o menu numerado (em vez de digitar "50x" e errar para "5x") e a
    confirmação em voz alta no resumo final."""

    nomes = list(OBJETIVAS)

    print("\n--- QUAL OBJETIVA ESTÁ MONTADA? ---")
    print("(o Micro-Manager não sabe responder isso sozinho -- confira no revólver)")

    for indice, nome in enumerate(nomes, start=1):
        faltam_exigidos = campos_faltando(nome, exigidos)
        faltam_autofoco = campos_faltando(nome, CAMPOS_AUTOFOCO)

        if faltam_exigidos:
            estado = f"[SEM CALIBRAÇÃO: falta {', '.join(faltam_exigidos)}]"
        elif faltam_autofoco:
            estado = f"[ok, mas sem autofoco: falta {', '.join(faltam_autofoco)}]"
        else:
            estado = "[calibrada]"

        print(f"  {indice}. {nome:<5} {estado}")

    while True:
        entrada = input(f"Escolha a objetiva (1 a {len(nomes)}): ").strip()

        try:
            escolha = int(entrada)
        except ValueError:
            print("[Erro] Digite o NÚMERO da linha, não o nome da objetiva.\n")
            continue

        if not 1 <= escolha <= len(nomes):
            print(f"[Erro] Escolha entre 1 e {len(nomes)}.\n")
            continue

        nome = nomes[escolha - 1]
        faltam = campos_faltando(nome, exigidos)

        if faltam:
            print(
                f"\n[Erro] A objetiva {nome} não serve aqui: falta {', '.join(faltam)} "
                "no parametros.py.\n"
                "        fov               -> rode calibrar_fov.py\n"
                "        exposicao/wb_*    -> rode calibrar_camera.py\n"
                "        Escolha outra objetiva ou calibre esta antes.\n"
            )
            continue

        print(f"[Sucesso] Objetiva {nome} selecionada.\n")

        return nome, dict(OBJETIVAS[nome])

def medir_foco_relativo(imagem):
    """Mede o quanto a imagem está em foco de um jeito COMPARÁVEL entre
    campos diferentes -- que é justamente o que a medir_nitidez não faz.

    Devolve a razão entre a energia de duas faixas de escala da própria
    imagem: a fina (1 a 4 px), que o desfoque mata primeiro, e a grossa
    (8 a 32 px), que ele quase não toca. Como as duas crescem juntas com a
    quantidade de estrutura no campo e com o contraste, esses dois fatores
    se cancelam na divisão e sobra uma grandeza adimensional que responde
    a foco.

    É por isso que aqui cabe um limiar fixo e na medir_nitidez não cabe:
    um campo com detrito fino pode pontuar 100x mais nitidez que um campo
    com um flake grande e liso, ambos em foco perfeito -- mas os dois dão
    razão parecida.

    NÃO substitui a medir_nitidez: dentro do autofocar, onde se compara o
    MESMO campo em vários Z, a nitidez bruta é a métrica certa e mais
    sensível. Esta serve para decidir SE vale chamar o autofocar.

    Devolve NaN quando não há denominador (campo uniforme, saturado, tampa
    fechada). NaN é honesto onde um número seria invenção -- e tem a
    propriedade útil de que `nan < limiar` é False em Python, então um
    gatilho escrito assim simplesmente não dispara nesses casos."""

    cinza = imagem if imagem.ndim == 2 else cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    # float32 antes de qualquer subtração: em uint8, 1 - 255 daria 2 por
    # overflow, e a banda passa-banda viraria lixo.
    cinza = cinza.astype(np.float32)

    # Banda fina: precisa da resolução total. É exatamente o detalhe de 1 a
    # 4 px que o desfoque apaga -- reduzir a imagem aqui apagaria o sinal.
    # Kernels pequenos (7 e 25 px), então o custo já é baixo.
    fina = (
        cv2.GaussianBlur(cinza, (0, 0), SIGMA_FINA_MIN)
        - cv2.GaussianBlur(cinza, (0, 0), SIGMA_FINA_MAX)
    )

    # Banda grossa: medida na imagem reduzida (ver REDUCAO_BANDA_GROSSA).
    # INTER_AREA é a interpolação correta para DIMINUIR: ela tira a média
    # do bloco de pixels, o que já funciona como filtro anti-aliasing. Usar
    # INTER_LINEAR aqui deixaria alias entrar e sujar a medida.
    # Os sigmas são divididos pelo mesmo fator para representarem a mesma
    # escala FÍSICA na imagem menor.
    pequena = cv2.resize(
        cinza,
        None,
        fx=1.0 / REDUCAO_BANDA_GROSSA,
        fy=1.0 / REDUCAO_BANDA_GROSSA,
        interpolation=cv2.INTER_AREA,
    )
    grossa = (
        cv2.GaussianBlur(pequena, (0, 0), SIGMA_GROSSA_MIN / REDUCAO_BANDA_GROSSA)
        - cv2.GaussianBlur(pequena, (0, 0), SIGMA_GROSSA_MAX / REDUCAO_BANDA_GROSSA)
    )

    energia_grossa = float(grossa.std())

    # Guarda puramente aritmética contra divisão por zero -- não é um
    # julgamento sobre a amostra. Quem decide se o campo tem estrutura de
    # verdade continua sendo a medir_textura.
    if energia_grossa < 1e-6:
        return float("nan")

    return float(fina.std()) / energia_grossa

def _matriz_projeto(xs, ys, tipo, pesos=None):
    """
    Monta a matriz de projeto do ajuste de foco para as coordenadas
    dadas e devolve (matriz, x0, y0, escala).

    O `pesos` só muda ONDE o ajuste é centrado, não a matriz em si. Isso
    importa no plano: como ele ajusta dz contra dx e dy SEM coluna de 1s,
    o ajuste é obrigado a passar pelo ponto (x0, y0, z0). Em mínimos
    quadrados comuns essa imposição é inofensiva -- a reta de regressão
    passa pelo centroide de qualquer jeito --, mas no ajuste PONDERADO o
    centroide que a solução ótima respeita é o centroide ponderado. Centrar
    na média simples ali enviesaria o resultado na direção dos pontos
    velhos, que é justamente o que o envelhecimento quer evitar.
    """

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    x0 = float(np.average(xs, weights=pesos))
    y0 = float(np.average(ys, weights=pesos))

    dx = xs - x0
    dy = ys - y0

    if tipo != "quadratica":
        # O plano ajusta dz (também centrado) contra dx e dy, por isso não
        # há coluna de 1s aqui: o z0 já é a média dos Z.
        return np.column_stack([dx, dy]), x0, y0, 1.0

    escala = float(max(np.abs(dx).max(), np.abs(dy).max()))

    if escala == 0:
        raise ValueError(
            "ERRO: todos os pontos de calibração estão na mesma posição XY."
        )

    ux = dx / escala
    uy = dy / escala

    matriz = np.column_stack([ux, uy, ux * uy, ux**2, uy**2, np.ones(len(xs))])

    return matriz, x0, y0, escala


def _condicionamento(xs, ys, tipo):
    """Fator de amplificação do erro de foco para uma dada geometria de
    pontos. Depende só de X e Y -- é por isso que dá para avaliar a
    qualidade de um conjunto de pontos antes de focar qualquer um deles."""

    matriz, _, _, _ = _matriz_projeto(xs, ys, tipo)

    return float(np.linalg.cond(matriz))


def _aplicar_pesos(matriz, alvo, pesos):
    """Transforma um problema de mínimos quadrados PONDERADO num problema
    comum, que o lstsq resolve sem saber que há pesos envolvidos.

    O truque é de uma linha de álgebra: minimizar a soma dos resíduos
    ponderados

        S = soma_i  w_i * r_i²  =  soma_i  (raiz(w_i) * r_i)²

    é o mesmo que minimizar a soma dos resíduos comuns de um sistema em
    que cada equação (cada LINHA da matriz e o seu alvo) foi multiplicada
    por raiz(w_i). É a raiz, e não o peso, justamente porque o resíduo
    entra ao quadrado."""

    if pesos is None:
        return matriz, alvo

    raiz = np.sqrt(np.asarray(pesos, dtype=float))

    # raiz[:, None] transforma o vetor de N pesos numa coluna N x 1, e o
    # numpy o "difunde" (broadcasting) por todas as colunas da matriz --
    # cada LINHA multiplicada pelo seu próprio fator.
    return matriz * raiz[:, None], alvo * raiz


def calibrar_plano_foco(pontos, verboso=True, pesos=None):
    """Ajusta um plano de foco da forma:
        Z = Z0 + ax*(X - X0) + ay*(Y - Y0).

    verboso=False silencia o resumo impresso -- usado no re-ajuste
    automático durante a varredura, que roda dezenas de vezes e encheria
    o console de tabelas.

    pesos=None mantém o ajuste clássico, com todos os pontos valendo o
    mesmo; quem passa pesos é o reajustar_modelo_foco, para envelhecer os
    pontos antigos (ver MEIA_VIDA_PONTOS)."""

    pontos = np.array(pontos, dtype=float)

    if len(pontos) < 3:
        raise ValueError(
            "ERRO: São necessários pelo menos 3 pontos para ajustar o plano de foco."
        )

    xs = pontos[:, 0]
    ys = pontos[:, 1]
    zs = pontos[:, 2]

    matriz, x0, y0, _ = _matriz_projeto(xs, ys, "plano", pesos)

    # Centro em Z dos pontos de calibração. O ajuste é feito sobre dz,
    # centrado, e por isso a matriz do plano não tem coluna de 1s.
    z0 = float(np.average(zs, weights=pesos))
    dz = zs - z0

    matriz, dz = _aplicar_pesos(matriz, dz, pesos)

    # Pontos alinhados (ou quase) deixam o plano mal determinado. O número
    # de condição pega tanto o caso exato quanto o "quase", que é o que
    # acontece na prática -- um teste de posto só pegaria o exato.
    #
    # Medido na matriz JÁ ponderada, que é a que o lstsq vai resolver: um
    # conjunto de pontos bem espalhado no papel pode estar mal espalhado
    # na prática se quase todo o peso estiver concentrado numa vizinhança.
    condicionamento = np.linalg.cond(matriz)

    if not np.isfinite(condicionamento) or condicionamento > LIMITE_CONDICIONAMENTO:
        raise ValueError(
            "ERRO: Os pontos de calibração estão alinhados ou quase alinhados "
            f"(condicionamento {condicionamento:.1e}, limite {LIMITE_CONDICIONAMENTO:.0e}). "
            "Escolha pelo menos 3 pontos bem espalhados pela amostra, formando "
            "um triângulo largo em vez de uma linha."
        )

    coeficientes, _, _, _ = np.linalg.lstsq(matriz, dz, rcond=None)

    # float() explícito: o lstsq devolve np.float64, que o json.dump não
    # sabe serializar. O modelo ajustado vai inteiro para o varredura.json.
    ax = float(coeficientes[0])
    ay = float(coeficientes[1])

    plano = {
        "tipo": "plano",
        "x0": x0,
        "y0": y0,
        "z0": z0,
        # O plano não precisa de normalização (dx e dy já estão na mesma
        # ordem de grandeza), mas o campo existe para o calcular_z poder
        # tratar os dois modelos do mesmo jeito.
        "escala": 1.0,
        "ax": ax,
        "ay": ay,
    }

    if verboso:
        _imprimir_modelo_foco(plano, pontos)

    return plano


def calibrar_superficie_foco(pontos, verboso=True, pesos=None):
    """Ajusta uma superfície quadrática de foco da forma:
        Z = Z0 + ax*dx + ay*dy + axy*dx*dy + axx*dx² + ayy*dy²
    onde dx = X - X0, dy = Y - Y0.

    verboso=False silencia o resumo impresso (ver calibrar_plano_foco).
    pesos=None mantém o ajuste clássico (ver calibrar_plano_foco)."""

    pontos = np.array(pontos, dtype=float)

    if len(pontos) < 6:
        raise ValueError(
            "ERRO: A superfície quadrática tem 6 coeficientes e precisa de pelo "
            "menos 6 pontos. Na prática, use mais: com exatamente 6 o ajuste "
            "passa por todos eles e reproduz o erro de foco de cada um, sem "
            "nenhuma média para amortecer."
        )

    xs = pontos[:, 0]
    ys = pontos[:, 1]
    zs = pontos[:, 2]

    matriz_teste, x0, y0, escala = _matriz_projeto(xs, ys, "quadratica", pesos)

    # A quadrática tem coluna de 1s, então o alvo do ajuste é o Z absoluto
    # (o z0 sai como coeficiente) -- e não o dz centrado, como no plano.
    matriz_teste, alvo = _aplicar_pesos(matriz_teste, zs, pesos)

    condicionamento = np.linalg.cond(matriz_teste)

    if not np.isfinite(condicionamento) or condicionamento > LIMITE_CONDICIONAMENTO:
        raise ValueError(
            "ERRO: Os pontos de calibração estão mal distribuídos para a "
            f"superfície quadrática (condicionamento {condicionamento:.1e}, "
            f"limite {LIMITE_CONDICIONAMENTO:.0e}).\n"
            "Duas causas possíveis: (a) faltam níveis -- a quadrática precisa "
            "de pelo menos 3 valores DISTINTOS de X e 3 de Y entre os pontos; "
            "(b) a área marcada é alongada demais para a quadrática, e nesse "
            "caso a saída é usar o plano."
        )

    coeficientes, _, _, _ = np.linalg.lstsq(matriz_teste, alvo, rcond=None)

    # Ver o comentário equivalente na calibrar_plano_foco.
    ax, ay, axy, axx, ayy, z0 = (float(c) for c in coeficientes)

    superficie = {
        "tipo": "quadratica",
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "escala": escala,
        "ax": ax,
        "ay": ay,
        "axy": axy,
        "axx": axx,
        "ayy": ayy,
    }

    if verboso:
        _imprimir_modelo_foco(superficie, pontos)

    return superficie


def pesos_por_idade(quantidade, qtd_ancoras, segurar_ancoras=False):
    """Peso de cada ponto de calibração em função de quão VELHO ele é.

    Os pontos chegam em ordem cronológica -- primeiro as âncoras coletadas
    à mão na Fase 2, depois um por checkpoint de autofoco --, então a
    posição na lista já é o relógio: não é preciso guardar timestamp.

    O peso cai pela metade a cada MEIA_VIDA_PONTOS pontos de idade.

    `segurar_ancoras` põe um piso de PESO_MINIMO_ANCORA nas primeiras
    qtd_ancoras posições. Ele NÃO é o padrão, e a razão é sutil: as âncoras
    seguram duas coisas ao mesmo tempo, a GEOMETRIA (o quanto a amostra é
    inclinada) e a ALTURA (onde ela está em Z). Segurar a geometria é o que
    se quer; segurar a altura é o contrário -- sob deriva térmica as
    âncoras dizem para sempre "a amostra está onde estava no começo", e o
    modelo fica preso ao passado por mais que o autofoco meça. Por isso o
    piso entra como PLANO B: só quando o ajuste sem ele não fecha por falta
    de espalhamento (ver reajustar_modelo_foco).

    Devolve None quando ainda não há nada a envelhecer -- e None é
    exatamente o que as funções de calibração entendem como 'sem pesos',
    o que preserva o comportamento clássico enquanto só existem âncoras.
    """

    if quantidade <= qtd_ancoras:
        return None

    idades = np.arange(quantidade - 1, -1, -1, dtype=float)
    pesos = 0.5 ** (idades / MEIA_VIDA_PONTOS)

    if segurar_ancoras:
        pesos[:qtd_ancoras] = np.maximum(pesos[:qtd_ancoras], PESO_MINIMO_ANCORA)

    return pesos


def residuos_modelo(pontos, modelo_foco):
    """Quanto cada ponto de calibração ficou ACIMA ou abaixo do modelo
    ajustado, em µm. É a matéria-prima de duas coisas: da escala de erro
    que decide se uma correção nova é espúria (ver checkpoint_autofoco) e
    do diagnóstico gravado no autofoco.csv."""

    return np.array(
        [z - calcular_z(x, y, modelo_foco) for x, y, z in pontos], dtype=float
    )


def desvio_robusto(valores):
    """Desvio típico de um conjunto, estimado pelo MAD (mediana dos
    desvios absolutos em torno da mediana) em vez do desvio padrão.

    Por que MAD aqui: o conjunto em questão são os resíduos do modelo de
    foco, e a pergunta que se faz a eles é 'esta medida nova é um
    outlier?'. Um desvio padrão é ele mesmo arrastado por outliers -- um
    único ponto ruim de 30 µm infla o sigma e passa a deixar entrar outros
    pontos ruins. A mediana não se move por isso.

    O fator 1,4826 converte MAD em desvio padrão equivalente para dados
    gaussianos, para que um critério escrito como '3 desvios' signifique
    a mesma coisa de sempre."""

    valores = np.asarray(valores, dtype=float)

    if valores.size == 0:
        return 0.0

    mad = float(np.median(np.abs(valores - np.median(valores))))

    return 1.4826 * mad


def reajustar_modelo_foco(pontos, tipo, qtd_ancoras=None):
    """
    Reajusta o modelo de foco com todos os pontos acumulados até agora:
    os coletados à mão no início mais os medidos pelo autofoco durante a
    varredura.

    É isso que faz o modelo melhorar sozinho ao longo da varredura.

    `qtd_ancoras` liga o envelhecimento dos pontos: informe quantos dos
    primeiros pontos da lista vieram da coleta manual e os demais passam a
    valer menos conforme envelhecem (ver pesos_por_idade). Sem ele o
    ajuste é o clássico, com todos os pontos valendo o mesmo -- que é o
    que os chamadores fora da varredura (a previsão durante a coleta, o
    deteccao.py) continuam querendo.
    """

    def ajustar(pesos):
        try:
            if tipo == "quadratica":
                return calibrar_superficie_foco(pontos, verboso=False, pesos=pesos)

            return calibrar_plano_foco(pontos, verboso=False, pesos=pesos)

        except (ValueError, np.linalg.LinAlgError):
            return None

    if qtd_ancoras is None:
        return ajustar(None)

    # Primeira tentativa: todo mundo envelhece, inclusive as âncoras. É o
    # ajuste que segue a deriva, porque nada o prende à altura inicial.
    modelo = ajustar(pesos_por_idade(len(pontos), qtd_ancoras))

    if modelo is not None:
        return modelo

    # Não fechou: os pontos com peso relevante estão concentrados demais
    # para determinar a superfície -- típico no começo da varredura, ou
    # quando a serpentina passou um bom tempo na mesma região. Aí as
    # âncoras voltam a valer, porque um modelo preso ao passado ainda é
    # melhor que nenhum modelo.
    return ajustar(
        pesos_por_idade(len(pontos), qtd_ancoras, segurar_ancoras=True)
    )


def calcular_z(x, y, modelo_foco):
    escala = modelo_foco.get("escala", 1.0)

    dx = (x - modelo_foco["x0"]) / escala
    dy = (y - modelo_foco["y0"]) / escala

    z = modelo_foco["z0"] + modelo_foco["ax"] * dx + modelo_foco["ay"] * dy

    if modelo_foco["tipo"] == "quadratica":
        z += (
            modelo_foco["axy"] * dx * dy
            + modelo_foco["axx"] * dx**2
            + modelo_foco["ayy"] * dy**2
        )

    return z


def _imprimir_modelo_foco(modelo_foco, pontos):
    """Imprime o resumo dos coeficientes do modelo de foco ajustado e o
    teste nos próprios pontos de calibração (Z real x Z previsto x erro).
    Usado tanto pelo ajuste de plano quanto pelo de superfície."""

    if modelo_foco["tipo"] == "quadratica":
        print("\n--- SUPERFÍCIE DE FOCO AJUSTADA ---")
    else:
        print("\n--- PLANO DE FOCO AJUSTADO ---")

    escala = modelo_foco.get("escala", 1.0)

    print(f"x0 = {modelo_foco['x0']:.3f}")
    print(f"y0 = {modelo_foco['y0']:.3f}")
    print(f"z0 = {modelo_foco['z0']:.3f}")
    print(f"dZ/dX = {modelo_foco['ax'] / escala:.6f}")
    print(f"dZ/dY = {modelo_foco['ay'] / escala:.6f}")

    if modelo_foco["tipo"] == "quadratica":
        print(f"dZ/dXdY = {modelo_foco['axy'] / escala**2:.9f}")
        print(f"dZ/dX² = {modelo_foco['axx'] / escala**2:.9f}")
        print(f"dZ/dY² = {modelo_foco['ayy'] / escala**2:.9f}")

    print("\n--- TESTE NOS PONTOS DE CALIBRAÇÃO ---")
    for k, (x, y, z_real) in enumerate(pontos, start=1):
        z_previsto = calcular_z(x, y, modelo_foco)
        erro = z_previsto - z_real

        print(
            f"Ponto {k}: "
            f"Z real = {z_real:.3f}, "
            f"Z calculado = {z_previsto:.3f}, "
            f"erro = {erro:.3f}"
        )


def _melhor_grid(qtd, minimo_niveis):
    """
    Escolhe (nx, ny): o maior grid com nx*ny <= qtd em que ambos os
    eixos tenham pelo menos minimo_niveis níveis. Empates são resolvidos
    pelo arranjo mais próximo de quadrado.
    """

    melhor = None

    for nx in range(minimo_niveis, qtd + 1):
        for ny in range(minimo_niveis, qtd + 1):
            if nx * ny > qtd:
                continue

            chave = (nx * ny, -abs(nx - ny))

            if melhor is None or chave > melhor[0]:
                melhor = (chave, (nx, ny))

    if melhor is None:
        return None

    return melhor[1]


def _ordenar_serpentina(pontos):
    """Reordena os pontos em serpentina -- colunas de X, com o sentido de
    Y invertido a cada coluna -- para o estágio não atravessar a amostra
    de ponta a ponta entre um ponto e o seguinte."""

    colunas = {}

    for x, y in pontos:
        # O arredondamento serve só para agrupar: pontos gerados na mesma
        # coluna podem diferir no último bit por causa do linspace.
        colunas.setdefault(round(x, 6), []).append((x, y))

    ordenados = []

    for indice, chave in enumerate(sorted(colunas)):
        descendo = indice % 2 == 1
        ordenados.extend(sorted(colunas[chave], key=lambda ponto: ponto[1], reverse=descendo))

    return ordenados


def sugerir_pontos_calibracao(malha, qtd, tipo):
    """
    Sugere qtd coordenadas (X, Y) bem distribuídas pela área que a
    varredura vai de fato percorrer.

    A ideia é tirar do usuário a escolha da GEOMETRIA dos pontos a serem coletados

    Devolve um dicionário com os pontos já na ordem de visita, a descrição
    do arranjo, o condicionamento previsto e o espaçamento do grid -- este
    último serve para dizer ao usuário quanto ele pode se afastar de um
    alvo sem estragar o ajuste.
    """

    malha_x = malha["malha_x"]
    malha_y = malha["malha_y"]

    # Extremos da MALHA, e não o ponto inicial/final que o usuário marcou:
    # o modelo de foco precisa ser bom onde as imagens serão tiradas, e o
    # arredondamento para cima do número de passos faz a malha extrapolar
    # um pouco a área marcada. Usar min/max também dispensa saber em que
    # sentido a varredura corre.
    x_min, x_max = min(malha_x), max(malha_x)
    y_min, y_max = min(malha_y), max(malha_y)

    # Varredura de uma coluna (ou de uma linha) só: sem extensão naquele
    # eixo todos os pontos cairiam no mesmo X, e a matriz seria singular
    # por falta de níveis. Abre-se a faixa de um passo, que é o pedaço da
    # amostra que aquele tile de fato cobre.
    if x_max == x_min:
        x_min -= malha["passo_x"] / 2
        x_max += malha["passo_x"] / 2

    if y_max == y_min:
        y_min -= malha["passo_y"] / 2
        y_max += malha["passo_y"] / 2


    minimo_niveis = 3 if tipo == "quadratica" else 2

    grid = _melhor_grid(qtd, minimo_niveis)

    if grid is None:
        pontos = [
            (x_min, y_min),
            (x_max, y_min),
            ((x_min + x_max) / 2, y_max),
        ]
        descricao = "triângulo largo"
        espacamento = min(x_max - x_min, y_max - y_min) / 2

    else:
        nx, ny = grid

        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)

        pontos = [(float(x), float(y)) for x in xs for y in ys]
        descricao = f"grid {nx}x{ny}"

        faltam = qtd - nx * ny

        if faltam:
            # Os pontos que sobram vão para os centros das células: eles
            # acrescentam níveis intermediários em vez de mexer nos que já existem
            centros = [
                (float((xs[i] + xs[i + 1]) / 2), float((ys[j] + ys[j + 1]) / 2), i + j)
                for i in range(nx - 1)
                for j in range(ny - 1)
            ]

            # Ordem de tabuleiro: pegando primeiro as células de paridade
            # par, os poucos centros usados saem diagonalmente opostos em
            # vez de amontoados num canto.
            centros.sort(key=lambda centro: (centro[2] % 2, centro[2]))

            pontos += [(x, y) for x, y, _ in centros[:faltam]]
            descricao += f" + {faltam} centro(s) de célula"

        espacamento = min((x_max - x_min) / (nx - 1), (y_max - y_min) / (ny - 1))

    pontos = _ordenar_serpentina(pontos)

    return {
        "pontos": pontos,
        "descricao": descricao,
        "condicionamento": _condicionamento(
            [ponto[0] for ponto in pontos], [ponto[1] for ponto in pontos], tipo
        ),
        "espacamento": espacamento,
    }


def perguntar_qtd_pontos(minimo_pontos):
    """Pergunta quantos pontos de calibração usar, validando o mínimo."""

    while True:
        entrada = input(f"Quantos pontos serão utilizados para ajustar a varredura? (Mínimo {minimo_pontos}): ")

        try:
            qtd = int(entrada)
        except ValueError:
            print("[Erro] Entrada inválida! Por favor, digite apenas números inteiros. Exemplo: 3, 4, 5.\n")
            continue

        if qtd < minimo_pontos:
            print(
                f"[Aviso] Você digitou {qtd}. "
                f"É necessário pelo menos {minimo_pontos} pontos para calibrar o estágio. "
                "Tente novamente.\n"
            )
            continue

        print(f"[Sucesso] O sistema será calibrado utilizando {qtd} pontos.\n")

        return qtd


def coletar_pontos_calibracao(
    core, xy_stage, z_stage, minimo_pontos, malha=None, tipo="plano", qtd_pontos=None
):
    """
    Pede a quantidade de pontos de calibração e depois coleta cada um.

    Com malha=None mantém o modo manual: o usuário escolhe sozinho onde
    colocar cada ponto. Recebendo a malha, o script passa a SUGERIR as
    posições e a levar o estágio até elas em XY -- ao usuário resta só
    ajustar o foco.

    `qtd_pontos` pula a pergunta. Serve para a varredura multi-amostra, que
    pergunta uma vez só e aplica a todas: repetir a mesma pergunta cinco
    vezes seguidas não dá nenhuma informação nova e só cria oportunidade de
    errar a digitação.

    Retorna a lista de pontos (X, Y, Z) coletados.
    """

    if qtd_pontos is None:
        qtd_pontos_calibracao = perguntar_qtd_pontos(minimo_pontos)
    else:
        qtd_pontos_calibracao = qtd_pontos

    sugestao = None

    if malha is not None:
        sugestao = sugerir_pontos_calibracao(malha, qtd_pontos_calibracao, tipo)

        print(
            f"Arranjo sugerido: {sugestao['descricao']} -- condicionamento previsto "
            f"{sugestao['condicionamento']:.1f} (limite {LIMITE_CONDICIONAMENTO:.0e})."
        )

        if sugestao["condicionamento"] > LIMITE_CONDICIONAMENTO:
            # Nenhum rearranjo salva este caso: o condicionamento também
            # depende da razão de aspecto da ÁREA, e uma área muito
            # alongada estoura o limite mesmo com os pontos ideais.
            print(
                "[Aviso] Nem o melhor arranjo possível cabe no limite nesta área --\n"
                "        ela é alongada demais para o ajuste escolhido. O problema é a\n"
                "        forma da área, não a posição dos pontos, e mudar os pontos não\n"
                "        resolve. Considere usar o plano em vez da quadrática, ou varrer\n"
                "        uma região menos alongada."
            )
        elif sugestao["condicionamento"] > LIMITE_CONDICIONAMENTO / 100:
            print(
                "[Aviso] Condicionamento previsto alto (área alongada). Passa no limite,\n"
                "        mas o erro ao focar cada ponto será amplificado no Z previsto."
            )

        print(
            f"\nO estágio será levado até cada ponto sugerido. Se algum cair num pedaço\n"
            f"sem nada em que focar, mova livremente até uma região próxima que tenha\n"
            f"estrutura: o que vale é onde você parar. Desvios de até uns\n"
            f"{sugestao['espacamento']:.0f} µm não atrapalham o ajuste."
        )

    pontos_coletados = []

    for p in range(qtd_pontos_calibracao):
        alvo_x = alvo_y = None

        if sugestao is None:
            input(f"\nMova o estágio para o Ponto de Calibração {p + 1}, AJUSTE O FOCO, e pressione ENTER...")

        else:
            alvo_x, alvo_y = sugestao["pontos"][p]

            print(f"\n--- Ponto {p + 1} de {qtd_pontos_calibracao} ---")
            print(f"Levando o estágio para X={alvo_x:.2f}, Y={alvo_y:.2f}...")

            core.set_xy_position(xy_stage, alvo_x, alvo_y)
            core.wait_for_device(xy_stage)

            # Com 3 pontos já dá para ajustar um plano provisório e dizer
            # em que Z o foco provavelmente está. É só um número impresso:
            # quem move o Z continua sendo o usuário.
            if len(pontos_coletados) >= 3:
                modelo_previo = reajustar_modelo_foco(pontos_coletados, "plano")

                if modelo_previo is not None:
                    print(
                        f"Z estimado pelos {len(pontos_coletados)} pontos anteriores: "
                        f"{calcular_z(alvo_x, alvo_y, modelo_previo):.3f} "
                        f"(estimativa; o script não mexe no Z)"
                    )

            input("AJUSTE O FOCO e pressione ENTER...")

        x_atual = core.get_x_position(xy_stage)
        y_atual = core.get_y_position(xy_stage)
        z_atual = core.get_position(z_stage)

        pontos_coletados.append((x_atual, y_atual, z_atual))

        print(f"-> Ponto {p + 1} salvo: X={x_atual:.1f}, Y={y_atual:.1f}, Z={z_atual:.2f}")

        if alvo_x is not None:
            desvio = math.hypot(x_atual - alvo_x, y_atual - alvo_y)

            if desvio > sugestao["espacamento"]:
                print(
                    f"   [Aviso] desvio de {desvio:.0f} µm do alvo, maior que o espaçamento "
                    f"do arranjo ({sugestao['espacamento']:.0f} µm). Registrado assim mesmo; "
                    "a conferência abaixo dirá se a geometria ainda serve."
                )

    if sugestao is not None:
        # O condicionamento previsto valia para os alvos. Este é o dos
        # pontos que o usuário de fato focou -- é ele que vale, e vê-lo
        # aqui evita descobrir um problema só quando o ajuste recusar.
        condicionamento = _condicionamento(
            [ponto[0] for ponto in pontos_coletados],
            [ponto[1] for ponto in pontos_coletados],
            tipo,
        )

        print(
            f"\nCondicionamento dos pontos realmente coletados: {condicionamento:.1f} "
            f"(previsto: {sugestao['condicionamento']:.1f}, limite {LIMITE_CONDICIONAMENTO:.0e})."
        )

    return pontos_coletados


def _ordenar_cantos(cantos):
    """Põe os 4 cantos em ordem cíclica ao redor do centro deles.

    Necessário porque você marca os cantos na ordem que for mais cômoda
    para o estágio, e tanto a área quanto as arestas dependem de percorrer
    o contorno na ordem certa -- ligar canto 1 a canto 3 daria uma
    'gravata borboleta' com área errada."""

    cx = sum(x for x, _ in cantos) / len(cantos)
    cy = sum(y for _, y in cantos) / len(cantos)

    return sorted(cantos, key=lambda ponto: math.atan2(ponto[1] - cy, ponto[0] - cx))


def _area_poligono(pontos):
    """Área pelo laço do sapateiro (shoelace), com os pontos já ordenados."""

    area = 0.0

    for k in range(len(pontos)):
        x1, y1 = pontos[k]
        x2, y2 = pontos[(k + 1) % len(pontos)]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def medir_rotacao(cantos):
    """Inclinação média das arestas da amostra em relação aos eixos do
    estágio, em graus, no intervalo (-45, 45].

    Serve para você decidir se vale realinhar o chip no suporte antes de
    gastar tempo: quanto mais torta a amostra, mais área a caixa reta
    descarta.

    A média é CIRCULAR de período 90°, não aritmética. As quatro arestas de
    um retângulo apontam para direções separadas por múltiplos de 90°, então
    elas não podem ser somadas diretamente: arestas a 89° e a 1° descrevem
    quase a mesma inclinação, mas a média aritmética daria 45°. Multiplicar
    o ângulo por 4 antes de somar faz as quatro coincidirem (4 x 90° = uma
    volta inteira); dividir por 4 no fim desfaz a operação."""

    ordenados = _ordenar_cantos(cantos)

    angulos = []

    for k in range(4):
        x1, y1 = ordenados[k]
        x2, y2 = ordenados[(k + 1) % 4]
        angulos.append(math.atan2(y2 - y1, x2 - x1))

    vetor_medio = np.mean([np.exp(4j * angulo) for angulo in angulos])

    return math.degrees(np.angle(vetor_medio)) / 4


def caixa_contida(cantos):
    """Maior retângulo alinhado aos eixos que cabe DENTRO do quadrilátero
    marcado -- a área azul do desenho.

    A regra é: em X, do SEGUNDO MENOR ao SEGUNDO MAIOR valor entre os
    quatro cantos; idem em Y.

    Por que isso fica garantidamente dentro: numa amostra retangular
    girada, cada um dos quatro cantos é o extremo de um lado -- um é o mais
    à esquerda, outro o mais acima, e assim por diante. Descartar o extremo
    de cada eixo e parar no segundo coloca cada borda da caixa exatamente
    na altura de um canto, que é o ponto em que aquela aresta está mais
    "para dentro". Nenhuma aresta pode cruzar a caixa depois disso.

    Sem rotação nenhuma os quatro valores viram dois pares repetidos, o
    segundo menor coincide com o menor, e a caixa reproduz a área marcada
    inteira -- o caso girado degenera no caso reto, como tem que ser.

    Levanta ValueError se a caixa degenerar: a partir de 45° de rotação num
    quadrado, as bordas se cruzam e não sobra retângulo nenhum."""

    xs = sorted(x for x, _ in cantos)
    ys = sorted(y for _, y in cantos)

    x_min, x_max = xs[1], xs[2]
    y_min, y_max = ys[1], ys[2]

    largura = x_max - x_min
    altura = y_max - y_min

    if largura <= 0 or altura <= 0:
        raise ValueError(
            "ERRO: os quatro cantos não deixam nenhum retângulo reto por dentro "
            f"(sobrou {largura:.1f} x {altura:.1f} µm). A amostra está girada "
            "perto de 45°, ou os cantos foram marcados fora de ordem. Realinhe "
            "o chip no suporte e marque de novo."
        )

    ordenados = _ordenar_cantos(cantos)
    area_marcada = _area_poligono(ordenados)
    area_caixa = largura * altura

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "largura": largura,
        "altura": altura,
        "rotacao_graus": medir_rotacao(cantos),
        "area_marcada": area_marcada,
        "area_caixa": area_caixa,
        # Fração da amostra marcada que fica FORA da varredura. É o preço da
        # caixa contida: nada é fotografado fora do chip, em troca de perder
        # as beiradas.
        "perda_area": 1.0 - area_caixa / area_marcada if area_marcada > 0 else 1.0,
    }


def marcar_cantos(core, xy_stage, rotulo=""):
    """Coleta os 4 cantos de uma amostra, um input() por canto."""

    cantos = []

    print(f"\n{rotulo}Marque os 4 CANTOS da amostra (a ordem não importa).")

    for k in range(4):
        input(f"  {rotulo}canto {k + 1} de 4 -- mova o estágio até ele e pressione ENTER...")

        x = core.get_x_position(xy_stage)
        y = core.get_y_position(xy_stage)

        cantos.append((x, y))

        print(f"     registrado: X={x:.2f}, Y={y:.2f}")

    return cantos


def definir_area_amostra(core, xy_stage, rotulo=""):
    """Marca os 4 cantos, calcula a caixa contida e mostra o custo da
    rotação -- repetindo enquanto você quiser realinhar a amostra.

    Repetir aqui, e não depois, é de propósito: realinhar o chip no suporte
    move a amostra, o que invalidaria qualquer ponto de foco já coletado.
    Este é o último momento em que dá para mexer no suporte de graça."""

    while True:
        cantos = marcar_cantos(core, xy_stage, rotulo)

        try:
            caixa = caixa_contida(cantos)
        except ValueError as erro:
            print(f"\n{erro}\n")
            continue

        print(
            f"\n  Área marcada:  {caixa['area_marcada'] / 1e6:.3f} mm²"
            f"  |  inclinação: {caixa['rotacao_graus']:+.1f}°"
        )
        print(
            f"  Área a varrer: {caixa['largura']:.0f} x {caixa['altura']:.0f} µm"
            f"  ({caixa['area_caixa'] / 1e6:.3f} mm²)"
        )
        print(f"  Descartado pela caixa reta: {caixa['perda_area']:.0%} da área marcada")

        if (
            abs(caixa["rotacao_graus"]) > ROTACAO_AVISO_GRAUS
            or caixa["perda_area"] > PERDA_AVISO
        ):
            print(
                f"\n  [Aviso] A amostra está {abs(caixa['rotacao_graus']):.1f}° torta, e por isso\n"
                f"          {caixa['perda_area']:.0%} dela ficará de fora da varredura. Se der para\n"
                "          realinhar o chip no suporte agora, você recupera essa área --\n"
                "          depois de coletar os pontos de foco, não dá mais."
            )

        resposta = input("\n  [ENTER] aceita  |  [r] realinhei a amostra, marcar de novo: ").strip().lower()

        if resposta != "r":
            return cantos, caixa


def gerar_malha(x_inicial, y_inicial, x_final, y_final, fov_x, fov_y, sobreposicao=0.0, verboso=True):
    """Gera a malha de posições (X, Y) que cobre o retângulo dado.

    Função PURA: não pergunta nada e não fala com o estágio. Quem coleta as
    coordenadas é a gerar_malha_varredura (2 pontos) ou a
    definir_area_amostra (4 cantos) -- separar as duas coisas é o que
    permite testar a geometria da varredura sem microscópio nenhum.

    O espaçamento entre posições é o campo de visão MENOS a sobreposição.
    """

    direcao_x = 1 if x_final > x_inicial else -1
    direcao_y = 1 if y_final > y_inicial else -1

    dist_x = abs(x_final - x_inicial)
    dist_y = abs(y_final - y_inicial)

    if not 0.0 <= sobreposicao < 1.0:
        raise ValueError(
            f"ERRO: sobreposicao deve estar em [0, 1), recebido {sobreposicao}."
        )

    passo_x = fov_x * (1.0 - sobreposicao)
    passo_y = fov_y * (1.0 - sobreposicao)

    qtd_passos_x = max(1, math.ceil(dist_x / passo_x))
    qtd_passos_y = max(1, math.ceil(dist_y / passo_y))

    malha_x = [x_inicial + (i * passo_x * direcao_x) for i in range(qtd_passos_x)]
    malha_y = [y_inicial + (j * passo_y * direcao_y) for j in range(qtd_passos_y)]

    if sobreposicao > 0 and verboso:
        print(
            f"\nSobreposição de {sobreposicao:.0%}: passo de "
            f"{passo_x:.2f} x {passo_y:.2f} µm num campo de {fov_x:.2f} x {fov_y:.2f} µm "
            f"(faixa compartilhada de {fov_x - passo_x:.2f} x {fov_y - passo_y:.2f} µm)."
        )

    return {
        "malha_x": malha_x,
        "malha_y": malha_y,
        "x_inicial": x_inicial,
        "y_inicial": y_inicial,
        "dist_x": dist_x,
        "dist_y": dist_y,
        "qtd_passos_x": qtd_passos_x,
        "qtd_passos_y": qtd_passos_y,
        "passo_x": passo_x,
        "passo_y": passo_y,
        "sobreposicao": sobreposicao,
        "fov_x": fov_x,
        "fov_y": fov_y,
    }


def gerar_malha_varredura(core, xy_stage, fov_x, fov_y, sobreposicao=0.0):
    """
    Pede o ponto inicial e final da área a varrer e gera a malha de
    posições (X, Y) cobrindo essa área.

    Modo de 2 pontos, mantido para o deteccao.py. A varredura multi-amostra
    usa a definir_area_amostra (4 cantos) no lugar disto.
    """

    print("--- CONFIGURAÇÃO DE MAPEAMENTO ---")

    input("1. Mova o estágio para o PONTO INICIAL (ex: canto superior esquerdo da amostra) e pressione ENTER...")
    x_inicial = core.get_x_position(xy_stage)
    y_inicial = core.get_y_position(xy_stage)

    print(f"Ponto Inicial registrado: X={x_inicial:.2f}, Y={y_inicial:.2f}")

    input("2. Mova o estágio para o PONTO FINAL (ex: canto inferior direito da amostra) e pressione ENTER...")
    x_final = core.get_x_position(xy_stage)
    y_final = core.get_y_position(xy_stage)

    print(f"Ponto Final registrado: X={x_final:.2f}, Y={y_final:.2f}")

    return gerar_malha(
        x_inicial, y_inicial, x_final, y_final, fov_x, fov_y, sobreposicao
    )


def ordem_serpentina(qtd_x, qtd_y):
    """Gera os índices (i, j) de uma grade na ordem em que o estágio deve
    visitá-los: descendo uma coluna, subindo a seguinte.

    Por que não varrer em raster (toda coluna começando em j=0): ao terminar
    uma coluna, o estágio teria que voltar do fim dela até o começo da
    próxima -- um deslocamento do TAMANHO DA COLUNA, repetido uma vez por
    coluna. Isso custa tempo e, pior, custa foco: é o maior salto de Z
    previsto de toda a varredura, e era exatamente por isso que o código
    antigo gastava um checkpoint de autofoco no início de cada coluna.

    Na serpentina o passo entre duas posições consecutivas é sempre de um
    campo, em qualquer ponto do percurso. O foco varia pouco de uma para a
    outra e o checkpoint por coluna deixa de ter razão de existir.

    Devolve os índices, não as coordenadas -- assim quem chama continua
    nomeando os arquivos pelo j GEOMÉTRICO (a linha da grade), e não pela
    ordem de visita. Trocar um pelo outro montaria o mosaico com as colunas
    ímpares invertidas.
    """

    for i in range(qtd_x):
        indices_j = range(qtd_y) if i % 2 == 0 else range(qtd_y - 1, -1, -1)

        for j in indices_j:
            yield i, j


def salvar_metadados_varredura(diretorio, malha, fov_x, fov_y, extras=None):
    """
    Grava na pasta da varredura o que o montar_mosaico.py precisa saber
    para remontar a amostra na escala certa.

    O `extras` acrescenta o que o montador não usa mas VOCÊ vai precisar
    depois: qual objetiva, com que exposição, onde no estágio, quais cantos
    foram marcados. Sem isso, uma pasta de tiles é só um monte de .tif sem
    procedência -- e com cinco amostras por rodada, descobrir qual era qual
    três semanas depois vira adivinhação.
    """

    metadados = {
        "fov_x": fov_x,
        "fov_y": fov_y,
        "passo_x": malha["passo_x"],
        "passo_y": malha["passo_y"],
        "sobreposicao": malha["sobreposicao"],
        "grid_x": malha["qtd_passos_x"],
        "grid_y": malha["qtd_passos_y"],
        # Origem absoluta da grade: a posição do tile (i=0, j=0). É o que
        # converte um índice de tile em coordenada de estágio -- e o que
        # permite voltar num flake interessante semanas depois.
        "x0": malha["malha_x"][0],
        "y0": malha["malha_y"][0],
    }

    if extras:
        metadados.update(extras)

    caminho = os.path.join(diretorio, "varredura.json")

    with open(caminho, "w", encoding="utf-8") as arquivo:
        # default=float é a rede de segurança contra escalares do numpy
        # (np.float64 e afins), que o json não serializa sozinho.
        json.dump(metadados, arquivo, indent=2, ensure_ascii=False, default=float)

    return caminho


def medir_nitidez(imagem):
    """Mede o quão nítida (em foco) uma imagem está, usando a variância do
    Laplaciano -- quanto maior o valor, mais nítida a imagem está.
    É a métrica clássica de autofoco por imagem em microscopia.

    ATENÇÃO: o valor só é comparável entre imagens do MESMO campo de visão.
    Ele depende fortemente do conteúdo: um campo vazio em foco perfeito
    pontua parecido com um campo cheio de flakes fora de foco, porque em
    campo vazio o Laplaciano mede o ruído do sensor, não o foco. Por isso
    existe a medir_textura -- para decidir se dá pra medir foco aqui."""

    cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) if imagem.ndim == 3 else imagem
    return cv2.Laplacian(cinza, cv2.CV_64F).var()


def medir_textura(imagem):
    """Mede quanta estrutura REAL existe na imagem -- serve para decidir se
    faz sentido rodar autofoco neste campo.

    Ao contrário da medir_nitidez, esse valor quase não muda com o foco:
    um campo com flakes continua pontuando alto mesmo bem desfocado. É
    exatamente o que se quer num portão de decisão -- ele não pode
    bloquear o autofoco justo quando a imagem está fora de foco."""

    cinza = imagem if imagem.ndim == 2 else cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    cinza = cinza.astype(np.float32)

    suave = cv2.GaussianBlur(cinza, (0, 0), 2.0)
    fundo = cv2.GaussianBlur(cinza, (0, 0), 32.0)

    return float((suave - fundo).std())


def buscar_foco(core, z_stage, camera, z_estimado, faixa_um, passo_um,
                faixa_nominal_um=None):
    """Procura o Z de melhor foco em torno de z_estimado (o Z que o modelo
    previu), varrendo no máximo +-faixa_um em passos de passo_um.

    `faixa_nominal_um` é a faixa CONFIGURADA para a objetiva, que pode ser
    menor que a faixa realmente percorrida (o autofoco de abertura e a busca
    de recuperação alargam a segunda sem tocar na primeira). Ela existe por
    causa do teste de pico chato -- ver a queda_esperada, mais abaixo. Sem
    ela a função se comporta como antes, usando a própria faixa percorrida.

    COMO A BUSCA ANDA
    -----------------
    Ela CRESCE DO CENTRO PARA FORA, em vez de varrer a faixa inteira:

        1. mede o centro e os dois vizinhos imediatos (3 fotos);
        2. se o maior dos três é o do meio, o pico está cercado -- para;
        3. se o maior está numa ponta, mede um passo além DAQUELE lado e
           repete;
        4. se a faixa acabar sem o pico ficar cercado, a busca falhou.

    O motivo é econômico. A curva de nitidez em função de Z tem um máximo
    só (é unimodal), então varrer os 21 pontos da faixa para depois pegar o
    argmax é gastar 18 fotos confirmando o que as 3 primeiras já diziam --
    e isso acontecia em TODO checkpoint, inclusive nos (muitos) em que o
    modelo estava certo e o foco já estava no centro. Aqui o custo passa a
    ser proporcional ao ERRO do modelo: 3 fotos quando ele acerta,
    degradando até o custo antigo quando ele erra feio.

    O critério de descarte é o mesmo de antes, só que dito de outro jeito:
    "o melhor Z caiu na ponta da faixa" virou "a faixa acabou antes de o
    pico ficar cercado". Idêntico em consequência -- não se sabe onde está
    o foco, então não se usa a medida.

    O REFINO SUB-PASSO
    ------------------
    Com o pico cercado, há três pontos (S-, S0, S+) igualmente espaçados em
    torno do máximo. Ajustando uma PARÁBOLA a eles e tomando o vértice,

        delta = (S- - S+) / (2 * (S- - 2*S0 + S+))     [em passos]

    o Z de melhor foco sai com precisão MELHOR que o passo da busca -- o
    pico verdadeiro quase nunca cai exatamente em cima de uma amostra. Isso
    tem duas consequências: a imagem final é registrada mais perto do foco
    real, e sobretudo o ponto que entra no modelo de calibração vem com
    menos ruído (e esse ruído se acumula, porque o modelo é re-ajustado com
    todos os pontos medidos).

    É também o que permitiria AUMENTAR o passo: hoje ele está em faixa/10
    em todas as objetivas, bem mais fino que a profundidade de campo. Com o
    vértice recuperando a precisão, um passo o dobro disso custaria menos
    fotos e daria um resultado igual ou melhor.

    Devolve um dicionário com o Z escolhido, a imagem lá, se a medida é
    confiável e o diagnóstico (quantas fotos custou, que queda de nitidez
    se observou). Ver `autofocar` para a forma antiga e enxuta."""

    passo_um = abs(float(passo_um))
    faixa_um = abs(float(faixa_um))

    # A faixa de onde a DOF é deduzida -- que NÃO é necessariamente a faixa
    # percorrida. Ver a queda_esperada.
    faixa_referencia = abs(float(faixa_nominal_um)) if faixa_nominal_um else faixa_um

    # Quantos passos inteiros cabem na faixa, para cada lado. O epsilon
    # cobre a representação binária: 71.25/7.125 pode sair 9.999999 e o
    # int() truncaria para 9, encolhendo a faixa sem ninguém pedir.
    k_maximo = max(1, int(faixa_um / passo_um + 1e-9))

    nitidez_por_k = {}
    imagem_por_k = {}

    def z_de(k):
        return z_estimado + k * passo_um

    def medir(k):
        core.set_position(z_stage, z_de(k))
        core.wait_for_device(z_stage)

        imagem = capturar_imagem(core, camera)

        nitidez_por_k[k] = medir_nitidez(imagem)
        imagem_por_k[k] = imagem

    for k in (0, -1, 1):
        if abs(k) <= k_maximo:
            medir(k)

    while True:
        k_pico = max(nitidez_por_k, key=nitidez_por_k.get)

        k_mais_baixo = min(nitidez_por_k)
        k_mais_alto = max(nitidez_por_k)

        # Cercado dos dois lados: é um máximo de verdade dentro do que foi
        # medido, e não apenas o melhor de uma ponta ainda inexplorada.
        if k_pico - 1 in nitidez_por_k and k_pico + 1 in nitidez_por_k:
            break

        if k_pico == k_mais_alto and k_mais_alto < k_maximo:
            medir(k_mais_alto + 1)
        elif k_pico == k_mais_baixo and k_mais_baixo > -k_maximo:
            medir(k_mais_baixo - 1)
        else:
            break

    k_pico = max(nitidez_por_k, key=nitidez_por_k.get)
    s_pico = nitidez_por_k[k_pico]

    cercado = k_pico - 1 in nitidez_por_k and k_pico + 1 in nitidez_por_k

    # A queda que se ESPERA ver a um passo do máximo, deduzida da relação
    # entre faixa, passo e profundidade de campo -- ver FRACAO_QUEDA_MINIMA
    # para a conta. Serve de régua: comparar a queda observada com um
    # número absoluto seria comparar com a objetiva errada.
    #
    # Repare que quem entra na conta é a faixa NOMINAL, e não a percorrida.
    # A fórmula deduz a DOF da faixa (DOF ~ faixa/2,5), e essa dedução só
    # vale para a faixa configurada na objetiva. Usar a faixa percorrida
    # fazia a busca alargada concluir que a DOF tinha triplicado junto --
    # e a DOF é uma propriedade óptica, não muda porque se resolveu
    # procurar mais longe. Como a queda esperada cai com o QUADRADO da
    # faixa, alargar 3x afrouxava o teste em 9x, justamente no checkpoint
    # de abertura, que é o mais importante e o menos verificável.
    #
    # Foi assim que a varredura de 17/08 aceitou ruído puro como medida boa
    # no primeiro checkpoint de três amostras: quedas de 0,0030 a 0,0068
    # passaram por um limiar de 0,00092 que, na faixa nominal, seria
    # 0,00825 -- e as três teriam sido corretamente descartadas.
    queda_esperada = 2.5 * (passo_um / faixa_referencia) ** 2

    resultado = {
        "z": z_de(k_pico),
        "imagem": imagem_por_k[k_pico],
        "confiavel": False,
        "motivo": "borda",
        "fotos": len(nitidez_por_k),
        "nitidez": s_pico,
        "queda": float("nan"),
        "queda_esperada": queda_esperada,
        "delta_passos": 0.0,
    }

    if s_pico <= 0:
        # Nitidez zero em toda a busca: imagem uniforme, tampa fechada,
        # sensor saturado. Não é "o foco está fora da faixa" -- é "não há
        # sinal nenhum aqui", que é o mesmo caso do pico chato. A distinção
        # importa porque só a borda aciona a busca em faixa alargada, e
        # alargar a busca de uma imagem sem sinal não acharia nada.
        resultado["motivo"] = "pico_chato"

        core.set_position(z_stage, resultado["z"])
        core.wait_for_device(z_stage)

        return resultado

    if not cercado:
        core.set_position(z_stage, resultado["z"])
        core.wait_for_device(z_stage)

        return resultado

    s_menos = nitidez_por_k[k_pico - 1]
    s_mais = nitidez_por_k[k_pico + 1]

    # Segunda diferença dos três pontos: a curvatura da parábola que passa
    # por eles, a menos de um fator 2. Negativa num máximo.
    curvatura = s_menos - 2.0 * s_pico + s_mais

    # Convertida em "queda equivalente a um passo do pico", que é a
    # grandeza que queda_esperada mede -- para uma parábola de queda q a um
    # passo, a segunda diferença vale 2q vezes a altura do pico.
    #
    # Por que a curvatura e não a queda observada no vizinho mais alto: se
    # o pico verdadeiro cai a MEIO CAMINHO entre duas amostras -- o que
    # acontece o tempo todo, não é caso raro --, os dois vizinhos do argmax
    # empatam e a queda observada é zero. Um teste feito assim recusaria
    # justamente as medidas bem centradas. A segunda diferença de uma
    # parábola não depende de onde ela foi amostrada, então vale o mesmo
    # com o pico em cima de uma amostra ou entre duas.
    queda = -curvatura / (2.0 * s_pico)
    resultado["queda"] = queda

    if queda < FRACAO_QUEDA_MINIMA * queda_esperada:
        # Curva chata: o campo tem textura (o portão deixou passar) mas a
        # nitidez mal responde ao Z. Pode ser estrutura em foco num outro
        # plano, sujeira na ótica, iluminação em gradiente. O argmax daqui
        # é ruído, e ruído não pode virar ponto de calibração.
        #
        # Uma curvatura positiva (mínimo, não máximo) também cai aqui: ela
        # produz queda negativa, que nunca alcança o limiar.
        resultado["motivo"] = "pico_chato"

        core.set_position(z_stage, resultado["z"])
        core.wait_for_device(z_stage)

        return resultado

    # Passou no teste acima => a curvatura é negativa, e o vértice existe.
    delta = 0.5 * (s_menos - s_mais) / curvatura

    # O vértice de uma parábola cujo maior valor amostrado é o do meio cai,
    # matematicamente, dentro de meio passo. Sair disso é sinal de ruído
    # dominando as contas -- o clamp evita que isso vire um salto de Z
    # inventado.
    delta = float(np.clip(delta, -0.5, 0.5))

    resultado["delta_passos"] = delta
    resultado["confiavel"] = True
    resultado["motivo"] = ""
    resultado["z"] = z_de(k_pico + delta)

    core.set_position(z_stage, resultado["z"])
    core.wait_for_device(z_stage)

    # A foto extra no vértice só se paga quando o deslocamento é
    # apreciável; para um vértice quase em cima da amostra central, a
    # imagem que já está na mão é indistinguível da que seria tirada.
    if abs(delta) >= DESLOCAMENTO_MINIMO_REFINO:
        resultado["imagem"] = capturar_imagem(core, camera)
        resultado["fotos"] += 1

    return resultado


def varrer_faixa_foco(core, z_stage, camera, z_central, faixa_um, passo_um):
    """Varre a faixa INTEIRA de Z, mede a nitidez em cada posição e fica com
    o máximo. É a busca grossa da re-ancoragem.

    POR QUE NÃO USAR A buscar_foco AQUI
    -----------------------------------
    A buscar_foco cresce do centro para fora e escolhe para que lado andar
    olhando qual dos vizinhos mediu mais. Isso embute uma premissa: a de que
    o centro já está na ENCOSTA do pico -- que existe um gradiente apontando
    o caminho.

    Quando o Z de partida está a centenas de µm do foco, a premissa não
    vale. Ali a nitidez está no piso de ruído do sensor, o platô não tem
    encosta nenhuma e o "maior dos três" é sorteio. A busca sai andando para
    um lado arbitrário e, se errar o lado, atravessa a faixa inteira, bate
    na borda e desiste -- sem nunca ter olhado para o outro lado.

    Varrer tudo custa mais fotos. É o preço de procurar onde não se sabe nem
    a direção, e é por isso que esta função só aparece na re-ancoragem: uma
    vez por amostra, contra as centenas de checkpoints da varredura.

    COMO O PICO É VALIDADO
    ----------------------
    Duas condições, e as duas dizem a mesma coisa por ângulos diferentes:

      proeminência -> o máximo tem de se destacar do platô, medido em
          desvios robustos do próprio conjunto (ver
          PROEMINENCIA_MINIMA_GROSSA). Sem isso, uma faixa inteira de imagem
          borrada devolveria o maior ruído dela como se fosse foco.
      não na borda -> um máximo na ponta da faixa varrida quer dizer que o
          foco pode estar mais além. A medida não é o pico, é o fim do que
          se olhou.

    Devolve um dicionário com o Z do máximo, a imagem lá, se a medida foi
    aceita e o diagnóstico. Ao contrário da buscar_foco, não há refino
    sub-passo: o passo aqui é grosso de propósito, e interpolar um vértice
    entre amostras tão espaçadas daria uma precisão que o dado não tem. Quem
    refina é a busca fina que roda depois, em cima deste resultado.
    """

    passo_um = abs(float(passo_um))
    faixa_um = abs(float(faixa_um))

    k_maximo = max(1, int(faixa_um / passo_um + 1e-9))

    # Só a MELHOR imagem é guardada, e não todas: uma faixa de ±300 µm com
    # passo de 2 µm são 301 posições, e 301 imagens de 536x536x3 seriam
    # ~260 MB parados na memória sem nenhum uso.
    nitidezes = []
    melhor = {"nitidez": -1.0, "z": z_central, "imagem": None, "indice": 0}

    for indice, k in enumerate(range(-k_maximo, k_maximo + 1)):
        z = z_central + k * passo_um

        core.set_position(z_stage, z)
        core.wait_for_device(z_stage)

        imagem = capturar_imagem(core, camera)
        nitidez = medir_nitidez(imagem)

        nitidezes.append(nitidez)

        if nitidez > melhor["nitidez"]:
            melhor = {"nitidez": nitidez, "z": z, "imagem": imagem, "indice": indice}

    valores = np.array(nitidezes, dtype=float)

    # A mediana e o MAD como régua, e não a média e o desvio padrão: a maior
    # parte da faixa varrida ESTÁ fora de foco, por construção. O pico é o
    # outlier que se procura -- ele não pode ser o mesmo que define a régua
    # contra a qual será julgado.
    piso = float(np.median(valores))
    espalhamento = desvio_robusto(valores)

    # O piso do denominador cobre o caso degenerado de um platô perfeitamente
    # constante, onde o MAD é zero e qualquer flutuação teria proeminência
    # infinita.
    proeminencia = (melhor["nitidez"] - piso) / max(espalhamento, 1e-9)

    resultado = {
        "z": melhor["z"],
        "imagem": melhor["imagem"],
        "aceito": False,
        "motivo": "",
        "fotos": len(valores),
        "nitidez": float(melhor["nitidez"]),
        "piso": piso,
        "proeminencia": float(proeminencia),
    }

    # A ordem das duas perguntas importa para o DIAGNÓSTICO, não para o
    # veredito: "existe um pico?" vem antes de "onde ele está?". Num campo
    # sem estrutura nenhuma o argmax do ruído cai em qualquer lugar,
    # inclusive na ponta -- e responder "borda" ali mandaria você alargar a
    # faixa, que é o conselho errado para um problema que não é de alcance.
    if proeminencia < PROEMINENCIA_MINIMA_GROSSA:
        resultado["motivo"] = "sem_pico"
    elif melhor["indice"] in (0, len(valores) - 1):
        resultado["motivo"] = "borda"
    else:
        resultado["aceito"] = True

    core.set_position(z_stage, resultado["z"])
    core.wait_for_device(z_stage)

    return resultado


def autofocar(core, z_stage, camera, z_estimado, faixa_um, passo_um):
    """Forma enxuta da busca de foco: devolve só
    (melhor_z, melhor_imagem, medicao_confiavel).

    É uma casca fina sobre a buscar_foco, mantida porque o deteccao.py
    ainda a chama assim. Código novo deve usar a buscar_foco direto, que
    devolve também o diagnóstico (quantas fotos custou, por que a medida
    foi ou não aceita)."""

    resultado = buscar_foco(core, z_stage, camera, z_estimado, faixa_um, passo_um)

    return resultado["z"], resultado["imagem"], resultado["confiavel"]


def checkpoint_autofoco(
    core,
    z_stage,
    camera,
    imagem,
    x,
    y,
    z_previsto,
    modelo_foco,
    pontos_foco,
    ajuste_z,
    config_autofoco,
    fator_faixa=1.0,
    qtd_ancoras=None,
    estado=None,
    abertura=False,
):
    """Um checkpoint de foco completo: mede a textura, decide se vale rodar
    o autofoco, roda, valida o resultado e devolve o modelo já re-ajustado.

    Concentra numa função só a lógica que estava copiada (e já divergida)
    entre o deteccao.py e o varredura_mosaico.py. Esta é a versão boa: a
    correção medida vira PONTO DE CALIBRAÇÃO e re-ajusta o modelo, em vez
    de virar um offset escalar. A diferença importa porque o offset corrige
    a altura mas não a INCLINAÇÃO -- e é a inclinação que faz o foco
    escapar ao longo de uma coluna.

    São cinco desfechos possíveis, e nenhum deles é "usar o número que veio":

      pulado_sem_textura -> campo vazio. Ali a nitidez mede o ruído do
          sensor, e o autofoco devolveria um Z aleatório que entraria no
          modelo e estragaria TODAS as posições seguintes. Não medir é
          melhor que medir mal.
      descartado_na_borda -> o melhor Z caiu na ponta da faixa varrida, ou
          seja, o foco de verdade está fora do alcance da busca. O que se
          achou foi o melhor de um intervalo todo ruim: uma medição
          falhada, não um resultado.
      descartado_pico_chato -> o campo tinha textura, mas a nitidez mal
          respondeu ao Z. Sem pico não há foco a medir; o argmax de uma
          curva chata é ruído.
      descartado_outlier -> a correção pedida é grande demais para o que
          esta amostra vinha errando. O caso típico é o autofoco ter
          travado em algo que não está no plano da amostra -- um detrito
          alto, uma borda de flake, um risco. Aceitar isso deslocaria o
          modelo inteiro.
      modelo_reajustado / offset -> medição boa. Vira ponto e o modelo é
          refeito com tudo que se sabe até agora; se o re-ajuste não for
          possível (pontos ainda quase alinhados, no começo da varredura),
          cai no offset escalar como plano B.

    `fator_faixa` alarga a busca sem mexer na configuração da objetiva --
    usado no autofoco de abertura de cada amostra, onde a deriva acumulada
    desde a calibração pode ser bem maior que a de um checkpoint no meio
    da varredura.

    `abertura` marca o PRIMEIRO checkpoint de uma amostra, e o que ele faz é
    tirar o filtro de salto do caminho. Ver o filtro de medida espúria, mais
    abaixo, para o porquê: em resumo, esse filtro julga a correção nova
    contra a última ACEITA, e no primeiro checkpoint não existe última --
    o default de 0,0 faz a correção ser julgada contra ZERO, que é
    exatamente a armadilha que o FATOR_OUTLIER_FAIXA descreve.

    `qtd_ancoras` informa quantos dos primeiros pontos vieram da coleta
    manual, o que liga o envelhecimento no re-ajuste (ver pesos_por_idade).

    `estado` é um dicionário que o CHAMADOR cria uma vez por amostra e
    passa em todos os checkpoints dela. É a memória curta do autofoco: qual
    foi a última correção aceita, quantas medidas seguidas foram recusadas,
    quantas bateram na borda. Sem ela cada checkpoint decidiria no vácuo --
    e as duas travas contra a espiral de desfoque (ver FATOR_OUTLIER_FAIXA
    e FATOR_FAIXA_RECUPERACAO) dependem justamente de lembrar do anterior.
    Omitir o estado faz cada checkpoint ser independente, que é o que os
    chamadores de uma medida avulsa querem.

    Devolve (imagem, z_final, modelo_foco, pontos_foco, ajuste_z, registro).
    """

    if estado is None:
        estado = {}

    # Depois de um checkpoint ter batido na borda, o próximo procura numa
    # faixa alargada: é a única forma de reencontrar um foco que saiu do
    # alcance da busca.
    if estado.get("bordas_seguidas", 0):
        fator_faixa = fator_faixa * FATOR_FAIXA_RECUPERACAO

    faixa = config_autofoco["faixa_um"] * fator_faixa
    passo = config_autofoco["passo_um"]

    textura = medir_textura(imagem)

    registro = {
        "textura": textura,
        "z_medido": float("nan"),
        "correcao": float("nan"),
        "confiavel": "",
        "acao": "",
        "pontos_modelo": len(pontos_foco),
        "fotos": 0,
        "queda": float("nan"),
        "residuo_rms": float("nan"),
    }

    if textura < config_autofoco["textura_minima"]:
        registro["acao"] = "pulado_sem_textura"

        print(
            f"[Autofoco] pulado: campo sem textura "
            f"({textura:.2f} < {config_autofoco['textura_minima']})"
        )

        return imagem, z_previsto, modelo_foco, pontos_foco, ajuste_z, registro

    busca = buscar_foco(
        core, z_stage, camera, z_previsto, faixa, passo,
        faixa_nominal_um=config_autofoco["faixa_um"],
    )

    z_medido = busca["z"]

    registro["z_medido"] = z_medido
    registro["correcao"] = z_medido - z_previsto
    registro["confiavel"] = int(busca["confiavel"])
    registro["fotos"] = busca["fotos"]
    registro["queda"] = busca["queda"]

    # Espalhamento dos pontos que o modelo ATUAL já explica: é a régua
    # contra a qual a correção nova é julgada. Vai também para o CSV --
    # um RMS que cresce ao longo da varredura é o sintoma de que a
    # superfície real não é a que o modelo supõe.
    residuos = residuos_modelo(pontos_foco, modelo_foco)
    registro["residuo_rms"] = float(np.sqrt(np.mean(residuos**2)))

    if not busca["confiavel"]:
        registro["acao"] = (
            "descartado_pico_chato"
            if busca["motivo"] == "pico_chato"
            else "descartado_na_borda"
        )

        # Só a borda pede faixa alargada da próxima vez: ela diz "o foco
        # está fora do alcance". Um pico chato diz outra coisa -- que
        # naquele campo não há foco a medir --, e procurar mais longe não
        # resolveria nada, só gastaria fotos.
        if busca["motivo"] == "borda":
            estado["bordas_seguidas"] = estado.get("bordas_seguidas", 0) + 1

        core.set_position(z_stage, z_previsto)
        core.wait_for_device(z_stage)

        if busca["motivo"] == "pico_chato":
            print(
                f"[Autofoco] descartado: nitidez quase não respondeu ao Z "
                f"(queda de {busca['queda']:.1%} contra "
                f"{busca['queda_esperada']:.1%} esperados)."
            )
        else:
            print(
                f"[Autofoco] descartado: melhor Z na borda da faixa (+-{faixa:.1f} µm). "
                "Foco real fora do alcance."
            )

        return imagem, z_previsto, modelo_foco, pontos_foco, ajuste_z, registro

    # --- filtro de medida espúria ---------------------------------------
    # Comparado contra o MODELO, não contra o z_previsto: o z_previsto já
    # embute o offset escalar, e é do modelo que os resíduos falam. As duas
    # coisas só diferem quando o offset está ativo, mas quando difere, é
    # esta a grandeza comparável.
    correcao_vs_modelo = z_medido - calcular_z(x, y, modelo_foco)

    # O salto em relação à última correção aceita -- ver FATOR_OUTLIER_FAIXA
    # para por que não é a correção crua que se julga aqui.
    salto = correcao_vs_modelo - estado.get("correcao_anterior", 0.0)

    limite_outlier = max(
        FATOR_OUTLIER_FAIXA * faixa,
        FATOR_OUTLIER_MAD * desvio_robusto(residuos),
    )

    rejeicoes = estado.get("rejeicoes_seguidas", 0)

    # Há duas situações em que o filtro de salto tem de sair do caminho, e
    # elas são a mesma coisa dita de dois jeitos: não existe "última
    # correção aceita" que sirva de referência.
    #
    # EM RECUPERAÇÃO (o checkpoint anterior bateu na borda): o filtro existe
    # para impedir que uma medida espúria contamine um modelo que estava bom
    # -- e aqui o modelo NÃO está bom, é justamente o que se tenta
    # consertar. Mantê-lo ligado faria as duas proteções brigarem: a faixa
    # alargada encontra o foco distante e o filtro o recusa exatamente por
    # estar distante, que era o ponto.
    #
    # NA ABERTURA (primeiro checkpoint da amostra): o estado está vazio, a
    # correcao_anterior cai no default de 0,0, e o "salto" avaliado passa a
    # ser a correção CRUA contra zero. É textualmente a armadilha que o
    # comentário do FATOR_OUTLIER_FAIXA diz existir para evitar -- e o único
    # ponto do código onde ela acontecia por construção, no checkpoint que
    # mais precisa funcionar. Na varredura de 17/08 isso custou o começo da
    # amostra 1: a busca achou o foco a -16,98 µm, com pico cercado e queda
    # cinco vezes acima do limiar, e o filtro descartou a medida por ela
    # passar de 15,00 µm. O modelo levou umas sete colunas para se recuperar
    # sozinho, e essas colunas saíram borradas.
    #
    # O que valida a medida nos dois casos é o mesmo: o pico ter ficado
    # cercado e proeminente, que é o que a buscar_foco já atesta em
    # busca["confiavel"] -- e a esta altura ele já passou por lá.
    em_recuperacao = estado.get("bordas_seguidas", 0) > 0
    sem_referencia = em_recuperacao or abertura

    if not sem_referencia and abs(salto) > limite_outlier and rejeicoes < REJEICOES_ATE_ACEITAR:
        registro["acao"] = "descartado_outlier"
        estado["rejeicoes_seguidas"] = rejeicoes + 1

        core.set_position(z_stage, z_previsto)
        core.wait_for_device(z_stage)

        print(
            f"[Autofoco] descartado: salto de {salto:+.2f} µm em relação à última "
            f"correção aceita, acima do limite de {limite_outlier:.2f} µm. "
            "Provável medida em algo fora do plano da amostra."
        )

        return imagem, z_previsto, modelo_foco, pontos_foco, ajuste_z, registro

    if abs(salto) > limite_outlier:
        if em_recuperacao:
            print(
                f"[Autofoco] salto de {salto:+.2f} µm aceito: o checkpoint anterior "
                "não achou o foco na faixa normal, e este achou na alargada. "
                "É a recuperação funcionando."
            )
        elif abertura:
            print(
                f"[Autofoco] salto de {salto:+.2f} µm aceito: é o primeiro checkpoint "
                "da amostra, e não há correção anterior contra a qual julgá-lo. "
                "O que o valida é o pico ter ficado cercado e proeminente."
            )
        else:
            print(
                f"[Autofoco] salto de {salto:+.2f} µm aceito: é a "
                f"{rejeicoes + 1}ª medida seguida a insistir que o foco está longe do "
                "previsto. Mais provável que o modelo tenha se perdido do que que "
                "todas tenham caído em detrito."
            )

    estado["correcao_anterior"] = correcao_vs_modelo
    estado["rejeicoes_seguidas"] = 0
    estado["bordas_seguidas"] = 0

    pontos_foco = pontos_foco + [(x, y, z_medido)]
    modelo_novo = reajustar_modelo_foco(
        pontos_foco, modelo_foco["tipo"], qtd_ancoras=qtd_ancoras
    )

    if modelo_novo is not None:
        modelo_foco = modelo_novo
        # A correção agora mora no modelo; manter o offset também seria
        # contá-la duas vezes.
        ajuste_z = 0.0
        registro["acao"] = "modelo_reajustado"

        print(
            f"[Autofoco] Z medido {z_medido:.3f} ({busca['fotos']} fotos) | "
            f"textura {textura:.2f} | "
            f"modelo re-ajustado com {len(pontos_foco)} pontos"
        )

    else:
        limite = AJUSTE_Z_MAXIMO_EM_FAIXAS * faixa
        ajuste_z = z_medido - calcular_z(x, y, modelo_foco)
        ajuste_z = max(-limite, min(limite, ajuste_z))
        registro["acao"] = "offset"

        print(
            f"[Autofoco] Z medido {z_medido:.3f} ({busca['fotos']} fotos) | "
            f"textura {textura:.2f} | "
            f"modelo não re-ajustado, usando offset {ajuste_z:+.3f}"
        )

    registro["pontos_modelo"] = len(pontos_foco)

    return busca["imagem"], z_medido, modelo_foco, pontos_foco, ajuste_z, registro


def _medir_foco_ancora(
    core, z_stage, camera, z_partida, faixa_fina, passo_fino, faixa_ampla, passo_grosso
):
    """Mede o Z de foco numa âncora: varredura exaustiva da faixa ampla,
    seguida de uma busca fina para o valor preciso.

    POR QUE NÃO TENTAR A BUSCA EXPANSIVA PRIMEIRO
    ---------------------------------------------
    Seria tentador: quando o Z de partida já está perto do foco -- que é o
    caso esperado aqui, porque quem chama já somou o deslocamento medido na
    sondagem --, a expansiva resolve em três fotos contra as ~30 da
    exaustiva.

    Um ensaio simulado mostrou por que não vale. Com a superfície deslocada
    de 150 µm E reinclinada, as âncoras cujo foco caía ~16 µm fora do Z de
    partida receberam medidas que erravam por exatamente esses ~16 µm: a
    expansiva parou num máximo local do platô, perto de onde começou, e o
    teste de pico chato ACEITOU o resultado. Quatro das quinze âncoras
    saíram assim, e o modelo ficou com 17 µm de erro máximo.

    A diferença em relação ao mesmo risco durante a varredura é o que uma
    medida ruim custa. Lá, um checkpoint errado é corrigido pelo seguinte,
    vinte imagens depois. Aqui, a âncora errada entra no modelo que rege a
    amostra INTEIRA, e não há mais nada para corrigi-la. Trinta fotos a mais
    por âncora, umas poucas vezes por amostra, é preço barato por não
    precisar confiar num pico que pode não existir.

    Devolve (z, fotos, ok). Com ok False não há medida em que confiar e o
    z devolvido é None.
    """

    fotos = 0

    grossa = varrer_faixa_foco(
        core, z_stage, camera, z_partida, faixa_ampla, passo_grosso
    )
    fotos += grossa["fotos"]

    if not grossa["aceito"]:
        return None, fotos, False

    fina = buscar_foco(
        core, z_stage, camera, grossa["z"], faixa_fina, passo_fino,
        faixa_nominal_um=faixa_fina,
    )
    fotos += fina["fotos"]

    if not fina["confiavel"]:
        return None, fotos, False

    return fina["z"], fotos, True


def reancorar_amostra(
    core,
    xy_stage,
    z_stage,
    camera,
    pontos_foco,
    tipo_modelo,
    config_autofoco,
    faixa_grossa_um,
):
    """Re-mede a altura dos pontos de calibração de uma amostra logo antes
    de varrê-la, e devolve o modelo de foco corrigido.

    O PROBLEMA QUE ELA RESOLVE
    --------------------------
    Ver o bloco RE-ANCORAGEM no topo do arquivo. Em uma frase: os pontos são
    coletados na Fase 2, todas as amostras de uma vez, e a última pode ser
    varrida horas depois -- tempo em que o foco anda muito mais do que a
    faixa do autofoco alcança. É um problema que o autofoco durante a
    varredura NÃO tem como resolver, porque ele já nasce procurando no lugar
    errado.

    COMO ELA TRABALHA
    -----------------
        1. sondagem, num ponto central: busca GROSSA e larga (varredura
           exaustiva, ver varrer_faixa_foco) seguida de busca FINA. A grossa
           diz em que região está o foco; a fina dá o Z com precisão.
        2. conferência, em QTD_PONTOS_CONFERENCIA pontos distantes: busca
           normal, partindo do Z da âncora já somado ao deslocamento da
           sondagem. Se o deslocamento estiver certo, cada uma custa ~3
           fotos.
        3. decisão. Se os deslocamentos concordam, o desvio é tratado como
           um OFFSET uniforme e somado a todas as âncoras -- a geometria que
           a Fase 2 validou é preservada de graça. Se discordam, a forma
           mudou e todas as âncoras são re-medidas, o que agora é barato
           porque o deslocamento grosso já é conhecido.

    POR QUE SOMAR EM VEZ DE ACRESCENTAR PONTOS
    ------------------------------------------
    Durante a varredura, o que o autofoco mede vira ponto NOVO, porque está
    num (x, y) que ainda não tinha medida. Aqui é o contrário: o (x, y) é o
    mesmo da âncora, e o que se tem é uma medida melhor do MESMO ponto.
    Acrescentar criaria dois pontos contraditórios na mesma posição, e o
    mínimos quadrados tiraria a média entre o certo e o errado.

    Mesmo (x, y) atualiza. (x, y) diferente acrescenta.

    O PORTÃO DE TEXTURA NÃO ENTRA AQUI
    ----------------------------------
    Ele existe para não rodar autofoco em campo vazio, onde a nitidez mede o
    ruído do sensor. Numa âncora esse risco não existe: você focou nela à
    mão na Fase 2, então sabe que há estrutura ali. O portão só poderia dar
    falso negativo -- e daria justamente quando o desfoque é grande, que é
    quando a re-ancoragem mais precisa acontecer.

    Devolve (pontos, modelo_ou_None, registro). Com modelo None nada foi
    aplicado, os pontos devolvidos são os mesmos que entraram, e o chamador
    deve seguir com o modelo que já tinha -- pior que corrigido, melhor que
    corrigido errado.
    """

    pontos = list(pontos_foco)

    registro = {
        "ok": False,
        "motivo": "",
        "estrategia": "",
        "offset_sondagem": None,
        "offsets": [],
        "dispersao": None,
        "offset_aplicado": None,
        "pontos_remedidos": 0,
        "medidas_falhas": 0,
        "fotos": 0,
    }

    if len(pontos) < 3:
        registro["motivo"] = "pontos_insuficientes"
        return pontos, None, registro

    passo_fino = config_autofoco["passo_um"]
    faixa_fina = config_autofoco["faixa_um"]

    # O passo da grossa vem da objetiva, e não da faixa larga: quem manda no
    # quão fino ele precisa ser é a profundidade de campo (ver
    # FATOR_PASSO_GROSSO). O max() só impede o caso degenerado de uma
    # objetiva cuja faixa configurada seja tão estreita que 0,2 dela fique
    # abaixo do próprio passo fino.
    passo_grosso = max(passo_fino, FATOR_PASSO_GROSSO * faixa_fina)

    xs = np.array([ponto[0] for ponto in pontos], dtype=float)
    ys = np.array([ponto[1] for ponto in pontos], dtype=float)

    # Uma ORDEM DE PREFERÊNCIA dos pontos, e não uma escolha única: a
    # sondagem pode falhar num campo sem estrutura e precisa de um plano B,
    # e os pontos de conferência saem da mesma lista.
    #
    # O primeiro é o mais CENTRAL -- é dele que sai o deslocamento aplicado
    # a todos os outros, e o centro é onde um erro de extrapolação (caso o
    # desvio não seja bem uniforme) machuca menos.
    #
    # Os seguintes vêm por escolha gulosa: cada um é o que está mais longe
    # do conjunto já escolhido, maximizando a distância ao MAIS PRÓXIMO
    # deles. É o critério usual para espalhar pontos por uma área, e é essa
    # distância que dá aos pontos de conferência poder de detectar uma
    # mudança de inclinação -- dois pontos vizinhos concordariam mesmo com o
    # chip torto, e a conferência não diria nada.
    candidatos = [int(np.argmin(np.hypot(xs - xs.mean(), ys - ys.mean())))]
    desejados = min(QTD_TENTATIVAS_SONDAGEM + QTD_PONTOS_CONFERENCIA, len(pontos))

    while len(candidatos) < desejados:
        distancia_ao_conjunto = np.full(len(pontos), np.inf)

        for k in candidatos:
            distancia_ao_conjunto = np.minimum(
                distancia_ao_conjunto, np.hypot(xs - xs[k], ys - ys[k])
            )

        candidatos.append(int(np.argmax(distancia_ao_conjunto)))

    # --- 1. sondagem -----------------------------------------------------

    fotos_previstas = int(2 * faixa_grossa_um / passo_grosso) + 1

    print(
        f"\n  [Re-ancoragem] medindo quanto o foco andou desde a Fase 2.\n"
        f"    busca grossa de +-{faixa_grossa_um:.0f} µm em passos de "
        f"{passo_grosso:.1f} µm ({fotos_previstas} fotos),\n"
        f"    depois busca fina para o valor preciso."
    )

    i_sondagem = None
    fina = None

    for tentativa, i in enumerate(candidatos[:QTD_TENTATIVAS_SONDAGEM], start=1):
        x, y, z_ancora = pontos[i]

        print(f"    sondagem {tentativa} em (X={x:.0f}, Y={y:.0f})...")

        core.set_xy_position(xy_stage, x, y)
        core.wait_for_device(xy_stage)

        grossa = varrer_faixa_foco(
            core, z_stage, camera, z_ancora, faixa_grossa_um, passo_grosso
        )
        registro["fotos"] += grossa["fotos"]

        if not grossa["aceito"]:
            registro["motivo"] = "busca_grossa_" + grossa["motivo"]

            if grossa["motivo"] == "borda":
                print(
                    f"      o pico ficou na borda dos +-{faixa_grossa_um:.0f} µm "
                    "varridos: o foco está mais\n"
                    "      longe que isso."
                )
            else:
                print(
                    f"      nenhum pico se destacou do platô (proeminência "
                    f"{grossa['proeminencia']:.1f}, mínimo\n"
                    f"      {PROEMINENCIA_MINIMA_GROSSA:.1f}): campo sem estrutura em que "
                    "medir foco, ou foco fora da faixa."
                )

            continue

        # A grossa só diz em que REGIÃO está o foco, com a precisão do passo
        # dela. O Z que vale é o desta busca fina, que tem o refino sub-passo.
        candidata = buscar_foco(
            core, z_stage, camera, grossa["z"], faixa_fina, passo_fino,
            faixa_nominal_um=faixa_fina,
        )
        registro["fotos"] += candidata["fotos"]

        if not candidata["confiavel"]:
            # A grossa achou um pico proeminente e a fina não confirmou. O
            # caso típico é o pico da grossa ser detrito num plano diferente
            # do da amostra. Sem confirmação não há deslocamento em que
            # confiar.
            registro["motivo"] = "busca_fina_" + candidata["motivo"]

            print(
                f"      a grossa achou um pico em Z={grossa['z']:.1f} e a fina não o "
                f"confirmou\n"
                f"      ({candidata['motivo']}): provável detrito num plano diferente do "
                "da amostra."
            )

            continue

        i_sondagem = i
        fina = candidata
        registro["motivo"] = ""
        break

    if i_sondagem is None:
        print(
            f"    [FALHA] nenhuma das {min(QTD_TENTATIVAS_SONDAGEM, len(pontos))} "
            "sondagens achou o foco.\n"
            "            A amostra será varrida com o modelo da Fase 2, SEM correção da\n"
            "            deriva acumulada. Se o motivo foi 'borda' em todas, aumente a\n"
            "            REANCORAGEM_FAIXA_UM; se foi 'platô', o problema é a amostra não\n"
            "            ter estrutura nesses pontos. Confira o autofoco.csv antes de\n"
            "            confiar nas imagens desta amostra."
        )

        return pontos, None, registro

    offset_sondagem = fina["z"] - pontos[i_sondagem][2]

    print(f"    deslocamento medido: {offset_sondagem:+.2f} µm")

    # --- 2. conferência --------------------------------------------------

    # Os pontos de conferência são os próximos da lista que não foram
    # gastos nas tentativas de sondagem.
    conferencia = [i for i in candidatos if i != i_sondagem][:QTD_PONTOS_CONFERENCIA]

    # A faixa da conferência é ALARGADA -- ver FATOR_FAIXA_CONFERENCIA. Ela
    # precisa alcançar uma mudança de forma maior que a faixa normal; se não
    # alcançar, devolve um pico de ruído perto do centro e a discordância
    # que deveria denunciar passa como concordância.
    faixa_conferencia = faixa_fina * FATOR_FAIXA_CONFERENCIA

    offsets = [offset_sondagem]
    medidos = {i_sondagem: fina["z"]}
    conferencias_falhas = 0

    for i in conferencia:
        x, y, z_ancora_k = pontos[i]

        core.set_xy_position(xy_stage, x, y)
        core.wait_for_device(xy_stage)

        # Parte do Z da âncora JÁ somado ao deslocamento da sondagem: se o
        # desvio for mesmo uniforme, o foco está a poucos µm daqui e a
        # tentativa barata resolve em três fotos.
        z_medido, fotos, ok = _medir_foco_ancora(
            core, z_stage, camera, z_ancora_k + offset_sondagem,
            faixa_fina, passo_fino, faixa_conferencia, passo_grosso,
        )
        registro["fotos"] += fotos

        if ok:
            offsets.append(z_medido - z_ancora_k)
            medidos[i] = z_medido

            print(
                f"    conferência em (X={x:.0f}, Y={y:.0f}): "
                f"{z_medido - z_ancora_k:+.2f} µm"
            )
        else:
            conferencias_falhas += 1

            print(
                f"    conferência em (X={x:.0f}, Y={y:.0f}): sem medida boa "
                f"nem em +-{faixa_conferencia:.0f} µm"
            )

    registro["offset_sondagem"] = float(offset_sondagem)
    registro["offsets"] = [float(valor) for valor in offsets]

    # --- 3. decisão ------------------------------------------------------

    tolerancia = FATOR_DISPERSAO_OFFSET * faixa_fina
    dispersao = float(max(offsets) - min(offsets)) if len(offsets) > 1 else None
    registro["dispersao"] = dispersao

    # TODAS as conferências precisam ter dado medida boa. Uma que falha não
    # é uma opinião a menos -- é um sinal de que o foco não estava onde o
    # deslocamento da sondagem previa, mesmo com a faixa alargada. Aceitar o
    # offset uniforme com base nas que sobraram seria ouvir só quem
    # concordou.
    concordam = (
        conferencias_falhas == 0
        and len(offsets) >= 2
        and dispersao <= tolerancia
    )

    if concordam:
        offset = float(np.median(offsets))

        # A geometria não é tocada: os mesmos (x, y), o mesmo espalhamento
        # que a Fase 2 validou e cujo condicionamento ela conferiu. Só a
        # altura muda.
        pontos_novos = [(px, py, pz + offset) for px, py, pz in pontos]

        registro["estrategia"] = "offset_uniforme"
        registro["offset_aplicado"] = offset
        registro["pontos_remedidos"] = len(medidos)

        print(
            f"    -> os {len(offsets)} deslocamentos concordam (dispersão "
            f"{dispersao:.2f} µm <= {tolerancia:.2f} µm).\n"
            f"       Tratado como offset uniforme de {offset:+.2f} µm, somado às "
            f"{len(pontos)} âncoras."
        )

    else:
        if conferencias_falhas:
            print(
                f"    -> {conferencias_falhas} conferência(s) não acharam o foco onde o "
                "deslocamento da\n"
                "       sondagem previa. Não dá para tratar como offset uniforme. "
                "Re-medindo\n"
                "       todas as âncoras."
            )
        else:
            print(
                f"    -> os deslocamentos DISCORDAM (dispersão {dispersao:.2f} µm > "
                f"{tolerancia:.2f} µm).\n"
                "       A forma da superfície mudou, não é só deslocamento. Re-medindo\n"
                "       todas as âncoras."
            )

        offset = float(np.median(offsets))
        pontos_novos = []
        falhas = 0

        for indice, (px, py, pz) in enumerate(pontos):
            if indice in medidos:
                pontos_novos.append((px, py, medidos[indice]))
                continue

            core.set_xy_position(xy_stage, px, py)
            core.wait_for_device(xy_stage)

            # Faixa ampla, pelo mesmo motivo da conferência: se chegamos a
            # este caminho é porque a forma mudou, então o deslocamento
            # comum não basta para prever onde cada âncora foi parar.
            z_medido, fotos, ok = _medir_foco_ancora(
                core, z_stage, camera, pz + offset,
                faixa_fina, passo_fino, faixa_conferencia, passo_grosso,
            )
            registro["fotos"] += fotos

            if ok:
                pontos_novos.append((px, py, z_medido))
                medidos[indice] = z_medido
            else:
                # Sem medida boa, o melhor palpite disponível é a âncora
                # velha somada ao deslocamento conhecido. Descartar o ponto
                # seria pior: ele é parte da geometria que sustenta o ajuste,
                # e tirá-lo pode deixar a matriz mal condicionada.
                pontos_novos.append((px, py, pz + offset))
                falhas += 1

        registro["estrategia"] = "todas_remedidas"
        registro["offset_aplicado"] = offset
        registro["pontos_remedidos"] = len(medidos)
        registro["medidas_falhas"] = falhas

        if falhas:
            print(
                f"       {falhas} de {len(pontos)} âncoras não deram medida boa; ficaram "
                f"com a altura\n"
                f"       antiga somada a {offset:+.2f} µm."
            )

    # --- o modelo novo ---------------------------------------------------

    # Sem qtd_ancoras: aqui TODOS os pontos acabaram de ser medidos, então
    # não há nada a envelhecer. O envelhecimento existe para dar mais voz ao
    # que foi medido há pouco -- e a esta altura isso vale para todos por
    # igual.
    modelo_novo = reajustar_modelo_foco(pontos_novos, tipo_modelo)

    if modelo_novo is None:
        registro["motivo"] = "ajuste_nao_fechou"

        print(
            "    [FALHA] o ajuste com as âncoras re-medidas não fechou. Seguindo com o\n"
            "            modelo da Fase 2."
        )

        return pontos, None, registro

    registro["ok"] = True

    return pontos_novos, modelo_novo, registro
