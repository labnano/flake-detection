import os
from tifffile import imwrite
from pycromanager import Core
import datetime
from caminhos import DIRETORIO_SAIDA_BASE
from parametros import FOV_MOSAICO, EXPOSICAO_MOSAICO, SOBREPOSICAO
from capturar_imagem import capturar_imagem
from comandos import (
    calibrar_plano_foco,
    calcular_z,
    coletar_pontos_calibracao,
    gerar_malha_varredura,
    autofocar,
    salvar_metadados_varredura,
)

# --- CONFIGURAÇÕES DE AUTOFOCO ---
# A cada INTERVALO_AUTOFOCO imagens, em vez de confiar só no plano de foco
# (ajustado com o mínimo de 3 pontos), o script tira um pequeno "stack" de
# fotos em Z ao redor do valor previsto e fica com a mais nítida -- mesma
# ideia já usada no deteccao.py.
# Os valores abaixo foram copiados do deteccao.py como ponto de partida --
# como a objetiva do mosaico tem magnificação diferente (FOV bem maior,
# ver FOV_MOSAICO), confira/ajuste FAIXA_UM e PASSO_UM pra profundidade de
# campo real dela antes de rodar uma varredura grande.
INTERVALO_AUTOFOCO = 20
AUTOFOCO_FAIXA_UM = 3.0   # quantos µm pra cada lado do Z previsto ele varre
AUTOFOCO_PASSO_UM = 0.5   # espaçamento entre as fotos do stack


# --- CONFIGURAÇÕES DE DIRETÓRIO ---
timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
diretorio_saida = os.path.join(DIRETORIO_SAIDA_BASE, "Varredura_" + timestamp)
os.makedirs(diretorio_saida, exist_ok = True)

core = Core()

# Identificação de Hardware
camera = core.get_camera_device()
xy_stage = core.get_xy_stage_device()
z_stage = core.get_focus_device()

fov_x, fov_y = FOV_MOSAICO

malha = gerar_malha_varredura(core, xy_stage, fov_x, fov_y, SOBREPOSICAO)
malha_x = malha["malha_x"]
malha_y = malha["malha_y"]
x_inicial = malha["x_inicial"]
y_inicial = malha["y_inicial"]
dist_x = malha["dist_x"]
dist_y = malha["dist_y"]
qtd_passos_x = malha["qtd_passos_x"]
qtd_passos_y = malha["qtd_passos_y"]

salvar_metadados_varredura(diretorio_saida, malha, fov_x, fov_y)

minimo_pontos = 3

# A malha vai junto porque é dela que saem as posições sugeridas: os pontos
# de calibração precisam cobrir a área que a varredura vai percorrer, não
# uma região qualquer da amostra. Aqui o modelo é sempre o plano.
pontos_coletados = coletar_pontos_calibracao(
    core, xy_stage, z_stage, minimo_pontos, malha=malha, tipo="plano"
)

modelo_foco = calibrar_plano_foco(pontos_coletados)
print("\n[Sucesso] Plano de foco calculado!")

#-----calculo do maior z-----
valores_z = []
for i, n in enumerate(malha_x):
    for j, m in enumerate(malha_y):
        valores_z.append(calcular_z(n, m, modelo_foco))


# Segurança: para o Live mode se estiver rodando
if core.is_sequence_running():
    core.stop_sequence_acquisition()

print(f"\nÁrea a ser mapeada: {dist_x:.2f} x {dist_y:.2f} µm")
print(f"Grade gerada: {qtd_passos_x} x {qtd_passos_y} imagens (Total: {qtd_passos_x * qtd_passos_y})")
print(f"Maior Z calculado: {max(valores_z):.3f}")
print(f"Menor Z calculado: {min(valores_z):.3f}")
input("Se estes valores forem seguros, aperte ENTER para iniciar a varredura...")
print(f"Iniciando captura. Arquivos serão salvos em: {diretorio_saida}")


core.set_exposure(EXPOSICAO_MOSAICO)

# O laço inteiro fica dentro de um try/finally: se der erro no meio da
# varredura (posição inválida, câmera travou, etc.), o estágio ainda assim
# volta para a posição inicial, em vez de ficar parado onde quebrou.
contador_imagem = 0
ajuste_z = 0.0  # correção aprendida pelo autofoco

try:
    for i, n in enumerate(malha_x):
        for j, m in enumerate(malha_y):
            contador_imagem += 1

            # 1. Movimentação XY
            core.set_xy_position(xy_stage, n, m)
            core.wait_for_device(xy_stage)

            z_previsto = calcular_z(n, m, modelo_foco) + ajuste_z

            if contador_imagem % INTERVALO_AUTOFOCO == 0:
                # 2-3. A cada INTERVALO_AUTOFOCO imagens, refina o foco:
                # tira um pequeno stack em Z ao redor do previsto e fica
                # com a mais nítida, corrigindo o que o plano de foco
                # (calibrado com só 3 pontos) não capturou.
                z_calculado, imagem_colorida, confiavel = autofocar(
                    core, z_stage, camera, z_previsto, AUTOFOCO_FAIXA_UM, AUTOFOCO_PASSO_UM
                )

                if confiavel:
                    # A diferença entre o Z realmente mais nítido e o que o
                    # modelo (sem ajuste) previa vira o novo ajuste, e passa a
                    # valer para as próximas posições, até o próximo checkpoint.
                    ajuste_z = z_calculado - calcular_z(n, m, modelo_foco)

                    print(
                        f"[Autofoco] Z ajustado para {z_calculado:.3f} "
                        f"(ajuste acumulado: {ajuste_z:+.3f})"
                    )
                else:
                    # Melhor Z na ponta da faixa: o foco real está fora do
                    # alcance da busca. Aceitar esse valor como ajuste
                    # estragaria todas as posições seguintes.
                    print(
                        f"[Autofoco] descartado: melhor Z na borda da faixa "
                        f"(+-{AUTOFOCO_FAIXA_UM} µm)."
                    )
            else:
                # 2. Movimentação Z direto pro valor previsto
                core.set_position(z_stage, z_previsto)
                core.wait_for_device(z_stage)

                # 3. Captura dos dados brutos, já organizados em RGB
                imagem_colorida = capturar_imagem(core, camera)

            # 4. Define o nome do arquivo dinamicamente
            nome_arquivo = f"img_pos_X{i}_Y{j}.tif"
            caminho_completo = os.path.join(diretorio_saida, nome_arquivo)

            # 5. Salva a matriz no formato nativo, sem corromper os bits
            imwrite(caminho_completo, imagem_colorida)
            print(f"Salvo: {nome_arquivo}")

    print("Escaneamento totalmente concluído.")

finally:
    # 7. Retorno à origem, sempre executado (mesmo se algo tiver falhado acima).
    print("Retornando o estágio à posição inicial...")
    core.set_xy_position(xy_stage, x_inicial, y_inicial)
    core.wait_for_device(xy_stage)

# A montagem do mosaico ficou no montar_mosaico.py, por dois motivos.
#
# O primeiro é memória: empilhar a varredura inteira com hstack/vstack exige
# o mosaico todo na RAM de uma vez, e uma falha aqui jogaria fora horas de
# microscópio. O montar_mosaico.py grava direto em disco (memmap) e, por ser
# um script à parte, pode ser repetido sem repetir a varredura.
#
# O segundo é correção: com SOBREPOSICAO > 0 os tiles NÃO são adjacentes, e
# colá-los lado a lado duplicaria a faixa compartilhada, esticando o mosaico.
# O montar_mosaico.py lê o varredura.json gravado acima e recorta cada tile
# ao passo real antes de encaixar.
print(f"\nImagens da grade completa em: {diretorio_saida}")
print("Para montar o mosaico, rode:")
print(f'    python montar_mosaico.py "{diretorio_saida}"')