import os
import sys
import numpy as np
from tifffile import imwrite, imread
from pycromanager import Core
import datetime
from caminhos import DIRETORIO_SAIDA_BASE
from parametros import FOV_MOSAICO, EXPOSICAO_MOSAICO
from capturar_imagem import capturar_imagem
from comandos import calibrar_plano_foco, calcular_z, coletar_pontos_calibracao, gerar_malha_varredura


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

malha = gerar_malha_varredura(core, xy_stage, fov_x, fov_y)
malha_x = malha["malha_x"]
malha_y = malha["malha_y"]
x_inicial = malha["x_inicial"]
y_inicial = malha["y_inicial"]
dist_x = malha["dist_x"]
dist_y = malha["dist_y"]
qtd_passos_x = malha["qtd_passos_x"]
qtd_passos_y = malha["qtd_passos_y"]

minimo_pontos = 3
pontos_coletados = coletar_pontos_calibracao(core, xy_stage, z_stage, minimo_pontos)

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
try:
    for i, n in enumerate(malha_x):
        for j, m in enumerate(malha_y):
            # 1. Movimentação
            core.set_xy_position(xy_stage, n, m)
            core.wait_for_device(xy_stage)
            core.set_position(z_stage, calcular_z(n, m, modelo_foco))
            core.wait_for_device(z_stage)
            # 2. Captura dos dados brutos, já organizados em RGB
            imagem_colorida = capturar_imagem(core, camera)

            # 5. Define o nome do arquivo dinamicamente
            nome_arquivo = f"img_pos_X{i}_Y{j}.tif"
            caminho_completo = os.path.join(diretorio_saida, nome_arquivo)

            # 6. Salva a matriz no formato nativo, sem corromper os bits
            imwrite(caminho_completo, imagem_colorida)
            print(f"Salvo: {nome_arquivo}")

    print("Escaneamento totalmente concluído.")

finally:
    # 7. Retorno à origem, sempre executado (mesmo se algo tiver falhado acima).
    print("Retornando o estágio à posição inicial...")
    core.set_xy_position(xy_stage, x_inicial, y_inicial)
    core.wait_for_device(xy_stage)

GRID_X = len(malha_x)  # Número de posições no eixo X
GRID_Y = len(malha_y) # Número de posições no eixo Y

print("Iniciando montagem do mosaico...")

linhas_do_mosaico = []

# 1. Monta o mosaico linha por linha (Eixo Y)
for j in range(GRID_Y):
    imagens_da_linha = []
    
    # 2. Pega todas as colunas daquela linha (Eixo X)
    for i in range(GRID_X):
        nome_arquivo = f"img_pos_X{i}_Y{j}.tif"
        caminho = os.path.join(diretorio_saida, nome_arquivo)
        
        try:
            # Lê a imagem do disco
            img = imread(caminho)
            imagens_da_linha.append(img)
        except FileNotFoundError:
            print(f"ERRO: Arquivo {caminho} não encontrado!")
            sys.exit()
            
    # 3. Gruda as imagens horizontalmente (lado a lado)
    linha_montada = np.hstack(imagens_da_linha)
    linhas_do_mosaico.append(linha_montada)

# 4. Gruda todas as linhas verticalmente (uma em cima da outra)
mosaico_final = np.vstack(linhas_do_mosaico)

# 5. Salva o super arquivo final
caminho_salvamento = os.path.join(diretorio_saida, f"mosaico_final_{timestamp}.tif")
imwrite(caminho_salvamento, mosaico_final)

print(f"Sucesso! Mosaico salvo com dimensões: {mosaico_final.shape}")