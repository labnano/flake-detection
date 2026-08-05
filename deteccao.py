import os
import numpy as np
import sys
from tifffile import imwrite, imread
from pycromanager import Core
import datetime
import csv
from modelo import obter_predictor, display_results
from capturar_imagem import capturar_imagem
from caminhos import DIRETORIO_SAIDA_BASE
from parametros import (
    FOV_DETECCAO,
    EXPOSICAO_DETECCAO,
    WHITE_BALANCE_RED,
    WHITE_BALANCE_BLUE,
)
from comandos import (
    calibrar_plano_foco,
    calibrar_superficie_foco,
    calcular_z,
    coletar_pontos_calibracao,
    gerar_malha_varredura,
    autofocar,
)

predictor = obter_predictor()

# --- CONFIGURAÇÕES DE DIRETÓRIO ---
timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
diretorio_saida = os.path.join(DIRETORIO_SAIDA_BASE, "Deteccao_" + timestamp)
diretorio_saida_flakes = os.path.join(diretorio_saida, "flakes")
diretorio_conferencia = os.path.join(diretorio_saida, "conferencia")

os.makedirs(diretorio_saida, exist_ok=True)
os.makedirs(diretorio_conferencia, exist_ok=True)
os.makedirs(diretorio_saida_flakes, exist_ok=True)

arquivo_log = os.path.join(diretorio_saida, "coordenadas_flakes_" + timestamp + ".csv")

with open(arquivo_log, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Nome_Arquivo", "X", "Y", "Z", "Qtd_Flakes", "caminho"])

# --- CONFIGURAÇÕES DE AUTOFOCO (parte B) ---
# A cada INTERVALO_AUTOFOCO imagens, em vez de confiar só no modelo de
# foco (plano/superfície), o script tira um pequeno "stack" de fotos em
# Z ao redor do valor previsto e fica com a mais nítida.
INTERVALO_AUTOFOCO = 15
AUTOFOCO_FAIXA_UM = 5.0   # quantos µm pra cada lado do Z previsto ele varre
AUTOFOCO_PASSO_UM = 0.5   # espaçamento entre as fotos do stack


core = Core()

# Identificação de Hardware
camera = core.get_camera_device()
xy_stage = core.get_xy_stage_device()
z_stage = core.get_focus_device()

fov_x, fov_y = FOV_DETECCAO

malha = gerar_malha_varredura(core, xy_stage, fov_x, fov_y)
malha_x = malha["malha_x"]
malha_y = malha["malha_y"]
x_inicial = malha["x_inicial"]
y_inicial = malha["y_inicial"]
dist_x = malha["dist_x"]
dist_y = malha["dist_y"]
qtd_passos_x = malha["qtd_passos_x"]
qtd_passos_y = malha["qtd_passos_y"]

# --- ESCOLHA DO METODO DE CALIBRAÇÃO DE FOCO ---
print("\n--- MÉTODO DE CALIBRAÇÃO DE FOCO ---")
print("1. Plano (mais rápido, funciona bem se a amostra for praticamente plana)")
print("2. Superfície curva/quadrática (mais precisa com lentes de magnificações maiores, porém exige mais pontos)")

while True:
    metodo = input("Escolha o método (1 ou 2): ").strip()

    if metodo == "1":
        minimo_pontos = 3
        break
    elif metodo == "2":
        minimo_pontos = 6
        break
    else:
        print("[Erro] Digite 1 ou 2.\n")

print(f"Esse método precisa de pelo menos {minimo_pontos} pontos de calibração.\n")

pontos_coletados = coletar_pontos_calibracao(core, xy_stage, z_stage, minimo_pontos)


if metodo == "1":
    modelo_foco = calibrar_plano_foco(pontos_coletados)
else:
    modelo_foco = calibrar_superficie_foco(pontos_coletados)

print("\n[Sucesso] Modelo de foco calculado!")


# ----- CÁLCULO DOS VALORES DE Z DA VARREDURA -----
valores_z = []

for n in malha_x:
    for m in malha_y:
        z_estimado = calcular_z(n, m, modelo_foco)
        valores_z.append(z_estimado)

if core.is_sequence_running():
    core.stop_sequence_acquisition()


print(f"\nÁrea a ser mapeada: {dist_x:.2f} x {dist_y:.2f} µm")
print(f"Grade gerada: {qtd_passos_x} x {qtd_passos_y} imagens. Total: {qtd_passos_x * qtd_passos_y}")

print("\n--- FAIXA DE FOCO ESTIMADA ---")
print(f"Maior Z calculado: {max(valores_z):.3f}")
print(f"Menor Z calculado: {min(valores_z):.3f}")

input("Se estes valores forem seguros, aperte ENTER para iniciar a varredura...")

print(f"\nIniciando captura. Arquivos serão salvos em: {diretorio_saida}")

core.set_exposure(EXPOSICAO_DETECCAO)
core.set_property(camera, "WhiteBalanceBlue", WHITE_BALANCE_BLUE)
core.set_property(camera, "WhiteBalanceRed", WHITE_BALANCE_RED)

# --- VARREDURA ---
contador_imagem = 0
ajuste_z = 0.0  #correção aprendida pelo autofoco

try:
    for i, n in enumerate(malha_x):
        for j, m in enumerate(malha_y):
            contador_imagem += 1

            # 1. Calcula o foco previsto para a posição atual
            z_previsto = calcular_z(n, m, modelo_foco) + ajuste_z

            if not np.isfinite(z_previsto):
                raise ValueError(f"Z calculado inválido em X={n}, Y={m}: Z={z_previsto}")

            print(f"\nIndo para X={n:.2f}, Y={m:.2f}, Z previsto={z_previsto:.3f}")

            # 2. Movimentação XY
            core.set_xy_position(xy_stage, n, m)
            core.wait_for_device(xy_stage)

            if contador_imagem % INTERVALO_AUTOFOCO == 0 or j == 0:
                # 3-4. A cada {INTERVALO_AUTOFOCO} imagens, refina o foco:
                # Tira um pequeno stack em Z e fica com a mais nítida.
                z_calculado, imagem_colorida = autofocar(
                    core, z_stage, camera, z_previsto, AUTOFOCO_FAIXA_UM, AUTOFOCO_PASSO_UM
                )

                # A diferença entre o Z realmente mais nítido e o que o
                # modelo (sem ajuste) previa vira o novo ajuste, e passa a
                # valer para as próximas posições, até o próximo checkpoint.
                ajuste_z = z_calculado - calcular_z(n, m, modelo_foco)

                print(
                    f"[Autofoco] Z ajustado para {z_calculado:.3f} "
                    f"(ajuste acumulado: {ajuste_z:+.3f})"
                )
            else:
                # 3. Movimentação Z direto pro valor previsto
                core.set_position(z_stage, z_previsto)
                core.wait_for_device(z_stage)

                # 4. Captura da imagem
                imagem_colorida = capturar_imagem(core, camera)
                z_calculado = z_previsto

            if j % 20 == 0:
                nome_arquivo = f"img_pos_X{i}_Y{j}.tif"
                caminho_completo = os.path.join(diretorio_conferencia, nome_arquivo)

                # 5. Salva a matriz no formato nativo, sem corromper os bits.
                # imagem_colorida já é RGB (confirmado comparando um arquivo
                imwrite(caminho_completo, imagem_colorida)
                print(f"Salvo arquivo de conferência: {caminho_completo}")

            # 6. Detecção
            # O MaskTerial espera BGR (ver FlakeClass.py), então convertemos
            # antes de mandar pro modelo. ascontiguousarray é necessário porque
            # a inversão de canais [:, :, ::-1] gera um array com stride
            # negativo, que o torch.tensor() usado dentro do MaskTerial não
            # aceita (ValueError: "tensors with negative strides").
            imagem_bgr = np.ascontiguousarray(imagem_colorida[:, :, ::-1])
            flakes = predictor.predict(imagem_bgr)

            if flakes:
                nome_arquivo = f"img_pos_X{i}_Y{j}.tif"
                caminho_completo = os.path.join(diretorio_saida_flakes, nome_arquivo)
                caminho_incompleto = os.path.join(diretorio_saida, nome_arquivo)

                nome_arquivo_flakes = f"flakes_pos_X{i}_Y{j}.tif"
                caminho_completo_flakes = os.path.join(diretorio_saida_flakes, nome_arquivo_flakes)

                # Salva imagem original
                imwrite(caminho_completo, imagem_colorida)
                imwrite(caminho_incompleto, imagem_colorida)

                # Salva imagem com flakes marcados.
                # display_results também espera BGR (mesmo motivo do predict acima).
                imagem_flakes = display_results(imagem_bgr, flakes)
                imwrite(caminho_completo_flakes, imagem_flakes[:, :, ::-1])

                # Registra no CSV
                with open(arquivo_log, mode='a', newline='', encoding='utf-8-sig') as file:
                    writer = csv.writer(file, delimiter=';')
                    writer.writerow([nome_arquivo, n, m, z_calculado, len(flakes), f"{diretorio_saida_flakes}"])

                print(
                    f"SUCESSO: {len(flakes)} flake(s) em "
                    f"X={n:.1f}, Y={m:.1f}, Z={z_calculado:.3f}. "
                    f"Salvo como {nome_arquivo}"
                )

            else:
                print(f"Sem flakes em X={n:.1f}, Y={m:.1f}, Z={z_calculado:.3f}.")
                nome_arquivo = f"img_pos_X{i}_Y{j}.tif"
                caminho_completo = os.path.join(diretorio_saida, nome_arquivo)
                imwrite(caminho_completo, imagem_colorida)

    print("\nVarredura totalmente concluída.")


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

finally:
    # 7. Retorno à origem, sempre executado (mesmo se algo tiver falhado acima).
    print("\nRetornando o estágio à posição inicial...")
    core.set_xy_position(xy_stage, x_inicial, y_inicial)
    core.wait_for_device(xy_stage)
