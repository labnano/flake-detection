import os
import numpy as np
from tifffile import imwrite
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
    SOBREPOSICAO,
)
from comandos import (
    calibrar_plano_foco,
    calibrar_superficie_foco,
    calcular_z,
    coletar_pontos_calibracao,
    gerar_malha_varredura,
    autofocar,
    medir_textura,
    reajustar_modelo_foco,
    salvar_metadados_varredura,
)

predictor = obter_predictor()

# --- CONFIGURAÇÕES DE DIRETÓRIO ---
timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
diretorio_saida = os.path.join(DIRETORIO_SAIDA_BASE, "Deteccao_" + timestamp)
# flakes/  -> só as posições que deram positivo, pra conferir os achados
#             enquanto a varredura ainda está rodando.
# mosaico/ -> TODAS as posições da grade, sem exceção. É a entrada do
#             montar_mosaico.py, que precisa da grade completa pra
#             remontar a amostra.
diretorio_saida_flakes = os.path.join(diretorio_saida, "flakes")
diretorio_saida_mosaico = os.path.join(diretorio_saida, "mosaico")
diretorio_conferencia = os.path.join(diretorio_saida, "conferencia")

os.makedirs(diretorio_saida, exist_ok=True)
os.makedirs(diretorio_conferencia, exist_ok=True)
os.makedirs(diretorio_saida_flakes, exist_ok=True)
os.makedirs(diretorio_saida_mosaico, exist_ok=True)

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

# Textura mínima (ver comandos.medir_textura) pra valer a pena rodar
# autofoco neste campo. Num campo vazio a nitidez mede só o ruído do
# sensor: o autofoco escolheria um Z aleatório, e esse Z entraria no
# modelo de foco e estragaria todas as posições seguintes.
# CALIBRAÇÃO: o valor medido é impresso a cada checkpoint. Rode uma
# varredura curta, veja que números aparecem em campo vazio e em campo
# com flake, e ponha o limiar no meio. Em simulação o vazio ficou ~0,8 e
# o campo com flake ~14 (8 mesmo bem desfocado), daí o 3.0 como partida.
TEXTURA_MINIMA = 3.0


core = Core()

# Identificação de Hardware
camera = core.get_camera_device()
xy_stage = core.get_xy_stage_device()
z_stage = core.get_focus_device()

fov_x, fov_y = FOV_DETECCAO

malha = gerar_malha_varredura(core, xy_stage, fov_x, fov_y, SOBREPOSICAO)
malha_x = malha["malha_x"]
malha_y = malha["malha_y"]
x_inicial = malha["x_inicial"]
y_inicial = malha["y_inicial"]
dist_x = malha["dist_x"]
dist_y = malha["dist_y"]
qtd_passos_x = malha["qtd_passos_x"]
qtd_passos_y = malha["qtd_passos_y"]

# Registra o passo real usado, para o montar_mosaico.py encaixar os tiles
# descontando a faixa sobreposta em vez de duplicá-la.
salvar_metadados_varredura(diretorio_saida, malha, fov_x, fov_y)

# --- ESCOLHA DO METODO DE CALIBRAÇÃO DE FOCO ---
print("\n--- MÉTODO DE CALIBRAÇÃO DE FOCO ---")
print("1. Plano (mais rápido, funciona bem se a amostra for praticamente plana)")
print("2. Superfície curva/quadrática (mais precisa com lentes de magnificações maiores, porém exige mais pontos)")
print("   Onde colocar os pontos não é mais escolha sua: o script calcula as")
print("   posições que dão o melhor ajuste e leva o estágio até cada uma.")
print("   Você só precisa focar (ou chegar perto, se ali não houver estrutura).")

while True:
    metodo = input("Escolha o método (1 ou 2): ").strip()

    if metodo == "1":
        minimo_pontos = 3
        tipo_modelo = "plano"
        break
    elif metodo == "2":
        # A quadrática tem 6 coeficientes, então 6 pontos é o mínimo
        # matemático -- mas com exatamente 6 o ajuste passa por todos eles
        # e reproduz o erro de foco de cada um sem nenhuma média para
        # amortecer. Medido: com 8 pontos o erro máximo ainda fica em ~2 µm;
        # com 10 cai para ~0,9 µm; com 12, ~0,65 µm. Daí o mínimo de 10.
        minimo_pontos = 10
        tipo_modelo = "quadratica"
        break
    else:
        print("[Erro] Digite 1 ou 2.\n")

print(f"Esse método precisa de pelo menos {minimo_pontos} pontos de calibração.\n")

# A malha vai junto porque é dela que saem as posições sugeridas: os pontos
# de calibração precisam cobrir a área que a varredura vai percorrer, não
# uma região qualquer da amostra.
pontos_coletados = coletar_pontos_calibracao(
    core, xy_stage, z_stage, minimo_pontos, malha=malha, tipo=tipo_modelo
)


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

# Offset escalar, usado só como PLANO B: enquanto o re-ajuste do modelo não
# for possível (pontos ainda alinhados na primeira coluna, por exemplo),
# ele segura o foco do jeito antigo. Assim que o modelo consegue ser
# re-ajustado, a correção passa a morar nele e este volta a zero.
ajuste_z = 0.0

# Pontos que alimentam o modelo de foco. Começa com os coletados à mão e
# cresce a cada autofoco confiável -- é isso que faz o modelo melhorar
# sozinho: você começa com 3 pontos "no olho" e termina com dezenas de
# pontos medidos pela própria métrica de nitidez.
pontos_foco = list(pontos_coletados)

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

            # 3. Movimentação Z para o valor previsto e captura.
            # Essa imagem serve para duas coisas: decidir se vale rodar o
            # autofoco aqui e, se não valer, ser a imagem final da posição.
            core.set_position(z_stage, z_previsto)
            core.wait_for_device(z_stage)

            imagem_colorida = capturar_imagem(core, camera)
            z_calculado = z_previsto

            # 4. Checkpoint de foco. O j == 0 existe porque o estágio acabou
            # de voltar de y_max para y_min começando uma coluna nova, e é
            # onde o foco mais se perdia.
            if contador_imagem % INTERVALO_AUTOFOCO == 0 or j == 0:
                textura = medir_textura(imagem_colorida)

                if textura < TEXTURA_MINIMA:
                    # Campo sem estrutura: aqui a nitidez mede ruído, não
                    # foco. Rodar autofoco devolveria um Z aleatório que
                    # contaminaria o modelo -- pior que não fazer nada.
                    print(f"[Autofoco] pulado: campo sem textura ({textura:.2f} < {TEXTURA_MINIMA})")

                else:
                    z_medido, imagem_medida, confiavel = autofocar(
                        core, z_stage, camera, z_previsto, AUTOFOCO_FAIXA_UM, AUTOFOCO_PASSO_UM
                    )

                    if not confiavel:
                        # Melhor Z na ponta da faixa: o foco real está fora
                        # do alcance da busca. Descarta e volta pro previsto.
                        print(
                            f"[Autofoco] descartado: melhor Z na borda da faixa "
                            f"(+-{AUTOFOCO_FAIXA_UM} µm). Foco real fora do alcance."
                        )
                        core.set_position(z_stage, z_calculado)
                        core.wait_for_device(z_stage)

                    else:
                        z_calculado = z_medido
                        imagem_colorida = imagem_medida

                        # Cada autofoco confiável é um ponto (X, Y, Z) medido
                        # de verdade. Entra no conjunto e o modelo é
                        # re-ajustado com tudo que se sabe até agora.
                        pontos_foco.append((n, m, z_calculado))
                        modelo_novo = reajustar_modelo_foco(pontos_foco, modelo_foco["tipo"])

                        if modelo_novo is not None:
                            modelo_foco = modelo_novo
                            # A correção agora mora no modelo; manter o
                            # offset também seria contá-la duas vezes.
                            ajuste_z = 0.0
                            print(
                                f"[Autofoco] Z medido {z_calculado:.3f} | textura {textura:.2f} | "
                                f"modelo re-ajustado com {len(pontos_foco)} pontos"
                            )
                        else:
                            # Re-ajuste ainda não é possível (pontos alinhados,
                            # por exemplo). Cai no offset escalar de antes.
                            ajuste_z = z_calculado - calcular_z(n, m, modelo_foco)
                            print(
                                f"[Autofoco] Z medido {z_calculado:.3f} | textura {textura:.2f} | "
                                f"modelo não re-ajustado, usando offset {ajuste_z:+.3f}"
                            )

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
                caminho_flake = os.path.join(diretorio_saida_flakes, nome_arquivo)
                caminho_mosaico = os.path.join(diretorio_saida_mosaico, nome_arquivo)

                nome_arquivo_flakes = f"flakes_pos_X{i}_Y{j}.tif"
                caminho_completo_flakes = os.path.join(diretorio_saida_flakes, nome_arquivo_flakes)

                # Salva a imagem original nos dois lugares de propósito: em
                # flakes/ pra análise imediata, e em mosaico/ porque toda
                # posição da grade precisa estar lá pro mosaico fechar.
                imwrite(caminho_flake, imagem_colorida)
                imwrite(caminho_mosaico, imagem_colorida)

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
                caminho_mosaico = os.path.join(diretorio_saida_mosaico, nome_arquivo)
                imwrite(caminho_mosaico, imagem_colorida)

    print("\nVarredura totalmente concluída.")

    # A montagem do mosaico NÃO acontece aqui de propósito: ela é a etapa
    # mais pesada em memória do pipeline, e rodá-la dentro deste processo
    # (que ainda tem o modelo do MaskTerial carregado) significaria perder
    # tudo caso ela falhe, depois de horas de microscópio. Como os tiles já
    # estão todos salvos em disco, ela virou um script à parte, repetível.
    print(f"\nImagens da grade completa em: {diretorio_saida_mosaico}")
    print("Para montar o mosaico, rode:")
    print(f'    python montar_mosaico.py "{diretorio_saida}"')

finally:
    # 7. Retorno à origem, sempre executado (mesmo se algo tiver falhado acima).
    print("\nRetornando o estágio à posição inicial...")
    core.set_xy_position(xy_stage, x_inicial, y_inicial)
    core.wait_for_device(xy_stage)
