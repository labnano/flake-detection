import cv2
import tifffile

CAMINHO = r"C:\Users\Pedro Guedes\Desktop\Faculdade\2026.2\IC\TesteDeteccao_28-08-2026_00-05-25\mosaico_final.tif"

N = 4  # cada eixo fica N vezes menor

img = tifffile.imread(CAMINHO)
altura, largura = img.shape[:2]          # shape vem como (altura, largura)
print(f"original: {largura} x {altura}")

# // é divisão inteira (descarta o resto). Se a dimensão não for múltipla de
# N, perde-se alguns pixels na borda -- irrelevante aqui.
nova_largura = largura // N
nova_altura = altura // N

# ATENÇÃO à ordem: resize quer (largura, altura), o inverso do shape.
reduzida = cv2.resize(img, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

saida = CAMINHO.replace(".tif", f"_reduzido_{N}x.tif")
tifffile.imwrite(saida, reduzida)
print(f"reduzido: {nova_largura} x {nova_altura} -> comando: &\"{saida}\"")